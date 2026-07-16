"""NanoNMR-M experiments.

This module implements a collection of NV/NNMR experiment routines used by
the NanoNMR-M setup. It contains the :class:`SpinMeasurements` class with
helpers for digitizer configuration, pulse sequence control, and a set of
experiment implementations (ODMR, Rabi, T1/T2, DEER, CASR, etc.).

Added lifecycle management via the @managed_experiment decorator, which standardizes
the setup and teardown of instruments, as well as error handling and status reporting.

Author: Evan Villafranca
Updated: 2026-03-23
"""
from __future__ import annotations

import functools
import logging
import math
import time
import warnings
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from digitizer_driver_2026_03_16 import SpectrumDigitizer
from nspyre import (
    DataSource, 
    InstrumentManager,
    StreamingList, 
    experiment_widget_process_queue,
)
from pulsestreamer import NextAction, PulseStreamer, When
from saveUtils import flexSave
import nv_dataclasses_2026_03_05 as nvcfg
# import nv_data_fitting_2026_03_24 as nvfit

_logger = logging.getLogger(__name__)
SUBSEQ_COUNT = {
    "DQ": 4,
    "DEER": 4,
    "DEER T1": 4,
    "DEER T2": 2,
    "MW T1": 2,
    "CD": 6,
    "CASR_alt": 2,
    "Cal": 1,
    "Opt T1": 1,
    "Pulsed ODMR RF": 4,
    "T2 RF": 4,
}

def format_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def format_minutes_seconds(seconds: float) -> str:
    seconds = int(round(seconds))

    if seconds < 60:
        return f"{seconds} sec"

    minutes, sec = divmod(seconds, 60)

    if minutes < 60:
        return f"{minutes} min {sec} sec"

    hours, minutes = divmod(minutes, 60)

    if hours < 24:
        return f"{hours} hr {minutes} min"

    days, hours = divmod(hours, 24)
    return f"{days} d {hours} hr"

def run_save(data_name, file_name, directory, file_format="json", seq=None):
    _logger.info("Saving file with flexSave...")
    if seq is not None:
        exp_name = f"{data_name}_{seq.lower()}"
    else:
        exp_name = data_name
    flexSave(datasetName=data_name, expType=exp_name, filename=file_name, dirs=directory, file_format=file_format)

@contextmanager
def _exclusive_ps(ps, token: str):
    ps.begin_exclusive(token, takeover=True, restore_on_release=True)
    try:
        yield
    finally:
        # Release exclusive control no matter how we exit
        try:
            ps.end_exclusive(token)
        except Exception as e:
            _logger.exception("ps.end_exclusive failed")

@contextmanager
def _rf_on(sig_gen):
    sig_gen.set_rf_toggle(1)
    try:
        yield
    finally:
        try:
            sig_gen.set_rf_toggle(0)
        except Exception as e:
            _logger.exception("sig_gen.set_rf_toggle(0) failed")

def _hdawg_seq_name(drive_type: str) -> str:
    return "DEER CD" if drive_type == "Continuous" else "DEER"

def _format_exception_msg(e: Exception) -> str:
    """Format an exception with its type and message for user display.
    
    Also logs the full traceback to help with debugging and prints to stderr.
    """
    import sys
    import traceback
    _logger.error(f"Exception occurred: {type(e).__name__}")
    
    # Extract useful info from exception
    exc_str = str(e) if str(e) else repr(e)
    if hasattr(e, 'args') and e.args:
        exc_args = "; ".join(str(arg) for arg in e.args)
    else:
        exc_args = ""
    
    # Print to stderr so user sees it immediately in terminal
    print(f"\n{'='*70}", file=sys.stderr, flush=True)
    print(f"❌ EXPERIMENT ERROR: {type(e).__name__}", file=sys.stderr, flush=True)
    print(f"{'='*70}", file=sys.stderr, flush=True)
    if exc_str:
        print(f"Message: {exc_str}", file=sys.stderr, flush=True)
    if exc_args:
        print(f"Args: {exc_args}", file=sys.stderr, flush=True)
    print(f"\nFull Traceback:", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr, flush=True)
    
    # Combine all info for return value
    if exc_str:
        msg = f"{type(e).__name__}: {exc_str}"
    elif exc_args:
        msg = f"{type(e).__name__}: {exc_args}"
    else:
        msg = f"{type(e).__name__}"
    
    return msg

def managed_experiment(*, token_prefix: str, dataset_key: str = "dataset"):
    """Decorator to run an experiment inside the standard lifecycle.

    Decorated methods should accept injected handles:
        def scan(self, *, mgr, data, pl_data, token, **kwargs): ...

    Public call remains: scan(**kwargs)
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            if dataset_key not in kwargs:
                raise KeyError(f"Missing required kwarg '{dataset_key}' for {fn.__name__}")
            
            dataset = kwargs[dataset_key]

            def body(*, mgr, data, pl_data, token):
                try:
                    return fn(self, mgr=mgr, data=data, pl_data=pl_data, token=token, **kwargs)
                except Exception as e:
                    # Print exception to stderr immediately
                    _format_exception_msg(e)
                    # Re-raise to be caught by run_experiment
                    raise

            enable_pl_trace = kwargs.pop("enable_pl_trace", False)

            return self.run_experiment(
                        dataset=dataset, 
                        token_prefix=token_prefix, 
                        body=body,
                        enable_pl_trace=enable_pl_trace,
            )

        return wrapped
    return decorator

################################
### --- Experiment class --- ###
################################

class SpinMeasurements:
    """NanoNMR-M experiments"""
    def __init__(self, queue_to_exp=None, queue_from_exp=None, queue_to_inst=None):
        """
        Args:
            queue_to_exp: A multiprocessing Queue object used to send messages
                to the experiment from the GUI.
            queue_from_exp: A multiprocessing Queue object used to send messages
                to the GUI from the experiment.
        """
        self.queue_to_exp = queue_to_exp
        self.queue_from_exp = queue_from_exp
        self.queue_to_inst = queue_to_inst

        self.dig = SpectrumDigitizer('dev/spcm0') # instantiate digitizer for high-speed data acquisition

    @staticmethod
    def choose_sideband(opt, nv_freq, side_freq, pulse_axis='x') -> tuple[float, list[float]]:
        """Return signal generator frequency & IQ phase values
        
        Assigns signal generator frequency based on sideband option and pulse phase
        """
        match pulse_axis:
            case 'y':
                delta = 90
            case _: 
                delta = 0

        match opt:
            case 'Upper':
                frequency = nv_freq - side_freq
                iq_phases = [delta+90, delta+0]
            case 'Both':
                frequency = nv_freq - side_freq
                iq_phases = [delta+0, delta+90, delta+90, delta+0] # lower sideband phases + upper sideband phases
            case _:
                frequency = nv_freq + side_freq
                iq_phases = [delta+0, delta+90]

        return frequency, iq_phases
    
    @staticmethod
    def _build_pi_times(pi_s: float) -> tuple[list[float], list[float]]:
        # returns ns lists like your current code expects
        pi = [pi_s * 1e9, pi_s * 1e9]
        pi_half = [0.5 * pi_s * 1e9, 0.5 * pi_s * 1e9]
        return pi, pi_half

    @staticmethod
    def _configure_sig_gen_iq(sig_gen, *, carrier_freq: float, rf_power: float) -> None:
        sig_gen.set_frequency(carrier_freq)
        sig_gen.set_rf_amplitude(rf_power)
        sig_gen.set_mod_type(7)            # QAM
        sig_gen.set_mod_subtype(1)         # no constellation mapping
        sig_gen.set_mod_function("IQ", 5)  # external modulation
        sig_gen.set_mod_toggle(1)

    @staticmethod
    def digitizer_configure(**kwargs) -> Dict[str, Any]:
        match kwargs['coupling']:
            case 'AC':
                acdc = 1
            case _:
                acdc = 0

        match kwargs['termination']:
            case '1M':
                term = 0
            case _:
                term = 1

        num_pts_in_exp = SUBSEQ_COUNT.get(kwargs['exp_type'], 2) * kwargs['num_pts']
        
        dig_config = {'num_pts_in_exp': num_pts_in_exp, # includes all subsequences
                      'num_iters': kwargs['iters'], # number of exp. iterations
                      'segment_size': kwargs['segment_size'],
                      'sampling_frequency': kwargs['sampling_freq']/1e9, # digitizer uses [GHz]
                      'AMP': int(kwargs['dig_amplitude']*1000), # digitizer uses [mV]
                      'readout_ch': int(kwargs['read_channel']),
                      'both_ch': kwargs['both_channels'],
                      'ACCOUPLE': acdc,
                      'HF_INPUT_50OHM': term,
                      'card_timeout': kwargs['dig_timeout'],
                      'pretrig_size': kwargs['pretrig_size'],
                      'runs': kwargs['runs']}
        
        return dig_config
    
    @staticmethod
    def _mag(x):
        if isinstance(x, tuple):
            return tuple(SpinMeasurements._mag(xx) for xx in x)
        if isinstance(x, list):
            return [SpinMeasurements._mag(xx) for xx in x]
        return x.magnitude if hasattr(x, "magnitude") else x

    def parse_analog_math(self, array, exp_type, pts):
        """Average point-interleaved analog data across runs for one or two channels.

        Parameters
        ----------
        array : array-like or tuple/list of array-like
            Single channel:
                - (num_segments,) scalar-per-segment data, or
                - (num_segments, segment_size) waveform data
            Dual channel:
                - (ch0, ch1), where each is either
                (num_segments,) or (num_segments, segment_size)
        exp_type : str
            Experiment type used to determine subsequence count.
        pts : int
            Number of experiment points per run.

        Returns
        -------
        Single channel:
            [sub0, sub1, ..., sub(n-1)]
        Dual channel:
            [
                [ch0_sub0, ch0_sub1, ..., ch0_sub(n-1)],
                [ch1_sub0, ch1_sub1, ..., ch1_sub(n-1)],
            ]

        Notes
        -----
        Expected scalar stream ordering within each channel:
            for each run:
                [pt0_sub0, pt0_sub1, ..., pt0_sub(n-1),
                pt1_sub0, pt1_sub1, ..., pt1_sub(n-1),
                ...
                pt(pts-1)_sub(n-1)]
        """
        n = SUBSEQ_COUNT.get(exp_type, 2)

        # Normalize input into a list of channels
        is_multi_channel = (
            isinstance(array, (tuple, list))
            and len(array) == 2
            and all(hasattr(ch, "shape") or hasattr(ch, "__array__") for ch in array)
        )
        channels = list(array) if is_multi_channel else [array]

        out_channels = []

        for ch in channels:
            # strip units, convert to ndarray
            arr = np.asarray(self._mag(ch), dtype=float)

            # Handle digitizer single-channel raw shape: (segments, digitizer samples, 1)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
    
            # if waveform data, reduce each segment to one scalar
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            elif arr.ndim != 1:
                raise ValueError(f"Expected 1D or 2D input per channel, got shape {arr.shape}")

            arr = arr.ravel() # dim (n * runs * pts,)

            block = n * pts
            size = arr.size

            if size % block != 0:
                raise ValueError(
                    f"Input length {size} not divisible by n*pts={block} "
                    f"(n={n}, pts={pts})"
                )

            runs = size // block

            # Shape: (runs*pts, n)
            per_point = arr.reshape(runs * pts, n)

            ch_out = []
            for j in range(n):
                avg = per_point[:, j].reshape(runs, pts).mean(axis=0) # dim (pts,)
                ch_out.append(avg)

            out_channels.append(ch_out)

        # Return single-channel legacy shape
        if len(out_channels) == 1:
            return out_channels[0] # list of [sub0, sub1, ..., sub(n-1)], each of shape (pts,)

        return out_channels # list of [[ch0_sub0, ch0_sub1, ..., ch0_sub(n-1)], [ch1_sub0, ch1_sub1, ..., ch1_sub(n-1)]], each of shape (pts,)

    def analog_pl_math(self, array, exp_type, time_pt, pts) -> np.ndarray:
        """ Analog math for extracting PL time traces from digitizer acquired data.
        """
        arr = np.asarray(self._mag(array)) # dim (n * runs * pts, dig samples, 1)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0] # dim (n * runs * pts, dig samples)

        n = SUBSEQ_COUNT.get(exp_type, 2)

        total_rows, num_dig_samples = arr.shape
        runs = total_rows // (n * pts)

        arr = arr.reshape(runs, pts, n, num_dig_samples) # dim (runs, pts, n, dig samples)

        traces = arr[:, time_pt, :, :].mean(axis=0) # dim (n, dig samples)

        return traces

    def get_plot_metadata(self, exp_type: str) -> dict:
        """Get plot metadata for the given experiment type."""
        metadata = {
            "CW ODMR": {
                'title': 'CW Optically Detected Magnetic Resonance',
                'xlabel': 'Frequency (GHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'CW ODMR PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "Rabi": {
                'title': 'Rabi Oscillation',
                'xlabel': 'MW Pulse Duration (ns)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'Rabi PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "Pulsed ODMR": {
                'title': 'Pulsed ODMR',
                'xlabel': 'Frequency (GHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'Pulsed ODMR PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "Pulsed ODMR RF": {
                'title': 'Pulsed ODMR RF',
                'xlabel': 'Frequency (GHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'Pulsed ODMR RF PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "Opt T1": {
                'title': 'Optical T1 Relaxation',
                'xlabel': 'Free Precession Interval (ms)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'Optical T1 PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "MW T1": {
                'title': 'MW T1',
                'xlabel': 'Free Precession Interval (ms)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'MW T1 PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "T2": {
                'title': 'T2 Coherence',
                'xlabel': 'Free Precession Interval (µs)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'T2 PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "T2 RF": {
                'title': 'T2 RF Coherence',
                'xlabel': 'Free Precession Interval (µs)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'T2 RF PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DQ": {
                'title': 'Double Quantum',
                'xlabel': 'Inter-pulse Delay (ns)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DQ PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER": {
                'title': 'DEER',
                'xlabel': 'Frequency (MHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER Rabi": {
                'title': 'DEER Rabi',
                'xlabel': 'MW Pulse Duration (ns)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER Rabi PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER FID": {
                'title': 'DEER FID',
                'xlabel': 'Free Precession Interval (µs)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER FID PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER FID CD": {
                'title': 'DEER FID CD',
                'xlabel': 'Free Precession Interval (µs)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER FID CD PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER Correlation": {
                'title': 'DEER Correlation',
                'xlabel': 'Frequency (MHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER Correlation PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER Correlation Rabi": {
                'title': 'DEER Correlation Rabi',
                'xlabel': 'MW Pulse Duration (ns)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER Correlation Rabi PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER T1": {
                'title': 'DEER T1',
                'xlabel': 'Free Precession Interval (µs)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER T1 PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "DEER T2": {
                'title': 'DEER T2',
                'xlabel': 'Inter-pulse Delay (ns)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'DEER T2 PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "Noise Spectroscopy": {
                'title': 'Noise Spectroscopy',
                'xlabel': 'Frequency (MHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'Noise Spectroscopy PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "CASR": {
                'title': 'CASR',
                'xlabel': 'Free Precession Interval (ms) or Frequency (kHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'CASR PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
            "CASR IR": {
                'title': 'CASR IR',
                'xlabel': 'Free Precession Interval (ms) or Frequency (kHz)',
                'ylabel': 'Signal (V) or Norm. Signal',
                'pl_title': 'CASR IR PL Time Trace',
                'pl_xlabel': 'Readout Window (ns)',
                'pl_ylabel': 'Signal (V)',
            },
        }
        return metadata.get(exp_type, {})

    def acquire_data(
        self,
        cfg,
        exp_type: str,
        x_data,
        data,
        fit_x,
        fit_y,
        fit_value,
        fit_error,
        pl_data=None,
        subseq_sweeps=None,
        subseq_2_sweeps=None,
        signal_pl_sweeps=None,
        background_pl_sweeps=None,
        iters_completed=0,
        exp_start_time=0,
        slice_start=None,
        slice_end=None,
        parse_type=None,
        **kwargs,
    ):
        """Standardized data acquisition and pushing to data server."""
        try:
            result_raw = self.dig.acquire()
        except Exception as e:
            raise

        if result_raw is None:
            return

        try:
            parsed = self.parse_analog_math(result_raw, parse_type or exp_type, cfg.num_pts)
        except ValueError:
            return

        metadata = self.get_plot_metadata(exp_type)
        if not metadata:
            raise ValueError(f"No metadata for exp_type {exp_type}")

        subseq_name_map = {
            2: ["signal", "background"],
            4: ["dark_signal", "dark_background", "echo_signal", "echo_background"],
            6: ["subseq0", "subseq1", "subseq2", "subseq3", "subseq4", "subseq5"],
        }

        if exp_type == "Pulsed ODMR RF":
            subseq_name_map[4] = ["rf_signal", "rf_background", "no_rf_signal", "no_rf_background"]

        if exp_type == "DEER T1":
            subseq_name_map[4] = ["with_py", "without_py", "with_ny", "without_ny"]

        datasets = {
            "x_fit": fit_x,
            "y_fit": fit_y,
            "fit_value": fit_value,
            "fit_error": fit_error,
        }

        def append_series(name, x_axis, value, container):
            if container is None:
                raise ValueError(
                    f"Container for '{name}' must be pre-created by the experiment function"
                )
            stacked = np.stack([x_axis, value])
            container.append(stacked)
            container.updated_item(-1)
            datasets[name] = container
            return container

        if subseq_sweeps is None:
            raise ValueError("subseq_sweeps must be provided by the calling experiment function")

        if cfg.both_channels and subseq_2_sweeps is None:
            raise ValueError("subseq_2_sweeps must be provided for both_channels experiments")

        # SLICING LOGIC: If slice_start/slice_end are specified, both x_data and parsed 
        # subsequence data are trimmed identically BEFORE stacking into StreamingLists.
        # This ensures x-axis and signal values always align. Any data before slice_start
        # is permanently discarded and never reaches fitting or plot export.

        if cfg.both_channels:
            if not (isinstance(parsed, (list, tuple)) and len(parsed) == 2):
                raise ValueError("Expected two-channel parsed data for both_channels")

            ch0, ch1 = parsed # dim (n, pts) for each channel
            n = len(ch0) # number of subsequences per channel (NOT number of channels or number of points)

            if len(subseq_sweeps) != n:
                raise ValueError(
                    f"subseq_sweeps length {len(subseq_sweeps)} does not match channel count {n}"
                )

            names = list(subseq_sweeps.keys())
            if len(names) != n:
                raise ValueError("subseq_sweeps must have exactly one container per subsequence")

            # Apply slicing to x_data if specified (CRITICAL: must align with data slicing)
            x_data_sliced = x_data
            if slice_start is not None or slice_end is not None:
                x_data_sliced = x_data[slice_start:slice_end]

            for idx, name in enumerate(names):
                value0 = ch0[idx]
                value1 = ch1[idx]
                
                if slice_start is not None or slice_end is not None:
                    value0 = value0[slice_start:slice_end]
                    value1 = value1[slice_start:slice_end]

                c0 = subseq_sweeps.get(name)
                if c0 is None:
                    raise ValueError(f"Missing subseq_sweeps container for '{name}'")
                append_series(name, x_data_sliced, value0, c0)

                c2 = subseq_2_sweeps.get(name) if subseq_2_sweeps is not None else None
                if c2 is None:
                    raise ValueError(f"Missing subseq_2_sweeps container for '{name}'")
                append_series(f"{name}_ch2", x_data_sliced, value1, c2)

        else:
            # Apply slicing to x_data if specified (CRITICAL: must align with data slicing)
            x_data_sliced = x_data
            if slice_start is not None or slice_end is not None:
                x_data_sliced = x_data[slice_start:slice_end]
            
            if isinstance(parsed, (list, tuple)) and len(parsed) != 2:  # one channel, num_sub_seqs != 2
                n = len(parsed)
                if len(subseq_sweeps) != n:
                    raise ValueError(
                        f"subseq_sweeps length {len(subseq_sweeps)} does not match extracted subsequences {n}"
                    )

                names = list(subseq_sweeps.keys())
                for idx, name in enumerate(names):
                    data_arr = parsed[idx]
                    if slice_start is not None or slice_end is not None:
                        data_arr = data_arr[slice_start:slice_end] # apply slicing to subsequence data if specified
                    c = subseq_sweeps.get(name)
                    append_series(name, x_data_sliced, data_arr, c) # populate the StreamingList for this subsequence
            else:
                sig, bg = parsed # divide up into standard two subsequences (signal and background)
                if slice_start is not None or slice_end is not None:
                    sig = sig[slice_start:slice_end] # apply slicing to subsequence data if specified
                    bg = bg[slice_start:slice_end] 

                if "signal" in subseq_sweeps and "background" in subseq_sweeps:
                    sig_name = "signal"
                    bg_name = "background"
                else:
                    keys = list(subseq_sweeps.keys())
                    if len(keys) < 2:
                        raise ValueError("subseq_sweeps must include at least two containers for 2-subsequence data")
                    sig_name, bg_name = keys[0], keys[1]

                append_series(sig_name, x_data_sliced, sig, subseq_sweeps[sig_name]) # populate the StreamingList for the signal subsequence
                append_series(bg_name, x_data_sliced, bg, subseq_sweeps[bg_name]) # populate the StreamingList for the background subsequence
 
        if pl_data is not None:
            if signal_pl_sweeps is None or background_pl_sweeps is None:
                raise ValueError("pl_data requires signal_pl_sweeps and background_pl_sweeps")
            sig_pl, bg_pl = self.analog_pl_math(result_raw, exp_type, cfg.pl_pt, cfg.num_pts)
            pl_trace_times = np.linspace(0, cfg.segment_size / cfg.dig_sampling_freq, cfg.segment_size) * 1e9
            signal_pl_sweeps = append_series("signal_pl", pl_trace_times, sig_pl, signal_pl_sweeps)
            background_pl_sweeps = append_series("background_pl", pl_trace_times, bg_pl, background_pl_sweeps)
            pl_data.push({
                "params": {"kwargs": kwargs, "iters_completed": iters_completed, "elapsed": time.perf_counter() - exp_start_time},
                "title": metadata['pl_title'],
                "xlabel": metadata['pl_xlabel'],
                "ylabel": metadata['pl_ylabel'],
                "datasets": {"signal_pl": signal_pl_sweeps, "background_pl": background_pl_sweeps},
            })

        data.push({
            "params": {"kwargs": kwargs, "iters_completed": iters_completed, "elapsed": time.perf_counter() - exp_start_time},
            "title": metadata['title'],
            "xlabel": metadata['xlabel'],
            "ylabel": metadata['ylabel'],
            "datasets": datasets,
        })

    @staticmethod
    def build_status_msg(
        *,
        status: str,
        percent_completed: int,
        fit_value: float | None,
        fit_error: float | None,
        start_time: float,
        total_iters: int,
        iters_completed: int,
        fit_value2: Optional[float] = None,
        fit_error2: Optional[float] = None,
        fit_units: Optional[list] = None,
        fit_units2: Optional[list] = None,
        exception: Optional[str] = None,
    ) -> dict:
        now = time.perf_counter()
        elapsed = now - start_time

        if iters_completed > 0:
            avg_loop = elapsed / iters_completed
            est_total = avg_loop * total_iters
        else:
            avg_loop = 0.0
            est_total = 0.0

        remaining = max(est_total - elapsed, 0.0)

        return {
            "status": status,
            "percent": int(percent_completed),

            "fit_value": fit_value,
            "fit_error": fit_error,
            "fit_units": fit_units,

            "fit_value2": fit_value2,
            "fit_error2": fit_error2,
            "fit_units2": fit_units2,

            "elapsed_s": elapsed,
            "remaining_s": remaining,
            "est_total_s": est_total,

            "elapsed_str": format_hhmmss(elapsed),
            "remaining_str": format_hhmmss(remaining),
            "est_total_str": format_minutes_seconds(est_total),

            "exception": exception,
        }
    
    @staticmethod
    def _safe(label: str, fn):
        """Run fn() and log exceptions, never raising."""
        try:
            fn()
        except Exception as e:
            _logger.exception("cleanup: %s failed", label)

    def emit_bpd_shutter_state(self, is_open: bool):
        if self.queue_to_inst is None:
            return

        try:
            self.queue_to_inst.put_nowait({
                "type": "bpd_shutter",
                "open": bool(is_open),
            })
        except Exception as e:
            _logger.warning(f"Could not emit BPD shutter state: {e}")

    @contextmanager
    def _shutter_open(self, shutter, daq, detector="APD"):
        shutter.open_shutter()
        if detector.lower() == "bpd":
            daq.open_do_task("shutter")
            daq.start_do_task()
            bpd_shutter_state = daq.read_do_task()
            daq.stop_do_task()
            daq.close_do_task()

            time.sleep(0.01)
            
            if not bpd_shutter_state: # if BPD shutter is closed, open it
                daq.open_do_task("shutter")
                daq.start_do_task()
                daq.write_do_task("shutter", shutter_status="open")
                daq.stop_do_task()
                daq.close_do_task()
            self.emit_bpd_shutter_state(is_open=True)
        try:
            yield
        finally:
            try:
                shutter.close_shutter()
                if detector.lower() == "bpd":
                    daq.open_do_task("shutter")
                    daq.start_do_task()
                    daq.write_do_task("shutter", shutter_status="close")
                    daq.stop_do_task()
                    daq.close_do_task()
                    self.emit_bpd_shutter_state(is_open=False)
            except Exception as e:
                _logger.exception("shutter.close_shutter failed")

    def equipment_off_handles(
        self,
        *,
        laser_shutter=None,
        sig_gen=None,
        ps=None,
        hdawg=None,
        token: str | None = None
    ):
        """Best-effort shutdown using existing instrument handles (no new InstrumentManager)."""

        # Recommended order: stop waveform sources first, then triggers, then digitizer, then shutter.
        if hdawg is not None:
            self._safe("hdawg.set_disabled", hdawg.set_disabled)

        if sig_gen is not None:
            self._safe("sig_gen.set_mod_toggle(0)", lambda: sig_gen.set_mod_toggle(0))
            self._safe("sig_gen.set_rf_toggle(0)", lambda: sig_gen.set_rf_toggle(0))

        # PulseStreamer
        if ps is not None: 
            if hasattr(ps, "Pulser"):       
                self._safe("ps.Pulser.forceFinal", ps.Pulser.forceFinal)
                self._safe("ps.Pulser.reset", ps.Pulser.reset)

            # Release experiment ownership of pulse streamer
            if token is not None and hasattr(ps, "end_exclusive"):
                self._safe("ps.end_exclusive", lambda: ps.end_exclusive(token))

        # Digitizer (self.dig is already your handle)
        self._safe("dig.stop_card", self.dig.stop_card)
        self._safe("dig.reset", self.dig.reset)

        if laser_shutter is not None:
           self._safe("laser_shutter.close_shutter", laser_shutter.close_shutter)

    """ Experiment logic """
    def run_experiment(
        self, 
        *, 
        dataset: str, 
        token_prefix: str, 
        body: Callable[..., Any],
        enable_pl_trace: bool = False,
        pl_dataset: Optional[str] = None,
    ) -> Any:
        
        """Run an experiment with a standardized lifecycle.

        This wrapper:
          - connects to InstrumentManager + DataSource
          - takes exclusive control of the Pulse Streamer
          - runs the experiment body(mgr=..., data=..., token=...)
          - always shuts equipment down using existing handles
        """
        token = f"{token_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

        try:
            with ExitStack() as stack:
                mgr = stack.enter_context(InstrumentManager())
                data = stack.enter_context(DataSource(dataset))

                pl_data = None
                if enable_pl_trace:
                    if pl_dataset is None:
                        pl_dataset = f"{dataset} pl"
                    pl_data = stack.enter_context(DataSource(pl_dataset))

                laser_shutter = mgr.laser_shutter
                sig_gen = mgr.sg
                ps = mgr.ps
                hdawg = mgr.awg

                with _exclusive_ps(ps, token):
                    try:
                        return body(mgr=mgr, data=data, pl_data=pl_data, token=token)
                    finally:
                        self.equipment_off_handles(
                            laser_shutter=laser_shutter,
                            sig_gen=sig_gen,
                            ps=ps,
                            hdawg=hdawg,
                        )
        except Exception as e:
            # Print to stderr immediately for visibility
            import sys
            print(f"\n{'='*70}", file=sys.stderr, flush=True)
            print(f"❌ UNCAUGHT EXCEPTION IN run_experiment", file=sys.stderr, flush=True)
            print(f"{'='*70}", file=sys.stderr, flush=True)
            print(f"Type: {type(e).__name__}", file=sys.stderr, flush=True)
            print(f"Message: {str(e)}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr, flush=True)
            raise

    def sigvstime_scan(self, **kwargs):     
        cfg = nvcfg.SignalScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety
        with InstrumentManager() as mgr, DataSource(cfg.dataset) as sigvstime_data:
            # run laser on continuously here from laser driver            
            # laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            daq = mgr.daq
            ps = mgr.ps

            token = f"SIGVSTIME_{time.strftime('%Y%m%d_%H%M%S')}"  # unique token for this experiment run
            ps.begin_exclusive(token, takeover=True, restore_on_release=True)  # take exclusive control of the pulse streamer

            sequence = ps.SigvsTime(1/cfg.exp_sampling_rate * 1e9) # pulse streamer sequence for CW ODMR
            
            # configure digitizer (need to use DC coupling for signal vs time)           
            dig_config = self.digitizer_configure(exp_type="Sig vs Time", num_pts = 1, iters = 1, 
                                                segment_size = cfg.segment_size, sampling_freq = cfg.dig_sampling_freq, dig_amplitude = cfg.dig_amplitude, 
                                                read_channel = cfg.read_channel, both_channels = cfg.both_channels, coupling = 'DC', termination = '1M', 
                                                pretrig_size = cfg.pretrig_size, dig_timeout = cfg.dig_timeout, runs = 400)
                
            time_start = time.time()

            signal_sweeps = StreamingList()

            # open laser shutter
            laser_shutter.open_shutter()
            
            # upload digitizer parameters
            self.dig.assign_param(dig_config)

            # configure laser settings and turn on
            # laser.set_diode_current_realtime(cfg.laser_power)
            
            # set pulsestreamer to start on software trigger & run infinitely
            ps.set_soft_trigger()
            ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) #cfg.runs*cfg.iters) # execute chosen sequence on Pulse Streamer
            
            # start digitizer --> waits for trigger from pulse sequence
            self.dig.config()
            self.dig.start_buffer()
            
            # start pulse sequence
            ps.start_now(owner=token)
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            for i in range(10000):                
                sig_result_raw = self.dig.acquire() # acquire data from digitizer

                # average all data over each trigger/segment 
                sig_result = np.mean(sig_result_raw,axis=1)
                sig_result = np.mean(sig_result)
                sig_val = sig_result.magnitude if hasattr(sig_result, "magnitude") else sig_result
                time_pt = time.time() - time_start

                # read the analog voltage levels received by the APD.
                # notify the streaminglist that this entry has updated so it will be pushed to the data server
                signal_sweeps.append(np.array([[time_pt], [sig_val]]))
                signal_sweeps.updated_item(-1) 
                
                # save the current data to the data server.
                sigvstime_data.push({'params': {'kwargs': kwargs, 'elapsed': time.perf_counter() - exp_start_time},
                                     'title': 'Signal Vs Time',
                                     'xlabel': 'Time step',
                                     'ylabel': 'APD Voltage (V)',
                                     'datasets': {'signal': signal_sweeps}})

                msg = self.build_status_msg(
                    status='in progress',
                    percent_completed=0,
                    fit_value=None,
                    fit_error=None,
                    start_time=exp_start_time,
                    total_iters=1,
                    iters_completed=0,
                )

                self.queue_from_exp.put_nowait(msg)
                
                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                    # the GUI has asked us nicely to exit. Save data if requested.
                    # self.equipment_off()

                    self.equipment_off_handles(
                        laser_shutter=laser_shutter,
                        ps=ps,
                        token=token
                    )

                    msg = self.build_status_msg(
                        status="stopped",
                        percent_completed=0,
                        fit_value=None,
                        fit_error=None,
                        start_time=exp_start_time,
                        total_iters=0,
                        iters_completed=0,
                    )
                    # self.queue_from_exp.put_nowait([percent_completed, 'stopped', [fit_value, fit_error]])
                    self.queue_from_exp.put_nowait(msg)

                    if cfg.save:
                        run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)
                    return

    @managed_experiment(token_prefix="CWODMR", dataset_key="dataset")           
    def odmr_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a CW ODMR sweep over a set of microwave frequencies."""  
        cfg = nvcfg.ODMRScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg

        ### --- Define NV drive parameters and configure SRS signal generator --- ###   
        delta = 0
        iq_phases = [delta + 0, delta + 90] # set IQ phase relations for lower sideband [lower I, lower Q]
        sig_gen_freq = cfg.center_freq + cfg.half_span_sideband_freq # set freq to sig gen 
        max_sideband_freq = 2 * cfg.half_span_sideband_freq # set span of ODMR sweep as max sideband modulation frequency --> ? MHz max. for AWG bandwidth

        ### --- Default parameter array for sweep --- ###
        mod_freqs = np.flip(np.linspace(0, max_sideband_freq, cfg.num_pts))
        real_freqs = sig_gen_freq - mod_freqs
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive
            
        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = real_freqs.copy()  # No scaling here; inline scaling in acquire_data call
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.CW_ODMR(cfg.num_pts, cfg.probe*1e9) # pulse streamer sequence for CW ODMR
        dig_cfg = self.digitizer_configure(
            exp_type="ODMR",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###                      
        try:
            hdawg.set_sequence(**{
                "seq": "CW ODMR",
                "i_offset": cfg.i_offset,
                "q_offset": cfg.q_offset,
                "probe_length": cfg.probe,
                "sideband_power": cfg.sideband_power,
                "sideband_freqs": mod_freqs,
                "iq_phases": iq_phases,
                "num_pts": cfg.num_pts,
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
            subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList() # for storing optional PL data --> list of numpy arrays of shape (2, dig segment_size)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power
        
        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):     
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return        
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            if pl_data is not None:
                if not (0 <= cfg.pl_pt < cfg.num_pts):
                    raise ValueError(
                        f"cfg.pl_pt ({cfg.pl_pt}) must be between 0 and {cfg.num_pts - 1} "
                        f"(num_pts={cfg.num_pts})"
                    )
            
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="CW ODMR",
                        x_data=real_freqs / 1e9,
                        data=data,
                        fit_x=fit_x / 1e9,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps,
                        background_pl_sweeps=background_pl_sweeps,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, 
                                cfg.dataset, subseq_sweeps["signal"], subseq_sweeps["background"], *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100
        
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, 
                        cfg.dataset, subseq_sweeps["signal"], subseq_sweeps["background"], *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))
        
    def odmr_smart_scan(self, **kwargs):
        """Run a CW ODMR smart scan sweep over a set of magnet angles to determine best alignment."""
    
        cfg = nvcfg.ODMRSmartScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety
        with InstrumentManager() as mgr, DataSource(cfg.dataset) as cw_odmr_data:
            ### --- Devices --- ###
            # laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            daq = mgr.daq
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            token = f"ODMRSMRTSCN_{time.strftime('%Y%m%d_%H%M%S')}"  # unique token for this experiment run
            ps.begin_exclusive(token, takeover=True, restore_on_release=True)  # take exclusive control of the pulse streamer

            num_angles = cfg.iters
            azi_angles = np.linspace(cfg.start_angle, cfg.stop_angle, num_angles)
            # print("azimuthal angles: ", azi_angles)
            # mgr.thor_polar.set_vel_params(3,7) # reset Thorlabs stages to default acceleration and velocity parameters
            mgr.thor_azi.set_vel_params(8,15)

            # move azimuthal stage to start angle
            mgr.thor_azi.move(cfg.start_angle, True)
            mgr.thor_azi.update_positions_callback() # update position
            
            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = cfg.center_freq + cfg.half_span_sideband_freq # set freq to sig gen 

            max_sideband_freq = 2*cfg.half_span_sideband_freq # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, cfg.num_pts)            
            mod_freqs = np.flip(mod_freqs)
                        
            real_freqs = sig_gen_freq - mod_freqs
            
            # define pulse sequence
            sequence = ps.CW_ODMR(cfg.num_pts, cfg.probe*1e9) # pulse streamer sequence for CW ODMR
            ps.probe_time = cfg.probe * 1e9

            # configure digitizer
            dig_config = self.digitizer_configure(exp_type = "ODMR", num_pts = cfg.num_pts, iters = cfg.iters, 
                                                  segment_size = cfg.segment_size, sampling_freq = cfg.dig_sampling_freq, dig_amplitude = cfg.dig_amplitude, 
                                                  read_channel = cfg.read_channel, both_channels = cfg.both_channels, coupling = cfg.dig_coupling, termination = cfg.dig_termination, 
                                                  pretrig_size = cfg.pretrig_size, dig_timeout = cfg.dig_timeout, runs = cfg.runs)
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(cfg.rf_power) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            try:
                hdawg.set_sequence(**{'seq': 'CW ODMR',
                                    'i_offset': cfg.i_offset,
                                    'q_offset': cfg.q_offset,
                                    'probe_length': cfg.probe, 
                                    'sideband_power': cfg.sideband_power,
                                    'sideband_freqs': mod_freqs, 
                                    'iq_phases': iq_phases,
                                    'num_pts': cfg.num_pts}) 
                time.sleep(2) # wait for AWG to finish setting sequence and magnet mount to get set
            except Exception as e:
                print(e)
            
            # run the experiment
            else:
                pos = mgr.thor_azi.current_position # set pos to be position as sweep is beginning (i.e., at start angle)
                mgr.thor_azi.update_positions_callback() # update position
                print(f"Moved to start angle: {pos} degrees")

                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()
                angle_fits = StreamingList()
                
                # Track previous fit frequency for adaptive initial guess
                prev_freq_fit = cfg.center_freq / 1e9  # Start with center frequency

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # mgr.thor_polar.set_vel_params(3,7) # TODO: speed up Thorlabs stage for quicker scan
                mgr.thor_azi.set_vel_params(25,35) # set azimuthal stage velocity and acceleration

                # configure laser settings and turn on
                # laser.set_diode_current_realtime(cfg.laser_power)

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                self.dig.config()
                self.dig.start_buffer()
                
                # start pulse sequence
                ps.start_now(owner=token)

                exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

                # start experiment loop
                for i in range(num_angles):
                    mgr.thor_azi.move(azi_angles[i], True)
                    mgr.thor_azi.update_positions_callback() # update position
                    pos = mgr.thor_azi.current_position # set pos to be position as move is beginning
                
                    odmr_result_raw = self.dig.acquire() # acquire data from digitizer

                    # average all data over each trigger/segment 
                    odmr_result = np.mean(odmr_result_raw,axis=1)

                    # partition buffer into signal and background datasets
                    try:
                        sig, bg = self.parse_analog_math(odmr_result, 'CW ODMR', cfg.num_pts)
                    except ValueError:
                        continue
                    
                    # notify the streaminglist that this entry has updated so it will be pushed to the data server
                    signal_sweeps.append(np.stack([real_freqs/1e9, sig]))
                    signal_sweeps.updated_item(-1) 
                    background_sweeps.append(np.stack([real_freqs/1e9, bg]))
                    background_sweeps.updated_item(-1)

                    # Fit the data
                    initial_guess = [0.01, prev_freq_fit, 6e-3, 1]  # [A, x0, gamma, c] — x0 uses previous fit
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            params, params_covariance = curve_fit(self.negative_lorentzian, real_freqs/1e9, sig/bg, p0=initial_guess)
                            # fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, cfg.dataset, signal_sweeps, background_sweeps, *cfg.fit_params)
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                    # Extract fitted parameters
                    freq_fit = params[1]
                    prev_freq_fit = freq_fit  # Update for next iteration
                    angle_fits.append(np.stack([azi_angles[i], freq_fit]))
                    angle_fits.updated_item(-1) 
                    print(f"ODMR = {round(freq_fit, 3)} GHz at angle {azi_angles[i]} degrees")

                    # update GUI progress bar & ETA
                    iter_completed = i + 1                      
                    percent_completed = int((iter_completed / cfg.iters) * 100)
                    
                    # save the current data to the data server
                    cw_odmr_data.push({'params': {'kwargs': kwargs, 'iters_completed': iter_completed, 'elapsed': time.perf_counter() - exp_start_time},
                                        'title': 'CW Optically Detected Magnetic Resonance',
                                        'xlabel': 'Frequency (GHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}})

                    msg = self.build_status_msg(
                            status='in progress',
                            percent_completed=percent_completed,
                            fit_value=None,
                            fit_error=None,
                            start_time=exp_start_time,
                            total_iters=cfg.iters,
                            iters_completed=iter_completed,
                        )

                    # self.queue_from_exp.put_nowait([percent_completed, 'in progress', [fit_value, fit_error]])
                    self.queue_from_exp.put_nowait(msg)

                    if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                        # the GUI has asked us nicely to exit. Save data if requested.
                        # self.equipment_off()
                        self.equipment_off_handles(
                            laser_shutter=laser_shutter,
                            ps=ps,
                            hdawg=hdawg,
                            token=token
                        )
                        mgr.thor_polar.set_vel_params(3,7) # reset Thorlabs stages to default acceleration and velocity parameters
                        mgr.thor_azi.set_vel_params(3,7)

                        msg = self.build_status_msg(
                            status="stopped",
                            percent_completed=int(percent_completed),
                            fit_value=None,
                            fit_error=None,
                            start_time=exp_start_time,
                            total_iters=cfg.iters,
                            iters_completed=iter_completed,
                        )
                        # self.queue_from_exp.put_nowait([percent_completed, 'stopped', [fit_value, fit_error]])
                        self.queue_from_exp.put_nowait(msg)
                        
                        if cfg.save:
                            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)
                        return
                    
                # save data if requested upon completion of experiment
                if cfg.save:
                    run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

                msg = self.build_status_msg(
                    status="complete",
                    percent_completed=int(percent_completed),
                    fit_value=None,
                    fit_error=None,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iter_completed,
                )
                # self.queue_from_exp.put_nowait([percent_completed, 'complete', [fit_value, fit_error]])
                self.queue_from_exp.put_nowait(msg)

            finally:
                # self.equipment_off() # turn off equipment regardless of if experiment started or failed
                # ps.end_exclusive(token)  # release exclusive control of the pulse streamer
                self.equipment_off_handles(
                        laser_shutter=laser_shutter,
                        ps=ps,
                        hdawg=hdawg,
                        token=token
                    )
                
                mgr.thor_polar.set_vel_params(3,7) # reset Thorlabs stages to default acceleration and velocity parameters
                mgr.thor_azi.set_vel_params(3,7)

                # Search for the minimum value of freq_fit
                min_freq_fit = float('inf')  # Initialize with a very large value
                min_azi_angle = None  # To store the corresponding azi_angle

                for array in angle_fits:
                    azi_angle, freq = array[0], array[1]  # Extract azi_angle and freq_fit
                    if freq < min_freq_fit:
                        min_freq_fit = freq
                        min_azi_angle = azi_angle

                # Print the results
                print(f"Best alignment at {min_azi_angle} degrees -> min. freq_fit: {min_freq_fit} GHz")

                time.sleep(1) # wait for Thorlabs stage to finish moving

                if min_azi_angle is not None:
                    lower = min(azi_angles)
                    upper = max(azi_angles)
                    # print(f"lower bound: {lower}, upper bound: {upper}")
                    if lower <= min_azi_angle <= upper:
                    # if min_azi_angle >= azi_angles[0] and min_azi_angle <= azi_angles[-1]:
                        mgr.thor_azi.move(min_azi_angle, True)
                        print(f"Stage alignment set at {min_azi_angle} degrees")
                    else:
                        print("Min azi angle out of bounds. Not moving stage.")
                else:
                    print("Min azi angle is None. Not moving stage.")
    
    @managed_experiment(token_prefix="RABI", dataset_key="dataset")
    def rabi_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a Rabi sweep over MW pulse durations."""
        cfg = nvcfg.RabiScanCfg(**kwargs)  # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg

        ### --- Default parameter array for sweep --- ###
        mw_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)

        ### --- Define NV drive parameters --- ###  
        sig_gen_freq, iq_phases = self.choose_sideband(
            cfg.sideband, cfg.freq, cfg.sideband_freq, cfg.pulse_axis
        )

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power)
    
        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = mw_times.copy()
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Rabi(cfg.laser_init * 1e9, mw_times * 1e9, cfg.laser_readout * 1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="Rabi",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                "seq": "Rabi",
                "i_offset": cfg.i_offset,
                "q_offset": cfg.q_offset,
                "sideband_power": cfg.sideband_power,
                "sideband_freq": cfg.sideband_freq,
                "iq_phases": iq_phases,
                "pi_pulses": mw_times,
                "num_pts": cfg.num_pts,
                "runs": cfg.runs,
            })   
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return

        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
            subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList() # for storing optional PL data --> list of numpy arrays of shape (2, dig segment_size)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()

            ### --- Main experiment loop --- ###
            if pl_data is not None:
                if not (0 <= cfg.pl_pt < cfg.num_pts):
                    raise ValueError(
                        f"cfg.pl_pt ({cfg.pl_pt}) must be between 0 and {cfg.num_pts - 1} "
                        f"(num_pts={cfg.num_pts})"
                    )
            
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="Rabi",
                        x_data=mw_times * 1e9,
                        data=data,
                        fit_x=fit_x * 1e9,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps,
                        background_pl_sweeps=background_pl_sweeps,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, 
                                cfg.dataset, subseq_sweeps["signal"], subseq_sweeps["background"], *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, 
                        cfg.dataset, subseq_sweeps["signal"], subseq_sweeps["background"], *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="PLSDODMR", dataset_key="dataset")
    def pulsed_odmr_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a Pulsed ODMR sweep over a set of microwave frequencies."""
        cfg = nvcfg.PulsedODMRScanCfg(**kwargs)  # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###            
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
        
        ### --- Define NV drive parameters --- ###           
        delta = 0
        iq_phases = [delta + 0, delta + 90] # set IQ phase relations for lower sideband [lower I, lower Q]   
        sig_gen_freq = cfg.center_freq + cfg.half_span_sideband_freq # set freq to sig gen 
        max_sideband_freq = 2 * cfg.half_span_sideband_freq # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

        ### --- Default parameter array for sweep --- ###
        mod_freqs = np.flip(np.linspace(0, max_sideband_freq, cfg.num_pts)) # define parameter array that will be swept over in experiment & shuffle
        real_freqs = sig_gen_freq - mod_freqs

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive
            
        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = real_freqs.copy()  # No scaling here; inline scaling in acquire_data call
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Pulsed_ODMR(cfg.laser_init*1e9, cfg.num_pts, cfg.pi*1e9, cfg.laser_readout*1e9) # pulse streamer sequence
        dig_cfg = self.digitizer_configure(
            exp_type="ODMR",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )
    
        ### --- Upload AWG sequence --- ### 
        try:
            hdawg.set_sequence(**{
                'seq': 'Pulsed ODMR',
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freqs': mod_freqs, 
                'iq_phases': iq_phases,
                'pi_pulse': cfg.pi, 
                'num_pts': cfg.num_pts,
                'runs': cfg.runs
            })    
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
            subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList() # for storing optional PL data --> list of numpy arrays of shape (2, dig segment_size)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence

            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return 
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            if pl_data is not None:
                if not (0 <= cfg.pl_pt < cfg.num_pts):
                    raise ValueError(
                        f"cfg.pl_pt ({cfg.pl_pt}) must be between 0 and {cfg.num_pts - 1} "
                        f"(num_pts={cfg.num_pts})"
                    )
            
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break
                
                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="Pulsed ODMR",
                        x_data=real_freqs / 1e9,
                        data=data,
                        fit_x=fit_x / 1e9,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps if pl_data is not None else None,
                        background_pl_sweeps=background_pl_sweeps if pl_data is not None else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, 
                                cfg.dataset, subseq_sweeps["signal"], subseq_sweeps["background"], *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, 
                        cfg.dataset, subseq_sweeps["signal"], subseq_sweeps["background"], *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="PLSDODMRRF", dataset_key="dataset")
    def pulsed_odmr_rf_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a Pulsed ODMR sweep over a set of microwave frequencies with coil RF tone for coil B field calibration."""
        cfg = nvcfg.PulsedODMRRFScanCfg(**kwargs)  # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define NV drive parameters --- ###      
        delta = 0
        iq_phases = [delta + 0, delta + 90] # set IQ phase relations for lower sideband [lower I, lower Q]   
        sig_gen_freq = cfg.center_freq + cfg.half_span_sideband_freq # set freq to sig gen 
        max_sideband_freq = 2 * cfg.half_span_sideband_freq # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

        ### --- Default parameter array for sweep --- ###
        mod_freqs = np.flip(np.linspace(0, max_sideband_freq, cfg.num_pts)) # define parameter array that will be swept over in experiment & shuffle
        real_freqs = sig_gen_freq - mod_freqs

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive
          
        ### --- Default fit parameters for live fitting --- ###
        fit_value = None
        fit_error = None
        fit_no_rf_value = None
        fit_no_rf_error = None
        fit_x = real_freqs.copy()  # No scaling here; inline scaling in acquire_data call
        fit_y = np.ones(len(fit_x))
        fit_no_rf_x = real_freqs.copy()  # No scaling here; inline scaling in acquire_data call
        fit_no_rf_y = fit_y.copy()
        fitted_diff = 0
        coil_b_field_gauss = None
        proton_pi_half = None
        fit_units = []  # Initialize for error handling
        fit_units2 = []  # Initialize for error handling

        pi_pulse = cfg.pi*1e9 # [ns] units for pulse streamer
        rf_period = 1/cfg.rf_pulse_freq*1e9 # rf pulse period [ns] units for pulse streamer
        
        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Pulsed_ODMR_RF(cfg.laser_init*1e9, cfg.num_pts, pi_pulse, rf_period, cfg.laser_readout*1e9) # pulse streamer sequence
        dig_cfg = self.digitizer_configure(
            exp_type="ODMR",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ### 
        try:
            hdawg.set_sequence(**{
                'seq': 'Pulsed ODMR RF',
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freqs': mod_freqs, 
                'iq_phases': iq_phases,
                'pi_pulse': pi_pulse/1e9, 
                'rf_freq': cfg.rf_pulse_freq,
                'rf_power': cfg.rf_pulse_power,
                'rf_phase': cfg.rf_pulse_phase,
                'rf_length': 3*rf_period/1e9,
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
            status="failed",
            percent_completed=0,
            fit_value=fit_value,
            fit_error=fit_error,
            fit_value2=fit_no_rf_value,
            fit_error2=fit_no_rf_error,
            fit_units2=fit_units2,
            start_time=time.perf_counter(),
            total_iters=cfg.iters,
            iters_completed=0,
            exception=_format_exception_msg(e),
        ))
            return
            
        subseq_sweeps = {
            "rf_signal": StreamingList(),
            "rf_background": StreamingList(),
            "no_rf_signal": StreamingList(),
            "no_rf_background": StreamingList(),
        }
        if cfg.both_channels:
            subseq_2_sweeps = {
                "rf_signal": StreamingList(),
                "rf_background": StreamingList(),
                "no_rf_signal": StreamingList(),
                "no_rf_background": StreamingList(),
            }
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList()
        
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence

            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_value2=fit_no_rf_value,
                    fit_error2=fit_no_rf_error,
                    fit_units2=fit_units2,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return 
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="Pulsed ODMR RF",
                        x_data=real_freqs / 1e9,
                        data=data,
                        fit_x=fit_x / 1e9,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps if pl_data is not None else None,
                        background_pl_sweeps=background_pl_sweeps if pl_data is not None else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type,
                                "odmr rf",
                                subseq_sweeps["rf_signal"],
                                subseq_sweeps["rf_background"],
                                *cfg.fit_params[:4],
                            )
                            fit_no_rf_value, fit_no_rf_error, fit_no_rf_x, fit_no_rf_y, fit_units2 = self.fit_data(
                                cfg.fit_type,
                                "odmr rf",
                                subseq_sweeps["no_rf_signal"],
                                subseq_sweeps["no_rf_background"],
                                *cfg.fit_params[4:],
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")
                        else:
                            fitted_diff = fit_no_rf_value[1] - fit_value[1]
                            coil_b_field_gauss = round(fitted_diff * 1000 / 2.8, 4)
                            proton_pi_half = round(1 / (42.577e-4 * coil_b_field_gauss) / 4, 2)

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    fit_value2=fit_no_rf_value,
                    fit_error2=fit_no_rf_error,
                    fit_units2=fit_units2,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                        cfg.fit_type, 
                        "odmr rf", 
                        subseq_sweeps["rf_signal"], 
                        subseq_sweeps["rf_background"], 
                        *cfg.fit_params[:4]
                    )
                    fit_no_rf_value, fit_no_rf_error, fit_no_rf_x, fit_no_rf_y, fit_units2 = self.fit_data(
                        cfg.fit_type, 
                        "odmr rf", 
                        subseq_sweeps["no_rf_signal"], 
                        subseq_sweeps["no_rf_background"], 
                        *cfg.fit_params[4:]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")
                else:
                    fitted_diff = fit_no_rf_value[1] - fit_value[1]
                    coil_b_field_gauss = round(fitted_diff*1000/2.8,4)
                    proton_pi_half = round(1/(42.577e-4*coil_b_field_gauss)/4,2)

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_value2=fit_no_rf_value,
            fit_error2=fit_no_rf_error,
            fit_units2=fit_units2,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

        if fit_value is not None and fit_no_rf_value is not None:
            print(f"Fitted resonances = {fit_value[1]} GHz (RF), {fit_no_rf_value[1]} GHz (no RF)")
            print(f"Fitted difference = {round(fitted_diff*1000,4)} MHz")
            print(f"Coil B field = {coil_b_field_gauss} G")
            print(f"1H pi/2 pulse = {proton_pi_half} us")

    @managed_experiment(token_prefix="OPTT1", dataset_key="dataset")
    def OPT_T1_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a T1 sweep without MW over a set of precession time intervals.
        """
        cfg = nvcfg.OptT1ScanCfg(**kwargs)
            
        ### --- Devices --- ###  
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        ps = mgr.ps

        ### --- Default parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                tau_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)
            case 'linspace':
                tau_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = tau_times
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###      
        sequence = ps.Optical_T1(tau_times * 1e9, cfg.laser_readout*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="Opt T1",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        opt_t1_sweeps = StreamingList()  # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        subseq_sweeps = {"opt_t1": opt_t1_sweeps}

        self.dig.assign_param(dig_cfg)  # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power)  # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="Opt T1",
                        x_data=tau_times * 1e3,
                        data=data,
                        fit_x=fit_x * 1e3,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                # OPT_T1 does not support fit_live via existing signal/background fit_data path.
                # If fit_live is requested, user should switch to a custom T1 fitter.
                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            _logger.warning("OPT_T1_scan: `fit` is not available for single-subsequence data using current fit_data (signal/background required).")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="MWT1", dataset_key="dataset")
    def MW_T1_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a T1 sweep with MW over a set of precession time intervals."""
        cfg = nvcfg.MWT1ScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###  
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
 
        ### --- Default parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                tau_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)
            case 'linspace':
                tau_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)
        
        # Validate that we have enough points for slicing
        if cfg.num_pts < 2:
            raise ValueError(f"cfg.num_pts must be at least 2 for MW T1 scan (got {cfg.num_pts})")
        
        ### --- Define NV drive parameters --- ###  
        sig_gen_freq, iq_phases = self.choose_sideband(
            cfg.sideband, cfg.freq, cfg.sideband_freq, cfg.pulse_axis
        )
        pi_pulse = cfg.pi*1e9 # [ns] units for pulse streamer
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = tau_times
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Diff_T1(cfg.laser_init*1e9, tau_times * 1e9, cfg.pulse_axis, pi_pulse, cfg.laser_readout*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="MW T1",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                'seq': 'T1',
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pi_pulse': pi_pulse/1e9, 
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters}
            )    
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
            subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList() # for storing optional PL data --> list of numpy arrays of shape (2, dig segment_size)


        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return
            
            exp_start_time = time.perf_counter()

            ### --- Main experiment loop --- ###
            if pl_data is not None:
                if not (0 <= cfg.pl_pt < cfg.num_pts):
                    raise ValueError(
                        f"cfg.pl_pt ({cfg.pl_pt}) must be between 0 and {cfg.num_pts - 1} "
                        f"(num_pts={cfg.num_pts})"
                    )
            
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="MW T1",
                        x_data=tau_times * 1e3,
                        data=data,
                        fit_x=fit_x * 1e3,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps if pl_data is not None else None,
                        background_pl_sweeps=background_pl_sweeps if pl_data is not None else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    _logger.error(f"MW_T1_scan acquire_data failed: {exception_type}: {str(e)}", exc_info=True)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break
                
                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type,
                                cfg.dataset, 
                                subseq_sweeps["signal"], 
                                subseq_sweeps["background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))
                            
        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                        cfg.fit_type, 
                        cfg.dataset, 
                        subseq_sweeps["signal"], 
                        subseq_sweeps["background"], 
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="T2", dataset_key="dataset")
    def T2_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a T2 sweep over a set of precession time intervals."""
        cfg = nvcfg.T2ScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                tau_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)
            case 'linspace':
                tau_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq)
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = tau_times
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        match cfg.t2_seq:
            case 'Ramsey':
                sequence = ps.Ramsey(cfg.laser_init*1e9, tau_times * 1e9, pi_half[0], pi_half[1], cfg.laser_readout*1e9)
                x_tau_times = tau_times * 1e9
            case 'Echo':
                sequence = ps.Echo(cfg.laser_init*1e9, tau_times * 1e9, pi_half[0], pi_half[1], 
                                        pi[0], pi[1], cfg.laser_readout*1e9)
                x_tau_times = 2 * tau_times * 1e9 + pi[1] # for definition of pi/2 - tau - pi - tau - pi/2
            case 'XY4':
                sequence = ps.XY4_N(cfg.laser_init*1e9, tau_times * 1e9, 'xy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
                x_tau_times = 4 * tau_times * 1e9 + 2 * pi[0] + 2 * pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + (2*(x_tau_times/2)/(4*cfg.n) + 2*pi[0] + \
                #             2*pi[1] + 3*x_tau_times/(4*cfg.n))*cfg.n
            case 'YY4':
                sequence = ps.XY4_N(cfg.laser_init*1e9, tau_times * 1e9, 'yy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
                x_tau_times = 4 * tau_times * 1e9 + 4 * pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + (2*(x_tau_times/2)/(4*cfg.n) + 4*pi[1] + 3*x_tau_times/(4*cfg.n))*cfg.n
            case 'XY8':
                sequence = ps.XY8_N(cfg.laser_init*1e9, tau_times * 1e9, 'xy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
                x_tau_times = 8 * tau_times * 1e9 + 4*pi[0] + 4*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*cfg.n) + 4*pi[0] + \
                #             4*pi[1] + 7*x_tau_times/(8*cfg.n) + (x_tau_times/2)/(8*cfg.n))*cfg.n
            case 'YY8':
                sequence = ps.XY8_N(cfg.laser_init*1e9, tau_times * 1e9, 'yy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
                x_tau_times = 8 * tau_times * 1e9 + 8 * pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*cfg.n) + 8*pi[1] + \
                #         7*x_tau_times/(8*cfg.n) + (x_tau_times/2)/(8*cfg.n))*cfg.n
            case 'CPMG':
                sequence = ps.CPMG_N(cfg.laser_init*1e9, tau_times * 1e9, cfg.pulse_axis, 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
                x_tau_times = tau_times * 1e9 + (cfg.n - 1) * (pi[1] + tau_times * 1e9) # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + x_tau_times/cfg.n + (pi[0] + x_tau_times/cfg.n)*(cfg.n-1) + pi[0]
            # case 'PulsePol':
            #     sequence = ps.PulsePol(tau_times, 
            #                         pi_half[0], pi_half[1], 
            #                         pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
                # x_tau_times = (x_tau_times + 2*pi_half[1] + pi[0] + 2*pi_half[0] + pi[1])*2*cfg.n
        dig_cfg = self.digitizer_configure(
            exp_type="T2",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                'seq': 'T2',
                'seq_dd': cfg.t2_seq,
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'n': cfg.n,
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters
            }) 
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
        
        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
            subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList() # for storing optional PL data --> list of numpy arrays of shape (2, dig segment_size)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return
                
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            if pl_data is not None:
                if not (0 <= cfg.pl_pt < cfg.num_pts):
                    raise ValueError(
                        f"cfg.pl_pt ({cfg.pl_pt}) must be between 0 and {cfg.num_pts - 1} "
                        f"(num_pts={cfg.num_pts})"
                    )
            
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="T2",
                        x_data=tau_times * 1e6,
                        data=data,
                        fit_x=fit_x * 1e6,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps if pl_data is not None else None,
                        background_pl_sweeps=background_pl_sweeps if pl_data is not None else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type,
                                cfg.dataset, 
                                subseq_sweeps["signal"], 
                                subseq_sweeps["background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                        cfg.fit_type,
                        cfg.dataset, 
                        subseq_sweeps["signal"], 
                        subseq_sweeps["background"], 
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format, seq=cfg.t2_seq)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))
    
    @managed_experiment(token_prefix="T2RF", dataset_key="dataset")
    def T2_rf_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a T2 sweep over a set of precession time intervalsb with constant RF applied."""
        cfg = nvcfg.T2RFScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                tau_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)
            case 'linspace':
                tau_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq)
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = tau_times
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.XY8_N_RF(cfg.laser_init*1e9, tau_times * 1e9, 'yy', 
                            pi_half[0], pi_half[1], 
                            pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
                            
        x_tau_times = 8 * tau_times * 1e9 + 8*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
        # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*cfg.n) + 8*pi[1] + \
        #         7*x_tau_times/(8*cfg.n) + (x_tau_times/2)/(8*cfg.n))*cfg.n

        # sequence = ps.XY8_N_RF(tau_times, 'xy', 
        #                     pi_half[0], pi_half[1], 
        #                     pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
        # x_tau_times = 8*tau_times + 4*pi[0] + 4*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
        # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*cfg.n) + 4*pi[0] + \
        #             4*pi[1] + 7*x_tau_times/(8*cfg.n) + (x_tau_times/2)/(8*cfg.n))*cfg.n

        rf_times = 500 + 2*pi_half[1] + (x_tau_times)*cfg.n + 100

        dig_cfg = self.digitizer_configure(
            exp_type="T2",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                'seq': 'T2 RF',
                # 'seq_dd': cfg.t2_seq,
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'rf_freq': cfg.rf_pulse_freq,
                'rf_power': cfg.rf_pulse_power,
                'rf_phase': cfg.rf_pulse_phase,
                'rf_length': rf_times/1e9,
                'n': cfg.n,
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters
            }) 
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return

        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
            subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList() # for storing optional PL data --> list of numpy arrays of shape (2, dig segment_size)


        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return
                
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="T2 RF",
                        x_data=tau_times * 1e6,
                        data=data,
                        fit_x=fit_x * 1e6,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps if pl_data is not None else None,
                        background_pl_sweeps=background_pl_sweeps if pl_data is not None else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type,
                                cfg.dataset, 
                                subseq_sweeps["signal"], 
                                subseq_sweeps["background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                        cfg.fit_type,
                        cfg.dataset,
                        subseq_sweeps["signal"],
                        subseq_sweeps["background"],
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DQ", dataset_key="dataset")
    def DQ_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DQ sweep over a set of precession time intervals."""
        cfg = nvcfg.DQScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                tau_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)
            case 'linspace':
                tau_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)

        ### --- Define NV drive parameters --- ###
        if cfg.pulse_axis == 'y':
            delta = 90
        else:
            delta = 0
        iq_phases = [delta + 0, delta + 90, delta + 90, delta + 0] # set IQ phase relations for upper and lower sidebands
        sig_gen_freq = (cfg.freq_minus + cfg.freq_plus) / 2 # set mean value freq to sig gen 
        sideband_freq = sig_gen_freq - cfg.freq_minus # set sideband freq to match the inputted values
        pi_pulse_minus = cfg.pi_minus*1e9 # [ns] units for pulse streamer
        pi_pulse_plus = cfg.pi_plus*1e9 # [ns] units for pulse streamer

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = tau_times
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.DQ(cfg.laser_init * 1e9, tau_times * 1e9, cfg.pulse_axis, pi_pulse_minus, pi_pulse_plus, cfg.laser_readout * 1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="DQ",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###            
        try:
            hdawg.set_sequence(**{
                'seq': 'DQ',
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': sideband_freq, 
                'iq_phases': iq_phases,
                'pi_minus1': pi_pulse_minus/1e9, 
                'pi_plus1': pi_pulse_plus/1e9, 
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return

        subseq_sweeps = {
            'S0,0': StreamingList(),
            'S0,-1': StreamingList(),
            'S-1,-1': StreamingList(),
            'S-1,+1': StreamingList(),
        }
        if cfg.both_channels:
            subseq_2_sweeps = {
                'S0,0': StreamingList(),
                'S0,-1': StreamingList(),
                'S-1,-1': StreamingList(),
                'S-1,+1': StreamingList(),
            }
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList()

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return    
                
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="DQ",
                        x_data=tau_times * 1e6,
                        data=data,
                        fit_x=fit_x * 1e6,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        pl_data=pl_data,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps if pl_data is not None else None,
                        background_pl_sweeps=background_pl_sweeps if pl_data is not None else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type,
                                cfg.dataset, 
                                subseq_sweeps["S0,0"], 
                                subseq_sweeps["S0,-1"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(cfg.fit_type, 
        #                 cfg.dataset, subseq_sweeps["S0,0"], subseq_sweeps["S0,-1"], *cfg.fit_params
        #             )
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))
                    
    @managed_experiment(token_prefix="DEER", dataset_key="dataset")
    def DEER_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DEER sweep over a set of MW frequencies."""
        cfg = nvcfg.DEERScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety
        
        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg

        ### --- Define NV drive parameters --- ###
        frequencies = np.linspace(cfg.start, cfg.stop, cfg.num_pts)
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq)
        pi, pi_half = self._build_pi_times(cfg.pi)

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power)

        ### --- Default fit parameters for live fitting --- ###
        fit_value: list[float] = []
        fit_error: list[float] = []
        fit_units = []
        fit_x = np.linspace(cfg.start, cfg.stop, cfg.num_pts) / 1e6
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer, digitizer and SRS signal generator for experiment --- ###
        if cfg.drive_type == "Continuous":
            sequence = ps.DEER_CD(
                cfg.laser_init * 1e9,
                pi_half[0], pi_half[1],
                pi[0], pi[1],
                cfg.tau * 1e9,
                cfg.num_pts,
                cfg.laser_readout * 1e9,
            )
            dark_pulse = (cfg.pi / 2) + cfg.tau + cfg.pi + cfg.tau + (cfg.pi / 2)
        else:
            sequence = ps.DEER(
                cfg.laser_init * 1e9,
                pi_half[0], pi_half[1],
                pi[0], pi[1],
                cfg.tau * 1e9,
                cfg.num_pts,
                cfg.laser_readout * 1e9,
            )
            dark_pulse = cfg.dark_pi

        dig_cfg = self.digitizer_configure(
            exp_type="DEER",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )
        
        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**dict(
                seq=_hdawg_seq_name(cfg.drive_type),
                i_offset=cfg.i_offset,
                q_offset=cfg.q_offset,
                sideband_power=cfg.sideband_power,
                sideband_freq=cfg.sideband_freq,
                iq_phases=iq_phases,
                pihalf_x=pi_half[0] / 1e9,
                pihalf_y=pi_half[1] / 1e9,
                pi_x=pi[0] / 1e9,
                pi_y=pi[1] / 1e9,
                pi_pulse=dark_pulse,
                mw_power=cfg.awg_power,
                num_pts=cfg.num_pts,
                runs=cfg.runs,
                iters=cfg.iters,
                freqs=frequencies,
            ))
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
        
        subseq_sweeps = {
            "dark_signal": StreamingList(),
            "dark_background": StreamingList(),
            "echo_signal": StreamingList(),
            "echo_background": StreamingList(),
        }
        if cfg.both_channels:
             subseq_2_sweeps = {
                "dark_signal": StreamingList(),
                "dark_background": StreamingList(),
                "echo_signal": StreamingList(),
                "echo_background": StreamingList(),
            }
        
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="DEER",
                        x_data=frequencies / 1e6,
                        data=data,
                        fit_x=fit_x,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if cfg.fit_live:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
                                cfg.fit_type,
                                cfg.dataset,
                                subseq_sweeps["dark_signal"],
                                subseq_sweeps["dark_background"],
                                subseq_sweeps["echo_signal"],
                                subseq_sweeps["echo_background"],
                                *cfg.fit_params,
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if cfg.fit:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
                        cfg.fit_type, 
                        cfg.dataset,
                        subseq_sweeps["dark_signal"],
                        subseq_sweeps["dark_background"],
                        subseq_sweeps["echo_signal"],
                        subseq_sweeps["echo_background"],
                        *cfg.fit_params,
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if cfg.save:
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERRABI", dataset_key="dataset")
    def DEER_rabi_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DEER Rabi sweep over a set of MW pulse durations."""
        cfg = nvcfg.DEERRabiScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
        
        ### --- Default parameter array for sweep --- ###
        dark_taus = np.linspace(cfg.start, cfg.stop, cfg.num_pts)        
 
        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(
            cfg.sideband, cfg.freq, cfg.sideband_freq
        ) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(cfg.pi * 1e9 / 2)
            pi.append(cfg.pi * 1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = dark_taus   
        fit_y = np.ones(len(fit_x))

        # define pulse sequence
        sequence = ps.DEER_Rabi(cfg.laser_init*1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], 
                            cfg.tau*1e9, cfg.num_pts, cfg.laser_readout*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="DEER",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                'seq': 'DEER Rabi',     
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': cfg.dark_freq,
                'mw_power': cfg.awg_power, 
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters,
                'pi_pulses': dark_taus
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {
            "dark_signal": StreamingList(),
            "dark_background": StreamingList(),
            "echo_signal": StreamingList(),
            "echo_background": StreamingList(),
        }
        if cfg.both_channels:
            subseq_2_sweeps = {
                "dark_signal": StreamingList(),
                "dark_background": StreamingList(),
                "echo_signal": StreamingList(),
                "echo_background": StreamingList(),
            }

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="DEER",
                        x_data=dark_taus * 1e9,
                        data=data,
                        fit_x=fit_x * 1e9,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
                                cfg.fit_type,
                                cfg.dataset,
                                subseq_sweeps["dark_signal"],
                                subseq_sweeps["dark_background"],
                                subseq_sweeps["echo_signal"],
                                subseq_sweeps["echo_background"],
                                *cfg.fit_params,
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100
      
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
                        cfg.fit_type, 
                        cfg.dataset, 
                        subseq_sweeps["dark_signal"], 
                        subseq_sweeps["dark_background"], 
                        subseq_sweeps["echo_signal"], 
                        subseq_sweeps["echo_background"], 
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERFID", dataset_key="dataset")
    def DEER_FID_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DEER FID sweep over a set of MW pulse durations."""
        cfg = nvcfg.DEERFIDScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                tau_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)
            case 'linspace':
                tau_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(
            cfg.sideband, cfg.freq, cfg.sideband_freq
        ) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = cfg.dark_pi
        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Define fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = tau_times
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.DEER_FID(cfg.laser_init*1e9, tau_times * 1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="DEER",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                'seq': 'DEER FID',
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,  
                'pi_pulse': dark_pi, 
                'dark_freq': cfg.dark_freq,
                'mw_power': cfg.awg_power, 
                'num_pts': cfg.num_pts,
                'n': cfg.n,
                'runs': cfg.runs, 
                'iters': cfg.iters
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {
            "dark_signal": StreamingList(),
            "dark_background": StreamingList(),
            "echo_signal": StreamingList(),
            "echo_background": StreamingList(),
        }
        if cfg.both_channels:
             subseq_2_sweeps = {
                "dark_signal": StreamingList(),
                "dark_background": StreamingList(),
                "echo_signal": StreamingList(),
                "echo_background": StreamingList(),
            }

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="DEER",
                        x_data=tau_times * 1e6,
                        data=data,
                        fit_x=fit_x * 1e6,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
                                cfg.fit_type, 
                                cfg.dataset, 
                                subseq_sweeps["dark_signal"], 
                                subseq_sweeps["dark_background"], 
                                subseq_sweeps["echo_signal"], 
                                subseq_sweeps["echo_background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
        #                 cfg.fit_type, 
        #                 cfg.dataset, 
        #                 subseq_sweeps["dark_signal"], 
        #                 subseq_sweeps["dark_background"], 
        #                 subseq_sweeps["echo_signal"], 
        #                 subseq_sweeps["echo_background"], 
        #                 *cfg.fit_params
        #             )
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {cfg.dataset} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERFIDCD", dataset_key="dataset")
    def DEER_FID_CD_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a continuous drive DEER FID sweep over a set of free precession intervals.""" 
        cfg = nvcfg.DEERFIDCDScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety
        
        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
           
        ### --- Default parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                tau_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)
            case 'linspace':
                tau_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = cfg.dark_pi
        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = tau_times * 1e9
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.DEER_FID_CD(cfg.laser_init*1e9, tau_times * 1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
        # dark_pulses = cfg.pi/2 + tau_times + (cfg.pi + 2*tau_times)*(cfg.n-1) + cfg.pi + tau_times + cfg.pi/2 
        
        dig_cfg = self.digitizer_configure(
            exp_type="CD",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                'seq': 'DEER FID CD',                
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': cfg.dark_freq,
                'pi_pulse': dark_pi,
                'taus': tau_times,
                'mw_power': cfg.awg_power,
                'cd_mw_power': cfg.awg_cd_power, 
                'num_pts': cfg.num_pts,
                'n': cfg.n,
                'runs': cfg.runs, 
                'iters': cfg.iters
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {
            "dark_signal": StreamingList(),
            "dark_background": StreamingList(),
            "echo_signal": StreamingList(),
            "echo_background": StreamingList(),
            "cd_signal": StreamingList(),
            "cd_background": StreamingList(),
        }
        if cfg.both_channels:
             subseq_2_sweeps = {
                "dark_signal": StreamingList(),
                "dark_background": StreamingList(),
                "echo_signal": StreamingList(),
                "echo_background": StreamingList(),
                "cd_signal": StreamingList(),
                "cd_background": StreamingList(),
            }

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="CD",
                        x_data=tau_times * 1e6,
                        data=data,
                        fit_x=fit_x * 1e6,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
                                cfg.fit_type, 
                                cfg.dataset, 
                                subseq_sweeps["dark_signal"], 
                                subseq_sweeps["dark_background"], 
                                subseq_sweeps["echo_signal"], 
                                subseq_sweeps["echo_background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100
      
        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_data(
        #                 cfg.fit_type, 
        #                 cfg.dataset, 
        #                 subseq_sweeps["dark_signal"], 
        #                 subseq_sweeps["dark_background"], 
        #                 subseq_sweeps["echo_signal"], 
        #                 subseq_sweeps["echo_background"], 
        #                 *cfg.fit_params
        #             )
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {cfg.dataset} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))                

    # TODO: update implementation of DEER correlation scan and add sequence to PS/AWG driver
    @managed_experiment(token_prefix="DEERCORR", dataset_key="dataset")
    def DEER_corr_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DEER Correlation sweep over a set of frequencies.""" 
        cfg = nvcfg.DEERCorrScanCfg(**kwargs)

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg

        ### --- Default parameter array for sweep --- ###
        frequencies = np.linspace(cfg.start, cfg.stop, cfg.num_pts)
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq)
        pi, pi_half = self._build_pi_times(cfg.pi)

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power)

        ### --- Default fit parameters for live fitting --- ###
        fit_value: list[float] = []
        fit_error: list[float] = []
        fit_x = np.linspace(cfg.start, cfg.stop, cfg.num_pts) / 1e6
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.DEER_Corr(
            cfg.laser_init*1e9, frequencies*1e6, cfg.tau*1e9, cfg.t_corr*1e9, 
            pi_half[0], pi_half[1], cfg.laser_readout*1e9) # send to PS in [ns] units for timing, [Hz] for frequency

        dig_cfg = self.digitizer_configure(
            exp_type="Corr",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            pass
            # hdawg.set_sequence(**{
            #     'seq': 'DEER Corr', # FIXME: add DEER Corr sequence to HDAWG               
            #     'i_offset': cfg.i_offset,
            #     'q_offset': cfg.q_offset,
            #     'sideband_power': cfg.sideband_power,
            #     'sideband_freq': cfg.sideband_freq, 
            #     'iq_phases': iq_phases,
            #     'pihalf_x': pi_half[0]/1e9,
            #     'pihalf_y': pi_half[1]/1e9,
            #     'pi_x': pi[0]/1e9, 
            #     'pi_y': pi[1]/1e9,
            #     'dark_freq': cfg.dark_freq,
            #     'dark_pulse': dark_pi,
            #     'mw_power': cfg.awg_power, 
            #     'num_pts': cfg.num_pts,
            #     'runs': cfg.runs, 
            #     'iters': cfg.iters,
            #     'pi_pulses': dark_taus
            # })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return

        subseq_sweeps = {
            "dark_signal": StreamingList(),
            "dark_background": StreamingList(),
            "echo_signal": StreamingList(),
            "echo_background": StreamingList(),
        }
        if cfg.both_channels:
             subseq_2_sweeps = {
                "dark_signal": StreamingList(),
                "dark_background": StreamingList(),
                "echo_signal": StreamingList(),
                "echo_background": StreamingList(),
            }
        

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="Corr",
                        x_data=frequencies/1e6,
                        data=data,
                        fit_x=fit_x,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type, 
                                cfg.dataset,
                                subseq_sweeps["dark_signal"], 
                                subseq_sweeps["dark_background"], 
                                subseq_sweeps["echo_signal"], 
                                subseq_sweeps["echo_background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100
      
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                        cfg.fit_type, 
                        cfg.dataset,
                        subseq_sweeps["dark_signal"], 
                        subseq_sweeps["dark_background"], 
                        subseq_sweeps["echo_signal"], 
                        subseq_sweeps["echo_background"], 
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERCORRRABI", dataset_key="dataset")
    def DEER_corr_rabi_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DEER Correlation Rabi sweep over a set of MW pulses."""
        cfg = nvcfg.DEERCorrRabiScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety
        
        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        dark_taus = np.linspace(cfg.start, cfg.stop, cfg.num_pts)          

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = cfg.dark_pi
        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = dark_taus
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###  
        sequence = ps.DEER_Corr_Rabi(
            cfg.laser_init*1e9, dark_taus*1e9, cfg.tau*1e9, cfg.t_corr*1e9, 
            pi_half[0], pi_half[1], pi[0], pi[1], cfg.laser_readout*1e9) # send to PS in [ns] units
        
        dig_cfg = self.digitizer_configure(
            exp_type="Corr Rabi",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sequence(**{
                'seq': 'DEER Corr Rabi',                
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': cfg.dark_freq,
                'dark_pulse': dark_pi,
                'mw_power': cfg.awg_power, 
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters,
                'pi_pulses': dark_taus
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
             subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="Corr",
                        x_data=dark_taus * 1e9,
                        data=data,
                        fit_x=fit_x * 1e9,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type, 
                                cfg.dataset, 
                                subseq_sweeps["signal"], 
                                subseq_sweeps["background"], 
                                *cfg.fit_params)
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100
      
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                        cfg.fit_type, 
                        cfg.dataset, 
                        subseq_sweeps["signal"], 
                        subseq_sweeps["background"], 
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERT1", dataset_key="dataset")
    def DEER_T1_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DEER Correlation T1 sweep over a set of correlation intervals."""
        cfg = nvcfg.DEERT1ScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                t_corr_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)     
            case 'linspace':
                t_corr_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)     

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq) # iq_phases for y pulse by default

        # define pi pulses
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = cfg.dark_pi

        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = t_corr_times
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Electron_T1(cfg.laser_init*1e9, t_corr_times*1e9, cfg.tau*1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], dark_pi*1e9, cfg.laser_readout*1e9) # send to PS in [ns] units
        dig_cfg = self.digitizer_configure(
            exp_type="DEER",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        ### --- Upload AWG sequence --- ##
        try:
            hdawg.set_sequence(**{
                'seq': 'DEER Corr T1',                  
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': cfg.dark_freq,
                'dark_pulse': dark_pi,
                'mw_power': cfg.awg_power, 
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters})
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
                
        subseq_sweeps = {
            "with_py": StreamingList(), 
            "without_py": StreamingList(), 
            "with_ny": StreamingList(), 
            "without_ny": StreamingList()
        }
        if cfg.both_channels:
            subseq_2_sweeps = {
                "with_py": StreamingList(), 
                "without_py": StreamingList(), 
                "with_ny": StreamingList(), 
                "without_ny": StreamingList()
            }
        
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="DEER T1",
                        parse_type="DEER",
                        x_data=t_corr_times * 1e6,
                        data=data,
                        fit_x=fit_x * 1e6,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        slice_start=1,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_t1_data(
                                cfg.fit_type, 
                                cfg.dataset, 
                                subseq_sweeps["with_py"], 
                                subseq_sweeps["without_py"], 
                                subseq_sweeps["with_ny"], 
                                subseq_sweeps["without_ny"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100
      
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_deer_t1_data(
                        cfg.fit_type, 
                        cfg.dataset, 
                        subseq_sweeps["with_py"], 
                        subseq_sweeps["without_py"], 
                        subseq_sweeps["with_ny"], 
                        subseq_sweeps["without_ny"], 
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERT2", dataset_key="dataset")
    def DEER_T2_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a DEER Correlation T1 sweep over a set of correlation intervals."""
        cfg = nvcfg.DEERT2ScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        match cfg.array_type:
            case 'geomspace':
                t_times = np.geomspace(cfg.start, cfg.stop, cfg.num_pts)     
            case 'linspace':
                t_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts)     

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi_half = cfg.dark_pi/2
        dark_pi = cfg.dark_pi

        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = t_times
        fit_y = np.ones(len(fit_x))

        # define pulse sequence
        sequence = ps.Electron_T2(cfg.laser_init*1e9, t_times*1e9, cfg.tau*1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], dark_pi_half*1e9, dark_pi*1e9, cfg.laser_readout*1e9, cfg.deer_t2_buffer*1e9) # send to PS in [ns] units
        dig_cfg = self.digitizer_configure(
            exp_type="DEER T2",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )
           
        ### --- Upload AWG sequence --- ##
        try:
            hdawg.set_sequence(**{
                'seq': 'DEER T2',                  
                'i_offset': cfg.i_offset,
                'q_offset': cfg.q_offset,
                'sideband_power': cfg.sideband_power,
                'sideband_freq': cfg.sideband_freq, 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': cfg.dark_freq,
                'dark_half_pulse': dark_pi_half,
                'dark_pulse': dark_pi,
                'mw_power': cfg.awg_power, 
                'num_pts': cfg.num_pts,
                'runs': cfg.runs, 
                'iters': cfg.iters
            })    
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
        
        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
             subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="DEER T2",
                        x_data=t_times * 1e9,
                        data=data,
                        fit_x=fit_x * 1e9,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type, 
                                cfg.dataset, 
                                subseq_sweeps["signal"], 
                                subseq_sweeps["background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100
      
        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
        #                 cfg.fit_type, 
        #                 cfg.dataset, 
        #                 subseq_sweeps["signal"], 
        #                 subseq_sweeps["background"], 
        #                 *cfg.fit_params
        #             )
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {cfg.dataset} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="CORRSPEC", dataset_key="dataset")
    def Corr_Spec_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """Run a Correlation Spectroscopy NMR sweep over a set of precession time intervals."""
        cfg = nvcfg.CorrSpecScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety
        
        ### --- Devices --- ###  
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        t_corr_times = np.linspace(cfg.start, cfg.stop, cfg.num_pts) 

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq) # iq_phases for x pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(cfg.pi*1e9/2)
            pi.append(cfg.pi*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_units = []
        fit_x = t_corr_times
        fit_y = np.ones(len(fit_x))
            
        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Corr_Spectroscopy(cfg.laser_init*1e9, t_corr_times*1e9, cfg.tau*1e9, 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], cfg.n, cfg.laser_readout*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="Corr Spec",
            num_pts=cfg.num_pts,
            iters=cfg.iters,
            segment_size=cfg.segment_size,
            sampling_freq=cfg.dig_sampling_freq,
            dig_amplitude=cfg.dig_amplitude,
            read_channel=cfg.read_channel,
            both_channels=cfg.both_channels,
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )

        corr_spec_time = pi_half[0] + (4*pi[0] + 4*pi[1] + 8*cfg.tau*1e9)*cfg.n + pi_half[1] + cfg.stop*1e9 + \
                            pi_half[0] + (4*pi[0] + 4*pi[1] + 8*cfg.tau*1e9)*cfg.n + pi_half[1]
        
        total_exp_time = 100 + cfg.laser_init*1e9 + 500 + corr_spec_time + 100 + cfg.laser_readout + 100

        ### --- Upload AWG sequence --- ###
        print(f"cfg.sig_opt: {cfg.sig_opt}")
        try:
            if cfg.sig_opt == "Coil":
                hdawg.set_sequence(**{
                    'seq': 'NMR RF',
                    'i_offset': cfg.i_offset,
                    'q_offset': cfg.q_offset,
                    'sideband_power': cfg.sideband_power,
                    'sideband_freq': cfg.sideband_freq, 
                    'iq_phases': iq_phases,
                    'pihalf_x': pi_half[0]/1e9,
                    'pihalf_y': pi_half[1]/1e9,
                    'pi_x': pi[0]/1e9, 
                    'pi_y': pi[1]/1e9,
                    'n': cfg.n,
                    'num_pts': cfg.num_pts,
                    'runs': cfg.runs, 
                    'iters': cfg.iters,
                    'total_exp_time': total_exp_time*1e-9,
                    'rf_power': 0.2,
                    'rf_freq': 2.88e6,
                    'rf_phase': 0
                })
            else:
                hdawg.set_sequence(**{
                    'seq': 'NMR',
                    'seq_nmr': 'Correlation Spectroscopy',
                    'i_offset': cfg.i_offset,
                    'q_offset': cfg.q_offset,
                    'sideband_power': cfg.sideband_power,
                    'sideband_freq': cfg.sideband_freq, 
                    'iq_phases': iq_phases,
                    'pihalf_x': pi_half[0]/1e9,
                    'pihalf_y': pi_half[1]/1e9,
                    'pi_x': pi[0]/1e9, 
                    'pi_y': pi[1]/1e9,
                    'n': cfg.n,
                    'num_pts': cfg.num_pts,
                    'runs': cfg.runs, 
                    'iters': cfg.iters
                })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                fit_units=fit_units,
                start_time=time.perf_counter(),
                total_iters=cfg.iters,
                iters_completed=0,
                exception=_format_exception_msg(e),
            ))
            return
            
        subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        if cfg.both_channels:
            subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}
        signal_pl_sweeps = background_pl_sweeps = None
        if pl_data is not None:
            signal_pl_sweeps, background_pl_sweeps = StreamingList(), StreamingList() # for storing optional PL data --> list of numpy arrays of shape (2, dig segment_size)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
            try:
                ps.set_soft_trigger() # set pulsestreamer to start on software trigger & run infinitely
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) # set up sequence for streaming
                self.dig.config() # start digitizer --> waits for trigger from pulse sequence
                self.dig.start_buffer() # start digitizer (enable trigger)  
                ps.start_now(owner=token) # start pulse sequence
            except Exception as e:
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            if pl_data is not None:
                if not (0 <= cfg.pl_pt < cfg.num_pts):
                    raise ValueError(
                        f"cfg.pl_pt ({cfg.pl_pt}) must be between 0 and {cfg.num_pts - 1} "
                        f"(num_pts={cfg.num_pts})"
                    )
            
            for i in range(cfg.iters):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    self.acquire_data(
                        cfg=cfg,
                        exp_type="Noise Spectroscopy",
                        x_data=t_corr_times * 1e6,
                        data=data,
                        pl_data=pl_data,
                        fit_x=fit_x * 1e6,
                        fit_y=fit_y,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        subseq_sweeps=subseq_sweeps,
                        subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                        signal_pl_sweeps=signal_pl_sweeps if pl_data is not None else None,
                        background_pl_sweeps=background_pl_sweeps if pl_data is not None else None,
                        iters_completed=i + 1,
                        exp_start_time=exp_start_time,
                        **kwargs,
                    )
                except Exception as e:
                    failed = True
                    exception_type = _format_exception_msg(e)
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type, 
                                cfg.dataset, 
                                subseq_sweeps["signal"], 
                                subseq_sweeps["background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = cfg.iters, 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                        cfg.fit_type, 
                        cfg.dataset, 
                        subseq_sweeps["signal"], 
                        subseq_sweeps["background"], 
                        *cfg.fit_params
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if kwargs.get("save", False):
            run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_units=fit_units,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))    

    @managed_experiment(token_prefix="CASR", dataset_key="dataset")
    def CASR_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """
        Run a Coherently Averaged Synchronized Readout NMR sweep over a range of frequencies.
        Choose either coil drive or RF pi/2 pulse for nuclear spin control."""
        cfg = nvcfg.CASRScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
        
        ### --- Intervals in pulse sequence used to define time points on x-axis in seconds --- ###
        laser_init_time = cfg.laser_init*1e9
        singlet_decay = 500 
        pi = [cfg.pi*1e9, cfg.pi*1e9]
        pi_half = [pi[0]/2, pi[1]/2]
        tau = int(cfg.tau*1e9) # half period of central frequency [ns] - used in DD blocks
        print(f"tau = {tau} ns")
        period = 2*tau

        num_subseqs = 2 * cfg.num_pts # number of subsequences in one full CASR sequence (signal and background subsequences interleaving points)

        dd_time = pi_half[0] + (4*pi[0] + 4*pi[1] + 8*tau)*cfg.n + pi_half[1]

        mw_buffer_time = 100 # buffer time between DD block and readout pulse [ns]
        laser_read_time = cfg.laser_readout*1e9 # laser readout pulse [ns]
        wait_time = 100 # dead time at end of sequence before next subsequence [ns]

        t_subseq = laser_init_time + singlet_decay + dd_time + mw_buffer_time + laser_read_time + wait_time # total time for one full CASR subsequence [ns]
        print(f"initial t_subseq = {t_subseq}")
        try:
            if t_subseq % period != 0:
                nearest_integer = np.ceil(t_subseq/period)
                new_t_subseq = nearest_integer * period
                wait_time = new_t_subseq - (t_subseq - wait_time)
                assert wait_time >= 0, "new wait_time is unphysical (negative)"
                t_subseq = new_t_subseq
        except AssertionError as e:
            self.queue_from_exp.put_nowait([0, 'failed', None, e])
        else:
            try:
                if t_subseq % period > 1e-6:
                    assert math.isclose(t_subseq % period, period, abs_tol=1e-9), "Adjusted 't_subseq' still not an integer multiple of 1/f0"
                print(f"New wait time = {wait_time}")
                print(f"t_subseq = {t_subseq} ns")
                # print(f"period = {period} ns")
                print(f"f0 = {(0.5/((tau+pi[0])*1e-9))} Hz")
                print(f"f = {cfg.rf_pulse_freq} Hz")
                print(f"\u0394f = f - f0 = {(0.5/((tau+pi[0])*1e-9) - cfg.rf_pulse_freq)/1000} kHz")
                print(f"Total sequence time = {num_subseqs * t_subseq * 1e-9} s") # multiply by 2 because of signal and background subsequences interleaving points
            except AssertionError as e:  
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=None,
                    fit_error=None,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                )) 
            else:
                ### --- Define time points for x-axis based on sequence parameters --- ###
                times = np.linspace(2 * t_subseq - wait_time - laser_read_time / 2, 2 * cfg.num_pts * t_subseq - wait_time - laser_read_time / 2, cfg.num_pts) * 1e-9 # start in middle of readout pulse and define one point as two subseqs for normalization (signal and background subseqs interleaving points)

                ### --- Define NV drive parameters --- ###
                sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq) # iq_phases for x pulse by default

                self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_power) # configure signal generator for NV drive

                ### --- Default fit parameters for live fitting --- ###
                fit_value, fit_error = [], []
                fit_units = []
                fit_x = times
                fit_y = np.ones(len(fit_x))

                ### --- Set up pulse streamer and digitizer for experiment --- ###
                if cfg.sig_opt == 'Coil':
                    print("Using CASR Coil sequence")
                    rf_duration = num_subseqs * t_subseq * 1e-9 # coil on for complete sequence duration (num_pts * 2 b/c signal and background subseqs interleaving points)
                    seq_rf = ps.CASR_RF_Coil() # coil drive version
                else:
                    rf_duration = cfg.rf_pi_half # RF pi/2 pulse duration for nuclear spin control
                    seq_rf = ps.CASR_RF(cfg.rf_pi_half * 1e9)
                
                seq_nv = ps.CASR_NV(laser_init_time, singlet_decay, 
                                pi_half[0], pi_half[1], pi[0], pi[1], 
                                tau, cfg.n, mw_buffer_time, cfg.laser_readout, wait_time) # NV DD subsequence
                
                dig_cfg = self.digitizer_configure(
                    exp_type="CASR",
                    num_pts=cfg.num_pts,
                    iters=cfg.iters,
                    segment_size=cfg.segment_size,
                    sampling_freq=cfg.dig_sampling_freq,
                    dig_amplitude=cfg.dig_amplitude,
                    read_channel=cfg.read_channel,
                    both_channels=cfg.both_channels,
                    coupling=cfg.dig_coupling,
                    termination=cfg.dig_termination,
                    pretrig_size=cfg.pretrig_size,
                    dig_timeout=cfg.dig_timeout,
                    runs=cfg.runs,
                )

                ### --- Upload AWG sequence --- ##
                try:
                    hdawg.set_sequence(**{
                        'seq': 'CASR',
                        'i_offset': cfg.i_offset,
                        'q_offset': cfg.q_offset,
                        'sideband_power': cfg.sideband_power,
                        'sideband_freq': cfg.sideband_freq, 
                        'iq_phases': iq_phases,
                        'pihalf_x': pi_half[0]/1e9,
                        'pihalf_y': pi_half[1]/1e9,
                        'pi_x': pi[0]/1e9, 
                        'pi_y': pi[1]/1e9,
                        'n_R': num_subseqs/2,  # number of repetitions of the NV DD subsequence in one normalized CASR data point (signal and background subsequences interleaving points)
                        'n': cfg.n,
                        'rf_freq': cfg.rf_pulse_freq,
                        'rf_power': cfg.rf_pulse_power,
                        'rf_phase': cfg.rf_pulse_phase,
                        'rf_pihalf': rf_duration})
                        # 'rf_pihalf': cfg.rf_pi_half})  
                        # 'rf_pihalf': cfg.num_pts*t_subseq*1e-9})
                except Exception as e:
                    self.queue_from_exp.put_nowait(self.build_status_msg(
                        status="failed",
                        percent_completed=0,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        fit_units=fit_units,
                        start_time=time.perf_counter(),
                        total_iters=cfg.iters,
                        iters_completed=0,
                        exception=_format_exception_msg(e),
                    ))
                    return
                
                subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
                if cfg.both_channels:
                    subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}

                self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
                # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

                ### --- Initialize experiment state variables --- ###
                stopped = False
                failed = False
                exception_type = None
                iters_completed = 0
                percent_completed = 0

                ### --- Open laser shutter and emit MW for NV drive --- ###
                with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
                    try:
                        ps.set_soft_trigger()

                        # new pulse streamer format:
                        # upload both sequences to pulse streamer
                        # 1. CASR RF sequence -> pi/2 on nuclear spins (n_runs = 1)
                        # 2. CASR NV sequence -> pi/2 on electron spins (n_runs = num_pts x 2)
                        ps.upload(slot_nr=0, data=seq_rf, n_runs=1, next_action=NextAction.SWITCH_SLOT, when=When.IMMEDIATE, owner=token) #upload initial sequence data with n_runs=1
                        ps.upload(slot_nr=1, data=seq_nv, n_runs=num_subseqs,next_action=NextAction.SWITCH_SLOT, when=When.IMMEDIATE, owner=token) #upload measurement sequence with n_runs=num_subseqs
                        
                        ps.start(slot_nr=0, slots_to_run=PulseStreamer.REPEAT_INFINITELY, owner=token) # start on slot 0 (seq_rf) and run rf and nv sequences alternatively infinitely, switching slots automatically 
                        
                        self.dig.config()
                        self.dig.start_buffer()
                        ps.start_now(owner=token)
                    except Exception as e:
                        self.queue_from_exp.put_nowait(self.build_status_msg(
                            status="failed",
                            percent_completed=0,
                            fit_value=fit_value,
                            fit_error=fit_error,
                            fit_units=fit_units,
                            start_time=time.perf_counter(),
                            total_iters=cfg.iters,
                            iters_completed=0,
                            exception=_format_exception_msg(e),
                        ))
                        return
            
                    exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

                    ### --- Main experiment loop --- ###
                    for i in range(cfg.iters):
                        if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                            stopped = True
                            iters_completed = i
                            percent_completed = int(100 * iters_completed / cfg.iters)
                            break

                        try:
                            self.acquire_data(
                                cfg=cfg,
                                exp_type="CASR",
                                x_data=times * 1e3,
                                data=data,
                                fit_x=fit_x * 1e3,
                                fit_y=fit_y,
                                fit_value=fit_value,
                                fit_error=fit_error,
                                subseq_sweeps=subseq_sweeps,
                                subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                                iters_completed=i + 1,
                                exp_start_time=exp_start_time,
                                slice_end=-1,
                                **kwargs,
                            )
                        except Exception as e:
                            failed = True
                            exception_type = _format_exception_msg(e)
                            iters_completed = i
                            percent_completed = int(100 * iters_completed / cfg.iters)
                            break

                        if kwargs.get("fit_live", False):  
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                        cfg.fit_type, 
                                        cfg.dataset, 
                                        subseq_sweeps["signal"], 
                                        subseq_sweeps["background"], 
                                        *cfg.fit_params
                                    )
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {cfg.dataset} measurement, {e}")
                        
                        iters_completed = i + 1
                        percent_completed = int(100 * iters_completed / cfg.iters)

                        self.queue_from_exp.put_nowait(self.build_status_msg(
                            status="in progress",
                            percent_completed=percent_completed,
                            fit_value=fit_value,
                            fit_error=fit_error,
                            fit_units=fit_units,
                            start_time=exp_start_time,
                            total_iters=cfg.iters,
                            iters_completed=iters_completed,
                        ))

                ### --- Experiment complete: final save and status update --- ###
                if not stopped and not failed:
                    iters_completed, percent_completed = cfg.iters, 100

                # if kwargs.get("fit", False):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         try:
                #             fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                #                 cfg.fit_type, 
                #                 cfg.dataset, 
                #                 subseq_sweeps["signal"], 
                #                 subseq_sweeps["background"], 
                #                 *cfg.fit_params
                #             )
                #         except (RuntimeError, OptimizeWarning) as e:
                #             _logger.warning(f"For {cfg.dataset} measurement, {e}")

                if kwargs.get("save", False):
                    run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

                status = "failed" if failed else ("stopped" if stopped else "complete")
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status=status,
                    percent_completed=int(percent_completed),
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                    exception=exception_type if failed else None,
                ))                
    
    @managed_experiment(token_prefix="CASRIR", dataset_key="dataset")
    def CASRIR_scan(self, *, mgr, data, pl_data, token, **kwargs):
        """
        Run a Coherently Averaged Synchronized Readout NMR sweep over a range of frequencies.
        Choose either coil drive or RF pi/2 pulse for nuclear spin control."""
        cfg = nvcfg.CASRIRScanCfg(**kwargs) # validate and parse kwargs into a dataclass for easier access and type safety

        ### --- Devices --- ###
        # laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        daq = mgr.daq
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
        
        ### --- Intervals in pulse sequence used to define time points on x-axis in seconds --- ###
        laser_init_time = cfg.laser_init*1e9
        singlet_decay = 500 
        pi = [cfg.pi*1e9, cfg.pi*1e9]
        pi_half = [pi[0]/2, pi[1]/2]
        tau = int(cfg.tau*1e9) # half period of central frequency [ns] - used in DD blocks
        print(f"tau = {tau} ns")
        period = 2*tau

        dd_time = pi_half[0] + (4*pi[0] + 4*pi[1] + 8*tau)*cfg.n + pi_half[1]

        mw_buffer_time = 100 # buffer time between DD block and readout pulse [ns]
        laser_read_time = cfg.laser_readout*1e9 # laser readout pulse [ns]
        wait_time = 100 # dead time at end of sequence before next subsequence [ns]

        t_seq = laser_init_time + singlet_decay + dd_time + mw_buffer_time + laser_read_time + wait_time # total time for one full CASR sequence [ns]
        print(f"initial t_seq = {t_seq}")
        try:
            if t_seq % period != 0:
                nearest_integer = np.ceil(t_seq/period)
                new_t_seq = nearest_integer * period
                wait_time = new_t_seq - (t_seq - wait_time)
                assert wait_time >= 0, "new wait_time is unphysical (negative)"
                t_seq = new_t_seq
        except AssertionError as e:
            self.queue_from_exp.put_nowait([0, 'failed', None, e])
        else:
            try:
                if t_seq % period > 1e-6:
                    assert math.isclose(t_seq % period, period, abs_tol=1e-9), "Adjusted 't_seq' still not an integer multiple of 1/f0"
                print(f"New wait time = {wait_time}")
                print(f"t_seq = {t_seq} ns")
                print(f"period = {period} ns")
                print(f"\u0394f = f - f0 = {(0.5/((tau+pi[0])*1e-9) - cfg.rf_pulse_freq)/1000} kHz")
            except AssertionError as e:  
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=None,
                    fit_error=None,
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=_format_exception_msg(e),
                ))
            else:
                ### --- Define time points for x-axis based on sequence parameters --- ###
                times = np.linspace(t_seq - wait_time - laser_read_time / 2, cfg.num_pts * t_seq, cfg.num_pts) * 1e-9

                ### --- Define NV drive parameters --- ###
                sig_gen_freq, iq_phases = self.choose_sideband(cfg.sideband, cfg.freq, cfg.sideband_freq) # iq_phases for x pulse by default

                self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=cfg.rf_pulse_power) # configure signal generator for NV drive

                ### --- Default fit parameters for live fitting --- ###
                fit_value, fit_error = [], []
                fit_units = []
                fit_x = times
                fit_y = np.ones(len(fit_x))

                ### --- Set up pulse streamer and digitizer for experiment --- ###
                if cfg.sig_opt == 'Coil':
                    print("Using CASR Coil sequence")
                    rf_duration = cfg.num_pts * t_seq * 1e-9 # coil on for entire sequence duration
                    seq_rf = ps.CASR_RF_Coil() # coil drive version
                else:
                    rf_duration = cfg.rf_pi_half # RF pi/2 pulse duration for nuclear spin control
                    seq_rf = ps.CASR_RF(cfg.rf_pi_half * 1e9)
                
                seq_nv = ps.CASR_NV(laser_init_time, singlet_decay, 
                                pi_half[0], pi_half[1], pi[0], pi[1], 
                                tau, cfg.n, mw_buffer_time, cfg.laser_readout, wait_time) # NV DD subsequence
                
                dig_cfg = self.digitizer_configure(
                    exp_type="CASR",
                    num_pts=cfg.num_pts,
                    iters=cfg.iters,
                    segment_size=cfg.segment_size,
                    sampling_freq=cfg.dig_sampling_freq,
                    dig_amplitude=cfg.dig_amplitude,
                    read_channel=cfg.read_channel,
                    both_channels=cfg.both_channels,
                    coupling=cfg.dig_coupling,
                    termination=cfg.dig_termination,
                    pretrig_size=cfg.pretrig_size,
                    dig_timeout=cfg.dig_timeout,
                    runs=cfg.runs,
                )

                ### --- Upload AWG sequence --- ##
                try:
                    hdawg.set_sequence(**{
                        'seq': 'CASR',
                        'i_offset': cfg.i_offset,
                        'q_offset': cfg.q_offset,
                        'sideband_power': cfg.sideband_power,
                        'sideband_freq': cfg.sideband_freq, 
                        'iq_phases': iq_phases,
                        'pihalf_x': pi_half[0]/1e9,
                        'pihalf_y': pi_half[1]/1e9,
                        'pi_x': pi[0]/1e9, 
                        'pi_y': pi[1]/1e9,
                        'n_R': cfg.num_pts,
                        'n': cfg.n,
                        'rf_freq': cfg.rf_pulse_freq,
                        'rf_power': cfg.rf_pulse_power,
                        'rf_phase': cfg.rf_pulse_phase,
                        'rf_pihalf': rf_duration})
                        # 'rf_pihalf': cfg.rf_pi_half})  
                        # 'rf_pihalf': cfg.num_pts*t_seq*1e-9})
                except Exception as e:
                    self.queue_from_exp.put_nowait(self.build_status_msg(
                        status="failed",
                        percent_completed=0,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        fit_units=fit_units,
                        start_time=time.perf_counter(),
                        total_iters=cfg.iters,
                        iters_completed=0,
                        exception=_format_exception_msg(e),
                    ))
                    return
                
                subseq_sweeps = {"signal": StreamingList(), "background": StreamingList()}
                if cfg.both_channels:
                    subseq_2_sweeps = {"signal": StreamingList(), "background": StreamingList()}

                self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
                # laser.set_diode_current_realtime(cfg.laser_power) # set laser power

                ### --- Initialize experiment state variables --- ###
                stopped = False
                failed = False
                exception_type = None
                iters_completed = 0
                percent_completed = 0

                ### --- Open laser shutter and emit MW for NV drive --- ###
                with self._shutter_open(laser_shutter, daq, cfg.detector), _rf_on(sig_gen):
                    try:
                        ps.set_soft_trigger()

                        # new pulse streamer format:
                        # upload both sequences to pulse streamer
                        # 1. CASR RF sequence -> pi/2 on nuclear spins (n_runs = 1)
                        # 2. CASR NV sequence -> pi/2 on electron spins (n_runs = num_pts x 2)
                        ps.upload(slot_nr=0, data=seq_rf, n_runs=1, next_action=NextAction.SWITCH_SLOT, when=When.IMMEDIATE, owner=token) #upload initial sequence data with n_runs=1
                        ps.upload(slot_nr=1, data=seq_nv, n_runs=cfg.num_pts*2,next_action=NextAction.SWITCH_SLOT, when=When.IMMEDIATE, owner=token) #upload measurement sequence with n_runs=num_pts*2
                        
                        ps.start(slot_nr=0, slots_to_run=PulseStreamer.REPEAT_INFINITELY, owner=token) # start on slot 0 (seq_rf) and run rf and nv sequences alternatively infinitely, switching slots automatically 
                        
                        self.dig.config()
                        self.dig.start_buffer()
                        ps.start_now(owner=token)
                    except Exception as e:
                        self.queue_from_exp.put_nowait(self.build_status_msg(
                            status="failed",
                            percent_completed=0,
                            fit_value=fit_value,
                            fit_error=fit_error,
                            fit_units=fit_units,
                            start_time=time.perf_counter(),
                            total_iters=cfg.iters,
                            iters_completed=0,
                            exception=_format_exception_msg(e),
                        ))
                        return
            
                    exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

                    ### --- Main experiment loop --- ###
                    for i in range(cfg.iters):
                        if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                            stopped = True
                            iters_completed = i
                            percent_completed = int(100 * iters_completed / cfg.iters)
                            break

                        try:
                            self.acquire_data(
                                cfg=cfg,
                                exp_type="CASR",
                                x_data=times * 1e3,
                                data=data,
                                fit_x=fit_x * 1e3,
                                fit_y=fit_y,
                                fit_value=fit_value,
                                fit_error=fit_error,
                                subseq_sweeps=subseq_sweeps,
                                subseq_2_sweeps=subseq_2_sweeps if cfg.both_channels else None,
                                iters_completed=i + 1,
                                exp_start_time=exp_start_time,
                                slice_end=-1,
                                **kwargs,
                            )
                        except Exception as e:
                            failed = True
                            exception_type = _format_exception_msg(e)
                            iters_completed = i
                            percent_completed = int(100 * iters_completed / cfg.iters)
                            break

                        if kwargs.get("fit_live", False):  
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                        cfg.fit_type, 
                                        cfg.dataset, 
                                        subseq_sweeps["signal"], 
                                        subseq_sweeps["background"], 
                                        *cfg.fit_params
                                    )
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {cfg.dataset} measurement, {e}")
                        
                        iters_completed = i + 1
                        percent_completed = int(100 * iters_completed / cfg.iters)

                        self.queue_from_exp.put_nowait(self.build_status_msg(
                            status="in progress",
                            percent_completed=percent_completed,
                            fit_value=fit_value,
                            fit_error=fit_error,
                            fit_units=fit_units,
                            start_time=exp_start_time,
                            total_iters=cfg.iters,
                            iters_completed=iters_completed,
                        ))

                ### --- Experiment complete: final save and status update --- ###
                if not stopped and not failed:
                    iters_completed, percent_completed = cfg.iters, 100

                if kwargs.get("fit", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y, fit_units = self.fit_data(
                                cfg.fit_type, 
                                cfg.dataset, 
                                subseq_sweeps["signal"], 
                                subseq_sweeps["background"], 
                                *cfg.fit_params
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                if kwargs.get("save", False):
                    run_save(cfg.dataset, cfg.filename, [cfg.directory], file_format=cfg.file_format)

                status = "failed" if failed else ("stopped" if stopped else "complete")
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status=status,
                    percent_completed=int(percent_completed),
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_units=fit_units,
                    start_time=exp_start_time,
                    total_iters=cfg.iters,
                    iters_completed=iters_completed,
                    exception=exception_type if failed else None,
                ))                
    


    ############################
    ### --- Data Fitting --- ###
    ############################

    @staticmethod
    def negative_lorentzian(x, A, x0, gamma, c):
        return -A / (1 + ((x - x0) / gamma)**2) + c
    
    @staticmethod
    def positive_lorentzian(x, A, x0, gamma, c):
        return A / (1 + ((x - x0) / gamma)**2) + c
    
    @staticmethod
    def decaying_cosine(x, A, t_decay, T, phi, c):
        return A * np.exp(-x / t_decay) * np.cos(2 * np.pi * x / T + phi) + c
    
    @staticmethod
    def stretched_exponential(x, A, T, n, c):
        return A*np.exp(-(x/T)**n) + c

    @staticmethod
    def mod_stretched_exponential(x, A, T, n, a1, f1, phi1, a2, f2, phi2):
        return A * np.exp(-(x / T)**n) * (1 - a1 * np.sin(2 * np.pi * f1 * x / 4 + phi1)**2) * (1 - a2 * np.sin(2 * np.pi * f2 * x / 4 + phi2)**2)

    # FIXME: make T1_nv and n_nv fixed parameters, not fit parameters
    @staticmethod
    def deer_t1_stretched_exponential(x, A, T1_nv, n_nv, T1_e, n_e, c):
        return A * np.exp(-(x / T1_nv)**n_nv - (x / T1_e)**n_e) + c

    def fit_data(self, fit_type, exp, sig_data, back_data, *args):
        """Fit experimental data to a model function.
        
        NOTE: sig_data and back_data are StreamingLists populated by acquire_data().
        If slice_start was specified during acquisition, these lists contain ONLY 
        the sliced data (first point already removed). Fitting operates on the 
        sliced data only, never on the discarded points.
        """
        # Check if data is available before attempting to fit
        if len(sig_data) == 0 or len(back_data) == 0:
            return [], [], [], [], []
        
        # Check all shapes are consistent before stacking
        sig_shapes = [arr.shape for arr in sig_data]
        back_shapes = [arr.shape for arr in back_data]
        
        if len(set(sig_shapes)) > 1:
            print(f"ERROR: sig_data has inconsistent shapes: {sig_shapes}")
            _logger.error(f"sig_data shape mismatch: {sig_shapes}")
            return [], [], [], [], []
        
        if len(set(back_shapes)) > 1:
            print(f"ERROR: back_data has inconsistent shapes: {back_shapes}")
            _logger.error(f"back_data shape mismatch: {back_shapes}")
            return [], [], [], [], []
        
        # Combine all signal sweeps into a single 3D array and average
        all_signal_data = np.stack(sig_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_sig = np.mean(all_signal_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all background sweeps into a single 3D array and average
        all_background_data = np.stack(back_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_bg = np.mean(all_background_data[1, :, :], axis=1)  # Shape: (10,)

        # Compute the microwave_times (assumed constant across sweeps)
        x_values = all_signal_data[0, :, 0]  # Shape: (10,) — x_data from experiment in display units
        
        # Convert x_values to SI units based on experiment type for consistent fitting
        if exp in ('odmr', 'odmr rf'):
            # ODMR experiments: x_values in GHz, convert to Hz (SI)
            x_values = x_values * 1e9
        elif exp in ('nmr', 'casr'):
            # NMR/CASR experiments: frequency in MHz/kHz, convert to Hz (SI)
            if exp == 'nmr':
                x_values = x_values / 1e6  # MHz → Hz
            else:  # casr
                x_values = x_values / 1e3  # kHz → Hz
        elif exp in ('rabi', 'corr rabi', 'deer t2'):
            # Time-domain in nanoseconds: convert to seconds (SI)
            x_values = x_values / 1e9
        elif exp == 't1':
            # T1 in milliseconds: convert to seconds (SI)
            x_values = x_values / 1e3
        elif exp in ('t2', 'dq'):
            # Time-domain in microseconds: convert to seconds (SI)
            x_values = x_values / 1e6
        else:
            # Default: assume time in milliseconds
            x_values = x_values / 1e3
        
        x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting in SI

        # Compute the ratio/difference of averaged signal and background for fitting
        if exp == 'odmr' or exp == 'rabi':
            y_values = averaged_sig / averaged_bg  # Shape: (10,)
        else:  # t1, t2, dq, and DEER time-domain experiments
            y_values = averaged_bg - averaged_sig

        # Normalize exp to lowercase for consistent matching throughout fit cases
        exp_lower = exp.lower() if isinstance(exp, str) else exp

        # Initial guesses: kept in SI units (no conversions before fitting)
        initial_guess = list(args)
        
        # Perform curve fitting 
        """
        "Neg. Lorentz.",
        "Pos. Lorentz.",
        "Two Neg. Lorentz."
        "Decaying Cos.",
        "Stretched Exp.",
        "Modulated Str. Exp.",
        "DEER T1 Str. Exp."
        """
        match fit_type:
            case 'Neg. Lorentz.':
                # Keep initial_guess in SI units (Hz) for fitting
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.negative_lorentzian, x_values, y_values, p0=initial_guess)
                y_fit = self.negative_lorentzian(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower in ('odmr', 'odmr rf'):
                    # Display center frequency and linewidth in GHz
                    display_params[1] /= 1e9  # x0: Hz → GHz
                    display_errors[1] /= 1e9
                    display_params[2] /= 1e9  # gamma: Hz → GHz
                    display_errors[2] /= 1e9
                elif exp_lower == 'deer':
                    # Display center frequency and linewidth in MHz
                    display_params[1] /= 1e6  # x0: Hz → MHz
                    display_errors[1] /= 1e6
                    display_params[2] /= 1e6  # gamma: Hz → MHz
                    display_errors[2] /= 1e6
                elif exp_lower == 'nmr':
                    # Display center frequency in MHz, linewidth in kHz
                    display_params[1] /= 1e6  # x0: Hz → MHz
                    display_errors[1] /= 1e6
                    display_params[2] /= 1e3  # gamma: Hz → kHz
                    display_errors[2] /= 1e3
                elif exp_lower == 'casr':
                    # Display center frequency and linewidth in kHz
                    display_params[1] /= 1e3  # x0: Hz → kHz
                    display_errors[1] /= 1e3
                    display_params[2] /= 1e3  # gamma: Hz → kHz
                    display_errors[2] /= 1e3
                
                fitted_values = [round(i, 4) for i in display_params]
                fitted_errors = [round(i, 4) for i in display_errors]
                # Units for Neg. Lorentz: [amplitude, freq, linewidth, vertical_offset]
                if exp_lower in ('odmr', 'odmr rf'):
                    fit_units = ['', 'GHz', 'GHz', '']
                elif exp_lower == 'deer':
                    fit_units = ['', 'MHz', 'MHz', '']
                elif exp_lower == 'nmr':
                    fit_units = ['', 'MHz', 'kHz', '']
                elif exp_lower == 'casr':
                    fit_units = ['', 'kHz', 'kHz', '']
                else:
                    fit_units = ['', '', '', '']
            
            case 'Pos. Lorentz.':
                # Keep initial_guess in SI units (Hz) for fitting
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.positive_lorentzian, x_values, y_values, p0=initial_guess)
                y_fit = self.positive_lorentzian(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower == 'nmr':
                    # Display center frequency in MHz, linewidth in kHz
                    display_params[1] /= 1e6  # x0: Hz → MHz
                    display_errors[1] /= 1e6
                    display_params[2] /= 1e3  # gamma: Hz → kHz
                    display_errors[2] /= 1e3
                elif exp_lower == 'casr':
                    # Display center frequency and linewidth in kHz
                    display_params[1] /= 1e3  # x0: Hz → kHz
                    display_errors[1] /= 1e3
                    display_params[2] /= 1e3  # gamma: Hz → kHz
                    display_errors[2] /= 1e3
                
                fitted_values = [round(i, 4) for i in display_params]
                fitted_errors = [round(i, 4) for i in display_errors]
                # Units for Pos. Lorentz: [amplitude, freq, linewidth, vertical_offset]
                if exp_lower == 'nmr':
                    fit_units = ['', 'MHz', 'kHz', '']
                elif exp_lower == 'casr':
                    fit_units = ['', 'kHz', 'kHz', '']
                else:
                    fit_units = ['', '', '', '']

            case 'Two Neg. Lorentz.':
                # Keep initial_guess in SI units (Hz) for fitting
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.negative_lorentzian, x_values, y_values, p0=initial_guess)
                y_fit = self.negative_lorentzian(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units (GHz for ODMR RF)
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower == 'odmr rf':
                    # Display frequencies in GHz
                    for idx in [1, 5]:  # x0 and x0_rf for two Lorentzians
                        if idx < len(display_params):
                            display_params[idx] /= 1e9
                            display_errors[idx] /= 1e9
                    for idx in [2, 6]:  # gamma and gamma_rf
                        if idx < len(display_params):
                            display_params[idx] /= 1e9
                            display_errors[idx] /= 1e9
                
                fitted_values = [round(i, 4) for i in display_params]
                fitted_errors = [round(i, 4) for i in display_errors]
                # Units for Two Neg. Lorentz: [amp1, freq1, linewidth1, offset1, amp2, freq2, linewidth2, offset2]
                if exp_lower == 'odmr rf':
                    fit_units = ['', 'GHz', 'GHz', '', '', 'GHz', 'GHz', '']
                else:
                    fit_units = ['', '', '', '', '', '', '', '']

            case 'Decaying Cos.':
                # Keep initial_guess in SI units (seconds) for fitting
                
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.decaying_cosine, x_values, y_values, p0=initial_guess)
                y_fit = self.decaying_cosine(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units based on experiment
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower in ('rabi', 'deer rabi', 'corr rabi', 'deer t2'):
                    # Display t_decay and T in nanoseconds
                    display_params[1] *= 1e9  # t_decay: s → ns
                    display_errors[1] *= 1e9
                    display_params[2] *= 1e9  # T: s → ns
                    display_errors[2] *= 1e9
                elif exp_lower in ('t2', 'dq'):
                    # Display t_decay and T in microseconds
                    display_params[1] *= 1e6  # t_decay: s → μs
                    display_errors[1] *= 1e6
                    display_params[2] *= 1e6  # T: s → μs
                    display_errors[2] *= 1e6
                    
                fitted_values = [round(i, 2) for i in display_params]
                fitted_errors = [round(i, 2) for i in display_errors]
                
                # For Rabi/DEER Rabi/FID experiments: Find pi pulse time as argmin of fitted curve
                # This method captures the first minimum regardless of inhomogeneity or parameter stretching
                if exp_lower in ('rabi', 'deer rabi', 'corr rabi', 'fid', 'fid cd'):
                    i_min = np.argmin(y_fit)
                    pi_time_si = x_fit[i_min]
                    
                    # Period uncertainty from covariance (param_errors[2])
                    # The minimum location is most sensitive to the period parameter T
                    pi_time_error_si = param_errors[2]
                    
                    # Convert to display units
                    if exp_lower in ('rabi', 'deer rabi', 'corr rabi', 'deer t2'):
                        scale = 1e9  # → ns
                    else:  # 'fid', 'fid cd'
                        scale = 1e6  # → μs
                    
                    fitted_values[2] = round(pi_time_si * scale, 2)
                    fitted_errors[2] = round(pi_time_error_si * scale, 2)
                
                # Units for Decaying Cos: [amplitude, decay_time, period, phase, offset]
                if exp_lower in ('rabi', 'deer rabi', 'corr rabi', 'deer t2'):
                    fit_units = ['', 'ns', 'ns', '', '']
                elif exp_lower in ('t2', 'dq'):
                    fit_units = ['', 'μs', 'μs', '', '']
                else:
                    fit_units = ['', '', '', '', '']
            
            case 'Stretched Exp.':
                # Keep initial_guess in SI units (seconds for time constants) for fitting
                
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.stretched_exponential, x_values, y_values, p0=initial_guess)
                y_fit = self.stretched_exponential(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units based on experiment
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower == 't1':
                    # Display time constant in milliseconds
                    display_params[1] *= 1e3  # T: s → ms
                    display_errors[1] *= 1e3
                elif exp_lower in ('t2', 'dq', 'fid', 'fid cd', 'deer t1'):
                    # Display time constant in microseconds
                    display_params[1] *= 1e6  # T: s → μs
                    display_errors[1] *= 1e6
                
                fitted_values = [round(i, 3) for i in display_params]
                fitted_errors = [round(i, 3) for i in display_errors]
                # Units for Stretched Exp: [amplitude, time_constant, exponent, offset]
                if exp_lower == 't1':
                    fit_units = ['', 'ms', '', '']
                elif exp_lower in ('t2', 'dq', 'fid', 'fid cd', 'deer t1'):
                    fit_units = ['', 'μs', '', '']
                else:
                    fit_units = ['', '', '', '']
            
            case 'Modulated Str. Exp.':
                # Keep initial_guess in SI units: T2 in seconds, frequencies in Hz (SI)
                
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.mod_stretched_exponential, x_values, y_values, p0=initial_guess)
                y_fit = self.mod_stretched_exponential(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units (T2 for T2 experiments)
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower == 't2':
                    # Display T2 in microseconds, modulation frequencies in MHz
                    display_params[1] *= 1e6  # T2: s → μs
                    display_errors[1] *= 1e6
                    for idx in [4, 7]:  # f1, f2
                        if idx < len(display_params):
                            display_params[idx] /= 1e6  # Hz → MHz
                            display_errors[idx] /= 1e6
                
                fitted_values = [round(i, 3) for i in display_params]
                fitted_errors = [round(i, 3) for i in display_errors]
                # Units for Modulated Str. Exp: [amplitude, T2, exponent, a1, f1, phi1, a2, f2, phi2]
                if exp_lower == 't2':
                    fit_units = ['', 'μs', '', '', 'MHz', '', '', 'MHz', '']
                else:
                    fit_units = ['', '', '', '', '', '', '', '', '']
            
            case _:
                print("No fit type selected.")
                return 0, 0, 0, 0, []
              
        return fitted_values, fitted_errors, x_fit, y_fit, fit_units
    
    def fit_deer_data(self, fit_type, exp, dark_sig_data, dark_back_data, echo_sig_data, echo_back_data, *args):
        """Fit DEER experimental data to a model function.
        
        NOTE: Input StreamingLists contain ONLY sliced data. If slice_start was specified
        during acquisition, the first points have already been permanently discarded 
        before reaching this function. Fitting operates exclusively on the sliced data.
        """
        # Combine all dark signal sweeps into a single 3D array and average
        all_dark_signal_data = np.stack(dark_sig_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_dark_sig = np.mean(all_dark_signal_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all dark background sweeps into a single 3D array and average
        all_dark_background_data = np.stack(dark_back_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_dark_bg = np.mean(all_dark_background_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all dark signal sweeps into a single 3D array and average
        all_echo_signal_data = np.stack(echo_sig_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_echo_sig = np.mean(all_echo_signal_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all dark background sweeps into a single 3D array and average
        all_echo_background_data = np.stack(echo_back_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_echo_bg = np.mean(all_echo_background_data[1, :, :], axis=1)  # Shape: (10,)

        # Compute the microwave_times (assumed constant across sweeps)
        x_values = all_dark_signal_data[0, :, 0]  # Shape: (10,) — x_data from experiment in display units
        
        # Map dataset name to experiment code and convert x_values to SI units
        exp_lower = exp.lower()
        exp_code = exp_lower.replace(' ', '')
        if exp_lower == 'deer':
            x_values = x_values / 1e6  # MHz → Hz (SI)
        elif exp_lower == 'deer rabi':
            x_values = x_values / 1e9  # ns → s (SI)
        elif exp_lower == 'fid':
            x_values = x_values / 1e6  # μs → s (SI)
        elif exp_lower == 'fid cd':
            x_values = x_values / 1e6  # μs → s (SI)
        
        x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting in SI

        # Compute the ratio/difference of averaged signal and background for fitting
        deer = (averaged_dark_bg - averaged_dark_sig) / (averaged_dark_bg + averaged_dark_sig)
        echo = (averaged_echo_bg - averaged_echo_sig) / (averaged_echo_bg + averaged_echo_sig)

        y_values = deer / echo

        initial_guess = list(args)

        # Perform curve fitting 
        match fit_type:
            case 'Neg. Lorentz.':
                # Keep initial_guess in SI units (Hz) for fitting
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.negative_lorentzian, x_values, y_values, p0=initial_guess)
                y_fit = self.negative_lorentzian(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units based on experiment
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower == 'deer':
                    # Display x0 and gamma in MHz
                    display_params[1] /= 1e6  # x0: Hz → MHz
                    display_errors[1] /= 1e6
                    display_params[2] /= 1e6  # gamma: Hz → MHz
                    display_errors[2] /= 1e6
                
                fitted_values = [round(i, 3) for i in display_params]
                fitted_errors = [round(i, 3) for i in display_errors]
                # Units for Neg. Lorentz (DEER): [amplitude, frequency, linewidth, offset]
                if exp_lower == 'deer':
                    fit_units = ['', 'MHz', 'MHz', '']
                else:
                    fit_units = ['', '', '', '']

            case 'Decaying Cos.':
                # Keep initial_guess in SI units (seconds) for fitting
                ### --- Perform fit --- ###
                params, covariance = curve_fit(self.decaying_cosine, x_values, y_values, p0=initial_guess)
                y_fit = self.decaying_cosine(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units based on experiment
                display_params = list(params)
                display_errors = list(param_errors)
                if exp_lower == 'deer rabi':
                    # Display t_decay and T in nanoseconds
                    display_params[1] *= 1e9  # t_decay: s → ns
                    display_errors[1] *= 1e9
                    display_params[2] *= 1e9  # T: s → ns
                    display_errors[2] *= 1e9
                elif exp_lower in ('fid', 'fid cd'):
                    # Display t_decay and T in microseconds
                    display_params[1] *= 1e6  # t_decay: s → μs
                    display_errors[1] *= 1e6
                    display_params[2] *= 1e6  # T: s → μs
                    display_errors[2] *= 1e6
                
                fitted_values = [round(i, 2) for i in display_params]
                fitted_errors = [round(i, 2) for i in display_errors]
                
                # Find pi pulse time (or T point) as argmin of fitted curve
                # This captures the first minimum
                i_min = np.argmin(y_fit)
                pi_time_si = x_fit[i_min]
                
                # Period uncertainty from covariance (param_errors[2])
                # The minimum location is most sensitive to the period parameter T
                pi_time_error_si = param_errors[2]
                
                # Convert to display units
                if exp_lower == 'deer rabi':
                    scale = 1e9  # → ns
                else:  # 'fid', 'fid cd'
                    scale = 1e6  # → μs
                
                fitted_values[2] = round(pi_time_si * scale, 2)
                fitted_errors[2] = round(pi_time_error_si * scale, 2)
                
                # Units for Decaying Cos (DEER): [amplitude, decay_time, period, phase, offset]
                if exp_lower == 'deer rabi':
                    fit_units = ['', 'ns', 'ns', '', '']
                elif exp_lower in ('fid', 'fid cd'):
                    fit_units = ['', 'μs', 'μs', '', '']
                else:
                    fit_units = ['', '', '', '', '']

            case _:
                print("No fit type selected.")
                return 0, 0, 0, 0

        return fitted_values, fitted_errors, x_fit, y_fit, fit_units

    # TODO: update this function to fit DEER T1
    def fit_deer_t1_data(self, fit_type, with_pulse_py_sweeps, without_pulse_py_sweeps, with_pulse_ny_sweeps, without_pulse_ny_sweeps, *args):
        """Fit DEER T1 experimental data to a stretched exponential model.
        
        NOTE: Input StreamingLists contain ONLY sliced data. If slice_start was specified
        during acquisition, the first points have already been permanently discarded 
        before reaching this function. Fitting operates exclusively on the sliced data.
        """
        # Combine all dark signal sweeps into a single 3D array and average
        all_with_pulse_py_data = np.stack(with_pulse_py_sweeps, axis=-1)  # Shape: (2, 10, 5)
        averaged_with_pulse_py = np.mean(all_with_pulse_py_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all dark background sweeps into a single 3D array and average
        all_without_pulse_py_data = np.stack(without_pulse_py_sweeps, axis=-1)  # Shape: (2, 10, 5)
        averaged_without_pulse_py = np.mean(all_without_pulse_py_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all dark signal sweeps into a single 3D array and average
        all_with_pulse_ny_data = np.stack(with_pulse_ny_sweeps, axis=-1)  # Shape: (2, 10, 5)
        averaged_with_pulse_ny = np.mean(all_with_pulse_ny_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all dark background sweeps into a single 3D array and average
        all_without_pulse_ny_data = np.stack(without_pulse_ny_sweeps, axis=-1)  # Shape: (2, 10, 5)
        averaged_without_pulse_ny = np.mean(all_without_pulse_ny_data[1, :, :], axis=1)  # Shape: (10,)

        # Compute the microwave_times (assumed constant across sweeps)
        x_values = all_with_pulse_py_data[0, :, 0]  # Shape: (10,) — x_data from experiment in display units
        
        # Convert x_values from display units (μs) to SI (seconds)
        x_values = x_values / 1e6  # μs → s (SI)
        
        x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting in SI

        # Compute the ratio/difference of averaged signal and background for fitting
        # deer = (averaged_dark_bg - averaged_dark_sig) / (averaged_dark_bg + averaged_dark_sig)
        # echo = (averaged_echo_bg - averaged_echo_sig) / (averaged_echo_bg + averaged_echo_sig)

        # y_values = deer / echo
        diff_overall = (averaged_without_pulse_py - averaged_with_pulse_py) - (averaged_without_pulse_ny - averaged_with_pulse_ny)
        y_values = diff_overall

        initial_guess = list(args)

        # Perform curve fitting 
        match fit_type:
            case 'DEER T1 Str. Exp.':
                # Keep initial_guess in SI units (seconds for time constants) for fitting
                ### --- Perform fit --- ### 
                params, covariance = curve_fit(self.deer_t1_stretched_exponential, x_values, y_values, p0=initial_guess)
                y_fit = self.deer_t1_stretched_exponential(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                
                # Convert fitted parameters to display units (microseconds for DEER T1)
                display_params = list(params)
                display_errors = list(param_errors)
                # T1_e and n_e are time constants in seconds, display as microseconds
                display_params[4] *= 1e6  # T1_e: s → μs
                display_errors[4] *= 1e6
                
                fitted_values = [round(i, 3) for i in display_params]
                fitted_errors = [round(i, 3) for i in display_errors]
                # Units for DEER T1 Str. Exp: [amplitude, T1_nv, n_nv, T1_e, n_e, offset]
                fit_units = ['', 'μs', '', 'μs', '', '']

            case _:
                print("No fit type selected.")
                return 0, 0, 0, 0

        return fitted_values, fitted_errors, x_fit, y_fit, fit_units
    


    # def __enter__(self):
    #     """Perform experiment setup."""
    #     # config logging messages
    #     # if running a method from the GUI, it will be run in a new process
    #     # this logging call is necessary in order to separate log messages
    #     # originating in the GUI from those in the new experiment subprocess
    #     nspyre_init_logger(
    #         log_level=logging.INFO,
    #         log_path=_HERE / '../logs',
    #         log_path_level=logging.DEBUG,
    #         prefix=Path(__file__).stem,
    #         file_size=10_000_000,
    #     )
    #     _logger.info('Created SpinMeasurements instance.')

    # def __exit__(self):
    #     """Perform experiment teardown."""
    #     _logger.info('Destroyed SpinMeasurements instance.')






