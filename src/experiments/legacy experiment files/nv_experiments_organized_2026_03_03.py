"""NanoNMR-M experiments.

This module implements a collection of NV/NNMR experiment routines used by
the NanoNMR-M setup. It contains the :class:`SpinMeasurements` class with
helpers for digitizer configuration, pulse sequence control, and a set of
experiment implementations (ODMR, Rabi, T1/T2, DEER, CASR, etc.).

Added lifecycle management via the @managed_experiment decorator, which standardizes
the setup and teardown of instruments, as well as error handling and status reporting.

Author: Evan Villafranca
Updated: 2026-03-02
"""
from __future__ import annotations
import logging
import math
import time
import functools
from pathlib import Path
from typing import List
from typing import Optional
from typing import Callable, Any
from contextlib import contextmanager

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import warnings
from scipy.optimize import OptimizeWarning, curve_fit

from digitizer_driver import SpectrumDigitizer
import daq_read_samples as daq
from pulsestreamer import (
    PulseStreamer,
    TriggerStart,
    NextAction,
    When,
    OnNoData,
)
from nspyre import DataSource, InstrumentManager
from nspyre import StreamingList, experiment_widget_process_queue

from saveUtils import flexSave

import nv_dataclasses_2026_03_02 as nvcfg

_HERE = Path(__file__).parent
_logger = logging.getLogger(__name__)
SUBSEQ_COUNT = {
    "DQ": 4,
    "DEER": 4,
    "CD": 6,
    "CASR_alt": 2,
    "Cal": 1,
    "Opt T1": 1,
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

def run_save(data_name, file_name, directory, seq=None):
    logging.info("Saving file with flexSave...")
    if seq is not None:
        exp_name = f"{data_name}_{seq.lower()}"
    else:
        exp_name = data_name
    flexSave(datasetName=data_name, expType=exp_name, filename=file_name, dirs=directory)

@contextmanager
def _exclusive_ps(ps, token: str):
    ps.begin_exclusive(token, takeover=True, restore_on_release=True)
    try:
        yield
    finally:
        # Release exclusive control no matter how we exit
        try:
            ps.end_exclusive(token)
        except Exception:
            _logger.exception("ps.end_exclusive failed")

@contextmanager
def _shutter_open(shutter):
    shutter.open_shutter()
    try:
        yield
    finally:
        try:
            shutter.close_shutter()
        except Exception:
            _logger.exception("shutter.close_shutter failed")

@contextmanager
def _rf_on(sig_gen):
    sig_gen.set_rf_toggle(1)
    try:
        yield
    finally:
        try:
            sig_gen.set_rf_toggle(0)
        except Exception:
            _logger.exception("sig_gen.set_rf_toggle(0) failed")

def _hdawg_seq_name(drive_type: str) -> str:
    return "DEER CD" if drive_type == "Continuous" else "DEER"

def managed_experiment(*, token_prefix: str, dataset_key: str = "dataset"):
    """Decorator to run an experiment inside the standard lifecycle.

    Decorated methods should accept injected handles:
        def scan(self, *, mgr, data, token, **kwargs): ...

    Public call remains: scan(**kwargs)
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            if dataset_key not in kwargs:
                raise KeyError(f"Missing required kwarg '{dataset_key}' for {fn.__name__}")
            dataset = kwargs[dataset_key]

            def body(*, mgr, data, token):
                return fn(self, mgr=mgr, data=data, token=token, **kwargs)

            return self.run_experiment(dataset=dataset, token_prefix=token_prefix, body=body)

        return wrapped
    return decorator

################################
### --- Experiment class --- ###
################################

class SpinMeasurements:
    """NanoNMR-M experiments"""
    def __init__(self, queue_to_exp=None, queue_from_exp=None):
        """
        Args:
            queue_to_exp: A multiprocessing Queue object used to send messages
                to the experiment from the GUI.
            queue_from_exp: A multiprocessing Queue object used to send messages
                to the GUI from the experiment.
        """
        self.queue_to_exp = queue_to_exp
        self.queue_from_exp = queue_from_exp
        self.dig = SpectrumDigitizer('dev/spcm0') # instantiate digitizer for high-speed data acquisition

    @staticmethod
    def choose_sideband(opt, nv_freq, side_freq, pulse_axis='x'):
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
    def _configure_sig_gen_iq(sig_gen, *, carrier_freq: float, rf_power: float):
        sig_gen.set_frequency(carrier_freq)
        sig_gen.set_rf_amplitude(rf_power)
        sig_gen.set_mod_type(7)            # QAM
        sig_gen.set_mod_subtype(1)         # no constellation mapping
        sig_gen.set_mod_function("IQ", 5)  # external modulation
        sig_gen.set_mod_toggle(1)

    @staticmethod
    def digitizer_configure(**kwargs):
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
                      # 'num_pts_in_exp': 2*kwargs['num_pts_in_exp'], # MW ON + OFF subsequences
                      'num_iters': kwargs['iters'], # number of exp. iterations
                      'segment_size': kwargs['segment_size'],
                      'sampling_frequency': kwargs['sampling_freq']/1e9, # digitizer uses [GHz]
                      'AMP': int(kwargs['dig_amplitude']*1000), # digitizer uses [mV]
                      'readout_ch': int(kwargs['read_channel']),
                      'ACCOUPLE': acdc,
                      'HF_INPUT_50OHM': term,
                      'card_timeout': kwargs['dig_timeout'],
                      'pretrig_size': kwargs['pretrig_size'],
                      'runs': kwargs['runs']}
        
        return dig_config
    
    @staticmethod
    def _mag(x):
        return x.magnitude if hasattr(x, "magnitude") else x

    def analog_math(self, array, exp_type, pts):
        """Fast subsequence averaging for point-interleaved data.

        Expected ordering (legacy): for each point -> [sub0, sub1, ..., sub(n-1)]
        repeated for pts points, repeated for runs.

        Returns:
        - n==1: ndarray (pts,)
        - n>1 : list of n ndarrays, each (pts,)
        """
        # Strip pint units early and explicitly (avoids UnitStrippedWarning)
        arr = np.asarray(self._mag(array)).ravel()

        n = SUBSEQ_COUNT.get(exp_type, 2)
        block = n * pts
        size = arr.size

        if size % block != 0:
            raise ValueError(f"Input length {size} not divisible by n*pts={block} (n={n}, pts={pts})")

        runs = size // block

        # Each row is one "point", columns are subseqs: (runs*pts, n)
        per_point = arr.reshape(runs * pts, n)

        # Compute (pts, n) by averaging across runs for each subseq column
        out_pts_n = np.empty((pts, n), dtype=per_point.dtype)
        for j in range(n):
            out_pts_n[:, j] = per_point[:, j].reshape(runs, pts).mean(axis=0)

        # Return legacy shape/format
        if n == 1:
            return out_pts_n[:, 0]

        return [out_pts_n[:, j] for j in range(n)]
  
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

            "fit_value2": fit_value2,
            "fit_error2": fit_error2,

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
        except Exception:
            _logger.exception("cleanup: %s failed", label)

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
    def run_experiment(self, *, dataset: str, token_prefix: str, body: Callable[..., Any]) -> Any:
        """Run an experiment with a standardized lifecycle.

        This wrapper:
          - connects to InstrumentManager + DataSource
          - takes exclusive control of the Pulse Streamer
          - runs the experiment body(mgr=..., data=..., token=...)
          - always shuts equipment down using existing handles
        """
        token = f"{token_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
        with InstrumentManager() as mgr, DataSource(dataset) as data:
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg

            with _exclusive_ps(ps, token):
                try:
                    return body(mgr=mgr, data=data, token=token)
                finally:
                    self.equipment_off_handles(
                        laser_shutter=laser_shutter,
                        sig_gen=sig_gen,
                        ps=ps,
                        hdawg=hdawg,
                    )

    def sigvstime_scan(self, **kwargs):     
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as sigvstime_data:
            # run laser on continuously here from laser driver
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            ps = mgr.ps

            token = f"SIGVSTIME_{time.strftime('%Y%m%d_%H%M%S')}"  # unique token for this experiment run
            ps.begin_exclusive(token, takeover=True, restore_on_release=True)  # take exclusive control of the pulse streamer

            sequence = ps.SigvsTime(1/kwargs['exp_sampling_rate'] * 1e9) # pulse streamer sequence for CW ODMR
            
            # configure digitizer (need to use DC coupling for signal vs time)           
            dig_config = self.digitizer_configure(exp_type="Sig vs Time", num_pts = 1, iters = 1, 
                                                segment_size = kwargs['segment_size'], sampling_freq = 0.5e9, dig_amplitude = 5, 
                                                read_channel = kwargs['read_channel'], coupling = 'DC', termination = '1M', 
                                                pretrig_size = kwargs['pretrig_size'], dig_timeout = 5, runs = 400)
                
            time_start = time.time()

            signal_sweeps = StreamingList()

            # open laser shutter
            laser_shutter.open_shutter()
            
            # upload digitizer parameters
            self.dig.assign_param(dig_config)

            # configure laser settings and turn on
            laser.set_diode_current_realtime(kwargs['laser_power'])
            
            # set pulsestreamer to start on software trigger & run infinitely
            ps.set_soft_trigger()
            ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY, owner=token) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
            
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

                    if kwargs['save'] == True:
                        run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    return

    @managed_experiment(token_prefix="CWODMR", dataset_key="dataset")           
    def odmr_scan(self, *, mgr, data, token, **kwargs):
        """Run a CW ODMR sweep over a set of microwave frequencies."""  
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg

        ### --- Define NV drive parameters and configure SRS signal generator --- ###   
        delta = 0
        iq_phases = [delta + 0, delta + 90] # set IQ phase relations for lower sideband [lower I, lower Q]
        sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 
        max_sideband_freq = 2 * kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> ? MHz max. for AWG bandwidth

        ### --- Default parameter array for sweep --- ###
        mod_freqs = np.flip(np.linspace(0, max_sideband_freq, kwargs['num_pts']))
        real_freqs = sig_gen_freq - mod_freqs
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive
            
        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = real_freqs / 1e9
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.CW_ODMR(kwargs['num_pts'], kwargs['probe']*1e9) # pulse streamer sequence for CW ODMR
        dig_cfg = self.digitizer_configure(
            exp_type="ODMR",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###                      
        try:
            hdawg.set_sampling_rate(0, kwargs["awg_samp_rate_1"])
            hdawg.set_sampling_rate(1, kwargs["awg_samp_rate_2"])
            hdawg.set_sequence(**{
                "seq": "CW ODMR",
                "i_offset": kwargs["i_offset"],
                "q_offset": kwargs["q_offset"],
                "probe_length": kwargs["probe"],
                "sideband_power": kwargs["sideband_power"],
                "sideband_freqs": mod_freqs,
                "iq_phases": iq_phases,
                "num_pts": kwargs["num_pts"],
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs['laser_power']) # set laser power
        
        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):     
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return        
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    odmr_result_raw = self.dig.acquire() # acquire data from digitizer
                    odmr_result = np.mean(odmr_result_raw,axis=1)
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break
   
                try:
                    sig, bg = self.analog_math(odmr_result, 'CW ODMR', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([real_freqs / 1e9, sig])); signal_sweeps.updated_item(-1)
                background_sweeps.append(np.stack([real_freqs / 1e9, bg])); background_sweeps.updated_item(-1)
       
                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(
                                kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({
                    "params": {"kwargs": kwargs, "iters_completed": iters_completed, "elapsed": time.perf_counter() - exp_start_time},
                    "title": "CW Optically Detected Magnetic Resonance",
                    "xlabel": "Frequency (GHz)",
                    "ylabel": "Signal (V) or Norm. Signal",
                    "datasets": {"signal": signal_sweeps, "background": background_sweeps, "x_fit": fit_x, "y_fit": fit_y, "fit_value": fit_value, "fit_error": fit_error},
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100
        
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(
                        kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))
        
    def odmr_smart_scan(self, **kwargs):
        """
        Run a CW ODMR sweep over a set of microwave frequencies.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as cw_odmr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            token = f"ODMRSMRTSCN_{time.strftime('%Y%m%d_%H%M%S')}"  # unique token for this experiment run
            ps.begin_exclusive(token, takeover=True, restore_on_release=True)  # take exclusive control of the pulse streamer

            num_angles = kwargs['iters']
            azi_angles = np.linspace(kwargs['start_angle'], kwargs['stop_angle'], num_angles)
            # print("azimuthal angles: ", azi_angles)
            # mgr.thor_polar.set_vel_params(3,7) # reset Thorlabs stages to default acceleration and velocity parameters
            mgr.thor_azi.set_vel_params(8,15)

            # move azimuthal stage to start angle
            mgr.thor_azi.move(kwargs['start_angle'], True)
            mgr.thor_azi.update_positions_callback() # update position
            
            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 

            max_sideband_freq = 2*kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
            mod_freqs = np.flip(mod_freqs)
                        
            real_freqs = sig_gen_freq - mod_freqs
            
            # define pulse sequence
            sequence = ps.CW_ODMR(kwargs['num_pts'], kwargs['probe']*1e9) # pulse streamer sequence for CW ODMR
            ps.probe_time = kwargs['probe'] * 1e9

            # configure digitizer
            dig_config = self.digitizer_configure(exp_type = "ODMR", num_pts = kwargs["num_pts"], iters = kwargs['iters'], 
                                                  segment_size = kwargs['segment_size'], sampling_freq = kwargs['dig_sampling_freq'], dig_amplitude = kwargs['dig_amplitude'], 
                                                  read_channel = kwargs['read_channel'], coupling = kwargs['dig_coupling'], termination = kwargs['dig_termination'], 
                                                  pretrig_size = kwargs['pretrig_size'], dig_timeout = kwargs['dig_timeout'], runs = kwargs['runs'])
            
            # configure signal generator for NV drive
            sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
            sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

            try:
                hdawg.set_sequence(**{'seq': 'CW ODMR',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'probe_length': kwargs['probe'], 
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freqs': mod_freqs, 
                                    'iq_phases': iq_phases,
                                    'num_pts': kwargs['num_pts']}) 
                hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
                hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
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

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # mgr.thor_polar.set_vel_params(3,7) # TODO: speed up Thorlabs stage for quicker scan
                mgr.thor_azi.set_vel_params(25,35) # set azimuthal stage velocity and acceleration

                # configure laser settings and turn on
                laser.set_diode_current_realtime(kwargs['laser_power'])

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
                    odmr_result=np.mean(odmr_result_raw,axis=1)

                    # partition buffer into signal and background datasets
                    try:
                        sig, bg = self.analog_math(odmr_result, 'CW ODMR', kwargs['num_pts'])
                    except ValueError:
                        continue
                    
                    # notify the streaminglist that this entry has updated so it will be pushed to the data server
                    signal_sweeps.append(np.stack([real_freqs/1e9, sig]))
                    signal_sweeps.updated_item(-1) 
                    background_sweeps.append(np.stack([real_freqs/1e9, bg]))
                    background_sweeps.updated_item(-1)

                    # Fit the data
                    initial_guess = [0.01, kwargs['center_freq']/1e9, 6e-3, 1]  # [A, x0, gamma, c]
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            params, params_covariance = curve_fit(self.negative_lorentzian, real_freqs/1e9, sig/bg, p0=initial_guess)
                            # fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                    # Extract fitted parameters
                    freq_fit = params[1]
                    angle_fits.append(np.stack([azi_angles[i], freq_fit]))
                    angle_fits.updated_item(-1) 
                    print(f"ODMR = {params} GHz at angle {azi_angles[i]} degrees")

                    # update GUI progress bar & ETA
                    iter_completed = i + 1                      
                    percent_completed = int((iter_completed / kwargs["iters"]) * 100)
                    
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
                            total_iters=kwargs['iters'],
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
                            total_iters=kwargs["iters"],
                            iters_completed=iter_completed,
                        )
                        # self.queue_from_exp.put_nowait([percent_completed, 'stopped', [fit_value, fit_error]])
                        self.queue_from_exp.put_nowait(msg)
                        
                        if kwargs['save'] == True:
                            run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                        return
                    
                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                msg = self.build_status_msg(
                    status="complete",
                    percent_completed=int(percent_completed),
                    fit_value=None,
                    fit_error=None,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
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
    def rabi_scan(self, *, mgr, data, token, **kwargs):
        """Run a Rabi sweep over MW pulse durations."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg

        ### --- Default parameter array for sweep --- ###
        mw_times = np.linspace(kwargs["start"], kwargs["stop"], kwargs["num_pts"]) * 1e9

        ### --- Define NV drive parameters --- ###  
        sig_gen_freq, iq_phases = self.choose_sideband(
            kwargs["sideband"], kwargs["freq"], kwargs["sideband_freq"], kwargs["pulse_axis"]
        )

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"])
    
        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = mw_times.copy()
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Rabi(kwargs["laser_init"] * 1e9, mw_times, kwargs["laser_readout"] * 1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="Rabi",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs["awg_samp_rate_1"])
            hdawg.set_sampling_rate(1, kwargs["awg_samp_rate_2"])
            hdawg.set_sequence(**{
                "seq": "Rabi",
                "i_offset": kwargs["i_offset"],
                "q_offset": kwargs["q_offset"],
                "sideband_power": kwargs["sideband_power"],
                "sideband_freq": kwargs["sideband_freq"],
                "iq_phases": iq_phases,
                "pi_pulses": mw_times / 1e9,
                "num_pts": kwargs["num_pts"],
                "runs": kwargs["runs"],
            })   
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return

        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()

            ### --- Main experiment loop --- ###
            for i in range(kwargs["iters"]):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    rabi_result_raw = self.dig.acquire() # acquire data from digitizer
                    rabi_result = np.mean(rabi_result_raw, axis=1)
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(rabi_result, "Rabi", kwargs["num_pts"]) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([mw_times, sig])); signal_sweeps.updated_item(-1)
                background_sweeps.append(np.stack([mw_times, bg])); background_sweeps.updated_item(-1)

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(
                                kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({
                    "params": {"kwargs": kwargs, "iters_completed": iters_completed, "elapsed": time.perf_counter() - exp_start_time},
                    "title": "Rabi Oscillation",
                    "xlabel": "MW Pulse Duration (ns)",
                    "ylabel": "Signal (V) or Norm. Signal",
                    "datasets": {"signal": signal_sweeps, "background": background_sweeps, "x_fit": fit_x, "y_fit": fit_y, "fit_value": fit_value, "fit_error": fit_error},
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(
                        kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="PLSDODMR", dataset_key="dataset")
    def pulsed_odmr_scan(self, *, mgr, data, token, **kwargs):
        """Run a Pulsed ODMR sweep over a set of microwave frequencies."""
        
        ### --- Devices --- ###            
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
        
        ### --- Define NV drive parameters --- ###           
        delta = 0
        iq_phases = [delta + 0, delta + 90] # set IQ phase relations for lower sideband [lower I, lower Q]   
        sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 
        max_sideband_freq = 2 * kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

        ### --- Default parameter array for sweep --- ###
        mod_freqs = np.flip(np.linspace(0, max_sideband_freq, kwargs['num_pts'])) # define parameter array that will be swept over in experiment & shuffle
        real_freqs = sig_gen_freq - mod_freqs

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive
            
        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = real_freqs / 1e9
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Pulsed_ODMR(kwargs['laser_init']*1e9, kwargs['num_pts'], kwargs['pi']*1e9, kwargs['laser_readout']*1e9) # pulse streamer sequence
        dig_cfg = self.digitizer_configure(
            exp_type="ODMR",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )
    
        ### --- Upload AWG sequence --- ### 
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'Pulsed ODMR',
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freqs': mod_freqs, 
                'iq_phases': iq_phases,
                'pi_pulse': kwargs['pi'], 
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs']
            })    
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs['laser_power']) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return 
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break
                
                try:
                    pulsed_odmr_result_raw = self.dig.acquire() # acquire data from digitizer   
                    pulsed_odmr_result = np.mean(pulsed_odmr_result_raw,axis=1) # average all data over each trigger/segment 
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(pulsed_odmr_result, 'Pulsed ODMR', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([real_freqs / 1e9, sig])); signal_sweeps.updated_item(-1)
                background_sweeps.append(np.stack([real_freqs / 1e9, bg])); background_sweeps.updated_item(-1)
       
                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(
                                kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                data.push({
                    "params": {"kwargs": kwargs, "iters_completed": iters_completed, "elapsed": time.perf_counter() - exp_start_time},
                    "title": "Pulsed Optically Detected Magnetic Resonance",
                    "xlabel": "Frequency (GHz)",
                    "ylabel": "Signal (V) or Norm. Signal",
                    "datasets": {"signal": signal_sweeps, "background": background_sweeps, "x_fit": fit_x, "y_fit": fit_y, "fit_value": fit_value, "fit_error": fit_error},
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(
                        kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="PLSDODMRRF", dataset_key="dataset")
    def pulsed_odmr_rf_scan(self, *, mgr, data, token, **kwargs):
        """Run a Pulsed ODMR sweep over a set of microwave frequencies with coil RF tone for coil B field calibration."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define NV drive parameters --- ###      
        delta = 0
        iq_phases = [delta + 0, delta + 90] # set IQ phase relations for lower sideband [lower I, lower Q]   
        sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 
        max_sideband_freq = 2 * kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

        ### --- Default parameter array for sweep --- ###
        mod_freqs = np.flip(np.linspace(0, max_sideband_freq, kwargs['num_pts'])) # define parameter array that will be swept over in experiment & shuffle
        real_freqs = sig_gen_freq - mod_freqs

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive
          
        ### --- Default fit parameters for live fitting --- ###
        fit_value = None
        fit_error = None
        fit_no_rf_value = None
        fit_no_rf_error = None
        fit_x = real_freqs / 1e9
        fit_y = np.ones(len(fit_x))
        fit_no_rf_x = real_freqs / 1e9
        fit_no_rf_y = fit_y.copy()
        fitted_diff = 0
        coil_b_field_gauss = None
        proton_pi_half = None

        pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer
        rf_period = 1/kwargs['rf_pulse_freq']*1e9 # rf pulse period [ns] units for pulse streamer
        
        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Pulsed_ODMR_RF(kwargs['laser_init']*1e9, kwargs['num_pts'], pi_pulse, rf_period, kwargs['laser_readout']*1e9) # pulse streamer sequence
        dig_cfg = self.digitizer_configure(
            exp_type="ODMR",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ### 
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'Pulsed ODMR RF',
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freqs': mod_freqs, 
                'iq_phases': iq_phases,
                'pi_pulse': pi_pulse/1e9, 
                'rf_freq': kwargs['rf_pulse_freq'],
                'rf_power': kwargs['rf_pulse_power'],
                'rf_phase': kwargs['rf_pulse_phase'],
                'rf_length': 3*rf_period/1e9,
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
            status="failed",
            percent_completed=0,
            fit_value=fit_value,
            fit_error=fit_error,
            fit_value2=fit_no_rf_value,
            fit_error2=fit_no_rf_error,
            start_time=time.perf_counter(),
            total_iters=kwargs["iters"],
            iters_completed=0,
            exception=type(e).__name__,
        ))
            return
            
        rf_signal_sweeps, rf_background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        no_rf_signal_sweeps, no_rf_background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs['laser_power']) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return 
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:       
                    pulsed_odmr_result_raw = self.dig.acquire() # acquire data from digitizer                
                    pulsed_odmr_result = np.mean(pulsed_odmr_result_raw,axis=1) # average all data over each trigger/segment 
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    rf_sig, rf_bg, no_rf_sig, no_rf_bg = self.analog_math(pulsed_odmr_result, 'DEER', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                rf_signal_sweeps.append(np.stack([real_freqs / 1e9, rf_sig])); rf_signal_sweeps.updated_item(-1)
                rf_background_sweeps.append(np.stack([real_freqs / 1e9, rf_bg])); rf_background_sweeps.updated_item(-1)
                no_rf_signal_sweeps.append(np.stack([real_freqs / 1e9, no_rf_sig])); no_rf_signal_sweeps.updated_item(-1)
                no_rf_background_sweeps.append(np.stack([real_freqs / 1e9, no_rf_bg])); no_rf_background_sweeps.updated_item(-1)      
                
                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(
                                'odmr', rf_signal_sweeps, rf_background_sweeps, *kwargs['fit_params'][:4]
                            )
                            fit_no_rf_value, fit_no_rf_error, fit_no_rf_x, fit_no_rf_y = self.fit_data(
                                'odmr', no_rf_signal_sweeps, no_rf_background_sweeps, *kwargs['fit_params'][4:]
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                        else:
                            fitted_diff = fit_no_rf_value[1] - fit_value[1]
                            coil_b_field_gauss = round(fitted_diff*1000/2.8,4)
                            proton_pi_half = round(1/(42.577e-4*coil_b_field_gauss)/4,2)

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'fit_vals': {'fit_value': fit_value, 'fit_error': fit_error, 
                                 'fit_no_rf_value': fit_no_rf_value, 'fit_no_rf_error': fit_no_rf_error, 
                                 'coil_B_field_gauss': coil_b_field_gauss, 'proton_pi_half_us': proton_pi_half},
                    'title': 'Pulsed Optically Detected Magnetic Resonance with RF',
                    'xlabel': 'Frequency (GHz)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'rf_signal' : rf_signal_sweeps, 'rf_background': rf_background_sweeps,
                                'signal' : no_rf_signal_sweeps, 'background': no_rf_background_sweeps,
                                'x_fit': fit_x, 'y_fit': fit_y, 'x_fit_no_rf': fit_no_rf_x, 'y_fit_no_rf': fit_no_rf_y}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    fit_value2=fit_no_rf_value,
                    fit_error2=fit_no_rf_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(
                        'odmr', rf_signal_sweeps, rf_background_sweeps, *kwargs['fit_params'][:4]
                    )
                    fit_no_rf_value, fit_no_rf_error, fit_no_rf_x, fit_no_rf_y = self.fit_data(
                        'odmr', no_rf_signal_sweeps, no_rf_background_sweeps, *kwargs['fit_params'][4:]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                else:
                    fitted_diff = fit_no_rf_value[1] - fit_value[1]
                    coil_b_field_gauss = round(fitted_diff*1000/2.8,4)
                    proton_pi_half = round(1/(42.577e-4*coil_b_field_gauss)/4,2)

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            fit_value2=fit_no_rf_value,
            fit_error2=fit_no_rf_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

        if fit_value is not None and fit_no_rf_value is not None:
            print(f"Fitted resonances = {fit_value[1]} GHz (RF), {fit_no_rf_value[1]} GHz (no RF)")
            print(f"Fitted difference = {round(fitted_diff*1000,4)} MHz")
            print(f"Coil B field = {coil_b_field_gauss} G")
            print(f"1H pi/2 pulse = {proton_pi_half} us")

    def OPT_T1_scan(self, **kwargs):
        """
        Run a T1 sweep without MW over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as t1_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg

            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            
            t1_buffer = self.generate_buffer('Opt T1', kwargs['runs'], kwargs['num_pts'])
            
            # configure digitizer
            dig_config = self.digitizer_configure(exp_type="Opt T1", num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
                                                  segment_size = kwargs['segment_size'], sampling_freq = kwargs['dig_sampling_freq'], dig_amplitude = kwargs['dig_amplitude'], 
                                                  read_channel = kwargs['read_channel'], coupling = kwargs['dig_coupling'], termination = kwargs['dig_termination'], 
                                                  pretrig_size = kwargs['pretrig_size'], dig_timeout = kwargs['dig_timeout'], runs = kwargs['runs'])
            
            sequence = ps.Optical_T1(tau_times, kwargs['laser_readout']*1e9)

            # configure devices used in scan
            laser.set_diode_current_realtime(kwargs['laser_power'])

            # daq.open_ai_task(kwargs['detector'], len(t1_buffer[0])) # kwargs['detector'] used for APD/BPD now, this line is not current

            self.dig.assign_param(dig_config)

            # for storing the experiment data
            # list of numpy arrays of shape (2, num_points)
            signal_sweeps = StreamingList()
            background_sweeps = StreamingList()
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            for i in range(kwargs['iters']):
                
                self.dig.config()
                self.dig.start_buffer()
                ps.stream(sequence, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                t1_result_raw = self.dig.acquire()
                
                t1_result = np.mean(t1_result_raw, axis=1)
                
                # partition buffer into signal and background datasets
                try:
                    sig, bg = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])
                except ValueError:
                    continue
                
                # notify the streaminglist that this entry has updated so it will be pushed to the data server                    
                signal_sweeps.append(np.stack([tau_times/1e6, sig]))
                signal_sweeps.updated_item(-1) 
                background_sweeps.append(np.stack([tau_times/1e6, bg]))
                background_sweeps.updated_item(-1)

                # update GUI progress bar & ETA
                iter_completed = i + 1                      
                percent_completed = int((iter_completed / kwargs["iters"]) * 100)
                
                # save the current data to the data server.
                t1_data.push({'params': {'kwargs': kwargs, 'iters_completed': iter_completed, 'elapsed': time.perf_counter() - exp_start_time},
                                'title': 'Optical T1 Relaxation',
                                'xlabel': 'Free Precession Interval (ms)',
                                'ylabel': 'Signal',
                                'datasets': {'signal' : signal_sweeps,
                                            'background': background_sweeps}
                })
                
                msg = self.build_status_msg(
                            status='in progress',
                            percent_completed=percent_completed,
                            fit_value=None,
                            fit_error=None,
                            start_time=exp_start_time,
                            total_iters=kwargs['iters'],
                            iters_completed=iter_completed,
                        )

                # self.queue_from_exp.put_nowait([percent_completed, 'in progress', [fit_value, fit_error]])
                self.queue_from_exp.put_nowait(msg)
                
                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                    # the GUI has asked us nicely to exit
                    if kwargs['save'] == True:
                        flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    
                    self.equipment_off()

                    return
                
                self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                    # the GUI has asked us nicely to exit. Save data if requested.
                    # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                    self.equipment_off()

                    msg = self.build_status_msg(
                        status="stopped",
                        percent_completed=int(percent_completed),
                        fit_value=None,
                        fit_error=None,
                        start_time=exp_start_time,
                        total_iters=kwargs["iters"],
                        iters_completed=iter_completed,
                    )
                    # self.queue_from_exp.put_nowait([percent_completed, 'stopped', [fit_value, fit_error]])
                    self.queue_from_exp.put_nowait(msg)

                    if kwargs['save'] == True:
                        run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    return
                    
            if kwargs['save'] == True:
                run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            self.equipment_off()

    @managed_experiment(token_prefix="MWT1", dataset_key="dataset")
    def MW_T1_scan(self, *, mgr, data, token, **kwargs):
        """Run a T1 sweep with MW over a set of precession time intervals."""

        ### --- Devices --- ###  
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
 
        ### --- Default parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            case 'linspace':
                tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
        
        ### --- Define NV drive parameters --- ###  
        sig_gen_freq, iq_phases = self.choose_sideband(
            kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'], kwargs['pulse_axis']
        )
        pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = tau_times[1:]/1e6
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Diff_T1(kwargs['laser_init']*1e9, tau_times, kwargs['pulse_axis'], pi_pulse, kwargs['laser_readout']*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="MW T1",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'T1',
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pi_pulse': pi_pulse/1e9, 
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']}
            )    
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return
            
            exp_start_time = time.perf_counter()

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:     
                    t1_result_raw = self.dig.acquire() # acquire data from digitizer
                    t1_result = np.mean(t1_result_raw,axis=1)
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break
                     
                try:
                    sig, bg = self.analog_math(t1_result, 'MW_T1', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue
                    
                signal_sweeps.append(np.stack([tau_times[1:] / 1e6, sig[1:]])); signal_sweeps.updated_item(-1)
                background_sweeps.append(np.stack([tau_times[1:] / 1e6, bg[1:]])); background_sweeps.updated_item(-1)  

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(
                                kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'MW T1 Relaxation',
                    'xlabel': 'Free Precession Interval (ms)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'signal' : signal_sweeps, 'background': background_sweeps, 'x_fit': fit_x, 'y_fit': fit_y, 'fit_value': fit_value, 'fit_error': fit_error}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))
                            
        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(
                        kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="T2", dataset_key="dataset")
    def T2_scan(self, *, mgr, data, token, **kwargs):
        """Run a T2 sweep over a set of precession time intervals."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            case 'linspace':
                tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'])
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = tau_times[1:]
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        match kwargs['t2_seq']:
            case 'Ramsey':
                sequence = ps.Ramsey(kwargs['laser_init']*1e9, tau_times, pi_half[0], pi_half[1], kwargs['laser_readout']*1e9)
                x_tau_times = tau_times
            case 'Echo':
                sequence = ps.Echo(kwargs['laser_init']*1e9, tau_times, pi_half[0], pi_half[1], 
                                        pi[0], pi[1], kwargs['laser_readout']*1e9)
                x_tau_times = 2*tau_times + pi[1] # for definition of pi/2 - tau - pi - tau - pi/2
            case 'XY4':
                sequence = ps.XY4_N(kwargs['laser_init']*1e9, tau_times, 'xy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
                x_tau_times = 4*tau_times + 2*pi[0] + 2*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + (2*(x_tau_times/2)/(4*kwargs['n']) + 2*pi[0] + \
                #             2*pi[1] + 3*x_tau_times/(4*kwargs['n']))*kwargs['n']
            case 'YY4':
                sequence = ps.XY4_N(kwargs['laser_init']*1e9, tau_times, 'yy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
                x_tau_times = 4*tau_times + 4*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + (2*(x_tau_times/2)/(4*kwargs['n']) + 4*pi[1] + 3*x_tau_times/(4*kwargs['n']))*kwargs['n']
            case 'XY8':
                sequence = ps.XY8_N(kwargs['laser_init']*1e9, tau_times, 'xy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
                x_tau_times = 8*tau_times + 4*pi[0] + 4*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 4*pi[0] + \
                #             4*pi[1] + 7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']
            case 'YY8':
                sequence = ps.XY8_N(kwargs['laser_init']*1e9, tau_times, 'yy', 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
                x_tau_times = 8*tau_times + 8*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 8*pi[1] + \
                #         7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']
            case 'CPMG':
                sequence = ps.CPMG_N(kwargs['laser_init']*1e9, tau_times, kwargs['pulse_axis'], 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
                x_tau_times = tau_times + (kwargs['n']-1)*(pi[1]+tau_times) # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
                # x_tau_times = 2*pi_half[0] + x_tau_times/kwargs['n'] + (pi[0] + x_tau_times/kwargs['n'])*(kwargs['n']-1) + pi[0]
            # case 'PulsePol':
            #     sequence = ps.PulsePol(tau_times, 
            #                         pi_half[0], pi_half[1], 
            #                         pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
                # x_tau_times = (x_tau_times + 2*pi_half[1] + pi[0] + 2*pi_half[0] + pi[1])*2*kwargs['n']
        dig_cfg = self.digitizer_configure(
            exp_type="T2",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'T2',
                'seq_dd': kwargs['t2_seq'],
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'n': kwargs['n'],
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']
            }) 
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
        
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points) 

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return
                
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:      
                    t2_result_raw = self.dig.acquire() # acquire data from digitizer
                    # t2_result_raw = t2_result_raw[:,50:]
                    t2_result = np.mean(t2_result_raw, axis=1)
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(t2_result, 'T2', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([tau_times[1:]/1e3, sig[1:]])); signal_sweeps.updated_item(-1)
                background_sweeps.append(np.stack([tau_times[1:]/1e3, bg[1:]])); background_sweeps.updated_item(-1)
     
                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(
                                kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'T2 Relaxation',
                    'xlabel': 'Free Precession Interval (\u03BCs)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'signal' : signal_sweeps, 'background': background_sweeps, 
                                'x_fit': fit_x, 'y_fit': fit_y, 'fit_value': fit_value, 'fit_error': fit_error}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(
                        kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))
    
    @managed_experiment(token_prefix="T2RF", dataset_key="dataset")
    def T2_rf_scan(self, *, mgr, data, token, **kwargs):
        """Run a T2 sweep over a set of precession time intervalsb with constant RF applied."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            case 'linspace':
                tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'])
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = tau_times[1:]
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.XY8_N_RF(kwargs['laser_init']*1e9, tau_times, 'yy', 
                            pi_half[0], pi_half[1], 
                            pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
        x_tau_times = 8*tau_times + 8*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
        # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 8*pi[1] + \
        #         7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']

        # sequence = ps.XY8_N_RF(tau_times, 'xy', 
        #                     pi_half[0], pi_half[1], 
        #                     pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
        # x_tau_times = 8*tau_times + 4*pi[0] + 4*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
        # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 4*pi[0] + \
        #             4*pi[1] + 7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']

        rf_times = 500 + 2*pi_half[1] + (x_tau_times)*kwargs['n'] + 100

        dig_cfg = self.digitizer_configure(
            exp_type="T2",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'T2 RF',
                # 'seq_dd': kwargs['t2_seq'],
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'rf_freq': kwargs['rf_pulse_freq'],
                'rf_power': kwargs['rf_pulse_power'],
                'rf_phase': kwargs['rf_pulse_phase'],
                'rf_length': rf_times/1e9,
                'n': kwargs['n'],
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']
            }) 
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return

        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points) 

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return
                
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:      
                    t2_result_raw = self.dig.acquire() # acquire data from digitizer
                    # t2_result_raw = t2_result_raw[:,50:]
                    t2_result = np.mean(t2_result_raw, axis=1)
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(t2_result, 'T2', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([tau_times[1:]/1e3, sig[1:]])); signal_sweeps.updated_item(-1)
                background_sweeps.append(np.stack([tau_times[1:]/1e3, bg[1:]])); background_sweeps.updated_item(-1)
     
                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(
                                kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'T2 Relaxation',
                    'xlabel': 'Free Precession Interval (\u03BCs)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'signal' : signal_sweeps, 'background': background_sweeps, 
                                'x_fit': fit_x, 'y_fit': fit_y, 'fit_value': fit_value, 'fit_error': fit_error}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(
                        kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))
    
    @managed_experiment(token_prefix="DQ", dataset_key="dataset")
    def DQ_scan(self, *, mgr, data, token, **kwargs):
        """Run a DQ sweep over a set of precession time intervals."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Define parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            case 'linspace':
                tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

        ### --- Define NV drive parameters --- ###
        if kwargs['pulse_axis'] == 'y':
            delta = 90
        else:
            delta = 0
        iq_phases = [delta + 0, delta + 90, delta + 90, delta + 0] # set IQ phase relations for upper and lower sidebands
        sig_gen_freq = (kwargs['freq_minus'] + kwargs['freq_plus']) / 2 # set mean value freq to sig gen 
        sideband_freq = sig_gen_freq - kwargs['freq_minus'] # set sideband freq to match the inputted values
        pi_pulse_minus = kwargs['pi_minus']*1e9 # [ns] units for pulse streamer
        pi_pulse_plus = kwargs['pi_plus']*1e9 # [ns] units for pulse streamer

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = tau_times[1:]/1e6
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.DQ(kwargs['laser_init'] * 1e9, tau_times, kwargs['pulse_axis'], pi_pulse_minus, pi_pulse_plus, kwargs['laser_readout'] * 1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="DQ",
            num_pts=kwargs["num_pts"],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###            
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'DQ',
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': sideband_freq, 
                'iq_phases': iq_phases,
                'pi_minus1': pi_pulse_minus/1e9, 
                'pi_plus1': pi_pulse_plus/1e9, 
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return

        s00_sweeps, s0m_sweeps, smm_sweeps, smp_sweeps = StreamingList(), StreamingList(), StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points) 

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return    
                
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:        
                    dq_result_raw = self.dig.acquire() # acquire data from digitizer
                    dq_result = np.mean(dq_result_raw,axis=1) # average all data over each trigger/segment 
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    # partition buffer into signal and background datasets FIXME: maybe need to use DEER option for 4 pts in analog math
                    s00, s0m, smm, smp = self.analog_math(dq_result, 'DQ', kwargs['num_pts']) # data for S0,0, S0,-1, S-1,-1, and S-1,+1 sequence
                except ValueError:
                    continue
                        
                s00_sweeps.append(np.stack([tau_times/1e3, s00])); s00_sweeps.updated_item(-1) 
                s0m_sweeps.append(np.stack([tau_times/1e3, s0m])); s0m_sweeps.updated_item(-1)
                smm_sweeps.append(np.stack([tau_times/1e3, smm])); smm_sweeps.updated_item(-1) 
                smp_sweeps.append(np.stack([tau_times/1e3, smp])); smp_sweeps.updated_item(-1)

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'DQ Relaxation',
                    'xlabel': 'Free Precession Interval (\u03BCs)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'S0,0' : s00_sweeps,
                                'S0,-1': s0m_sweeps,
                                'S-1,-1' : smm_sweeps,
                                'S-1,+1': smp_sweeps}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y = self.fit_data(
        #                 kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
        #             )
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))
                    
    @managed_experiment(token_prefix="DEER", dataset_key="dataset")
    def DEER_scan(self, *, mgr, data, token, **kwargs):
        """Run a DEER sweep over a set of MW frequencies."""
        cfg = nvcfg.DeerScanCfg(**kwargs)
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
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
            coupling=cfg.dig_coupling,
            termination=cfg.dig_termination,
            pretrig_size=cfg.pretrig_size,
            dig_timeout=cfg.dig_timeout,
            runs=cfg.runs,
        )
        
        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, cfg.awg_samp_rate_1)
            hdawg.set_sampling_rate(1, cfg.awg_samp_rate_2)
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
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
        
        dark_signal_sweeps, dark_background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        echo_signal_sweeps, echo_background_sweeps = StreamingList(), StreamingList()

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0

        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=cfg.iters,
                    iters_completed=0,
                    exception=type(e).__name__,
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
                    deer_result_raw = self.dig.acquire()
                    deer_result = np.mean(deer_result_raw, axis=1)
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / cfg.iters)
                    break

                try:
                    dark_sig, dark_bg, echo_sig, echo_bg = self.analog_math(deer_result, "DEER", cfg.num_pts) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                dark_signal_sweeps.append(np.stack([frequencies / 1e6, dark_sig])); dark_signal_sweeps.updated_item(-1)
                dark_background_sweeps.append(np.stack([frequencies / 1e6, dark_bg])); dark_background_sweeps.updated_item(-1)
                echo_signal_sweeps.append(np.stack([frequencies / 1e6, echo_sig])); echo_signal_sweeps.updated_item(-1)
                echo_background_sweeps.append(np.stack([frequencies / 1e6, echo_bg])); echo_background_sweeps.updated_item(-1)

                if cfg.fit_live:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(
                                cfg.dataset,
                                dark_signal_sweeps, dark_background_sweeps,
                                echo_signal_sweeps, echo_background_sweeps,
                                *cfg.fit_params,
                            )
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {cfg.dataset} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / cfg.iters)

                data.push({
                    "params": {"kwargs": kwargs, "iters_completed": iters_completed, "elapsed": time.perf_counter() - exp_start_time},
                    "title": "DEER",
                    "xlabel": "Surface Electron Resonance (MHz)",
                    "ylabel": "Signal (V) or Norm. Signal",
                    "datasets": {
                        "dark_signal": dark_signal_sweeps,
                        "dark_background": dark_background_sweeps,
                        "echo_signal": echo_signal_sweeps,
                        "echo_background": echo_background_sweeps,
                        "x_fit": fit_x,
                        "y_fit": fit_y,
                        "fit_value": fit_value,
                        "fit_error": fit_error,
                    },
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
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
                    fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(
                        cfg.dataset,
                        dark_signal_sweeps, dark_background_sweeps,
                        echo_signal_sweeps, echo_background_sweeps,
                        *cfg.fit_params,
                    )
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {cfg.dataset} measurement, {e}")

        if cfg.save:
            run_save(cfg.dataset, cfg.filename, [cfg.directory])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=cfg.iters,
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERRABI", dataset_key="dataset")
    def DEER_rabi_scan(self, *, mgr, data, token, **kwargs):
        """Run a DEER Rabi sweep over a set of MW pulse durations."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
        
        ### --- Default parameter array for sweep --- ###
        dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9            
 
        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(
            kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']
        ) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(kwargs['pi'] * 1e9 / 2)
            pi.append(kwargs['pi'] * 1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = dark_taus   
        fit_y = np.ones(len(fit_x))

        # define pulse sequence
        sequence = ps.DEER_Rabi(kwargs['laser_init']*1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], 
                            kwargs['tau']*1e9, kwargs['num_pts'], kwargs['laser_readout']*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="DEER",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'DEER Rabi',     
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': kwargs['dark_freq'],
                'mw_power': kwargs['awg_power'], 
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters'],
                'pi_pulses': dark_taus/1e9
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        dark_signal_sweeps, dark_background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        echo_signal_sweeps, echo_background_sweeps = StreamingList(), StreamingList()

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:   
                    deer_result_raw = self.dig.acquire() # acquire data from digitizer
                    deer_result = np.mean(deer_result_raw,axis=1) # average all data over each trigger/segment
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    dark_sig, dark_bg, echo_sig, echo_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue
                        
                dark_signal_sweeps.append(np.stack([dark_taus, dark_sig])); dark_signal_sweeps.updated_item(-1) 
                dark_background_sweeps.append(np.stack([dark_taus, dark_bg])); dark_background_sweeps.updated_item(-1)
                echo_signal_sweeps.append(np.stack([dark_taus, echo_sig])); echo_signal_sweeps.updated_item(-1) 
                echo_background_sweeps.append(np.stack([dark_taus, echo_bg])); echo_background_sweeps.updated_item(-1)

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'DEER Rabi',
                    'xlabel': 'MW Pulse Duration (ns)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'dark_signal' : dark_signal_sweeps, 'dark_background': dark_background_sweeps,
                                'echo_signal' : echo_signal_sweeps, 'echo_background': echo_background_sweeps,
                                'x_fit': fit_x, 'y_fit': fit_y, 'fit_value': fit_value, 'fit_error': fit_error}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100
      
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERFID", dataset_key="dataset")
    def DEER_FID_scan(self, *, mgr, data, token, **kwargs):
        """Run a DEER FID sweep over a set of MW pulse durations."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            case 'linspace':
                tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(
            kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']
        ) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = kwargs['dark_pi']
        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Define fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = tau_times/1e6
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.DEER_FID(kwargs['laser_init']*1e9, tau_times, pi_half[0], pi_half[1], 
                            pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="DEER",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'DEER FID',
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,  
                'pi_pulse': dark_pi, 
                'dark_freq': kwargs['dark_freq'],
                'mw_power': kwargs['awg_power'], 
                'num_pts': kwargs['num_pts'],
                'n': kwargs['n'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        dark_signal_sweeps, dark_background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        echo_signal_sweeps, echo_background_sweeps = StreamingList(), StreamingList()

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:   
                    deer_result_raw = self.dig.acquire() # acquire data from digitizer
                    deer_result = np.mean(deer_result_raw,axis=1) # average all data over each trigger/segment
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    dark_sig, dark_bg, echo_sig, echo_bg = self.analog_math(deer_result, 'DEER', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue
                        
                # notify the streaminglist that this entry has updated so it will be pushed to the data server
                dark_signal_sweeps.append(np.stack([tau_times, dark_sig])); dark_signal_sweeps.updated_item(-1) 
                dark_background_sweeps.append(np.stack([tau_times, dark_bg])); dark_background_sweeps.updated_item(-1)
                echo_signal_sweeps.append(np.stack([tau_times, echo_sig])); echo_signal_sweeps.updated_item(-1) 
                echo_background_sweeps.append(np.stack([tau_times, echo_bg])); echo_background_sweeps.updated_item(-1)

                # if kwargs.get("fit_live", False):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         try:
                #             fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                #         except (RuntimeError, OptimizeWarning) as e:
                #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'DEER Free Induction Decay',
                    'xlabel': 'Free Precession Interval (ns)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'dark_signal' : dark_signal_sweeps,
                                'dark_background': dark_background_sweeps,
                                'echo_signal' : echo_signal_sweeps,
                                'echo_background': echo_background_sweeps,}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERFIDCD", dataset_key="dataset")
    def DEER_FID_CD_scan(self, *, mgr, data, token, **kwargs):
        """Run a continuous drive DEER FID sweep over a set of free precession intervals.""" 
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
           
        ### --- Default parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            case 'linspace':
                tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = kwargs['dark_pi']
        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = tau_times/1e6
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.DEER_FID_CD(kwargs['laser_init']*1e9, tau_times, pi_half[0], pi_half[1], 
                                pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
        # dark_pulses = kwargs['pi']/2 + tau_times/1e9 + (kwargs['pi'] + 2*tau_times/1e9)*(kwargs['n']-1) + kwargs['pi'] + tau_times/1e9 + kwargs['pi']/2 
        
        dig_cfg = self.digitizer_configure(
            exp_type="CD",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'DEER FID CD',                
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': kwargs['dark_freq'],
                'pi_pulse': dark_pi,
                'taus': tau_times/1e9,
                'mw_power': kwargs['awg_power'],
                'cd_mw_power': kwargs['awg_cd_power'], 
                'num_pts': kwargs['num_pts'],
                'n': kwargs['n'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        dark_signal_sweeps, dark_background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
        echo_signal_sweeps, echo_background_sweeps = StreamingList(), StreamingList()
        cd_signal_sweeps, cd_background_sweeps = StreamingList(), StreamingList()

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:    
                    cd_result_raw = self.dig.acquire() # acquire data from digitizer
                    cd_result = np.mean(cd_result_raw,axis=1) # average all data over each trigger/segment 
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    dark_sig, dark_bg, echo_sig, echo_bg, cd_sig, cd_bg = self.analog_math(cd_result, 'CD', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                dark_signal_sweeps.append(np.stack([tau_times, dark_sig])); dark_signal_sweeps.updated_item(-1) 
                dark_background_sweeps.append(np.stack([tau_times, dark_bg])); dark_background_sweeps.updated_item(-1)
                echo_signal_sweeps.append(np.stack([tau_times, echo_sig])); echo_signal_sweeps.updated_item(-1) 
                echo_background_sweeps.append(np.stack([tau_times, echo_bg])); echo_background_sweeps.updated_item(-1)
                cd_signal_sweeps.append(np.stack([tau_times, cd_sig])); cd_signal_sweeps.updated_item(-1) 
                cd_background_sweeps.append(np.stack([tau_times, cd_bg])); cd_background_sweeps.updated_item(-1)

                # if kwargs.get("fit_live", False):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         try:
                #             fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                #         except (RuntimeError, OptimizeWarning) as e:
                #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'DEER Free Induction Decay Continuous Drive',
                    'xlabel': 'Free Precession Interval (ns)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'dark_signal' : dark_signal_sweeps,
                                'dark_background': dark_background_sweeps,
                                'echo_signal' : echo_signal_sweeps,
                                'echo_background': echo_background_sweeps,
                                'cd_signal' : cd_signal_sweeps,
                                'cd_background': cd_background_sweeps,}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100
      
        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))                

    @managed_experiment(token_prefix="DEERCORRRABI", dataset_key="dataset")
    def DEER_corr_rabi_scan(self, *, mgr, data, token, **kwargs):
        """Run a DEER Correlation Rabi sweep over a set of MW pulses."""
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])          

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = kwargs['dark_pi']
        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = dark_taus
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###  
        sequence = ps.DEER_Corr_Rabi(kwargs['laser_init']*1e9, dark_taus*1e9, kwargs['tau']*1e9, kwargs['t_corr']*1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], kwargs['laser_readout']*1e9) # send to PS in [ns] units
        dig_cfg = self.digitizer_configure(
            exp_type="Corr Rabi",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'DEER Corr Rabi',                
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': kwargs['dark_freq'],
                'dark_pulse': dark_pi,
                'mw_power': kwargs['awg_power'], 
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters'],
                'pi_pulses': dark_taus
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:     
                    corr_result_raw = self.dig.acquire() # acquire data from digitizer
                    corr_result=np.mean(corr_result_raw,axis=1) # average all data over each trigger/segment
                    # segments=(np.shape(corr_result))[0]
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(corr_result, 'Corr', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([dark_taus*1e9, sig])); signal_sweeps.updated_item(-1) 
                background_sweeps.append(np.stack([dark_taus*1e9, bg])); background_sweeps.updated_item(-1)

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                data.push({
                    'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                    'title': 'DEER Correlation Rabi',
                    'xlabel': 'MW Pulse Duration (ns)',
                    'ylabel': 'Signal (V) or Norm. Signal',
                    'datasets': {'signal' : signal_sweeps,
                                'background': background_sweeps}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100
      
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERT1", dataset_key="dataset")
    def DEER_T1_scan(self, *, mgr, data, token, **kwargs):
        """Run a DEER Correlation T1 sweep over a set of correlation intervals."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                t_corr_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     
            case 'linspace':
                t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

        # define pi pulses
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi = kwargs['dark_pi']

        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)

        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = t_corr_times/1e6
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Electron_T1(kwargs['laser_init']*1e9, t_corr_times*1e9, kwargs['tau']*1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], dark_pi*1e9, kwargs['laser_readout']*1e9) # send to PS in [ns] units
        dig_cfg = self.digitizer_configure(
            exp_type="DEER",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ##
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'DEER Corr T1',                  
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': kwargs['dark_freq'],
                'dark_pulse': dark_pi,
                'mw_power': kwargs['awg_power'], 
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']})
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
                
        with_pulse_py_sweeps, without_pulse_py_sweeps = StreamingList(), StreamingList()
        with_pulse_ny_sweeps, without_pulse_ny_sweeps = StreamingList(), StreamingList()
        
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    corr_result_raw = self.dig.acquire() # acquire data from digitizer
                    corr_result = np.mean(corr_result_raw,axis=1) # average all data over each trigger/segment 
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    with_py, without_py, with_ny, without_ny = self.analog_math(corr_result, 'DEER', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                with_pulse_py_sweeps.append(np.stack([t_corr_times[1:]*1e6, with_py[1:]])); with_pulse_py_sweeps.updated_item(-1) 
                without_pulse_py_sweeps.append(np.stack([t_corr_times[1:]*1e6, without_py[1:]])); without_pulse_py_sweeps.updated_item(-1)
                with_pulse_ny_sweeps.append(np.stack([t_corr_times[1:]*1e6, with_ny[1:]])); with_pulse_ny_sweeps.updated_item(-1) 
                without_pulse_ny_sweeps.append(np.stack([t_corr_times[1:]*1e6, without_ny[1:]])); without_pulse_ny_sweeps.updated_item(-1)

                if kwargs.get("fit_live", False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizeWarning)
                        try:
                            fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], with_pulse_py_sweeps, without_pulse_py_sweeps, with_pulse_ny_sweeps, without_pulse_ny_sweeps, *kwargs['fit_params'])
                        except (RuntimeError, OptimizeWarning) as e:
                            _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                                'title': 'DEER Correlation T1',
                                'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                'ylabel': 'Signal (V) or Norm. Signal',
                                'datasets': {'with_py': with_pulse_py_sweeps,
                                            'without_py': without_pulse_py_sweeps,
                                            'with_ny': with_pulse_ny_sweeps,
                                            'without_ny': without_pulse_ny_sweeps}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100
      
        if kwargs.get("fit", False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                except (RuntimeError, OptimizeWarning) as e:
                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="DEERT2", dataset_key="dataset")
    def DEER_T2_scan(self, *, mgr, data, token, **kwargs):
        """Run a DEER Correlation T1 sweep over a set of correlation intervals."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        match kwargs['array_type']:
            case 'geomspace':
                t_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     
            case 'linspace':
                t_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        dark_pi_half = kwargs['dark_pi']/2
        dark_pi = kwargs['dark_pi']

        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = t_times/1e6
        fit_y = np.ones(len(fit_x))

        # define pulse sequence
        sequence = ps.Electron_T2(kwargs['laser_init']*1e9, t_times*1e9, kwargs['tau']*1e9, pi_half[0], pi_half[1], 
                            pi[0], pi[1], dark_pi_half*1e9, dark_pi*1e9, kwargs['laser_readout']*1e9, kwargs['deer_t2_buffer']*1e9) # send to PS in [ns] units
        dig_cfg = self.digitizer_configure(
            exp_type="DEER T2",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )
           
        ### --- Upload AWG sequence --- ##
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'DEER T2',                  
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'dark_freq': kwargs['dark_freq'],
                'dark_half_pulse': dark_pi_half,
                'dark_pulse': dark_pi,
                'mw_power': kwargs['awg_power'], 
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']
            })    
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
        
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)

        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return
            
            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    deer_t2_result_raw = self.dig.acquire() # acquire data from digitizer
                    deer_t2_result = np.mean(deer_t2_result_raw, axis=1) # average all data over each trigger/segment
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(deer_t2_result, 'DEER T2', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([t_times*1e9, sig])); signal_sweeps.updated_item(-1) 
                background_sweeps.append(np.stack([t_times*1e9, bg])); background_sweeps.updated_item(-1)

                # if kwargs.get("fit_live", False):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         try:
                #             fit_value, fit_error, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                #         except (RuntimeError, OptimizeWarning) as e:
                #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])

                # save the current data to the data server
                data.push({'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                                'title': 'DEER T2',
                                'xlabel': 'Free Precession Interval (ns)',
                                'ylabel': 'Signal (V) or Norm. Signal',
                                'datasets': {'signal': signal_sweeps,
                                            'background': background_sweeps}
                })

                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100
      
        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
        
        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))

    @managed_experiment(token_prefix="CORRSPECRF", dataset_key="dataset")
    def Corr_Spec_RF_scan(self, *, mgr, data, token, **kwargs):
        """Run a Correlation Spectroscopy NMR sweep over a set of precession time intervals."""
        
        ### --- Devices --- ###  
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = t_corr_times/1e9
        fit_y = np.ones(len(fit_x))

        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Corr_Spectroscopy_RF(kwargs['laser_init']*1e9, t_corr_times, kwargs['tau']*1e9, 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="Corr Spec",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        corr_spec_time = pi_half[0] + (4*pi[0] + 4*pi[1] + 8*kwargs['tau']*1e9)*kwargs['n'] + pi_half[1] + kwargs['stop']*1e9 + \
                            pi_half[0] + (4*pi[0] + 4*pi[1] + 8*kwargs['tau']*1e9)*kwargs['n'] + pi_half[1]
        
        total_exp_time = 100 + kwargs['laser_init']*1e9 + 500 + corr_spec_time + 100 + kwargs['laser_readout'] + 100

        ### --- Upload AWG sequence --- ##
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'NMR',
                'seq_nmr': 'Correlation Spectroscopy',
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'n': kwargs['n'],
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters'],
                'total_exp_time': total_exp_time*1e-9,
                'rf_power': 0.2,
                'rf_freq': 2.88e6,
                'rf_phase': 0
            })
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
    
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break
                
                try:
                    nmr_result_raw = self.dig.acquire() # acquire data from digitizer
                    nmr_result = np.mean(nmr_result_raw,axis=1) # average all data over each trigger/segment 
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(nmr_result, 'NMR', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([t_corr_times / 1e3, sig])); signal_sweeps.updated_item(-1) 
                background_sweeps.append(np.stack([t_corr_times / 1e3, bg])); background_sweeps.updated_item(-1)

                # if kwargs.get("fit_live", False):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         try:
                #             fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                #         except (RuntimeError, OptimizeWarning) as e:
                #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])
   
                data.push({'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                                'title': 'NMR Time Domain Data',
                                'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                'ylabel': 'Signal (V) or Norm. Signal',
                                'datasets': {'signal' : signal_sweeps,
                                            'background': background_sweeps}
                })
                        
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y = self.fit_data(
        #                 kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
        #             )
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))                

    @managed_experiment(token_prefix="CORRSPEC", dataset_key="dataset")
    def Corr_Spec_scan(self, *, mgr, data, token, **kwargs):
        """Run a Correlation Spectroscopy NMR sweep over a set of precession time intervals."""
        
        ### --- Devices --- ###  
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
            
        ### --- Default parameter array for sweep --- ###
        t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

        ### --- Define NV drive parameters --- ###
        sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default
        pi: List[float] = []
        pi_half: List[float] = []
        for i in range(2):
            pi_half.append(kwargs['pi']*1e9/2)
            pi.append(kwargs['pi']*1e9)
        
        self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_power"]) # configure signal generator for NV drive

        ### --- Default fit parameters for live fitting --- ###
        fit_value, fit_error = [], []
        fit_x = t_corr_times/1e9
        fit_y = np.ones(len(fit_x))
            
        ### --- Set up pulse streamer and digitizer for experiment --- ###
        sequence = ps.Corr_Spectroscopy(kwargs['laser_init']*1e9, t_corr_times, kwargs['tau']*1e9, 
                                    pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
        dig_cfg = self.digitizer_configure(
            exp_type="Corr Spec",
            num_pts=kwargs['num_pts'],
            iters=kwargs["iters"],
            segment_size=kwargs["segment_size"],
            sampling_freq=kwargs["dig_sampling_freq"],
            dig_amplitude=kwargs["dig_amplitude"],
            read_channel=kwargs["read_channel"],
            coupling=kwargs["dig_coupling"],
            termination=kwargs["dig_termination"],
            pretrig_size=kwargs["pretrig_size"],
            dig_timeout=kwargs["dig_timeout"],
            runs=kwargs["runs"],
        )

        ### --- Upload AWG sequence --- ###
        try:
            hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
            hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])
            hdawg.set_sequence(**{
                'seq': 'NMR',
                'seq_nmr': 'Correlation Spectroscopy',
                'i_offset': kwargs['i_offset'],
                'q_offset': kwargs['q_offset'],
                'sideband_power': kwargs['sideband_power'],
                'sideband_freq': kwargs['sideband_freq'], 
                'iq_phases': iq_phases,
                'pihalf_x': pi_half[0]/1e9,
                'pihalf_y': pi_half[1]/1e9,
                'pi_x': pi[0]/1e9, 
                'pi_y': pi[1]/1e9,
                'n': kwargs['n'],
                'num_pts': kwargs['num_pts'],
                'runs': kwargs['runs'], 
                'iters': kwargs['iters']})
        except Exception as e:
            self.queue_from_exp.put_nowait(self.build_status_msg(
                status="failed",
                percent_completed=0,
                fit_value=fit_value,
                fit_error=fit_error,
                start_time=time.perf_counter(),
                total_iters=kwargs["iters"],
                iters_completed=0,
                exception=type(e).__name__,
            ))
            return
            
        signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
    
        self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
        laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

        ### --- Initialize experiment state variables --- ###
        stopped = False
        failed = False
        exception_type = None
        iters_completed = 0
        percent_completed = 0
                
        ### --- Open laser shutter and emit MW for NV drive --- ###
        with _shutter_open(laser_shutter), _rf_on(sig_gen):
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
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
                return

            exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

            ### --- Main experiment loop --- ###
            for i in range(kwargs['iters']):
                if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                    stopped = True
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break
                
                try:
                    nmr_result_raw = self.dig.acquire() # acquire data from digitizer
                    nmr_result = np.mean(nmr_result_raw,axis=1) # average all data over each trigger/segment 
                except Exception as e:
                    failed = True
                    exception_type = type(e).__name__
                    iters_completed = i
                    percent_completed = int(100 * iters_completed / kwargs["iters"])
                    break

                try:
                    sig, bg = self.analog_math(nmr_result, 'NMR', kwargs['num_pts']) # partition buffer into signal and background datasets
                except ValueError:
                    continue

                signal_sweeps.append(np.stack([t_corr_times / 1e3, sig])); signal_sweeps.updated_item(-1) 
                background_sweeps.append(np.stack([t_corr_times / 1e3, bg])); background_sweeps.updated_item(-1)

                # if kwargs.get("fit_live", False):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         try:
                #             fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                #         except (RuntimeError, OptimizeWarning) as e:
                #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                iters_completed = i + 1
                percent_completed = int(100 * iters_completed / kwargs["iters"])
   
                data.push({'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                                'title': 'NMR Time Domain Data',
                                'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                'ylabel': 'Signal (V) or Norm. Signal',
                                'datasets': {'signal' : signal_sweeps,
                                            'background': background_sweeps}
                })
                        
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="in progress",
                    percent_completed=percent_completed,
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                ))

        ### --- Experiment complete: final save and status update --- ###
        if not stopped and not failed:
            iters_completed, percent_completed = kwargs["iters"], 100

        # if kwargs.get("fit", False):
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             fit_value, fit_error, fit_x, fit_y = self.fit_data(
        #                 kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
        #             )
        #         except (RuntimeError, OptimizeWarning) as e:
        #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

        if kwargs.get("save", False):
            run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

        status = "failed" if failed else ("stopped" if stopped else "complete")
        self.queue_from_exp.put_nowait(self.build_status_msg(
            status=status,
            percent_completed=int(percent_completed),
            fit_value=fit_value,
            fit_error=fit_error,
            start_time=exp_start_time,
            total_iters=kwargs["iters"],
            iters_completed=iters_completed,
            exception=exception_type if failed else None,
        ))    

    @managed_experiment(token_prefix="CASR", dataset_key="dataset")
    def CASR_scan(self, *, mgr, data, token, **kwargs):
        """
        Run a Coherently Averaged Synchronized Readout NMR sweep over a range of frequencies.
        Choose either coil drive or RF pi/2 pulse for nuclear spin control."""
        
        ### --- Devices --- ###
        laser = mgr.laser
        laser_shutter = mgr.laser_shutter
        sig_gen = mgr.sg
        ps = mgr.ps
        hdawg = mgr.awg
        
        
        ### --- Intervals in pulse sequence used to define time points on x-axis in seconds --- ###
        laser_init_time = kwargs['laser_init']*1e9
        singlet_decay = 500 
        pi = [kwargs['pi']*1e9, kwargs['pi']*1e9]
        pi_half = [pi[0]/2, pi[1]/2]
        tau = int(kwargs['tau']*1e9) # half period of central frequency [ns] - used in DD blocks
        print(f"tau = {tau} ns")
        period = 2*tau

        dd_time = pi_half[0] + (4*pi[0] + 4*pi[1] + 8*tau)*kwargs['n'] + pi_half[1]

        mw_buffer_time = 100 # buffer time between DD block and readout pulse [ns]
        laser_read_time = kwargs['laser_readout']*1e9 # laser readout pulse [ns]
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
                print(f"\u0394f = f - f0 = {(0.5/((tau+pi[0])*1e-9) - kwargs['rf_pulse_freq'])/1000} kHz")
            except AssertionError as e:  
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status="failed",
                    percent_completed=0,
                    fit_value=None,
                    fit_error=None,
                    start_time=time.perf_counter(),
                    total_iters=kwargs["iters"],
                    iters_completed=0,
                    exception=type(e).__name__,
                ))
            else:
                ### --- Define time points for x-axis based on sequence parameters --- ###
                times = np.linspace(t_seq - wait_time - laser_read_time / 2, kwargs['num_pts'] * t_seq, kwargs['num_pts']) * 1e-9

                ### --- Define NV drive parameters --- ###
                sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default

                self._configure_sig_gen_iq(sig_gen, carrier_freq=sig_gen_freq, rf_power=kwargs["rf_pulse_power"]) # configure signal generator for NV drive

                ### --- Default fit parameters for live fitting --- ###
                fit_value, fit_error = [], []
                fit_x = times
                fit_y = np.ones(len(fit_x))

                ### --- Set up pulse streamer and digitizer for experiment --- ###
                if kwargs['sig_opt'] == 'Coil':
                    print("Using CASR Coil sequence")
                    rf_duration = kwargs['num_pts'] * t_seq * 1e-9 # coil on for entire sequence duration
                    seq_rf = ps.CASR_RF_Coil() # coil drive version
                else:
                    rf_duration = kwargs['rf_pi_half'] # RF pi/2 pulse duration for nuclear spin control
                    seq_rf = ps.CASR_RF(kwargs['rf_pi_half'] * 1e9)
                
                seq_nv = ps.CASR_NV(laser_init_time, singlet_decay, 
                                pi_half[0], pi_half[1], pi[0], pi[1], 
                                tau, kwargs['n'], mw_buffer_time, kwargs['laser_readout'], wait_time) # NV DD subsequence
                
                dig_cfg = self.digitizer_configure(
                    exp_type="CASR",
                    num_pts=kwargs['num_pts'],
                    iters=kwargs["iters"],
                    segment_size=kwargs["segment_size"],
                    sampling_freq=kwargs["dig_sampling_freq"],
                    dig_amplitude=kwargs["dig_amplitude"],
                    read_channel=kwargs["read_channel"],
                    coupling=kwargs["dig_coupling"],
                    termination=kwargs["dig_termination"],
                    pretrig_size=kwargs["pretrig_size"],
                    dig_timeout=kwargs["dig_timeout"],
                    runs=kwargs["runs"],
                )

                ### --- Upload AWG sequence --- ##
                try:
                    hdawg.set_sampling_rate(0, kwargs['awg_samp_rate_1'])
                    hdawg.set_sampling_rate(1, kwargs['awg_samp_rate_2'])   
                    hdawg.set_sequence(**{
                        'seq': 'CASR',
                        'i_offset': kwargs['i_offset'],
                        'q_offset': kwargs['q_offset'],
                        'sideband_power': kwargs['sideband_power'],
                        'sideband_freq': kwargs['sideband_freq'], 
                        'iq_phases': iq_phases,
                        'pihalf_x': pi_half[0]/1e9,
                        'pihalf_y': pi_half[1]/1e9,
                        'pi_x': pi[0]/1e9, 
                        'pi_y': pi[1]/1e9,
                        'n_R': kwargs['num_pts'],
                        'n': kwargs['n'],
                        'rf_freq': kwargs['rf_pulse_freq'],
                        'rf_power': kwargs['rf_pulse_power'],
                        'rf_phase': kwargs['rf_pulse_phase'],
                        'rf_pihalf': rf_duration})
                        # 'rf_pihalf': kwargs['rf_pi_half']})  
                        # 'rf_pihalf': kwargs['num_pts']*t_seq*1e-9})
                except Exception as e:
                    self.queue_from_exp.put_nowait(self.build_status_msg(
                        status="failed",
                        percent_completed=0,
                        fit_value=fit_value,
                        fit_error=fit_error,
                        start_time=time.perf_counter(),
                        total_iters=kwargs["iters"],
                        iters_completed=0,
                        exception=type(e).__name__,
                    ))
                    return
                
                signal_sweeps, background_sweeps = StreamingList(), StreamingList() # for storing the experiment data --> list of numpy arrays of shape (2, num_points)

                self.dig.assign_param(dig_cfg) # upload digitizer parameters for experiment
                laser.set_diode_current_realtime(kwargs["laser_power"]) # set laser power

                ### --- Initialize experiment state variables --- ###
                stopped = False
                failed = False
                exception_type = None
                iters_completed = 0
                percent_completed = 0

                ### --- Open laser shutter and emit MW for NV drive --- ###
                with _shutter_open(laser_shutter), _rf_on(sig_gen):
                    try:
                        ps.set_soft_trigger()

                        # new pulse streamer format:
                        # upload both sequences to pulse streamer
                        # 1. CASR RF sequence -> pi/2 on nuclear spins (n_runs = 1)
                        # 2. CASR NV sequence -> pi/2 on electron spins (n_runs = num_pts x 2)
                        ps.upload(slot_nr=0, data=seq_rf, n_runs=1, next_action=NextAction.SWITCH_SLOT, when=When.IMMEDIATE, owner=token) #upload initial sequence data with n_runs=1
                        ps.upload(slot_nr=1, data=seq_nv, n_runs=kwargs['num_pts']*2,next_action=NextAction.SWITCH_SLOT, when=When.IMMEDIATE, owner=token) #upload measurement sequence with n_runs=num_pts*2
                        
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
                            start_time=time.perf_counter(),
                            total_iters=kwargs["iters"],
                            iters_completed=0,
                            exception=type(e).__name__,
                        ))
                        return
            
                    exp_start_time = time.perf_counter()  # start timer for experiment (used for time remaining estimate)

                    ### --- Main experiment loop --- ###
                    for i in range(kwargs['iters']):
                        if experiment_widget_process_queue(self.queue_to_exp) == "stop":
                            stopped = True
                            iters_completed = i
                            percent_completed = int(100 * iters_completed / kwargs["iters"])
                            break

                        try:     
                            casr_result_raw = self.dig.acquire() # acquire data from digitizer
                            casr_result = np.mean(casr_result_raw,axis=1) # average all data over each trigger/segment 
                        except Exception as e:
                            failed = True
                            exception_type = type(e).__name__
                            iters_completed = i
                            percent_completed = int(100 * iters_completed / kwargs["iters"])
                            break

                        try:
                            sig, bg = self.analog_math(casr_result, 'CASR', kwargs['num_pts']) # partition buffer into signal and background datasets
                        except ValueError:
                            continue

                        signal_sweeps.append(np.stack([times[:-1]*1e3, sig[:-1]])); signal_sweeps.updated_item(-1) # exclude last point (outlier)
                        background_sweeps.append(np.stack([times[:-1]*1e3, bg[:-1]])); background_sweeps.updated_item(-1) # exclude last point (outlier)
                        
                        if kwargs.get("fit_live", False):  
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_error, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                        
                        iters_completed = i + 1
                        percent_completed = int(100 * iters_completed / kwargs["iters"])

                        data.push({'params': {'kwargs': kwargs, 'iters_completed': iters_completed, 'elapsed': time.perf_counter() - exp_start_time},
                                        'title': 'CASR Time Domain Data',
                                        'xlabel': 'Free Precession Interval (ms) or Frequency (kHz)',
                                        'ylabel': 'Signal (V) or Norm. Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        self.queue_from_exp.put_nowait(self.build_status_msg(
                            status="in progress",
                            percent_completed=percent_completed,
                            fit_value=fit_value,
                            fit_error=fit_error,
                            start_time=exp_start_time,
                            total_iters=kwargs["iters"],
                            iters_completed=iters_completed,
                        ))

                ### --- Experiment complete: final save and status update --- ###
                if not stopped and not failed:
                    iters_completed, percent_completed = kwargs["iters"], 100

                # if kwargs.get("fit", False):
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("error", OptimizeWarning)
                #         try:
                #             fit_value, fit_error, fit_x, fit_y = self.fit_data(
                #                 kwargs["dataset"], signal_sweeps, background_sweeps, *kwargs["fit_params"]
                #             )
                #         except (RuntimeError, OptimizeWarning) as e:
                #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                if kwargs.get("save", False):
                    run_save(kwargs["dataset"], kwargs["filename"], [kwargs["directory"]])

                status = "failed" if failed else ("stopped" if stopped else "complete")
                self.queue_from_exp.put_nowait(self.build_status_msg(
                    status=status,
                    percent_completed=int(percent_completed),
                    fit_value=fit_value,
                    fit_error=fit_error,
                    start_time=exp_start_time,
                    total_iters=kwargs["iters"],
                    iters_completed=iters_completed,
                    exception=exception_type if failed else None,
                ))                
    

    """ Data Fitting """

    @staticmethod
    def negative_lorentzian(x, A, x0, gamma, c):
        return -A / (1 + ((x - x0) / gamma)**2) + c
    
    @staticmethod
    def positive_lorentzian(x, A, x0, gamma, c):
        return A / (1 + ((x - x0) / gamma)**2) + c
    
    @staticmethod
    def decaying_cosine(x, A, gamma, T, phi, c):
        return A * np.exp(-gamma * x) * np.cos(2 * np.pi * x / T + phi) + c
    
    @staticmethod
    def stretched_exponential(x, A, T, n, c):
        return A*np.exp(-(x/T)**n) + c

    @staticmethod
    def mod_stretched_exponential(x, A, T, n, a1, f1, phi1, a2, f2, phi2):
        return A*np.exp(-(x/T)**n)*(1-a1*np.sin(2*np.pi*f1*x/4 + phi1)**2)*(1-a2*np.sin(2*np.pi*f2*x/4 + phi2)**2)

    def fit_data(self, exp, sig_data, back_data, *args):
        # Combine all signal sweeps into a single 3D array and average
        all_signal_data = np.stack(sig_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_sig = np.mean(all_signal_data[1, :, :], axis=1)  # Shape: (10,)

        # Combine all background sweeps into a single 3D array and average
        all_background_data = np.stack(back_data, axis=-1)  # Shape: (2, 10, 5)
        averaged_bg = np.mean(all_background_data[1, :, :], axis=1)  # Shape: (10,)

        # Compute the microwave_times (assumed constant across sweeps)
        x_values = all_signal_data[0, :, 0]  # Shape: (10,)
        x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting

        # Compute the ratio/difference of averaged signal and background for fitting
        if exp == 'odmr' or exp == 'rabi':
            y_values = averaged_sig / averaged_bg  # Shape: (10,)
        elif exp == 't1' or exp == 't2':
            y_values = averaged_bg - averaged_sig

        # Initial guesses for parameters: A, gamma, f, phi, C
        # initial_guess = [0.02, 0.001, 200, 0, 1]
        initial_guess = list(args)

        # Perform curve fitting 
        match exp:
            case 'odmr':
                params, covariance = curve_fit(self.negative_lorentzian, x_values, y_values, p0=initial_guess)
                y_fit = self.negative_lorentzian(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                # fitted_value = round(x_fit[np.argmin(y_fit)],4)
                fitted_values = [round(i, 4) for i in params]
                fitted_errors = [round(i, 4) for i in param_errors]
            case 'rabi':
                params, covariance = curve_fit(self.decaying_cosine, x_values, y_values, p0=initial_guess)
                y_fit = self.decaying_cosine(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                fitted_values = [round(i, 2) for i in params]
                fitted_errors = [round(i, 2) for i in param_errors]
                fitted_values[2] = round(x_fit[np.argmin(y_fit)],2)
                fitted_errors[2] = 0
            case 'odmr rf':
                params, covariance = curve_fit(self.negative_lorentzian, x_values, y_values, p0=initial_guess)
                y_fit = self.negative_lorentzian(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                # fitted_value = round(x_fit[np.argmin(y_fit)],4)
                fitted_values = [round(i, 4) for i in params]
                fitted_errors = [round(i, 4) for i in param_errors]
            case 't1':
                params, covariance = curve_fit(self.stretched_exponential, x_values, y_values, p0=initial_guess)
                y_fit = self.stretched_exponential(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                fitted_values = [round(i, 3) for i in params]
                fitted_errors = [round(i, 3) for i in param_errors]
            case 't2':
                params, covariance = curve_fit(self.mod_stretched_exponential, x_values, y_values, p0=initial_guess)
                y_fit = self.mod_stretched_exponential(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                fitted_values = [round(i, 3) for i in params]
                fitted_errors = [round(i, 3) for i in param_errors]

        return fitted_values, fitted_errors, x_fit, y_fit
    
    def fit_deer_data(self, exp, dark_sig_data, dark_back_data, echo_sig_data, echo_back_data, *args):
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
        x_values = all_dark_signal_data[0, :, 0]  # Shape: (10,)
        x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting

        # Compute the ratio/difference of averaged signal and background for fitting
        deer = (averaged_dark_bg - averaged_dark_sig) / (averaged_dark_bg + averaged_dark_sig)
        echo = (averaged_echo_bg - averaged_echo_sig) / (averaged_echo_bg + averaged_echo_sig)

        y_values = deer / echo

        # Initial guesses for parameters: A, gamma, f, phi, C
        # initial_guess = [0.02, 0.001, 200, 0, 1]
        initial_guess = list(args)

        # Perform curve fitting 
        match exp:
            case 'deer':
                params, covariance = curve_fit(self.negative_lorentzian, x_values, y_values, p0=initial_guess)
                y_fit = self.negative_lorentzian(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                # fitted_value = round(x_fit[np.argmin(y_fit)],3)
                fitted_values = [round(i, 3) for i in params]
                fitted_errors = [round(i, 3) for i in param_errors]
                # fitted_values[0] = 100*round(params[0], 3) # to display DEER contrast as percent
                # fitted_errors[0] = 100*round(param_errors[0], 3)
            case 'deer rabi':
                params, covariance = curve_fit(self.decaying_cosine, x_values, y_values, p0=initial_guess)
                y_fit = self.decaying_cosine(x_fit, *params)
                param_errors = np.sqrt(np.diag(covariance))
                # fitted_value = round(x_fit[np.argmin(y_fit)],2)
                fitted_values = [round(i, 2) for i in params]
                fitted_errors = [round(i, 2) for i in param_errors]
                # fitted_values[2] = round(params[2]/2, 2)
                # fitted_errors[2] = round(param_errors[2]/2, 2)
                fitted_values[2] = round(x_fit[np.argmin(y_fit)],2)
                fitted_errors[2] = 0

        # # Extract fitted parameters
        # A_fit, gamma_fit, f_fit, phi_fit, C_fit = params

        # print(f"Fitted pi pulse = {fitted_value} ns")
        return fitted_values, fitted_errors, x_fit, y_fit


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
