"""
NanoNMR-M Experiments

Written by Evan Villafranca
Updated on 2/20/2025

"""
import time
import logging

from pathlib import Path
from typing import List

from pulsestreamer import PulseStreamer
from spcm import units
from digitizer_driver import SpectrumDigitizer

import math
import numpy as np
import warnings
from scipy.optimize import curve_fit, OptimizeWarning
from numba import njit 

from nspyre import DataSource, DataSink
from nspyre import InstrumentServer, InstrumentManager
from nspyre import experiment_widget_process_queue
from nspyre import StreamingList
from nspyre.data.save import save_json
# from nspyre import nspyre_init_logger
# from saver import DataSaver

from customUtils import flexSave

_HERE = Path(__file__).parent
_logger = logging.getLogger(__name__)

class SpinMeasurements:
    """NanoNMR-M experiments
    
    """
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
        
    def run_save(self, data_name, file_name, directory):
        logging.info("Saving file with flexSave...")
        flexSave(data_name, data_name, file_name, directory)
    
    # @njit(parallel=True)
    def analog_math(self, array, exp_type, pts):        
        """Return list of data arrays for each experiment subsequence.
        
        Slices input data array into constituent subsequences (alike data pts)
        Example: mw on & mw off subsequences --> array[::2], array[1::2]
        """
        if exp_type == 'Cal' or exp_type == 'Optical_T1':
            sig = sig + bg
            runs = len(sig) // pts  # Number of runs
            sig_array = sig.reshape(runs, pts).T.mean(axis=1).magnitude
            
            return sig_array
        
        elif exp_type == 'DEER':
            sliced_arrays = array.reshape(-1, 4)
            dark_sig, dark_bg, echo_sig, echo_bg = sliced_arrays[:, 0], sliced_arrays[:, 1], sliced_arrays[:, 2], sliced_arrays[:, 3]

            runs = len(dark_sig) // pts  # Number of runs
            # Reshape arrays to have `runs` rows and `pts` columns, then compute mean along axis 0
            dark_ms1_array = np.mean(dark_sig.reshape(runs, pts), axis=0).magnitude
            dark_ms0_array = np.mean(dark_bg.reshape(runs, pts), axis=0).magnitude
            echo_ms1_array = np.mean(echo_sig.reshape(runs, pts), axis=0).magnitude
            echo_ms0_array = np.mean(echo_bg.reshape(runs, pts), axis=0).magnitude

            return [dark_ms1_array, dark_ms0_array, echo_ms1_array, echo_ms0_array]

        elif exp_type == 'CD':
            sliced_arrays = array.reshape(-1, 6)
            dark_sig, dark_bg, echo_sig, echo_bg, cd_sig, cd_bg = sliced_arrays[:, 0], sliced_arrays[:, 1], sliced_arrays[:, 2], sliced_arrays[:, 3], sliced_arrays[:, 4], sliced_arrays[:, 5]

            runs = len(dark_sig) // pts  # Number of runs
            # Reshape arrays to have `runs` rows and `pts` columns, then compute mean along axis 0
            dark_ms1_array = np.mean(dark_sig.reshape(runs, pts), axis=0).magnitude
            dark_ms0_array = np.mean(dark_bg.reshape(runs, pts), axis=0).magnitude
            echo_ms1_array = np.mean(echo_sig.reshape(runs, pts), axis=0).magnitude
            echo_ms0_array = np.mean(echo_bg.reshape(runs, pts), axis=0).magnitude
            cd_ms1_array = np.mean(cd_sig.reshape(runs, pts), axis=0).magnitude
            cd_ms0_array = np.mean(cd_bg.reshape(runs, pts), axis=0).magnitude

            return [dark_ms1_array, dark_ms0_array, echo_ms1_array, echo_ms0_array, cd_ms1_array, cd_ms0_array]
        
        elif exp_type == 'CASR':
            sliced_arrays = array.reshape(-1, pts)
            sig, bg = sliced_arrays[::2, :], sliced_arrays[1::2, :]

            runs = len(sig) # Number of rows in sig = runs

            # Reshape arrays to have `runs` rows and `pts` columns, then compute mean along axis 0
            ms1_array = np.mean(sig, axis=0).magnitude
            ms0_array = np.mean(bg, axis=0).magnitude

            return [ms1_array, ms0_array]
        
        else:
            sliced_arrays = array.reshape(-1, 2)
            sig, bg = sliced_arrays[:, 0], sliced_arrays[:, 1] 

            runs = len(sig) // pts  # Number of runs

            # Reshape arrays to have `runs` rows and `pts` columns, then compute mean along axis 0
            ms1_array = np.mean(sig.reshape(runs, pts), axis=0).magnitude
            ms0_array = np.mean(bg.reshape(runs, pts), axis=0).magnitude

            return [ms1_array, ms0_array]



    """ Configure equipment """

    def choose_sideband(self, opt, nv_freq, side_freq, pulse_axis='x'):
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
    
    def digitizer_configure(self, **kwargs):
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

        dig_config = {'num_pts_in_exp': 2*kwargs['num_pts_in_exp'], # MW ON + OFF subsequences
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
    
    def volt_factor(self, detector):
        if detector == 'APD':
            factor = 1
        else:
            factor = -1
        return factor 
    
    def equipment_off(self, detector):
        """Shut off equipment at experiment end/stop.
        """
        with InstrumentManager() as mgr:
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg

            laser.laser_off()
            laser_shutter.close_shutter()

            sig_gen.set_rf_toggle(0)
            sig_gen.set_mod_toggle(0)

            ps.Pulser.reset()
            self.dig.stop_card()
            self.dig.reset()
            hdawg.set_disabled()

    def set_srs396(self, freq, power):
        with InstrumentManager() as mgr:
            sig_gen = mgr.sg
            sig_gen.set_frequency(freq) # set carrier frequency
            sig_gen.set_rf_amplitude(power) # set MW power
            sig_gen.set_mod_type(7) # quadrature amplitude modulation
            sig_gen.set_mod_subtype(1) # no constellation mapping
            sig_gen.set_mod_function('IQ', 5) # external modulation
            sig_gen.set_mod_toggle(1) # turn on modulation mode

    """ Experiment logic """

    # def experiment_scan(self, **kwargs):
    #     with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as data:
    #         # load devices used in scan
    #         laser = mgr.laser
    #         laser_shutter = mgr.laser_shutter
    #         pickoff_shutter = mgr.pickoff_shutter
    #         sig_gen = mgr.sg
    #         ps = mgr.ps
    #         hdawg = mgr.awg

    #         # define pulse sequence
    #         sequence = ps.CW_ODMR(kwargs['num_pts'], kwargs['probe']*1e9) # pulse streamer sequence for CW ODMR

    #         # configure digitizer (need to set impedance = 1 Mohm for AC coupling)
    #         dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
    #                                               segment_size = kwargs['segment_size'], sampling_freq = kwargs['dig_sampling_freq'], dig_amplitude = kwargs['dig_amplitude'], 
    #                                               read_channel = kwargs['read_channel'], coupling = kwargs['dig_coupling'], termination = kwargs['dig_termination'], 
    #                                               pretrig_size = kwargs['pretrig_size'], dig_timeout = kwargs['dig_timeout'], runs = kwargs['runs'])
            
    #         # configure signal generator for NV drive
    #         sig_gen.set_frequency(sig_gen_freq) # set carrier frequency
    #         sig_gen.set_rf_amplitude(kwargs['rf_power']) # set MW power
    #         sig_gen.set_mod_type(7) # quadrature amplitude modulation
    #         sig_gen.set_mod_subtype(1) # no constellation mapping
    #         sig_gen.set_mod_function('IQ', 5) # external modulation
    #         sig_gen.set_mod_toggle(1) # turn on modulation mode

    def sigvstime_scan(self, **kwargs):     
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as sigvstime_data:
            # run laser on continuously here from laser driver
            laser_shutter = mgr.laser_shutter
            ps = mgr.ps

            sequence = ps.SigvsTime(1/kwargs['exp_sampling_rate'] * 1e9) # pulse streamer sequence for CW ODMR

            # configure digitizer (need to use DC coupling for signal vs time)           
            if kwargs['sigvstime_detector'] == 'BPD':
                dig_config = self.digitizer_configure(num_pts_in_exp = 1, iters = 1, 
                                                    segment_size = kwargs['segment_size'], sampling_freq = 0.5e9, dig_amplitude = 5, 
                                                    read_channel = kwargs['read_channel'], coupling = 'DC', termination = '1M', 
                                                    pretrig_size = kwargs['pretrig_size'], dig_timeout = 5, runs = 400)
            else:
                dig_config = self.digitizer_configure(num_pts_in_exp = 1, iters = 1, 
                                                    segment_size = kwargs['segment_size'], sampling_freq = 0.5e9, dig_amplitude = 5, 
                                                    read_channel = kwargs['read_channel'], coupling = 'DC', termination = '1M', 
                                                    pretrig_size = kwargs['pretrig_size'], dig_timeout = 5, runs = 400)
                
            time_start = time.time()

            signal_sweeps = StreamingList()

            # open laser shutter
            laser_shutter.open_shutter()
            
            # upload digitizer parameters
            self.dig.assign_param(dig_config)

            # set pulsestreamer to start on software trigger & run infinitely
            ps.set_soft_trigger()
            ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
            
            # start digitizer --> waits for trigger from pulse sequence
            self.dig.config()
            self.dig.start_buffer()
            
            # start pulse sequence
            ps.start_now()
            
            for i in range(10000):                
                sig_result_raw = self.dig.acquire() # acquire data from digitizer

                # average all data over each trigger/segment 
                sig_result = np.mean(sig_result_raw,axis=1)
                sig_result = np.mean(sig_result)

                time_pt = time.time() - time_start

                # read the analog voltage levels received by the APD.
                # notify the streaminglist that this entry has updated so it will be pushed to the data server
                signal_sweeps.append(np.array([[time_pt], [sig_result]]))
                signal_sweeps.updated_item(-1) 
                
                # save the current data to the data server.
                sigvstime_data.push({'params': {'kwargs': kwargs},
                                     'title': 'Signal Vs Time',
                                     'xlabel': 'Time step',
                                     'ylabel': 'APD Voltage (V)',
                                     'datasets': {'signal': signal_sweeps}})

                self.queue_from_exp.put_nowait(['0', 'in progress', None])

                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                    # the GUI has asked us nicely to exit. Save data if requested.
                    self.equipment_off(kwargs['sigvstime_detector'])
                    self.queue_from_exp.put_nowait(['0', 'stopped', None])
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    return
                
    def odmr_scan(self, **kwargs):
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

            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 

            max_sideband_freq = 2*kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> ? MHz max. for AWG bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
            mod_freqs = np.flip(mod_freqs)
                        
            real_freqs = sig_gen_freq - mod_freqs
            
            # default fit parameters
            fit_value = None
            fit_x = real_freqs/1e9
            fit_y = np.ones(len(fit_x))

            # define pulse sequence
            sequence = ps.CW_ODMR(kwargs['num_pts'], kwargs['probe']*1e9) # pulse streamer sequence for CW ODMR

            # configure digitizer (need to set impedance = 1 Mohm for AC coupling)
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
                                                  segment_size = kwargs['segment_size'], sampling_freq = kwargs['dig_sampling_freq'], dig_amplitude = kwargs['dig_amplitude'], 
                                                  read_channel = kwargs['read_channel'], coupling = kwargs['dig_coupling'], termination = kwargs['dig_termination'], 
                                                  pretrig_size = kwargs['pretrig_size'], dig_timeout = kwargs['dig_timeout'], runs = kwargs['runs'])
            
            # configure signal generator for NV drive
            self.set_srs396(sig_gen_freq, kwargs['rf_power'])
                        
            volt_factor = self.volt_factor(kwargs['detector']) # if using PMT, make negative voltages positive  

            try:
                hdawg.set_sequence(**{'seq': 'CW ODMR',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'probe_length': kwargs['probe'], 
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freqs': mod_freqs, 
                                    'iq_phases': iq_phases,
                                    'num_pts': kwargs['num_pts']})
            except Exception as e:
                print(e)
            
            # run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('cw')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer() # start digitizer (enable trigger)  
                    ps.start_now() # start pulse sequence

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        odmr_result_raw = self.dig.acquire() # acquire data from digitizer
                        odmr_result = np.mean(odmr_result_raw,axis=1)
                        
                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(odmr_result, 'CW ODMR', kwargs['num_pts'])
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([real_freqs/1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([real_freqs/1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # produce live fitting if requested
                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                        # save the current data to the data server
                        cw_odmr_data.push({'params': {'kwargs': kwargs},
                                            'title': 'CW Optically Detected Magnetic Resonance',
                                            'xlabel': 'Frequency (GHz)',
                                            'ylabel': 'Signal',
                                            'datasets': {'signal' : signal_sweeps, 'background': background_sweeps,
                                                        'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            self.equipment_off(kwargs['detector'])
                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                        
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

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
            
            num_angles = (kwargs['stop_angle'] - kwargs['start_angle']) * 2 + 1
            azi_angles = np.linspace(kwargs['start_angle'], kwargs['stop_angle'], num_angles)

            mgr.thor_polar.set_vel_params(3,7) # reset Thorlabs stages to default acceleration and velocity parameters
            mgr.thor_azi.set_vel_params(3,7)

            mgr.thor_polar.move(kwargs['start_angle'], True)
            mgr.thor_polar.update_positions_callback() # update position
            pos = mgr.thor_polar.current_position # set pos to be position as move is beginning
            
            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 

            max_sideband_freq = 2*kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
            mod_freqs = np.flip(mod_freqs)
            
            # np.random.shuffle(mod_freqs)
            
            real_freqs = sig_gen_freq - mod_freqs
            
            # define pulse sequence
            sequence = ps.CW_ODMR(kwargs['num_pts']) # pulse streamer sequence for CW ODMR
            ps.probe_time = kwargs['probe'] * 1e9

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector']) # if using PMT, make negative voltages positive    

            try:
                hdawg.set_sequence(**{'seq': 'CW ODMR',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'probe_length': kwargs['probe'], 
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freqs': mod_freqs, 
                                    'iq_phases': iq_phases,
                                    'num_pts': kwargs['num_pts']}) 
            except Exception as e:
                print(e)
            
            # run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()
                angle_fits = []

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                mgr.thor_polar.set_vel_params(3,7) # TODO: speed up Thorlabs stage for quicker scan
                mgr.thor_azi.set_vel_params(3,7)

                # configure laser settings and turn on
                # laser.set_modulation_state('cw')
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                self.dig.config()
                self.dig.start_buffer()
                
                # start pulse sequence
                ps.start_now()

                # start experiment loop
                for i in range(num_angles):
                    mgr.thor_polar.move(azi_angles[i], True)
                    mgr.thor_polar.update_positions_callback() # update position
                    pos = mgr.thor_polar.current_position # set pos to be position as move is beginning
                
                    odmr_result_raw = self.dig.acquire() # acquire data from digitizer

                    # average all data over each trigger/segment 
                    odmr_result=np.mean(odmr_result_raw,axis=1)
                    segments=(np.shape(odmr_result))[0]

                    # partition buffer into signal and background datasets
                    try:
                        sig, bg = volt_factor*self.analog_math(odmr_result, 'CW ODMR', kwargs['num_pts'])
                    except ValueError:
                        continue
                    
                    # notify the streaminglist that this entry has updated so it will be pushed to the data server
                    signal_sweeps.append(np.stack([real_freqs/1e9, sig]))
                    signal_sweeps.updated_item(-1) 
                    background_sweeps.append(np.stack([real_freqs/1e9, bg]))
                    background_sweeps.updated_item(-1)

                    # Fit the data
                    initial_guess = [0.1, kwargs['center_freq']/1e9, 5]  # [A, x0, gamma]
                    params, params_covariance = curve_fit(self.negative_lorentzian, real_freqs/1e9, sig/bg, p0=initial_guess)

                    # Extract fitted parameters
                    freq_fit = params[1]
                    angle_fits.append(np.stack([azi_angles[i], freq_fit]))
                    angle_fits.updated_item(-1) 

                    # save the current data to the data server
                    cw_odmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'CW Optically Detected Magnetic Resonance',
                                        'xlabel': 'Frequency (GHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}})

                    # update GUI progress bar                        
                    percent_completed = str(int(((i+1)/num_angles)*100))
                    self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                # save data if requested upon completion of experiment
                if kwargs['save'] == True:
                    self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
            
            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

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
                print(f"Minimum freq_fit: {min_freq_fit}, Corresponding azi_angle: {min_azi_angle}")

    def rabi_scan(self, **kwargs):
        """
        Run a Rabi sweep over a set of microwave pulse durations.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as rabi_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg

            # define parameter array that will be swept over in experiment & shuffle
            mw_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'], kwargs['pulse_axis'])

            # default fit parameters
            fit_value = None
            fit_x = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
            fit_y = np.ones(len(fit_x))

            # define pulse sequence
            sequence = ps.Rabi(kwargs['laser_init']*1e9, mw_times, kwargs['laser_readout']*1e9) # pulse streamer sequence

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector']) # if using PMT, make negative voltages positive    

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'Rabi',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pi_pulses': mw_times/1e9, 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs']})  
            except Exception as e:
                _logger.info("HDAWG disconnected. Restart in Instrument Server with 'restart awg'.")
                exception_type = type(e).__name__
                self.queue_from_exp.put_nowait([0, 'failed', None, e])
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                    
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        rabi_result_raw = self.dig.acquire() # acquire data from digitizer
                        # average all data over each trigger/segment 
                        # rabi_result_raw = rabi_result_raw[:,50:]
                        rabi_result=np.mean(rabi_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(rabi_result, 'Rabi', kwargs['num_pts'])
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([mw_times, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([mw_times, bg]))
                        background_sweeps.updated_item(-1)

                        # produce live fitting if requested
                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                        # save the current data to the data server
                        rabi_data.push({'params': {'kwargs': kwargs},
                                        'title': 'Rabi Oscillation',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps,
                                                    'x_fit': fit_x,
                                                    'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            self.equipment_off(kwargs['detector'])
                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                            
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                    
                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed
                
    def pulsed_odmr_scan(self, **kwargs):
        """Run a Pulsed ODMR sweep over a set of microwave frequencies.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as pulsed_odmr_data:
            # devices
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 
            max_sideband_freq = 2*kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
            mod_freqs = np.flip(mod_freqs)      
            real_freqs = sig_gen_freq - mod_freqs

            # default fit parameters
            fit_value = None
            fit_x = real_freqs/1e9
            fit_y = np.ones(len(fit_x))

            # define pulse sequence
            sequence = ps.Pulsed_ODMR(kwargs['laser_init']*1e9, kwargs['num_pts'], kwargs['pi']*1e9, kwargs['laser_readout']*1e9) # pulse streamer sequence

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector']) # if using PMT, make negative voltages positive

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'Pulsed ODMR',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freqs': mod_freqs, 
                                    'iq_phases': iq_phases,
                                    'pi_pulse': kwargs['pi'], 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs']})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        pulsed_odmr_result_raw = self.dig.acquire() # acquire data from digitizer
                        # average all data over each trigger/segment 
                        pulsed_odmr_result = np.mean(pulsed_odmr_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(pulsed_odmr_result, 'Pulsed ODMR', kwargs['num_pts'])
                        except ValueError:
                            continue

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([real_freqs/1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([real_freqs/1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # produce live fitting if requested
                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                        # save the current data to the data server
                        pulsed_odmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'Pulsed Optically Detected Magnetic Resonance',
                                        'xlabel': 'Frequency (GHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps, 'background': background_sweeps,
                                                    'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])

                            # produce live fitting if requested
                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                        
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    
                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def pulsed_odmr_rf_scan(self, **kwargs):
        """
        Run a Pulsed ODMR sweep over a set of microwave frequencies with coil RF tone for coil B field calibration.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as pulsed_odmr_rf_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define NV drive frequency & sideband           
            delta = 0
            iq_phases = [delta+0, delta+90] # set IQ phase relations for lower sideband [lower I, lower Q]
                
            sig_gen_freq = kwargs['center_freq'] + kwargs['half_span_sideband_freq'] # set freq to sig gen 

            max_sideband_freq = 2*kwargs['half_span_sideband_freq'] # set span of ODMR sweep as max sideband modulation frequency --> 100 MHz max. for SG396 IQ bandwidth

            # define parameter array that will be swept over in experiment & shuffle
            mod_freqs = np.linspace(0, max_sideband_freq, kwargs['num_pts'])            
            mod_freqs = np.flip(mod_freqs)
                  
            real_freqs = sig_gen_freq - mod_freqs

            # default fit parameters
            fit_value = None
            fit_x = real_freqs/1e9
            fit_y = np.ones(len(fit_x))
            fitted_diff = 0

            pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer
            rf_period = 1/kwargs['rf_pulse_freq']*1e9 # rf pulse period [ns] units for pulse streamer

            # define pulse sequence
            sequence = ps.Pulsed_ODMR_RF(kwargs['laser_init']*1e9, kwargs['num_pts'], pi_pulse, rf_period, kwargs['laser_readout']*1e9) # pulse streamer sequence

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector']) 

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'Pulsed ODMR RF',
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
                                    'iters': kwargs['iters']})

            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                rf_signal_sweeps = StreamingList()
                rf_background_sweeps = StreamingList()
                no_rf_signal_sweeps = StreamingList()
                no_rf_background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        pulsed_odmr_result_raw = self.dig.acquire() # acquire data from digitizer                
                        # average all data over each trigger/segment 
                        pulsed_odmr_result = np.mean(pulsed_odmr_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            rf_sig, rf_bg, no_rf_sig, no_rf_bg = volt_factor*self.analog_math(pulsed_odmr_result, 'DEER', kwargs['num_pts']) # same math logic as DEER sequence
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        rf_signal_sweeps.append(np.stack([real_freqs/1e9, rf_sig]))
                        rf_signal_sweeps.updated_item(-1) 
                        rf_background_sweeps.append(np.stack([real_freqs/1e9, rf_bg]))
                        rf_background_sweeps.updated_item(-1)
                        no_rf_signal_sweeps.append(np.stack([real_freqs/1e9, no_rf_sig]))
                        no_rf_signal_sweeps.updated_item(-1) 
                        no_rf_background_sweeps.append(np.stack([real_freqs/1e9, no_rf_bg]))
                        no_rf_background_sweeps.updated_item(-1)

                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_data('odmr', rf_signal_sweeps, rf_background_sweeps, 0.005, 1, 0.006, 1)
                                    fit_no_rf_value, fit_no_rf_x, fit_no_rf_y = self.fit_data('odmr', no_rf_signal_sweeps, no_rf_background_sweeps, 0.005, 1, 0.006, 1)
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                                else:
                                    fitted_diff = fit_no_rf_value - fit_value

                        # save the current data to the data server
                        pulsed_odmr_rf_data.push({'params': {'kwargs': kwargs},
                                        'title': 'Pulsed Optically Detected Magnetic Resonance',
                                        'xlabel': 'Frequency (GHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'rf_signal' : rf_signal_sweeps, 'rf_background': rf_background_sweeps,
                                                    'signal' : no_rf_signal_sweeps, 'background': no_rf_background_sweeps,
                                                    'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])

                            # produce live fitting if requested
                            # if kwargs['fit'] == True:
                            #     with warnings.catch_warnings():
                            #         warnings.simplefilter("error", OptimizeWarning)
                            #         try:
                            #             fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], rf_signal_sweeps, rf_background_sweeps, *kwargs['fit_params'])
                            #         except (RuntimeError, OptimizeWarning) as e:
                            #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                        
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    
                    # if kwargs['fit'] == True:
                    #     with warnings.catch_warnings():
                    #         warnings.simplefilter("error", OptimizeWarning)
                    #         try:
                    #             fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], rf_signal_sweeps, rf_background_sweeps, *kwargs['fit_params'])
                    #         except (RuntimeError, OptimizeWarning) as e:
                    #             _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed
                print(f"Fitted resonances = {fit_value} GHz (RF), {fit_no_rf_value} GHz (no RF)")
                print(f"Fitted difference = {round(fitted_diff*1000,4)} MHz")
                print(f"Coil B field = {round(fitted_diff*1000/2.8,4)} G")

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
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
                                                  segment_size = kwargs['segment_size'], sampling_freq = kwargs['dig_sampling_freq'], dig_amplitude = kwargs['dig_amplitude'], 
                                                  read_channel = kwargs['read_channel'], coupling = kwargs['dig_coupling'], termination = kwargs['dig_termination'], 
                                                  pretrig_size = kwargs['pretrig_size'], dig_timeout = kwargs['dig_timeout'], runs = kwargs['runs'])
            
            sequence = ps.Optical_T1(tau_times, kwargs['laser_readout']*1e9)

            # configure devices used in scan
            laser.set_modulation_state('pulsed')
            laser.set_analog_control_mode('current')
            laser.set_diode_current_realtime(kwargs['laser_power'])
            laser.laser_on()

            volt_factor = self.volt_factor(kwargs['detector'])

            daq.open_ai_task(kwargs['detector'], len(t1_buffer[0]))

            self.dig.assign_param(dig_config)

            # for storing the experiment data
            # list of numpy arrays of shape (2, num_points)
            signal_sweeps = StreamingList()
            background_sweeps = StreamingList()

            for i in range(kwargs['iters']):
                
                self.dig.config()
                self.dig.start_buffer()
                ps.stream(sequence, kwargs['runs']) # execute chosen sequence on Pulse Streamer

                t1_result_raw = self.dig.acquire()
                
                t1_result = np.mean(t1_result_raw, axis=1)
                
                # partition buffer into signal and background datasets
                try:
                    sig, bg = volt_factor*self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])
                except ValueError:
                    continue
                
                # notify the streaminglist that this entry has updated so it will be pushed to the data server                    
                signal_sweeps.append(np.stack([tau_times/1e6, sig]))
                signal_sweeps.updated_item(-1) 
                background_sweeps.append(np.stack([tau_times/1e6, bg]))
                background_sweeps.updated_item(-1)

                # save the current data to the data server.
                t1_data.push({'params': {'kwargs': kwargs},
                                'title': 'Optical T1 Relaxation',
                                'xlabel': 'Free Precession Interval (ms)',
                                'ylabel': 'Signal',
                                'datasets': {'signal' : signal_sweeps,
                                            'background': background_sweeps}
                })

                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                    # the GUI has asked us nicely to exit
                    if kwargs['save'] == True:
                        flexSave(kwargs['dataset'], kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    
                    self.equipment_off(kwargs['detector'])

                    return
                
                percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                    # the GUI has asked us nicely to exit. Save data if requested.
                    # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                    self.equipment_off(kwargs['detector'])
                    self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    return
                    
            if kwargs['save'] == True:
                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

            self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            self.equipment_off(kwargs['detector'])

    def MW_T1_scan(self, **kwargs):
        """
        Run a T1 sweep with MW over a set of precession time intervals.

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

            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # default fit parameters
            fit_value = None
            fit_x = tau_times[1:]/1e6
            fit_y = np.ones(len(fit_x))

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'], kwargs['pulse_axis'])

            pi_pulse = kwargs['pi']*1e9 # [ns] units for pulse streamer

            # define pulse sequence
            sequence = ps.Diff_T1(kwargs['laser_init']*1e9, tau_times, kwargs['pulse_axis'], pi_pulse, kwargs['laser_readout']*1e9)

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'T1',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': kwargs['sideband_freq'], 
                                    'iq_phases': iq_phases,
                                    'pi_pulse': pi_pulse/1e9, 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})  
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        t1_result_raw = self.dig.acquire() # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        t1_result = np.mean(t1_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(t1_result, 'MW_T1', kwargs['num_pts'])
                        except ValueError:
                            continue
                    
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([tau_times[1:]/1e6, sig[1:]]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([tau_times[1:]/1e6, bg[1:]]))
                        background_sweeps.updated_item(-1)

                        print(f"signal_sweeps size: {np.shape(signal_sweeps)}")

                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                    print(f"fitted T1: {fit_value}")
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                
                        # save the current data to the data server
                        t1_data.push({'params': {'kwargs': kwargs},
                                        'title': 'MW T1 Relaxation',
                                        'xlabel': 'Free Precession Interval (ms)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps, 'background': background_sweeps, 
                                                    'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])

                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])

                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                    
                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def T2_scan(self, **kwargs):
        """
        Run a T2 sweep over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as t2_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # default fit parameters
            fit_value = None
            fit_x = tau_times[1:]
            fit_y = np.ones(len(fit_x))

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'])

            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            # define pulse sequence
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

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'T2',
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
                                    'iters': kwargs['iters']}) 
            except Exception as e:
                print(f"AWG ERROR: {e}")
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        t2_result_raw = self.dig.acquire() # acquire data from digitizer
                        # t2_result_raw = t2_result_raw[:,50:]
                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        t2_result = np.mean(t2_result_raw, axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(t2_result, 'T2', kwargs['num_pts'])
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([tau_times[1:]/1e3, sig[1:]]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([tau_times[1:]/1e3, bg[1:]]))
                        background_sweeps.updated_item(-1)

                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                        # save the current data to the data server
                        t2_data.push({'params': {'kwargs': kwargs},
                                        'title': 'T2 Relaxation',
                                        'xlabel': 'Free Precession Interval (\u03BCs)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps, 'background': background_sweeps, 
                                                    'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])
                        
                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            self.equipment_off(kwargs['detector'])
                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                    
                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed
    
    def T2_rf_scan(self, **kwargs):
        """
        Run a T2 sweep over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as t2_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # default fit parameters
            fit_value = None
            fit_x = tau_times[1:]
            fit_y = np.ones(len(fit_x))

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq'])

            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)

            # define pulse sequence
            
            # sequence = ps.XY8_N_RF(tau_times, 'xy', 
            #                     pi_half[0], pi_half[1], 
            #                     pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
            # x_tau_times = 8*tau_times + 4*pi[0] + 4*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
            # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 4*pi[0] + \
            #             4*pi[1] + 7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']

            sequence = ps.XY8_N_RF(kwargs['laser_init']*1e9, tau_times, 'yy', 
                                pi_half[0], pi_half[1], 
                                pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
            x_tau_times = 8*tau_times + 8*pi[1] # for definition of (tau/2 - pi - tau - pi - tau - pi - tau - pi - tau/2)*n
            # x_tau_times = 2*pi_half[0] + ((x_tau_times/2)/(8*kwargs['n']) + 8*pi[1] + \
            #         7*x_tau_times/(8*kwargs['n']) + (x_tau_times/2)/(8*kwargs['n']))*kwargs['n']

            rf_times = 500 + 2*pi_half[1] + (x_tau_times)*kwargs['n'] + 100

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'T2 RF',
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
                                    'iters': kwargs['iters']}) 
            except Exception as e:
                print(f"AWG ERROR: {e}")
                # print(f"DEVICES: {InstrumentServer._devs}")
                # # InstrumentServer.restart(self, name="awg")
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        t2_result_raw = self.dig.acquire() # acquire data from digitizer

                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        t2_result = np.mean(t2_result_raw, axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(t2_result, 'T2', kwargs['num_pts'])
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([tau_times[1:]/1e3, sig[1:]]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([tau_times[1:]/1e3, bg[1:]]))
                        background_sweeps.updated_item(-1)

                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                        # print(f"type signal_sweeps = {type(signal_sweeps)}")
                        # print(f"type sig: {type(sig)}")

                        # save the current data to the data server
                        t2_data.push({'params': {'kwargs': kwargs},
                                        'title': 'T2 Relaxation',
                                        'xlabel': 'Free Precession Interval (\u03BCs)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps, 'background': background_sweeps, 
                                                    'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])
                        
                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            self.equipment_off(kwargs['detector'])
                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_data(kwargs['dataset'], signal_sweeps, background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                    
                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed
    
    def DQ_scan(self, **kwargs):
        """
        Run a DQ sweep over a set of precession time intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as dq_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            if kwargs['pulse_axis'] == 'y':
                delta = 90
            else:
                delta = 0
            
            iq_phases = [delta+0, delta+90, delta+90, delta+0] # set IQ phase relations for upper and lower sidebands
                
            sig_gen_freq = (kwargs['freq_minus'] + kwargs['freq_plus'])/2 # set mean value freq to sig gen 

            sideband_freq = sig_gen_freq - kwargs['freq_minus'] # set sideband freq to match the inputted values

            pi_pulse_minus = kwargs['pi_minus']*1e9 # [ns] units for pulse streamer
            pi_pulse_plus = kwargs['pi_plus']*1e9 # [ns] units for pulse streamer

            # define pulse sequence
            sequence = ps.DQ(kwargs['laser_init']*1e9, tau_times, kwargs['pulse_axis'], pi_pulse_minus, pi_pulse_plus, kwargs['laser_readout']*1e9)

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = 2*kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'DQ',
                                    'i_offset': kwargs['i_offset'],
                                    'q_offset': kwargs['q_offset'],
                                    'sideband_power': kwargs['sideband_power'],
                                    'sideband_freq': sideband_freq, 
                                    'iq_phases': iq_phases,
                                    'pi_minus1': pi_pulse_minus/1e9, 
                                    'pi_plus1': pi_pulse_plus/1e9, 
                                    'num_pts': kwargs['num_pts'],
                                    'runs': kwargs['runs'], 
                                    'iters': kwargs['iters']})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                s00_sweeps = StreamingList()
                s0m_sweeps = StreamingList()
                smm_sweeps = StreamingList()
                smp_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        dq_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        dq_result = np.mean(dq_result_raw,axis=1)

                        # partition buffer into signal and background datasets FIXME: maybe need to use DEER option for 4 pts in analog math
                        try:
                            s00, s0m, smm, smp = volt_factor*self.analog_math(dq_result, 'DQ', kwargs['num_pts']) # data for S0,0, S0,-1, S-1,-1, and S-1,+1 sequence
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        s00_sweeps.append(np.stack([tau_times/1e3, s00]))
                        s00_sweeps.updated_item(-1) 
                        s0m_sweeps.append(np.stack([tau_times/1e3, s0m]))
                        s0m_sweeps.updated_item(-1)

                        smm_sweeps.append(np.stack([tau_times/1e3, smm]))
                        smm_sweeps.updated_item(-1) 
                        smp_sweeps.append(np.stack([tau_times/1e3, smp]))
                        smp_sweeps.updated_item(-1)

                        # save the current data to the data server
                        dq_data.push({'params': {'kwargs': kwargs},
                                        'title': 'T1 Relaxation',
                                        'xlabel': 'Free Precession Interval (\u03BCs)',
                                        'ylabel': 'Signal',
                                        'datasets': {'S0,0' : s00_sweeps,
                                                    'S0,-1': s0m_sweeps,
                                                    'S-1,-1' : smm_sweeps,
                                                    'S-1,+1': smp_sweeps}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                        
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed
    
    def DEER_scan(self, **kwargs):
        """
        Run a DEER sweep over a set of MW frequencies.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # default fit parameters
            fit_value = None
            fit_x = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])/1e6   
            fit_y = np.ones(len(fit_x))

            # define parameter array that will be swept over in experiment & shuffle
            frequencies = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])            

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            if kwargs['drive_type'] == 'Continuous':
                sequence = ps.DEER_CD(kwargs['laser_init']*1e9, pi_half[0], pi_half[1], 
                                    pi[0], pi[1], 
                                    kwargs['tau']*1e9, kwargs['num_pts'], kwargs['laser_readout']*1e9) # send to PS in [ns] units
                dark_pulse = kwargs['pi']/2 + kwargs['tau'] + kwargs['pi'] + kwargs['tau'] + kwargs['pi']/2 # send to AWG in [s] units 
            else:
                sequence = ps.DEER(kwargs['laser_init']*1e9, pi_half[0], pi_half[1], 
                                    pi[0], pi[1], 
                                    kwargs['tau']*1e9, kwargs['num_pts'], kwargs['laser_readout']*1e9)
                dark_pulse = dark_pi

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = 2*kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                if kwargs['drive_type'] == 'Continuous':
                    hdawg.set_sequence(**{'seq': 'DEER CD',
                                        'i_offset': kwargs['i_offset'],
                                        'q_offset': kwargs['q_offset'],
                                        'sideband_power': kwargs['sideband_power'],
                                        'sideband_freq': kwargs['sideband_freq'], 
                                        'iq_phases': iq_phases,
                                        'pihalf_x': pi_half[0]/1e9,
                                        'pihalf_y': pi_half[1]/1e9,
                                        'pi_x': pi[0]/1e9, 
                                        'pi_y': pi[1]/1e9,
                                        'pi_pulse': dark_pulse, 
                                        'mw_power': kwargs['awg_power'], 
                                        'num_pts': kwargs['num_pts'],
                                        'runs': kwargs['runs'], 
                                        'iters': kwargs['iters'],
                                        'freqs': frequencies})
                else:
                    hdawg.set_sequence(**{'seq': 'DEER',
                                        'i_offset': kwargs['i_offset'],
                                        'q_offset': kwargs['q_offset'],
                                        'sideband_power': kwargs['sideband_power'],
                                        'sideband_freq': kwargs['sideband_freq'], 
                                        'iq_phases': iq_phases,
                                        'pihalf_x': pi_half[0]/1e9,
                                        'pihalf_y': pi_half[1]/1e9,
                                        'pi_x': pi[0]/1e9, 
                                        'pi_y': pi[1]/1e9,
                                        'pi_pulse': dark_pulse, 
                                        'mw_power': kwargs['awg_power'], 
                                        'num_pts': kwargs['num_pts'],
                                        'runs': kwargs['runs'], 
                                        'iters': kwargs['iters'],
                                        'freqs': frequencies})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    time_start = time.time()
                    # start experiment loop
                    for i in range(kwargs['iters']):
                        deer_result_raw = self.dig.acquire() # acquire data from digitizer
                        # print(np.shape(deer_result_raw))
                        # deer_result_raw = deer_result_raw[:,50:]
                        # define dummy array to contain experiment data --> size (runs*num_pts) --> (segment_size)
                        deer_result = np.mean(deer_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            dark_sig, dark_bg, echo_sig, echo_bg = volt_factor*self.analog_math(deer_result, 'DEER', kwargs['num_pts'])
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([frequencies/1e6, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([frequencies/1e6, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([frequencies/1e6, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([frequencies/1e6, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                        # save the current data to the data server
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER',
                                        'xlabel': 'Surface Electron Resonance (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps, 'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps, 'echo_background': echo_background_sweeps,
                                                    'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            self.equipment_off(kwargs['detector'])
                            print(f"Experiment stopped after: {round(time.time() - time_start,2)} seconds")   
                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return

                    print(f"Completed experiment time: {round(time.time() - time_start,2)} seconds")        
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                    
                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def DEER_rabi_scan(self, **kwargs):
        """
        Run a DEER Rabi sweep over a set of MW pulse durations.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # default fit parameters
            fit_value = None
            fit_x = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9   
            fit_y = np.ones(len(fit_x))

            # define parameter array that will be swept over in experiment & shuffle
            dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9            

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.DEER_Rabi(kwargs['laser_init']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1], 
                                kwargs['tau']*1e9, kwargs['num_pts'], kwargs['laser_readout']*1e9)

            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = 2*kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'DEER Rabi',     
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
                                    'pi_pulses': dark_taus/1e9})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        deer_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        deer_result=np.mean(deer_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            dark_sig, dark_bg, echo_sig, echo_bg = volt_factor*self.analog_math(deer_result, 'DEER', kwargs['num_pts'])
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([dark_taus, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([dark_taus, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([dark_taus, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([dark_taus, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        if kwargs['fit_live'] == True:
                            with warnings.catch_warnings():
                                warnings.simplefilter("error", OptimizeWarning)
                                try:
                                    fit_value, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                                except (RuntimeError, OptimizeWarning) as e:
                                    _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                        # save the current data to the data server
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Rabi',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps, 'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps, 'echo_background': echo_background_sweeps,
                                                    'x_fit': fit_x, 'y_fit': fit_y}})

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', fit_value])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            if kwargs['fit'] == True:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", OptimizeWarning)
                                    try:
                                        fit_value, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                                    except (RuntimeError, OptimizeWarning) as e:
                                        _logger.warning(f"For {kwargs['dataset']} measurement, {e}")

                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', fit_value])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    if kwargs['fit'] == True:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", OptimizeWarning)
                            try:
                                fit_value, fit_x, fit_y = self.fit_deer_data(kwargs['dataset'], dark_signal_sweeps, dark_background_sweeps, echo_signal_sweeps, echo_background_sweeps, *kwargs['fit_params'])
                            except (RuntimeError, OptimizeWarning) as e:
                                _logger.warning(f"For {kwargs['dataset']} measurement, {e}")
                    
                    self.queue_from_exp.put_nowait([percent_completed, 'complete', fit_value])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def DEER_FID_scan(self, **kwargs):
        """
        Run a DEER FID sweep over a set of MW pulse durations.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.DEER_FID(kwargs['laser_init']*1e9, tau_times, pi_half[0], pi_half[1], 
                                pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = 2*kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'DEER FID',
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
                                    'iters': kwargs['iters']})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        deer_result_raw = self.dig.acquire() # acquire data from digitizer
                        # average all data over each trigger/segment 
                        deer_result = np.mean(deer_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            dark_sig, dark_bg, echo_sig, echo_bg = volt_factor*self.analog_math(deer_result, 'DEER', kwargs['num_pts'])
                        except ValueError:
                            continue
                        
                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([tau_times, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([tau_times, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([tau_times, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([tau_times, echo_bg]))
                        echo_background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        deer_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER FID',
                                        'xlabel': 'Free Precession Interval (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def DEER_FID_CD_scan(self, **kwargs):
        """
        Run a continuous drive DEER FID sweep over a set of free precession intervals.

        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as cd_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    tau_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9
                case 'linspace':
                    tau_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.DEER_FID_CD(kwargs['laser_init']*1e9, tau_times, pi_half[0], pi_half[1], 
                                    pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
            # dark_pulses = kwargs['pi']/2 + tau_times/1e9 + (kwargs['pi'] + 2*tau_times/1e9)*(kwargs['n']-1) + kwargs['pi'] + tau_times/1e9 + kwargs['pi']/2 
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = 3*kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'DEER FID CD',                
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
                                    'iters': kwargs['iters']})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                dark_signal_sweeps = StreamingList()
                dark_background_sweeps = StreamingList()
                echo_signal_sweeps = StreamingList()
                echo_background_sweeps = StreamingList()
                cd_signal_sweeps = StreamingList()
                cd_background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        cd_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        cd_result=np.mean(cd_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            dark_sig, dark_bg, echo_sig, echo_bg, cd_sig, cd_bg = volt_factor*self.analog_math(cd_result, 'CD', kwargs['num_pts'])
                        except ValueError:
                            continue

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        dark_signal_sweeps.append(np.stack([tau_times, dark_sig]))
                        dark_signal_sweeps.updated_item(-1) 
                        dark_background_sweeps.append(np.stack([tau_times, dark_bg]))
                        dark_background_sweeps.updated_item(-1)
                        echo_signal_sweeps.append(np.stack([tau_times, echo_sig]))
                        echo_signal_sweeps.updated_item(-1) 
                        echo_background_sweeps.append(np.stack([tau_times, echo_bg]))
                        echo_background_sweeps.updated_item(-1)
                        cd_signal_sweeps.append(np.stack([tau_times, cd_sig]))
                        cd_signal_sweeps.updated_item(-1) 
                        cd_background_sweeps.append(np.stack([tau_times, cd_bg]))
                        cd_background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        cd_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER FID Continuous Drive',
                                        'xlabel': 'Free Precession Interval (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'dark_signal' : dark_signal_sweeps,
                                                    'dark_background': dark_background_sweeps,
                                                    'echo_signal' : echo_signal_sweeps,
                                                    'echo_background': echo_background_sweeps,
                                                    'cd_signal' : cd_signal_sweeps,
                                                    'cd_background': cd_background_sweeps,}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def DEER_corr_rabi_scan(self, **kwargs):
        """
        Run a DEER Correlation Rabi sweep over a set of MW pulses.
        
        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as corr_rabi_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            dark_taus = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])          

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.DEER_Corr_Rabi(kwargs['laser_init']*1e9, dark_taus*1e9, kwargs['tau']*1e9, kwargs['t_corr']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1], kwargs['laser_readout']*1e9) # send to PS in [ns] units
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'DEER Corr Rabi',                
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
                                    'pi_pulses': dark_taus})

            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        corr_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        corr_result=np.mean(corr_result_raw,axis=1)
                        segments=(np.shape(corr_result))[0]

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(corr_result, 'Corr', kwargs['num_pts'])
                        except ValueError:
                            continue

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([dark_taus*1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([dark_taus*1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        corr_rabi_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Correlation Rabi',
                                        'xlabel': 'MW Pulse Duration (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                    
                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def DEER_T1_scan(self, **kwargs):
        """
        Run a DEER Correlation T1 sweep over a set of correlation intervals.
        
        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_t1_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    t_corr_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     
                case 'linspace':
                    t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.Electron_T1(kwargs['laser_init']*1e9, t_corr_times*1e9, kwargs['tau']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1], dark_pi*1e9, kwargs['laser_readout']*1e9) # send to PS in [ns] units
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = 2*kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'DEER Corr T1',                  
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
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                with_pulse_py_sweeps = StreamingList()
                without_pulse_py_sweeps = StreamingList()
                with_pulse_ny_sweeps = StreamingList()
                without_pulse_ny_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        corr_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        corr_result = np.mean(corr_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            with_py, without_py, with_ny, without_ny = volt_factor*self.analog_math(corr_result, 'DEER', kwargs['num_pts'])
                        except ValueError:
                            continue

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        with_pulse_py_sweeps.append(np.stack([t_corr_times*1e6, with_py]))
                        with_pulse_py_sweeps.updated_item(-1) 
                        without_pulse_py_sweeps.append(np.stack([t_corr_times*1e6, without_py]))
                        without_pulse_py_sweeps.updated_item(-1)
                        with_pulse_ny_sweeps.append(np.stack([t_corr_times*1e6, with_ny]))
                        with_pulse_ny_sweeps.updated_item(-1) 
                        without_pulse_ny_sweeps.append(np.stack([t_corr_times*1e6, without_ny]))
                        without_pulse_ny_sweeps.updated_item(-1)

                        # save the current data to the data server
                        deer_t1_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER Correlation T1',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'with_py': with_pulse_py_sweeps,
                                                    'without_py': without_pulse_py_sweeps,
                                                    'with_ny': with_pulse_ny_sweeps,
                                                    'without_ny': without_pulse_ny_sweeps}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def DEER_T2_scan(self, **kwargs):
        """
        Run a DEER Correlation T1 sweep over a set of correlation intervals.
        
        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as deer_t2_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            match kwargs['array_type']:
                case 'geomspace':
                    t_times = np.geomspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     
                case 'linspace':
                    t_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts'])     

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for y pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []
            dark_pi_half = kwargs['dark_pi']/2
            dark_pi = kwargs['dark_pi']

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.Electron_T2(kwargs['laser_init']*1e9, t_times*1e9, kwargs['tau']*1e9, pi_half[0], pi_half[1], 
                                pi[0], pi[1], dark_pi_half*1e9, dark_pi*1e9, kwargs['laser_readout']*1e9, kwargs['deer_t2_buffer']*1e9) # send to PS in [ns] units
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'DEER T2',                  
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
                                    'iters': kwargs['iters']})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        deer_t2_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        deer_t2_result = np.mean(deer_t2_result_raw, axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(deer_t2_result, 'DEER T2', kwargs['num_pts'])
                        except ValueError:
                            continue

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([t_times*1e9, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([t_times*1e9, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        deer_t2_data.push({'params': {'kwargs': kwargs},
                                        'title': 'DEER T2',
                                        'xlabel': 'Free Precession Interval (ns)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal': signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def Corr_Spec_scan(self, **kwargs):
        """
        Run a Correlation Spectroscopy NMR sweep over a set of precession time intervals.
        
        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as nmr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.Corr_Spectroscopy_RF(kwargs['laser_init']*1e9, t_corr_times, kwargs['tau']*1e9, 
                                     pi_half[0], pi_half[1], 
                                     pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            corr_spec_time = pi_half[0] + (4*pi[0] + 4*pi[1] + 8*kwargs['tau']*1e9)*kwargs['n'] + pi_half[1] + kwargs['stop']*1e9 + \
                             pi_half[0] + (4*pi[0] + 4*pi[1] + 8*kwargs['tau']*1e9)*kwargs['n'] + pi_half[1]
            
            total_exp_time = 100 + kwargs['laser_init']*1e9 + 500 + corr_spec_time + 100 + kwargs['laser_readout'] + 100

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'NMR',
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
                                    'rf_phase': 0})
            except Exception as e:
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        nmr_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        nmr_result=np.mean(nmr_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])
                        except ValueError:
                            continue

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([t_corr_times/1e3, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([t_corr_times/1e3, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        nmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'NMR Time Domain Data',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def Corr_Spec_scan_orig(self, **kwargs):
        """
        Run a Correlation Spectroscopy NMR sweep over a set of precession time intervals.
        
        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as nmr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # define parameter array that will be swept over in experiment & shuffle
            t_corr_times = np.linspace(kwargs['start'], kwargs['stop'], kwargs['num_pts']) * 1e9

            # define NV drive frequency & sideband
            sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default

            # define pi pulses
            pi: List[float] = []
            pi_half: List[float] = []

            for i in range(2):
                pi_half.append(kwargs['pi']*1e9/2)
                pi.append(kwargs['pi']*1e9)
            
            # define pulse sequence
            sequence = ps.Corr_Spectroscopy(kwargs['laser_init']*1e9, t_corr_times, kwargs['tau']*1e9, 
                                     pi_half[0], pi_half[1], 
                                     pi[0], pi[1], kwargs['n'], kwargs['laser_readout']*1e9)
            
            # configure digitizer
            dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

            volt_factor = self.volt_factor(kwargs['detector'])

            # upload AWG sequence first
            try:
                hdawg.set_sequence(**{'seq': 'NMR',
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
                print(e)
            
            # if successfully uploaded, run the experiment
            else:
                # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                signal_sweeps = StreamingList()
                background_sweeps = StreamingList()

                # open laser shutter
                laser_shutter.open_shutter()

                # upload digitizer parameters
                self.dig.assign_param(dig_config)

                # emit MW for NV drive
                sig_gen.set_rf_toggle(1) # turn on NV signal generator

                # configure laser settings and turn on
                laser.set_modulation_state('pulsed')
                laser.set_analog_control_mode('current')
                laser.set_diode_current_realtime(kwargs['laser_power'])
                laser.laser_on()

                # set pulsestreamer to start on software trigger & run infinitely
                ps.set_soft_trigger()
                ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                
                # start digitizer --> waits for trigger from pulse sequence
                try:
                    self.dig.config()
                except Exception as e:
                    print(f"Digitizer exception: {e}")
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                else:
                    self.dig.start_buffer()
                
                    # start pulse sequence
                    ps.start_now()

                    # start experiment loop
                    for i in range(kwargs['iters']):
                        
                        nmr_result_raw = self.dig.acquire() # acquire data from digitizer

                        # average all data over each trigger/segment 
                        nmr_result=np.mean(nmr_result_raw,axis=1)

                        # partition buffer into signal and background datasets
                        try:
                            sig, bg = volt_factor*self.analog_math(nmr_result, 'NMR', kwargs['num_pts'])
                        except ValueError:
                            continue

                        # notify the streaminglist that this entry has updated so it will be pushed to the data server
                        signal_sweeps.append(np.stack([t_corr_times/1e3, sig]))
                        signal_sweeps.updated_item(-1) 
                        background_sweeps.append(np.stack([t_corr_times/1e3, bg]))
                        background_sweeps.updated_item(-1)

                        # save the current data to the data server
                        nmr_data.push({'params': {'kwargs': kwargs},
                                        'title': 'NMR Time Domain Data',
                                        'xlabel': 'Free Precession Interval (\u03BCs) or Frequency (MHz)',
                                        'ylabel': 'Signal',
                                        'datasets': {'signal' : signal_sweeps,
                                                    'background': background_sweeps}
                        })

                        # update GUI progress bar                        
                        percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                        self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                        if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                            # the GUI has asked us nicely to exit. Save data if requested.
                            # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                            self.equipment_off(kwargs['detector'])
                            self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                            return
                            
                    # save data if requested upon completion of experiment
                    if kwargs['save'] == True:
                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                    self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

            finally:
                self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed

    def CASR_scan(self, **kwargs):
        """
        Run a Coherently Averaged Synchronized Readout NMR sweep over a range of frequencies.
        
        Keyword args:
            dataset: name of the dataset to push data to
            start (float): start frequency
            stop (float): stop frequency
            num_pts (int): number of points between start-stop (inclusive)
            iterations: number of times to repeat the experiment
        """
        # connect to the instrument server & the data server.
        # create a data set, or connect to an existing one with the same name if it was created earlier.
        with InstrumentManager() as mgr, DataSource(kwargs['dataset']) as casr_data:
            # load devices used in scan
            laser = mgr.laser
            laser_shutter = mgr.laser_shutter
            sig_gen = mgr.sg
            ps = mgr.ps
            hdawg = mgr.awg
            
            # period = (1/kwargs['central_freq'])*1e9 # reference for ensuring sequence is a multiple of this central period [ns]
            
            # intervals in pulse sequence used to define time points on x-axis in seconds
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

            t_seq = laser_init_time + singlet_decay + dd_time + mw_buffer_time + laser_read_time + wait_time
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
                    exception_type = type(e).__name__
                    self.queue_from_exp.put_nowait([0, 'failed', None, e])

                else:
                    # x-axis time values array for experiment [s]
                    times = np.linspace(t_seq - wait_time - laser_read_time/2, kwargs['num_pts']*t_seq, kwargs['num_pts']) * 1e-9

                    # define NV drive frequency & sideband
                    sig_gen_freq, iq_phases = self.choose_sideband(kwargs['sideband'], kwargs['freq'], kwargs['sideband_freq']) # iq_phases for x pulse by default

                    # define pulse sequence
                    sequence = ps.CASR(kwargs['rf_pi_half']*1e9, laser_init_time, singlet_decay, 
                                    pi_half[0], pi_half[1], pi[0], pi[1], 
                                    tau, kwargs['n'], mw_buffer_time, kwargs['laser_readout'], wait_time, kwargs['num_pts'])
                    # sequence = ps.CASR_RF(laser_init_time, singlet_decay, 
                    #                 pi_half[0], pi_half[1], pi[0], pi[1], 
                    #                 tau, kwargs['n'], mw_buffer_time, kwargs['laser_readout'], wait_time, kwargs['num_pts'])
                    
                    # configure digitizer
                    dig_config = self.digitizer_configure(num_pts_in_exp = kwargs['num_pts'], iters = kwargs['iters'], 
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

                    volt_factor = self.volt_factor(kwargs['detector'])

                    # upload AWG sequence first
                    try:
                        hdawg.set_sequence(**{'seq': 'CASR',
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
                                            'rf_pihalf': kwargs['rf_pi_half']})
                                            # 'rf_pihalf': kwargs['num_pts']*t_seq*1e-9})
                    except Exception as e:
                        print(e)
                    
                    # if successfully uploaded, run the experiment
                    else:
                        # for storing the experiment data --> list of numpy arrays of shape (2, num_points)
                        signal_sweeps = StreamingList()
                        background_sweeps = StreamingList()

                        # open laser shutter
                        laser_shutter.open_shutter()
                        time.sleep(0.1)

                        # upload digitizer parameters
                        self.dig.assign_param(dig_config)

                        # emit MW for NV drive
                        sig_gen.set_rf_toggle(1) # turn on NV signal generator

                        # configure laser settings and turn on
                        laser.set_modulation_state('pulsed')
                        laser.set_analog_control_mode('current')
                        laser.set_diode_current_realtime(kwargs['laser_power'])
                        laser.laser_on()

                        # set pulsestreamer to start on software trigger & run infinitely
                        ps.set_soft_trigger()
                        ps.stream(sequence, PulseStreamer.REPEAT_INFINITELY) #kwargs['runs']*kwargs['iters']) # execute chosen sequence on Pulse Streamer
                        
                        # start digitizer --> waits for trigger from pulse sequence
                        try:
                            self.dig.config()
                        except Exception as e:
                            print(f"Digitizer exception: {e}")
                            exception_type = type(e).__name__
                            self.queue_from_exp.put_nowait([0, 'failed', None, exception_type])
                        else:
                            self.dig.start_buffer()
                        
                            # start pulse sequence
                            ps.start_now()

                            # start experiment loop
                            for i in range(kwargs['iters']):
                                
                                casr_result_raw = self.dig.acquire() # acquire data from digitizer

                                # average all data over each trigger/segment 
                                casr_result=np.mean(casr_result_raw,axis=1)

                                # partition buffer into signal and background datasets
                                try:
                                    sig, bg = volt_factor*self.analog_math(casr_result, 'CASR', kwargs['num_pts'])
                                except ValueError:
                                    continue

                                # notify the streaminglist that this entry has updated so it will be pushed to the data server
                                signal_sweeps.append(np.stack([times*1e3, sig]))
                                signal_sweeps.updated_item(-1) 
                                background_sweeps.append(np.stack([times*1e3, bg]))
                                background_sweeps.updated_item(-1)

                                # save the current data to the data server
                                casr_data.push({'params': {'kwargs': kwargs},
                                                'title': 'CASR Time Domain Data',
                                                'xlabel': 'Free Precession Interval (ms) or Frequency (kHz)',
                                                'ylabel': 'Signal',
                                                'datasets': {'signal' : signal_sweeps,
                                                            'background': background_sweeps}
                                })

                                # update GUI progress bar                        
                                percent_completed = str(int(((i+1)/kwargs['iters'])*100))
                                self.queue_from_exp.put_nowait([percent_completed, 'in progress', None])

                                if experiment_widget_process_queue(self.queue_to_exp) == 'stop':
                                    # the GUI has asked us nicely to exit. Save data if requested.
                                    # print(f"is there a queue to exp? {self.queue_to_exp.get()}")
                                    self.equipment_off(kwargs['detector'])
                                    self.queue_from_exp.put_nowait([percent_completed, 'stopped', None])
                                    if kwargs['save'] == True:
                                        self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])
                                    return
                                    
                            # save data if requested upon completion of experiment
                            if kwargs['save'] == True:
                                self.run_save(kwargs['dataset'], kwargs['filename'], [kwargs['directory']])

                            self.queue_from_exp.put_nowait([percent_completed, 'complete', None])

                    finally:
                        self.equipment_off(kwargs['detector']) # turn off equipment regardless of if experiment started or failed 

    

    """ Data Fitting """

    def negative_lorentzian(self, x, A, x0, gamma, c):
        return -A / (1 + ((x - x0) / gamma) ** 2) + c
    
    def positive_lorentzian(self, x, A, x0, gamma, c):
        return A / (1 + ((x - x0) / gamma) ** 2) + c
    
    def decaying_cosine(self, x, A, gamma, T, phi, c):
        return A * np.exp(-gamma * x) * np.cos(2 * np.pi * x / T + phi) + c
    
    def stretched_exponential(self, x, A, T, n, c):
        return A*np.exp(-(x/T)**n) + c

    def mod_stretched_exponential(self, x, A, T, n, a1, f1, phi1, a2, f2, phi2):
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
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                fitted_value = round(x_fit[np.argmin(y_fit)],4)
            case 'rabi':
                params, covariance = curve_fit(self.decaying_cosine, x_values, y_values, p0=initial_guess)
                y_fit = self.decaying_cosine(x_fit, *params)
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                fitted_value = round(x_fit[np.argmin(y_fit)],2)
            case 't1':
                params, covariance = curve_fit(self.stretched_exponential, x_values, y_values, p0=initial_guess)
                y_fit = self.stretched_exponential(x_fit, *params)
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                fitted_value = round(params[1],3)
            case 't2':
                params, covariance = curve_fit(self.mod_stretched_exponential, x_values, y_values, p0=initial_guess)
                y_fit = self.mod_stretched_exponential(x_fit, *params)
                # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
                fitted_value = round(params[1],3)

        return fitted_value, x_fit, y_fit
    
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
                fitted_value = round(x_fit[np.argmin(y_fit)],3)
            case 'deer rabi':
                params, covariance = curve_fit(self.decaying_cosine, x_values, y_values, p0=initial_guess)
                y_fit = self.decaying_cosine(x_fit, *params)
                fitted_value = round(x_fit[np.argmin(y_fit)],2)
        # # Extract fitted parameters
        # A_fit, gamma_fit, f_fit, phi_fit, C_fit = params

        # print(f"Fitted pi pulse = {fitted_value} ns")
        return fitted_value, x_fit, y_fit


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
