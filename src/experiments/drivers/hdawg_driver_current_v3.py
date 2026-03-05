"""
Zurich Instruments HDAWG driver for integration into Nspyre

Author: Evan Villafranca 
Created: 2023-3-29
"""

# import zhinst.toolkit as tk
import zhinst.utils
import zhinst.core

import os
import time
import textwrap
import numpy as np
import time
from typing import List

class HDAWG:

    def __init__(self, device_id, server_host, server_port):

        # self.awg = tk.HDAWG('name', 'dev8181', interface = 'USB')
        api_level = 6
        # self.awg.setup()
        # self.awg.connect_device()
        (self.daq, self.device, self.props) = zhinst.utils.create_api_session(device_id, api_level, server_host, server_port)

        zhinst.utils.api_server_version_check(self.daq)

        self.set_channels()
        self.sampling_rate = [2.4e9, 2.4e9] # Hz
        self.awg_sampling_rate_opts_dict = {"2.4 GHz": [0, 2.4e9], "1.2 GHz": [1, 1.2e9], "600 MHz": [2, 600e6], "300 MHz": [3, 300e6],
                                            "150 MHz": [4, 150e6], "75 MHz": [5, 75e6], "37.5 MHz": [6, 37.5e6], "18.75 MHz": [7, 18.75e6],
                                            "9.37 MHz": [8, 9.37e6], "4.68 MHz": [9, 4.68e6], "2.34 MHz": [10, 2.34e6], "1.17 MHz": [11, 1.17e6],
                                            "585.93 kHz": [12, 585930], "292.96 kHz": [13, 292960]}
        self.voltage_range = 0.8
        self.voltage_range_rf = 0.8
        self.control_both_groups = -1

        zhinst.utils.disable_everything(self.daq, self.device)

        exp_setting = [
            [f"/{self.device}/awgs/0/time", 0],
            [f"/{self.device}/awgs/1/time", 0],

            [f"/{self.device}/system/clocks/referenceclock/source", 1], # external reference clock from Rb standard

            [f"/{self.device}/awgs/0/auxtriggers/0/channel", 0],
            [f"/{self.device}/awgs/0/auxtriggers/1/channel", 0],
            [f"/{self.device}/awgs/0/auxtriggers/0/slope", 1],
            [f"/{self.device}/awgs/0/auxtriggers/1/slope", 1],           
            
            [f"/{self.device}/awgs/1/auxtriggers/0/channel", 2],
            [f"/{self.device}/awgs/1/auxtriggers/1/channel", 3],
            [f"/{self.device}/awgs/1/auxtriggers/0/slope", 1],
            [f"/{self.device}/awgs/1/auxtriggers/1/slope", 1],

            [f"/{self.device}/awgs/0/outputs/0/modulation/mode", 1], # set for SRS IQ modulation
            [f"/{self.device}/awgs/0/outputs/1/modulation/mode", 2], # set for SRS IQ modulation
            [f"/{self.device}/awgs/1/outputs/0/modulation/mode", 1],
            [f"/{self.device}/awgs/1/outputs/1/modulation/mode", 2],
            # [f"/{self.device}/awgs/1/outputs/0/modulation/mode", 0],
            # [f"/{self.device}/awgs/1/outputs/1/modulation/mode", 0], # for DC signal test debugging coil ODMR

            # for SRS IQ modulation
            [f"/{self.device}/sigouts/0/on", 1],
            [f"/{self.device}/sigouts/1/on", 1],
            [f"/{self.device}/sigouts/0/direct", 1],
            [f"/{self.device}/sigouts/1/direct", 1],
            # [f"/{self.device}/sigouts/3/direct", 1], # TODO: see if this is necessary (comment out when done with RF coil sensitivity test signal)

            # for electron/nuclear spin driving
            [f"/{self.device}/sigouts/2/on", 1],
            [f"/{self.device}/sigouts/3/on", 1],

            [f"/{self.device}/dios/0/output", 0],
            [f"/{self.device}/dios/0/drive", 1],

            [f"/{self.device}/sigouts/0/delay", 0],
            [f"/{self.device}/sigouts/1/delay", 0],

            [f"/{self.device}/sigouts/0/range", self.voltage_range],
            [f"/{self.device}/sigouts/1/range", self.voltage_range],
            [f"/{self.device}/sigouts/2/range", self.voltage_range],
            [f"/{self.device}/sigouts/3/range", self.voltage_range_rf]
            # [f"/{self.device}/awgs/0/outputs/{self.awg_channel}/amplitude", self.amplitude]

            # ['/%s/awgs/0/outputs/%d/amplitude' % (self.device, self.awg_channel), self.amplitude],
            # ['/%s/awgs/0/outputs/0/modulation/mode' % self.device, 0],
            # ['/%s/awgs/0/time'                 % self.device, 0],
            # ['/%s/awgs/0/userregs/0'           % self.device, 0]
        ]
        
        self.daq.set(exp_setting)

        self.awgModule = self.daq.awgModule()
        
    def set_channels(self):
        ### CHANNEL SELECTION ###
        # 'system/awg/channelgrouping' : Configure how many independent sequencers
        #   should run on the AWG and how the outputs are grouped by sequencer.
        #   0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
        #   1 : 2x4 with HDAWG8; 1x4 with HDAWG4.
        #   2 : 1x8 with HDAWG8. 
        # Configure the HDAWG to use one sequencer with the same waveform on all output channels.
        self.daq.setInt(f"/{self.device}/system/awg/channelgrouping", 0)
        self.daq.setInt(f"/{self.device}/triggers/in/0/imp50", 1)
        self.daq.setInt(f"/{self.device}/triggers/in/1/imp50", 1)
        self.daq.setInt(f"/{self.device}/triggers/in/2/imp50", 1)
        self.daq.setInt(f"/{self.device}/triggers/in/3/imp50", 1)

    def set_awg_oscillator_control(self, state):
        match state:
            case 'on':
                self.daq.setInt(f"/{self.device}/system/awg/oscillatorcontrol", 1)
            case _:
                self.daq.setInt(f"/{self.device}/system/awg/oscillatorcontrol", 0)
    
    def set_sampling_rate(self, group, val):
        try:
            print("setting sampling rate for group ", group, " to idx ", self.awg_sampling_rate_opts_dict[val][0], " which is ", self.awg_sampling_rate_opts_dict[val][1], " Hz")
            self.daq.setInt(f"/{self.device}/awgs/{group}/time", self.awg_sampling_rate_opts_dict[val][0]) # set sampling rate option for specific group
            self.sampling_rate[group] = self.awg_sampling_rate_opts_dict[val][1] # set sampling rate in Hz for specific group
        except Exception as e:
            print(e)

    def wait_ready(self, group: int, timeout: float = 60.0, require_transition: bool = False):
        """
        Wait until /awgs/{group}/ready == 1.

        If require_transition=True, we first wait for ready to become 0 at least once
        (useful to avoid accepting a stale '1' from a previously-loaded ELF).
        """
        dev = self.device
        node = f"/{dev}/awgs/{group}/ready"

        t0 = time.time()

        if require_transition:
            # Wait until it is NOT ready (0) first
            while self.daq.getInt(node) == 1:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"Timed out waiting for awgs/{group}/ready to drop from 1 to 0.")
                time.sleep(0.01)

        # Then wait until it becomes ready (1)
        while self.daq.getInt(node) == 0:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"Timed out waiting for awgs/{group}/ready == 1.")
            time.sleep(0.01)
        
    def set_group_enabled(self, group):
        self.daq.setInt(f"/{self.device}/awgs/{group}/single", 1)
        self.daq.setInt(f"/{self.device}/awgs/{group}/enable", 1)        
        # self.daq.setInt(f"/{self.device}/awgs/1/single", 1)
        # self.daq.setInt(f"/{self.device}/awgs/1/enable", 1)
    
    def set_awg_enabled(self, group: int | None = None, enable: bool = True):
        val = 1 if enable else 0
        if group is None:
            for g in (0, 1):
                self.daq.setInt(f"/{self.device}/awgs/{g}/enable", val)
        else:
            self.daq.setInt(f"/{self.device}/awgs/{group}/enable", val)
        self.daq.sync()

    def set_disabled(self):
        self.daq.setInt(f"/{self.device}/awgs/0/single", 0)
        self.daq.setInt(f"/{self.device}/awgs/0/enable", 0)
        self.daq.setInt(f"/{self.device}/awgs/1/single", 0)
        self.daq.setInt(f"/{self.device}/awgs/1/enable", 0)
        self.awgModule.set('awg/enable', 0)

    def set_voltage_offsets(self, i_offset, q_offset):
        self.daq.setDouble(f"/{self.device}/sigouts/1/offset", i_offset) # units of [V]
        self.daq.setDouble(f"/{self.device}/sigouts/0/offset", q_offset)

    def set_sequence(self, **kwargs):

        self.set_voltage_offsets(kwargs['i_offset'], kwargs['q_offset']) # set I and Q offset voltages to suppress carrier freq

        match kwargs['seq']:
            case 'Test':
                # wave I_pihalf_x = rect(10000.0, 0.5625);
                # wave Q_pihalf_x = rect(10000.0, 0.5625);

                # setSinePhase(0, 0); setSinePhase(1, 90);

                # while(1){
                # playWave(1, I_pihalf_x, 2, Q_pihalf_x); waitWave();
                
                # }
                match kwargs['channel']:
                    case 'IQ':
                        i_wave = [self.create_rect_wave("I", 10000, self.convert_mw_power(kwargs['sideband_power']))]
                        q_wave = [self.create_rect_wave("Q", 10000, self.convert_mw_power(kwargs['sideband_power']))]

                        i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                        q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                        awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                        for i in range(1):
                            awg_pulses.append(f"""while(1){{{self.create_iq_pulses_no_trig(1, "I", 2, "Q")} waitWave();}}""")

                        self.awg_iq_program_text = "\n".join(i_wave + q_wave + i_phase + q_phase + awg_pulses)

                        self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                        self.control_both_groups = 0

                    case 'DEER':
                        print("running deer output")
                        deer_wave = [self.create_rect_wave("waveRF", 10000, self.convert_mw_power(kwargs['deer_power']))]

                        awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                        for i in range(1):
                            awg_pulses.append(f"""while(1){{{self.create_pulses_no_trig(1, "waveRF")} waitWave();}}""")

                        self.awg_program_text = "\n".join(deer_wave + awg_pulses)

                        self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['RF_Frequency'])

                        self.control_both_groups = 1

                self.set_awg_oscillator_control('off')

            # FIXME: outdated loop structure
            case 'Calibrate':
                i_wave = [self.create_rect_wave("I", kwargs['pi']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q", kwargs['pi']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                for i in range(kwargs['num_pts']):
                    awg_pulses.append(f"""repeat(1){{{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();}}""")
                
                iter_repeat = [self.repeat(kwargs['iters'])] 
                runs_repeat = [self.repeat(kwargs['runs'])]

                iters_end = [f"}}}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + i_phase + q_phase + iter_repeat + runs_repeat + awg_pulses + iters_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0

            case 'CW ODMR':           
                odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
                # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
                i_wave = [self.create_rect_wave("I", kwargs['probe_length']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q", kwargs['probe_length']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                
                odmr_pulses: List[str] = []
                
                # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
                for i in range(kwargs['num_pts']):
                    odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 0

            case 'CW ODMR central freq':           
                odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
                # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
                i_wave = [self.create_rect_wave("I", kwargs['probe_length']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q", kwargs['probe_length']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                
                odmr_pulses: List[str] = []
                
                # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
                for i in range(kwargs['num_pts']//2):
                    odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();
{self.set_sine_phase(0, kwargs['iq_phases'][2])}
{self.set_sine_phase(1, kwargs['iq_phases'][3])}
{self.create_iq_pulses(1, 1, "I", 2, "Q")} waitWave();""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 0

            case 'Rabi':           
                rabi_pulses = kwargs['pi_pulses']

                i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                i_waves: List[str] = [] # defines wave structures
                q_waves: List[str] = []
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    # awg_waves.append(f"wave {i} = rect({rabi_pulses[i]}, {kwargs['awg_power']});")
                    i_waves.append(self.create_rect_wave(f"I{i}", rabi_pulses[i]*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power'])))
                    q_waves.append(self.create_rect_wave(f"Q{i}", rabi_pulses[i]*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power'])))
                    
                    awg_pulses.append(f"""{self.create_iq_pulses(1, 1, f"I{i}", 2, f"Q{i}")} waitWave();""")

                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_phase + q_phase + i_waves + q_waves + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0
            
            case 'Pulsed ODMR':           
                odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
                # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
                i_wave = [self.create_rect_wave("I_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                
                odmr_pulses: List[str] = []
                
                # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
                for i in range(kwargs['num_pts']):
                    odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
{self.create_iq_pulses(1, 1, "I_pi", 2, "Q_pi")} waitWave();""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 0

            case 'Pulsed ODMR RF':   
                print("running pulsed odmr COIL")        
                odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
                # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
                i_wave = [self.create_rect_wave("I_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                
                odmr_pulses: List[str] = []
                
                # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
                for i in range(kwargs['num_pts']):
                    odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
repeat(2){{{self.create_iq_pulses(1, 1, "I_pi", 2, "Q_pi")} waitWave();}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

                awg_wave = [self.create_rect_wave("waveRF", kwargs['rf_length']*self.sampling_rate[1], self.convert_mw_power_rf(kwargs['rf_power']))]
                awg_pulses = [f"""repeat({kwargs['num_pts']}){{
repeat(1){{resetOscPhase();
{self.set_sine_phase(1, kwargs['rf_phase'])}
{self.create_pulses(2, 2, "waveRF")} waitWave();}}}}"""]
                
                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['rf_freq'])

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 2

#             case 'Pulsed ODMR RF':   
#                 print("running odmr COIL dc")        
#                 odmr_side_freqs = kwargs['sideband_freqs'] # define frequency array
                
#                 # define the sideband frequencies for the ODMR experiment over range ("start", "stop", step defined by num_pts)
#                 i_wave = [self.create_rect_wave("I_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
#                 q_wave = [self.create_rect_wave("Q_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                
#                 odmr_pulses: List[str] = []
                
#                 # loop through each sideband frequency, set lower frequency and output first then set upper frequency and output
#                 for i in range(kwargs['num_pts']):
#                     odmr_pulses.append(f"""setDouble('oscs/0/freq', {odmr_side_freqs[i]}); 
# {self.set_sine_phase(0, kwargs['iq_phases'][0])}
# {self.set_sine_phase(1, kwargs['iq_phases'][1])}
# repeat(2){{{self.create_iq_pulses(1, 1, "I_pi", 2, "Q_pi")} waitWave();}}""")
                
#                 inf_repeat = [self.repeat_inf()]
#                 inf_end = [f"}}"]

#                 self.awg_iq_program_text = "\n".join(i_wave + q_wave + inf_repeat + odmr_pulses + inf_end)

#                 awg_wave = [self.create_rect_wave("waveRF", 2000, 0.5)] # DC signal
#                 awg_pulses = [f"""while(true){{
# waitDigTrigger(2);
# playWave(2, waveRF);
# waitWave();}}"""]
                
#                 self.awg_program_text = "\n".join(awg_wave + awg_pulses)
                
#                 # self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['rf_freq'])

#                 self.set_awg_oscillator_control('on')

#                 self.control_both_groups = 2

            case 'T1':
                i_wave = [self.create_rect_wave("I_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave = [self.create_rect_wave("Q_pi", kwargs['pi_pulse']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                i_phase = [self.set_sine_phase(0, kwargs['iq_phases'][0])]
                q_phase = [self.set_sine_phase(1, kwargs['iq_phases'][1])]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                # for i in range(kwargs['num_pts']):
                awg_pulses.append(f"""repeat({kwargs['num_pts']}){{{self.create_iq_pulses(1, 1, "I_pi", 2, "Q_pi")} waitWave();}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave + q_wave + i_phase + q_phase + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0

            case 'DQ':
                i_wave_minus1 = [self.create_rect_wave("I_minus1", kwargs['pi_minus1']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_minus1 = [self.create_rect_wave("Q_minus1", kwargs['pi_minus1']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_plus1 = [self.create_rect_wave("I_plus1", kwargs['pi_plus1']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_plus1 = [self.create_rect_wave("Q_plus1", kwargs['pi_plus1']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                
                awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_sine_phase(0, kwargs['iq_phases'][0])}
{self.set_sine_phase(1, kwargs['iq_phases'][1])}
repeat(4){{{self.create_iq_pulses(1, 1, "I_minus1", 2, "Q_minus1")} waitWave();}}
{self.set_sine_phase(0, kwargs['iq_phases'][2])}
{self.set_sine_phase(1, kwargs['iq_phases'][3])}
repeat(1){{{self.create_iq_pulses(1, 1, "I_plus1", 2, "Q_plus1")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_minus1 + q_wave_minus1 + i_wave_plus1 + q_wave_plus1 + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0
                
            case 'T2':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                
                match kwargs['seq_dd']:
                    case 'Ramsey':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'Echo':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();  
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'XY4':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();        
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'XY8':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();              
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                    
                    case 'YY8':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")
                            
                    case 'CPMG':
                        awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();                     
repeat({kwargs['n']}){{
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")

                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 0

            case 'T2 RF':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                
                # XY8
#                 awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();              
# repeat({kwargs['n']}){{
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
# repeat({kwargs['n']}){{
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
# {self.set_pulse_phases('y', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
# {self.set_pulse_phases('x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
# {self.set_pulse_phases('-x', kwargs['iq_phases'])}
# {self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")

                # YY8    
                awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();}}
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}""")

                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                rf_times = kwargs['rf_length']

                awg_phase = [self.set_sine_phase(1, kwargs['rf_phase'])]
                awg_waves: List[str] = [] # defines wave structures
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse
                
                for i in range(kwargs['num_pts']):
                    # awg_waves.append(f"wave {i} = rect({rabi_pulses[i]}, {kwargs['awg_power']});")
                    awg_waves.append(self.create_rect_wave(f"waveRF{i}", rf_times[i]*self.sampling_rate[1], self.convert_mw_power_rf(kwargs['rf_power'])))
                    
                    awg_pulses.append(f"""repeat(2){{{self.create_pulses(2, 2, f"waveRF{i}")} waitWave();}}""")

#                 awg_wave = [self.create_rect_wave("waveRF", kwargs['rf_length']*self.sampling_rate[1], self.convert_mw_power_rf(kwargs['rf_power']))]
#                 awg_pulses = [f"""repeat({kwargs['num_pts']}){{
# repeat(1){{
# {self.set_sine_phase(0, kwargs['rf_phase'])}
# {self.create_pulses(2, 2, "waveRF")} waitWave();}}}}"""]
                
                self.awg_program_text = "\n".join(awg_phase + awg_waves + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['rf_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER':  
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();     
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                

                deer_freqs = kwargs['freqs'] # define frequency array
                # print("DEER FREQUENCIES: ", deer_freqs)
                #define the frequency for the deer experiment over range ("start", "stop", step defined by num_pts)
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate[1]}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['pi_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses: List[str] = []

                for i in range(kwargs['num_pts']):
                    awg_pulses.append(f"""setDouble('oscs/1/freq', {deer_freqs[i]}); 
repeat(4){{{self.create_pulses(1, 1, "waveRF")} waitWave();}}""")

                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)

                self.set_awg_oscillator_control('on')
                
                self.control_both_groups = 2

            case 'DEER CD':           
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(3){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();        
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                deer_freqs = kwargs['freqs'] # define frequency array
                # print("DEER FREQUENCIES: ", deer_freqs)
                #define the frequency for the deer experiment over range ("start", "stop", step defined by num_pts)
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate[1]}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['pi_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses: List[str] = []

                for i in range(kwargs['num_pts']):
                    awg_pulses.append(f"""setDouble('oscs/1/freq', {deer_freqs[i]}); 
repeat(2){{{self.create_pulses(1, 1, "waveRF")} waitWave();}}""")

                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 2

            case 'DEER Rabi':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();   
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                rabi_pulses = kwargs['pi_pulses']
                
                awg_waves: List[str] = [] # defines wave structures
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    # awg_waves.append(f"wave {i} = rect({rabi_pulses[i]}, {kwargs['awg_power']});")
                    awg_waves.append(self.create_rect_wave(f"w{i}", rabi_pulses[i]*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power'])))
                    # awg_pulses.append(f"waitDigTrigger(1); playWave(1, {i});")
                    
                    awg_pulses.append(f"""repeat(4){{
{self.create_pulses(1, 1, f"w{i}")}}}""")

                self.awg_program_text = "\n".join(awg_waves + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER Corr Rabi':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(3){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();           
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                rabi_pulses = kwargs['pi_pulses']
                
                awg_wave = [self.create_rect_wave("waveRF", kwargs['dark_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]

                awg_waves: List[str] = [] # defines wave structures
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    # awg_waves.append(f"wave {i} = rect({rabi_pulses[i]}, {kwargs['awg_power']});")
                    awg_waves.append(self.create_rect_wave(f"w{i}", rabi_pulses[i]*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power'])))
                    # awg_pulses.append(f"waitDigTrigger(1); playWave(1, {i});")
                    awg_pulses.append(f"""repeat(2){{{self.create_pulses(1, 1, "waveRF")} {self.create_pulses(1, 1, "waveRF")}
{self.create_pulses(1, 1, f"w{i}")} {self.create_pulses(1, 1, "waveRF")} {self.create_pulses(1, 1, "waveRF")}}}""")

                self.awg_program_text = "\n".join(awg_wave + awg_waves + inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')
                
                self.control_both_groups = 2

            case 'DEER FID':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate[1]}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['pi_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses = [f"""repeat({kwargs['num_pts']}){{
repeat({4*kwargs['n']}){{
{self.create_pulses(1, 1, "waveRF")}}}}}"""]
                
                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER FID CD':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(3){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('-x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                taus = kwargs['taus']

                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate[1]}, {kwargs['awg_power']});"] 
                awg_wave_DEER = [self.create_rect_wave("waveDEER", kwargs['pi_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]
                
                awg_CD_waves: List[str] = [] # defines wave structures
                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                for i in range(kwargs['num_pts']):
                    awg_CD_waves.append(self.create_rect_wave(f"waveCD{i}", taus[i]*self.sampling_rate[1], self.convert_mw_power(kwargs['cd_mw_power'])))

                    awg_pulses.append(f"""repeat({4*kwargs['n']}){{
{self.create_pulses(1, 1, "waveDEER")}}}
repeat({4*kwargs['n']}){{
{self.create_pulses(1, 1, f"waveCD{i}")}}}""")
                
                self.awg_program_text = "\n".join(awg_wave_DEER + awg_CD_waves + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER Corr T1':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(4){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}

repeat(2){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate[1]}, {kwargs['awg_power']});"] 
                awg_wave = [self.create_rect_wave("waveRF", kwargs['dark_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]
                awg_pulses = [f"""repeat({kwargs['num_pts']}){{
repeat(1){{
repeat(18){{
{self.create_pulses(1, 1, "waveRF")}}}}}}}"""]
                
                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'DEER T2':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_iq_pulses: List[str] = []

                awg_iq_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(3){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}

repeat(1){{{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();

{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();

{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}}}""")
                
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_iq_pulses + inf_end)
                
                #awg_wave = [f"wave waveRF = rect({kwargs['dark_pi']*self.sampling_rate[1]}, {kwargs['awg_power']});"] 
                awg_pihalf_wave = [self.create_rect_wave("wave_pihalf_RF", kwargs['dark_half_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]
                awg_pi_wave = [self.create_rect_wave("wave_pi_RF", kwargs['dark_pulse']*self.sampling_rate[1], self.convert_mw_power(kwargs['mw_power']))]

                awg_pulses = [f"""repeat({kwargs['num_pts']}){{
repeat(2){{
repeat(2){{
{self.create_pulses(1, 1, "wave_pi_RF")} waitWave();}}

repeat(1){{
{self.create_pulses(1, 1, "wave_pihalf_RF")} waitWave();
{self.create_pulses(1, 1, "wave_pi_RF")} waitWave();
{self.create_pulses(1, 1, "wave_pihalf_RF")} waitWave();
}}

repeat(2){{
{self.create_pulses(1, 1, "wave_pi_RF")} waitWave();}}}}}}"""]
                
                self.awg_program_text = "\n".join(awg_pihalf_wave + awg_pi_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['dark_freq'])

                self.set_awg_oscillator_control('off')

                self.control_both_groups = 2

            case 'NMR':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                awg_pulses.append(f"""repeat({kwargs['num_pts']}){{
repeat(2){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}""")

                    
                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                ### RF coil signal
                awg_wave = [self.create_rect_wave("waveRF_pihalf", kwargs['total_exp_time']*self.sampling_rate[1], self.convert_mw_power_rf(kwargs['rf_power']))]
                awg_pulses = [f"""repeat(1){{{self.set_sine_phase(1, kwargs['rf_phase'])}
{self.create_pulses(2, 2, "waveRF_pihalf")} waitWave();}}"""]
                
                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['rf_freq'])
                ##

                self.set_awg_oscillator_control('on')

                # self.control_both_groups = 0
                self.control_both_groups = 2

            case 'CASR':
                i_wave_pihalf_x = [self.create_rect_wave("I_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pihalf_x = [self.create_rect_wave("Q_pihalf_x", kwargs['pihalf_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pihalf_y = [self.create_rect_wave("I_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]          
                q_wave_pihalf_y = [self.create_rect_wave("Q_pihalf_y", kwargs['pihalf_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]            
                i_wave_pi_x = [self.create_rect_wave("I_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_x = [self.create_rect_wave("Q_pi_x", kwargs['pi_x']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                i_wave_pi_y = [self.create_rect_wave("I_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]
                q_wave_pi_y = [self.create_rect_wave("Q_pi_y", kwargs['pi_y']*self.sampling_rate[0], self.convert_mw_power(kwargs['sideband_power']))]

                awg_pulses: List[str] = [] # lines telling AWG to emit pulse

                awg_pulses.append(f"""repeat({kwargs['n_R']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();

{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_x", 2, "Q_pihalf_x")} waitWave();
repeat({kwargs['n']}){{
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();
{self.set_pulse_phases('y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_y", 2, "Q_pi_y")} waitWave();
{self.set_pulse_phases('x', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pi_x", 2, "Q_pi_x")} waitWave();}}
{self.set_pulse_phases('-y', kwargs['iq_phases'])}
{self.create_iq_pulses(1, 1, "I_pihalf_y", 2, "Q_pihalf_y")} waitWave();}}""")

                inf_repeat = [self.repeat_inf()]
                inf_end = [f"}}"]

                self.awg_iq_program_text = "\n".join(i_wave_pihalf_x + q_wave_pihalf_x + i_wave_pihalf_y + q_wave_pihalf_y + 
                                             i_wave_pi_x + q_wave_pi_x + i_wave_pi_y + q_wave_pi_y + 
                                             inf_repeat + awg_pulses + inf_end)

                self.daq.setDouble(f"/{self.device}/oscs/0/freq", kwargs['sideband_freq'])

                # awg_wave = [self.create_rect_wave("waveRF_pihalf", kwargs['rf_pihalf']*75e6, self.convert_mw_power_rf(kwargs['rf_power']))]
                awg_wave = [self.create_rect_wave("waveRF_pihalf", kwargs['rf_pihalf']*self.sampling_rate[1], self.convert_mw_power_rf(kwargs['rf_power']))]
                awg_pulses = [f"""repeat(1){{resetOscPhase();
{self.set_sine_phase(1, kwargs['rf_phase'])}
{self.create_pulses(2, 2, "waveRF_pihalf")} waitWave();}}"""]
                
                self.awg_program_text = "\n".join(awg_wave + inf_repeat + awg_pulses + inf_end)
                
                self.daq.setDouble(f"/{self.device}/oscs/1/freq", kwargs['rf_freq'])

                self.set_awg_oscillator_control('on')

                self.control_both_groups = 2

        # Disable AWG cores before compilation
        if self.control_both_groups == 2:
            self.set_awg_enabled(None, False) # disables both via device nodes
        elif self.control_both_groups == 0:
            self.set_awg_enabled(0, False)
        elif self.control_both_groups == 1:
            self.set_awg_enabled(1, False)

        if self.control_both_groups == 0:
            self.awg_iq_program = textwrap.dedent(f"{self.awg_iq_program_text}")
            self.compile_sequence(0, self.awg_iq_program)
        elif self.control_both_groups == 1:
            self.awg_program = textwrap.dedent(f"{self.awg_program_text}")
            self.compile_sequence(1, self.awg_program)
        else:
            self.awg_iq_program = textwrap.dedent(f"{self.awg_iq_program_text}")
            self.compile_sequence(0, self.awg_iq_program)
            time.sleep(0.01)
            self.awg_program = textwrap.dedent(f"{self.awg_program_text}")
            self.compile_sequence(1, self.awg_program)

        # Enable required AWG cores
        if self.control_both_groups == 2:
            self.set_awg_enabled(None, True)
        elif self.control_both_groups == 0:
            self.set_awg_enabled(0, True)
        else:
            self.set_awg_enabled(1, True)

    def compile_sequence(self, group: int, program: str, timeout: float = 60.0):
        dev = self.device

        # Disable this core before reloading
        self.daq.setInt(f"/{dev}/awgs/{group}/enable", 0)
        self.daq.setInt(f"/{dev}/awgs/{group}/single", 0)
        self.daq.sync()

        awgModule = self.daq.awgModule()
        awgModule.set("device", dev)
        awgModule.set("index", group)
        awgModule.set("compiler/upload", 1)
        awgModule.execute()

        # Start compile (+ upload)
        awgModule.set("compiler/sourcestring", program)

        # Wait compile done
        t0 = time.time()
        while True:
            st = awgModule.getInt("compiler/status")
            if st != -1:
                break
            if time.time() - t0 > timeout:
                raise TimeoutError("AWG compilation timed out (compiler/status stayed -1).")
            time.sleep(0.01)

        if st == 1:
            raise RuntimeError(f"AWG compile error: {awgModule.getString('compiler/statusstring')}")
        if st == 2:
            print("AWG compile warning:", awgModule.getString("compiler/statusstring"))

        # Wait upload progress done
        t0 = time.time()
        while True:
            prog = awgModule.getDouble("progress")
            if prog >= 1.0:
                break
            if time.time() - t0 > timeout:
                raise TimeoutError("AWG upload timed out (progress never reached 1.0).")
            time.sleep(0.02)

        # Wait device core ready
        t0 = time.time()
        while self.daq.getInt(f"/{dev}/awgs/{group}/ready") == 0:
            if time.time() - t0 > timeout:
                raise TimeoutError("AWG core did not become ready after upload.")
            time.sleep(0.02)

    def convert_mw_power(self, power):
        I = 500/np.sqrt(2)
        if power > I:
            raise Exception("Power exceeds IQ port handling for SRS 396 signal generator.")
        else:
            return power/self.voltage_range

    def convert_mw_power_rf(self, power):
        I = 500/np.sqrt(2)
        if power > I:
            raise Exception("Power exceeds IQ port handling for SRS 396 signal generator.")
        else:
            return power/self.voltage_range_rf
        
# generate separate one-task functions for each command: create rectangular waves; repeat iterations; repeat runs; create pulses
    def create_rect_wave(self, name, length, power):
        return f"wave {name} = rect({length}, {power});"
    
    def set_sine_phase(self, ch, phase):
        return f"setSinePhase({ch}, {phase});"
    
    def set_pulse_phases(self, axis, iq_phases):
        match axis:
            case 'x':
                phase_offset = 0
            case '-x':
                phase_offset = 180
            case 'y':
                phase_offset = 90
            case '-y':
                phase_offset = 270

        return f"setSinePhase(0, {phase_offset + iq_phases[0]}); setSinePhase(1, {phase_offset + iq_phases[1]});"

    def repeat_inf(self):
        return f"while(true){{"
    
    def repeat(self, var):
        return f"repeat({var}){{"

    def runs(self, runs):
        return f"repeat({runs}){{"
    
    def create_pulses(self, trigger_num, wave_num, wave_name):
        return f"waitDigTrigger({trigger_num}); playWave({wave_num}, {wave_name});"
        # return f"playWave({wave_num}, {wave_name});"
    
    def create_pulses_no_trig(self, wave_num, wave_name):
        return f"playWave({wave_num}, {wave_name});"
    
    def create_iq_pulses(self, trigger_num, i_wave_num, i_wave_name, q_wave_num, q_wave_name):
        return f"waitDigTrigger({trigger_num}); playWave({i_wave_num}, {i_wave_name}, {q_wave_num}, {q_wave_name});"

    def create_iq_pulses_no_trig(self, i_wave_num, i_wave_name, q_wave_num, q_wave_name):
        return f"playWave({i_wave_num}, {i_wave_name}, {q_wave_num}, {q_wave_name});"
    
    def create_gauss_wave(self, name, sample_num, amplitude, center_position, standard_deviation):
        return f"wave {name} = gauss({sample_num}, {amplitude}, {center_position}, {standard_deviation});"
    
    def create_sine_wave(self, name, sample_num, amplitude, phase_offset, period_num):
        return f"wave {name} = sine({sample_num}, {amplitude}, {phase_offset}, {period_num});"



    def set_voltage_range(self):
        pass

    def set_sine_generators_state(self):
        # TODO:
        # control when to bypass internal sine generators
        pass


