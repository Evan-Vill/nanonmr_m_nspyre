"""
Spectrum Instrumentation GmbH (c)

3_acq_multi.py

Shows a simple Standard multiple recording mode example using only the few necessary commands
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0
- triggering is done with a channel trigger on channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.

Edited by Evan Villafranca - 2026-03-16
"""

import logging

import numpy as np
import spcm
from spcm import units 

logger = logging.getLogger(__name__)

class SpectrumDigitizer(): 
    def __init__(self,ip_address):
        self._init_config = dict(
            ACCOUPLE=1,
            AMP=5000, # [mV]
            card_timeout=20 * units.s,
            DCCOUPLE=1,
            HF_INPUT_50OHM=1, # 1 = 50 ohm, 0 = 1 Mohm
            num_pts_in_exp=None,
            num_iters=None,
            mem_size=None, 
            num_segment=None,
            pretrig_size=32 * units.Sa, 
            posttrig_size=None,
            readout_ch=None,
            both_ch=False,
            ip_address=ip_address,
            runs=None, 
            sampling_frequency=0.5 * units.GHz,
            segment_size=512 * units.Sa,
            TERMINATE50OHM=1,
            conversion_offset=0.1340332,
            conversion_amp=0.8819033287026647,
        )
       
        # assign the configuration to self
        for key, value in self._init_config.items():
            setattr(self, key, value)
        
        self.connect() # open the connection with the digitizer 

    def connect(self):
        self.card = spcm.Card(self.ip_address) 
        self.card.__enter__()

        if self.card._closed:
            raise RuntimeError("Failed to connect to the digitizer.")

        logger.info("Successfully connected to the digitizer.")

    def disconnect(self):
        self.card.__exit__()

        if self.card._closed:
            logger.info("Card disconnection successful")
            for key, value in self._init_config.items():
                setattr(self, key, value)
        else:
            logger.warning("Card disconnection unsuccessful")

    def stop_card(self):
        self.card.stop(spcm.M2CMD_DATA_STOPDMA)
        logger.info("Card stopped")

    def reset(self):
        self.card.reset()

    def assign_param(self, settings_dict):
        for key, value in settings_dict.items():
            if not hasattr(self, key):
                continue
            
            if key == "readout_ch":
                setattr(self, key, int(value))
            elif key in ("segment_size", "pretrig_size", "posttrig_size", "num_pts_in_exp", "runs"):
                setattr(self, key, value * units.Sa)
            elif key == "sampling_frequency":
                setattr(self, key, value * units.GHz)  # or units.Hz depending on how you pass it
            elif key == "card_timeout":
                setattr(self, key, value * units.s)
            else:
                setattr(self, key, value)
   
    def config(self):  
        # handling memory assignment         
        self.posttrig_size = self.segment_size - self.pretrig_size
        self.num_segment = self.runs * self.num_pts_in_exp 
        self.mem_size = self.num_segment * self.segment_size

        # setup clock engine
        clock = spcm.Clock(self.card)
        clock.mode(spcm.SPC_CM_EXTREFCLOCK)
        clock.reference_clock(10000000) # reference clk = 10 MHz
        clock.sample_rate(self.sampling_frequency)
        
        ch_mapping = {
            0 : spcm.CHANNEL0,
            1 : spcm.CHANNEL1,
        }
        acdc_mapping = {
            0: spcm.SPC_ACDC0,
            1: spcm.SPC_ACDC1,
        }
        path_mapping={
            0: spcm.SPC_PATH0,
            1: spcm.SPC_PATH1,
        }
        amp_mapping={
            0: spcm.SPC_AMP0,
            1: spcm.SPC_AMP1,
        }

        # setup card mode
        self.card.card_mode(spcm.SPC_REC_FIFO_MULTI) # SPC_REC_FIFO_MULTI
        self.card.timeout(self.card_timeout) # get rid of samples [Sa] units on card timeout param

        # set analog input parameters
        if not self.both_ch:
            if self.readout_ch not in ch_mapping:
                raise ValueError(f"Invalid readout_ch: {self.readout_ch}. Expected 0 or 1.")
            
            self.card.set_i(spcm.SPC_CHENABLE, ch_mapping[self.readout_ch])
            self.card.set_i(path_mapping[self.readout_ch], int(self.HF_INPUT_50OHM))
            self.card.set_i(amp_mapping[self.readout_ch], int(self.AMP))
            self.card.set_i(acdc_mapping[self.readout_ch], int(self.ACCOUPLE))
        else:
            self.card.set_i(spcm.SPC_CHENABLE, spcm.CHANNEL0 | spcm.CHANNEL1)

            self.card.set_i(spcm.SPC_PATH0, int(self.HF_INPUT_50OHM))
            self.card.set_i(spcm.SPC_PATH1, int(self.HF_INPUT_50OHM))

            self.card.set_i(spcm.SPC_AMP0, int(self.AMP))
            self.card.set_i(spcm.SPC_AMP1, int(self.AMP))

            self.card.set_i(spcm.SPC_ACDC0, int(self.ACCOUPLE))
            self.card.set_i(spcm.SPC_ACDC1, int(self.ACCOUPLE))
        
        # setup trigger engine
        trigger = spcm.Trigger(self.card)
        trigger.ext0_mode(spcm.SPC_TM_POS)   # set trigger mode
        trigger.or_mask(spcm.SPC_TMASK_EXT0) # trigger set to external
        trigger.termination(self.TERMINATE50OHM)
        trigger.ext0_coupling(spcm.COUPLING_DC)  # trigger coupling
        trigger.ext0_level0(2 * units.V)
                
        self.multiple_recording = spcm.Multi(self.card)
        self.multiple_recording.memory_size(self.mem_size * 10)
        self.multiple_recording.allocate_buffer(self.segment_size, num_segments=self.num_segment)
        self.multiple_recording.post_trigger(self.posttrig_size)
        
        # record forever 
        self.multiple_recording.to_transfer_samples(0)
        self.multiple_recording.notify_samples(self.mem_size)
        self.multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    def check_connection(self):
        if self.card._closed:
            raise RuntimeError("Digitizer is not connected.")
   
    def start_buffer(self):
        self.card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    # TODO: check dual channel functionality and array interleaving to confirm correct processing   
    def acquire(self):
        try:
            data_block = next(self.multiple_recording) # dim (mem_size, segment_size, 1)
            # print(f"shape of dig data block: {np.shape(data_block)}")  
            # print(f"ravel shape: {np.shape(np.asarray(data_block).ravel())}")      

        except spcm.SpcmTimeout:
            self.card.stop(spcm.M2CMD_DATA_STOPDMA)
            self.card.__exit__()
            self.card.__enter__()
            return None

        raw_data = np.asarray(data_block).copy()
        scale = (self.AMP / 1000) / np.abs(self.card.max_sample_value())

        if not self.both_ch:    
            return raw_data * scale * units.V
        
        ch0_data = raw_data[:, :, 0]
        ch1_data = raw_data[:, :, 1]
        
        return ch0_data * scale * units.V, ch1_data * scale * units.V