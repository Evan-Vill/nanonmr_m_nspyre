'''
Python file for controlling magnet mount equipment in NanoNMR experiments
in UChicago Maurer Lab

Copyright (c) 2022, Evan Villafranca
All rights reserved.
'''

import time
from contextlib import suppress

import numpy as np
from nspyre import InstrumentManager

from multiprocessing import Process
# from drivers.nnmr_stagecontrol_zaber import NanoNMRZaber
# from drivers.nnmr_stagecontrol_thorlabs import NanoNMRThorlabs

import logging
logger = logging.getLogger(__name__)
 
class NanoNMRMagnetMotion:
    '''
    Perform magnet mount movements & align magnetic fields using 
    functinality from nnmr_stagecontrol_.py files.
    '''
    def __init__(self):
        super().__init__()

    # def get_all_positions(self):
    #     with InstrumentManager() as mgr:
    #         self.curr_zab_pos = mgr.zaber.current_positions
    #         self.curr_polar_pos = mgr.thor_polar.current_position
    #         self.curr_azi_pos = mgr.thor_azi.current_position

    #     return [self.curr_zab_pos, self.curr_polar_pos, self.curr_azi_pos]

    def get_parked_status(self):
        with InstrumentManager() as mgr:
            is_parked = mgr.zaber.check_parked()
            print("IS ZABER PARKED? ", is_parked)
            return is_parked
        
    def enable_all(self):
        with InstrumentManager() as mgr: 
            mgr.zaber.unpark()
            mgr.thor_polar.enable()
            mgr.thor_azi.enable()

    def disable_all(self):
        with InstrumentManager() as mgr:
            mgr.zaber.park()
            mgr.thor_polar.disable()
            mgr.thor_azi.disable()

    def move_to_home(self, stage):

        with InstrumentManager() as mgr:
            if stage == "zaber":
                mgr.zaber.home(mgr.zaber.update_positions_callback())
                # is_home = mgr.zaber.check_homed()
                # if is_home: 
                #     "Edit status bar here saying all devices already homed"
                #     logger.info("Zaber stages already homed.")
                # else:
                #     mgr.zaber.home(mgr.zaber.update_positions_callback())
            elif stage == "thor_polar":
                mgr.thor_polar.home(mgr.thor_polar.update_positions_callback())
                # is_home = mgr.thor_polar.check_homed()
                # if is_home:
                #     logger.info("Thorlabs polar stage already homed.")
                # else:
                #     mgr.thor_polar.home(mgr.thor_polar.update_positions_callback())
            elif stage == "thor_azi":
                mgr.thor_azi.home(mgr.thor_azi.update_positions_callback())
                # is_home = mgr.thor_azi.check_homed()
                # if is_home:
                #     logger.info("Thorlabs azimuthal stage already homed.")
                # else:
                #     mgr.thor_azi.home(mgr.thor_azi.update_positions_callback())

    def enable(self, stage):
        with InstrumentManager() as mgr:
            if stage == 'zaber':
                mgr.zaber.unpark()
            elif stage == 'thor_polar':
                mgr.thor_polar.enable()
            elif stage == 'thor_azi':
                mgr.thor_azi.enable()
            
    def disable(self, stage):
        with InstrumentManager() as mgr:
            if stage == 'zaber':
                mgr.zaber.park()
            elif stage == 'thor_polar':
                mgr.thor_polar.disable()
            elif stage == 'thor_azi':
                mgr.thor_azi.disable()
            
    def _angle_diff_deg(self, current, target):
        '''
        Calculate the minimum difference between two angles in degrees,
        accounting for wrap-around at 360 degrees.
        '''
        diff = (target - current + 180.0) % 360.0 - 180.0
        return diff # if diff != -180 else 180
    
    def standby_all(self, **kwargs):
        TARGET_AZI = 100.0  # degrees
        TOL = 0.1  # degrees tolerance for at standby position
        POLL_S = 0.5  # seconds between polling position
        TIMEOUT_S = 60.0  # safety timeout

        with InstrumentManager() as mgr:
            try:
                kwargs['queue'].put_nowait(['start standby', 
                                            mgr.zaber.current_positions, 
                                            mgr.thor_polar.current_position, 
                                            mgr.thor_azi.current_position])
                
                # move all stages to standby positions
                mgr.zaber.standby()
                mgr.thor_polar.standby()
                mgr.thor_azi.standby()

                # update positions of stages to display in GUI
                mgr.thor_azi.update_positions_callback()
                pos_azi = mgr.thor_azi.current_position 
                
                t0 = time.time()
                last_pos = None

                # poll until azimuthal stage reaches standby position (or timeout)
                while True:
                    # check position
                    err = abs(self._angle_diff_deg(pos_azi, TARGET_AZI))
                    if err <= TOL:
                        break  # at standby position

                    # optional: check for no movement (stuck)
                    if last_pos is not None and abs(pos_azi - last_pos) < 1e-6:
                        logger.warning("Azimuthal stage position not changing; possible stall.")
                    
                    # update for next iteration
                    last_pos = pos_azi

                    # check for timeout
                    if (time.time() - t0) > TIMEOUT_S:
                        logger.error("Azimuthal stage timed out while moving to standby position.")
                        break

                    time.sleep(POLL_S)
                    mgr.thor_azi.update_positions_callback() # update azimuthal position
                    pos_azi = mgr.thor_azi.current_position

                    mgr.thor_polar.update_positions_callback() # update polar position

                    kwargs['queue'].put_nowait(['in motion to standby', 
                                            mgr.zaber.current_positions, 
                                            mgr.thor_polar.current_position, 
                                            mgr.thor_azi.current_position])
                    
                # while not round(pos_azi,1) == round(100.0, 1):
                #     time.sleep(0.8)
                #     mgr.thor_azi.update_positions_callback() # update position
                #     pos_azi = mgr.thor_azi.current_position

            except Exception as e:
                # with suppress(Exception):
                #     logger.error(f"Error during standby_all: {e}")
                pass
            finally:
                time.sleep(0.05)
                # update positions of stages to display in GUI
                mgr.zaber.update_positions_callback()
                mgr.thor_polar.update_positions_callback()
                mgr.thor_azi.update_positions_callback()

                time.sleep(0.05)
                kwargs['queue'].put(['done', 
                                     mgr.zaber.current_positions, 
                                     mgr.thor_polar.current_position, 
                                     mgr.thor_azi.current_position])
                time.sleep(0.1)

    def stop_motion(self, stage):
        with InstrumentManager() as mgr:
            if stage == "zaber":
                mgr.zaber.stop_motion()
            elif stage == "thor_polar":
                mgr.thor_polar.stop_motion()
            elif stage == "thor_azi":
                mgr.thor_azi.stop_motion()

    def stop_all_motion(self):
        with InstrumentManager() as mgr:
            mgr.zaber.stop_motion()
            mgr.thor_polar.stop_motion()
            mgr.thor_azi.stop_motion()

    def move_to_B_field(self):
        pass

    def move_to_orientation(self, **kwargs):
        TOL = 0.1  # degrees tolerance for at standby position
        POLL_S = 0.5  # seconds between polling position
        TIMEOUT_S = 60.0  # safety timeout
        
        with InstrumentManager() as mgr:    
            try:
                # if kwargs['stage'] == 'all': 
                #     # reset zaber and polar positions to furthest apart and neutral respectively
                #     mgr.zaber.move(100, kwargs['abs'])
                #     mgr.thor_polar.move(0, kwargs['abs'])
                #     time.sleep(0.1)
                #     # after resetting stages to safe orientation, move to new positions
                #     mgr.thor_azi.move(kwargs['new_pos'][0], kwargs['abs'])
                #     mgr.thor_polar.move(kwargs['new_pos'][1], kwargs['abs'])
                #     mgr.zaber.move(kwargs['new_pos'][2], kwargs['abs'])
    
                if kwargs['stage'] == 'zaber':
                    # print(dir(mgr.zaber))
                    kwargs['queue'].put_nowait(['start z move', 
                                        mgr.zaber.current_positions, 
                                        mgr.thor_polar.current_position, 
                                        mgr.thor_azi.current_position, 
                                        kwargs['stage'], 
                                        kwargs['new_pos'], 
                                        kwargs['abs']])
                    
                    mgr.zaber.move(kwargs['new_pos'], kwargs['abs'])
                    # print("NEW ZABER POSITION = ", kwargs['new_pos'])

                elif kwargs['stage'] == 'thor_polar':
                    mgr.thor_polar.update_positions_callback() # update position

                    if kwargs['abs']:
                        TARGET_POLAR = kwargs['new_angle']  # degrees
                    else:
                        TARGET_POLAR = mgr.thor_polar.current_position + kwargs['new_angle']  # degrees

                    kwargs['queue'].put_nowait(['start rotation move', 
                                        mgr.zaber.current_positions, 
                                        mgr.thor_polar.current_position, 
                                        mgr.thor_azi.current_position, 
                                        kwargs['stage'], 
                                        kwargs['new_angle'],
                                        kwargs['abs']])

                    mgr.thor_polar.move(kwargs['new_angle'], kwargs['abs'])

                    mgr.thor_polar.update_positions_callback() # update position
                    pos_polar = mgr.thor_polar.current_position # set pos to be position as move is beginning

                    t0 = time.time()
                    last_pos_polar = None

                    # poll until polar stage reaches new position (or timeout)
                    while True:
                        # check position
                        err = abs(self._angle_diff_deg(pos_polar, TARGET_POLAR))
                        if err <= TOL:
                            break  # at new position

                        # optional: check for no movement (stuck)
                        if last_pos_polar is not None and abs(pos_polar - last_pos_polar) < 1e-6:
                            logger.warning("Polar stage position not changing; possible stall.")
                        
                        # update for next iteration
                        last_pos_polar = pos_polar

                        # check for timeout
                        if (time.time() - t0) > TIMEOUT_S:
                            logger.error("Polar stage timed out while moving to new position.")
                            break

                        time.sleep(POLL_S)
                        mgr.thor_polar.update_positions_callback() # update position
                        pos_polar = mgr.thor_polar.current_position

                        kwargs['queue'].put_nowait(['in rotation motion to target', 
                                                mgr.zaber.current_positions, 
                                                mgr.thor_polar.current_position, 
                                                mgr.thor_azi.current_position, 
                                                kwargs['stage'], 
                                                kwargs['new_angle'],
                                                kwargs['abs']])
                        
                    # while not round(pos,1) == round(kwargs['new_angle'], 1):
                    #     time.sleep(0.8)
                    #     mgr.thor_polar.update_positions_callback() # update position
                    #     pos = mgr.thor_polar.current_position

                elif kwargs['stage'] == 'thor_azi':
                    mgr.thor_azi.update_positions_callback() # update position

                    if kwargs['abs']:
                        TARGET_AZI = kwargs['new_angle']  # degrees
                    else:
                        TARGET_AZI = mgr.thor_azi.current_position + kwargs['new_angle']  # degrees

                    kwargs['queue'].put_nowait(['start rotation move', 
                                        mgr.zaber.current_positions, 
                                        mgr.thor_polar.current_position, 
                                        mgr.thor_azi.current_position, 
                                        kwargs['stage'], 
                                        kwargs['new_angle'],
                                        kwargs['abs']])

                    mgr.thor_azi.move(kwargs['new_angle'], kwargs['abs'])

                    mgr.thor_azi.update_positions_callback() # update position
                    pos_azi = mgr.thor_azi.current_position # set pos to be position as move is beginning

                    t1 = time.time()
                    last_pos_azi = None

                    # poll until azimuthal stage reaches new position (or timeout)
                    while True:
                        # check position
                        err = abs(self._angle_diff_deg(pos_azi, TARGET_AZI))
                        if err <= TOL:
                            break  # at new position

                        # optional: check for no movement (stuck)
                        if last_pos_azi is not None and abs(pos_azi - last_pos_azi) < 1e-6:
                            logger.warning("Azimuthal stage position not changing; possible stall.")
                        
                        # update for next iteration
                        last_pos_azi = pos_azi

                        # check for timeout
                        if (time.time() - t1) > TIMEOUT_S:
                            logger.error("Azimuthal stage timed out while moving to new position.")
                            break

                        time.sleep(POLL_S)
                        mgr.thor_azi.update_positions_callback() # update position
                        pos_azi = mgr.thor_azi.current_position

                        kwargs['queue'].put_nowait(['in rotation motion to target', 
                                                mgr.zaber.current_positions, 
                                                mgr.thor_polar.current_position, 
                                                mgr.thor_azi.current_position, 
                                                kwargs['stage'], 
                                                kwargs['new_angle'],
                                                kwargs['abs']])

                    # while not round(pos,1) == round(kwargs['new_angle'], 1):
                    #     time.sleep(0.8)
                    #     mgr.thor_azi.update_positions_callback() # update position
                    #     pos = mgr.thor_azi.current_position
                        
                else:
                    logger.info(f"Invalid stage name {kwargs['stage']}.")
                    pass
            except:
                pass
            finally:
                time.sleep(0.05)
                # update positions of stages to display in GUI
                mgr.zaber.update_positions_callback()
                mgr.thor_polar.update_positions_callback()
                mgr.thor_azi.update_positions_callback()

                time.sleep(0.05)
                kwargs['queue'].put(['done', 
                                     mgr.zaber.current_positions, 
                                     mgr.thor_polar.current_position, 
                                     mgr.thor_azi.current_position])

                time.sleep(0.1)

    def field_calibration(self):
        pass


    