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

from find_magnet_position_2 import find_z_for_b_from_fits, exclusion_zone_check

import logging
logger = logging.getLogger(__name__)
 
class NanoNMRMagnetMotion:
    '''
    Perform magnet mount movements & align magnetic fields using 
    functinality from nnmr_stagecontrol_.py files.
    '''
    def __init__(self):
        super().__init__()

    # def get_parked_status(self):
    #     with InstrumentManager() as mgr:
    #         is_parked = mgr.zaber.check_parked()
    #         print("IS ZABER PARKED? ", is_parked)
    #         return is_parked
        
    def enable_all(self, **kwargs):
        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']

        zaber.unpark()
        thor_polar.enable()
        thor_azi.enable()

    def disable_all(self, **kwargs):
        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']

        zaber.park()
        thor_polar.disable()
        thor_azi.disable()

    # def move_to_home(self, stage):
    #     with InstrumentManager() as mgr:
    #         if stage == "zaber":
    #             mgr.zaber.home(mgr.zaber.update_positions_callback())
    #             # is_home = mgr.zaber.check_homed()
    #             # if is_home: 
    #             #     "Edit status bar here saying all devices already homed"
    #             #     logger.info("Zaber stages already homed.")
    #             # else:
    #             #     mgr.zaber.home(mgr.zaber.update_positions_callback())
    #         elif stage == "thor_polar":
    #             mgr.thor_polar.home(mgr.thor_polar.update_positions_callback())
    #             # is_home = mgr.thor_polar.check_homed()
    #             # if is_home:
    #             #     logger.info("Thorlabs polar stage already homed.")
    #             # else:
    #             #     mgr.thor_polar.home(mgr.thor_polar.update_positions_callback())
    #         elif stage == "thor_azi":
    #             mgr.thor_azi.home(mgr.thor_azi.update_positions_callback())
    #             # is_home = mgr.thor_azi.check_homed()
    #             # if is_home:
    #             #     logger.info("Thorlabs azimuthal stage already homed.")
    #             # else:
    #             #     mgr.thor_azi.home(mgr.thor_azi.update_positions_callback())

    def enable(self, **kwargs):
        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']
        stage = kwargs['stage']
        
        if stage == 'zaber':
            zaber.unpark()
        elif stage == 'thor_polar':
            thor_polar.enable()
        elif stage == 'thor_azi':
            thor_azi.enable()

    def disable(self, **kwargs):
        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']
        stage = kwargs['stage']

        if stage == 'zaber':
            zaber.park()
        elif stage == 'thor_polar':
            thor_polar.disable()
        elif stage == 'thor_azi':
            thor_azi.disable()
            
    def _angle_diff_deg(self, current, target):
        '''
        Calculate the minimum difference between two angles in degrees,
        accounting for wrap-around at 360 degrees.
        '''
        diff = (target - current + 180.0) % 360.0 - 180.0
        return diff # if diff != -180 else 180
    
    def standby_all(self, **kwargs):
        TARGET_AZI = 100.0  # degrees
        TOL = 0.05  # degrees tolerance for at standby position
        POLL_S = 0.5  # seconds between polling position
        TIMEOUT_S = 60.0  # safety timeout

        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']

        try:
            kwargs['queue'].put_nowait(['start standby', 
                                        zaber.current_positions, 
                                        thor_polar.current_position, 
                                        thor_azi.current_position])
            
            # move all stages to standby positions
            zaber.standby()
            thor_polar.standby()
            thor_azi.standby()

            # update positions of stages to display in GUI
            thor_azi.update_positions_callback()
            pos_azi = thor_azi.current_position 
            
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
                thor_azi.update_positions_callback() # update azimuthal position
                pos_azi = thor_azi.current_position

                thor_polar.update_positions_callback() # update polar position

                kwargs['queue'].put_nowait(['in motion to standby', 
                                        zaber.current_positions, 
                                        thor_polar.current_position, 
                                        thor_azi.current_position])

        except Exception as e:
            logger.error(f"Error during standby_all: {e}")
            return
        finally:
            time.sleep(0.05)
            # update positions of stages to display in GUI
            zaber.update_positions_callback()
            thor_polar.update_positions_callback()
            thor_azi.update_positions_callback()

            time.sleep(0.05)
            kwargs['queue'].put(['done', 
                                    zaber.current_positions, 
                                    thor_polar.current_position, 
                                    thor_azi.current_position])
            time.sleep(0.1)

    def stop_motion(self, **kwargs):
        """Stop motion of specified stage immediately."""
        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']
        
        if kwargs['stage'] == "zaber":
            zaber.stop_motion()
        elif kwargs['stage'] == "thor_polar":
            thor_polar.stop_motion()
        elif kwargs['stage'] == "thor_azi":
            thor_azi.stop_motion()

        time.sleep(0.05)
        # update positions of stages to display in GUI
        zaber.update_positions_callback()
        thor_polar.update_positions_callback()
        thor_azi.update_positions_callback()

        time.sleep(0.05)
        kwargs['queue'].put([
            'stopped', 
            zaber.current_positions, 
            thor_polar.current_position, 
            thor_azi.current_position
        ])

        time.sleep(0.1)
        
    def stop_all_motion(self, **kwargs):
        """Stop motion of all stages immediately."""
        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']
        
        zaber.stop_motion()
        time.sleep(0.05)
        thor_polar.stop_motion()
        time.sleep(0.05)
        thor_azi.stop_motion()
        time.sleep(0.05)

        # update positions of stages to display in GUI
        zaber.update_positions_callback()
        thor_polar.update_positions_callback()
        thor_azi.update_positions_callback()

        time.sleep(0.05)
        kwargs['queue'].put([
            'stopped', 
            zaber.current_positions, 
            thor_polar.current_position, 
            thor_azi.current_position
        ])

        time.sleep(0.1)

    def move_to_orientation(self, **kwargs):
        """Move specified stage to target position or angle, with safety checks for bounds and exclusion zones."""
        TOL = 0.05  # degrees tolerance for at standby position
        POLL_S = 0.5  # seconds between polling position
        TIMEOUT_S = 60.0  # safety timeout

        thor_azi = kwargs['thor_azi']
        thor_polar = kwargs['thor_polar']
        zaber = kwargs['zaber']

        try:
            thor_azi.update_positions_callback()
            thor_polar.update_positions_callback()
            zaber.update_positions_callback()

            start_azi_pos = thor_azi.current_position
            start_polar_pos = thor_polar.current_position
            start_zaber_pos = zaber.current_positions

            if kwargs['stage'] == 'zaber':
                out_of_bounds = None
                z_min = None
                z_max = None

                ### --- Determine target position based on whether move is absolute or relative --- ###
                if kwargs['move_type'] == "Absolute":
                    target_pos = kwargs['new_pos']
                elif kwargs['move_type'] == "Jog":
                    target_pos = start_zaber_pos + kwargs['new_pos']
                elif kwargs['move_type'] == "B Field Target":
                    target_pos, out_of_bounds, z_min, z_max = find_z_for_b_from_fits(
                        b_target=kwargs['new_pos'],
                        azi_choice=start_azi_pos,
                        polar_choice=start_polar_pos
                    )
                else:
                    logger.warning(f"Invalid move type {kwargs['move_type']} for Zaber stage move.")
                    return
                
                ### --- Check if proposed move is safe --- ###
                move_is_safe, move_msg = self.is_move_safe(
                    stage=zaber,
                    stage_name=kwargs['stage'], 
                    stage_current_positions=[start_azi_pos, start_polar_pos, start_zaber_pos], 
                    stage_target_position=target_pos,
                    new_pos=kwargs['new_pos'], 
                    move_type=kwargs['move_type'],
                    out_of_bounds=out_of_bounds,
                    z_min=z_min,
                    z_max=z_max,
                    b_target=kwargs['new_pos'] if kwargs['move_type'] == "B Field Target" else None
                )
                if not move_is_safe:
                    kwargs['queue'].put_nowait(['unsafe z move', 
                                        zaber.current_positions,
                                        thor_polar.current_position,
                                        thor_azi.current_position,
                                        kwargs['stage'],
                                        kwargs['new_pos'],
                                        kwargs['abs'],
                                        move_msg])
                    return
                else:
                    kwargs['queue'].put_nowait(['start z move', 
                                        zaber.current_positions, 
                                        thor_polar.current_position, 
                                        thor_azi.current_position, 
                                        kwargs['stage'], 
                                        kwargs['new_pos'], 
                                        kwargs['abs'],
                                        move_msg])
                
                    ### --- Move Zaber stages to new position --- ###
                    zaber.move(kwargs['new_pos'], kwargs['abs'])

            elif kwargs['stage'] == 'thor_polar':
                if kwargs['move_type'] == "Absolute":
                    target_pos = kwargs['new_angle']  # degrees
                else:
                    target_pos = thor_polar.current_position + kwargs['new_angle']  # degrees

                move_is_safe, move_msg = self.is_move_safe(
                    stage=thor_polar,
                    stage_name=kwargs['stage'], 
                    stage_current_positions=[start_azi_pos, start_polar_pos, start_zaber_pos], 
                    stage_target_position=target_pos,
                    new_pos=kwargs['new_angle'], 
                    move_type=kwargs['move_type']
                )
                if not move_is_safe:
                    kwargs['queue'].put_nowait(['unsafe rotation move', 
                                        zaber.current_positions,
                                        thor_polar.current_position,
                                        thor_azi.current_position,
                                        kwargs['stage'],
                                        kwargs['new_angle'],
                                        kwargs['abs'],
                                        move_msg])
                    return
                else:
                    kwargs['queue'].put_nowait(['start rotation move', 
                                        zaber.current_positions, 
                                        thor_polar.current_position, 
                                        thor_azi.current_position, 
                                        kwargs['stage'], 
                                        kwargs['new_angle'],
                                        kwargs['abs'],
                                        move_msg])

                    thor_polar.move(kwargs['new_angle'], kwargs['abs'])

                    thor_polar.update_positions_callback() # update position
                    pos_polar = thor_polar.current_position # set pos to be position as move is beginning

                    t0 = time.time()
                    last_pos_polar = None

                    # poll until polar stage reaches new position (or timeout)
                    while True:
                        # check position
                        err = abs(self._angle_diff_deg(pos_polar, target_pos))
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
                        thor_polar.update_positions_callback() # update position
                        pos_polar = thor_polar.current_position

                        kwargs['queue'].put_nowait(['in rotation motion to target', 
                                                zaber.current_positions, 
                                                thor_polar.current_position, 
                                                thor_azi.current_position, 
                                                kwargs['stage'], 
                                                kwargs['new_angle'],
                                                kwargs['abs'],
                                                move_msg])

            elif kwargs['stage'] == 'thor_azi':
                if kwargs['move_type'] == "Absolute":
                    target_pos = kwargs['new_angle']  # degrees
                else:
                    target_pos = thor_azi.current_position + kwargs['new_angle']  # degrees

                move_is_safe, move_msg = self.is_move_safe(
                    stage=thor_azi,
                    stage_name=kwargs['stage'], 
                    stage_current_positions=[start_azi_pos, start_polar_pos, start_zaber_pos], 
                    stage_target_position=target_pos,
                    new_pos=kwargs['new_angle'], 
                    move_type=kwargs['move_type']
                )
                if not move_is_safe:
                    kwargs['queue'].put_nowait(['unsafe rotation move', 
                                        zaber.current_positions,
                                        thor_polar.current_position,
                                        thor_azi.current_position,
                                        kwargs['stage'],
                                        kwargs['new_angle'],
                                        kwargs['abs'],
                                        move_msg])
                    return
                else:
                    kwargs['queue'].put_nowait(['start rotation move', 
                                        zaber.current_positions, 
                                        thor_polar.current_position, 
                                        thor_azi.current_position, 
                                        kwargs['stage'], 
                                        kwargs['new_angle'],
                                        kwargs['abs'],
                                        move_msg])

                    thor_azi.move(kwargs['new_angle'], kwargs['abs'])

                    thor_azi.update_positions_callback() # update position
                    pos_azi = thor_azi.current_position # set pos to be position as move is beginning

                    t1 = time.time()
                    last_pos_azi = None

                    # poll until azimuthal stage reaches new position (or timeout)
                    while True:
                        # check position
                        err = abs(self._angle_diff_deg(pos_azi, target_pos))
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
                        thor_azi.update_positions_callback() # update position
                        pos_azi = thor_azi.current_position

                        kwargs['queue'].put_nowait(['in rotation motion to target', 
                                                zaber.current_positions, 
                                                thor_polar.current_position, 
                                                thor_azi.current_position, 
                                                kwargs['stage'], 
                                                kwargs['new_angle'],
                                                kwargs['abs'],
                                                move_msg])
                    
            else:
                logger.info(f"Invalid stage name {kwargs['stage']}.")
                return
        except Exception as e:
            logger.error(f"Error during move_to_orientation for stage {kwargs['stage']}.")
            logger.error(f"Exception: {e}")
            return
        finally:
            time.sleep(0.05)
            # update positions of stages to display in GUI
            zaber.update_positions_callback()
            thor_polar.update_positions_callback()
            thor_azi.update_positions_callback()

            time.sleep(0.05)
            kwargs['queue'].put(['done', 
                                    zaber.current_positions, 
                                    thor_polar.current_position, 
                                    thor_azi.current_position])

            time.sleep(0.1)

    def is_move_safe(self, stage, stage_name, stage_current_positions, stage_target_position, new_pos, move_type=None, out_of_bounds=None, z_min=None, z_max=None, b_target=None):
        """Check if a proposed move is safe based on stage limits and exclusion zones.
        Args:            
            stage: the stage object to be moved (e.g., zaber, thor_polar, thor_azi)
            stage_name: string name of the stage to be moved (e.g., 'zaber', 'thor_polar', 'thor_azi')
            stage_current_positions: list of current positions for all stages [thor_azi, thor_polar, zaber]
            stage_target_position: the proposed new position for the stage being moved (degrees for polar/azimuthal, mm for zaber)
            new_pos: new angle/position or delta angle/position for the proposed move (used for messaging/logging)
            move_type: string reserved for specifying when Zaber stage is being moved to "B Field Target". This gets treated differently.
            out_of_bounds: boolean indicating whether the target position is out of bounds
            z_min: minimum z-position for the target position
            z_max: maximum z-position for the target position
            b_target: the target B field value in Gauss if move_type is "B Field Target"
           
        Returns:
            move_is_safe: boolean indicating whether the proposed move is considered safe based on bounds and exclusion zones
                True if move is safe, False if move is not safe
            msg: string message providing details on the safety check results
        """
        
        move_in_bounds = False  # check if move is within stage limits
        move_in_space = False  # check if move is safe within physical microscope space (e.g., won't cause magnets to crash into objective or sample holder)

        current_positions = {
            'thor_azi': stage_current_positions[0],
            'thor_polar': stage_current_positions[1],
            'zaber': stage_current_positions[2]
        }

        logger.info(f"Checking if move requested for stage {stage_name} is safe...")

        ### --- Specific handling of Zaber moves to B field targets --- ###
        if move_type == "B Field Target":
            if out_of_bounds:
                if stage_target_position < z_min:
                    msg = f"Move not permitted. Target position z = {stage_target_position} mm for move to B = {b_target} G is in exclusion zone (z_min = {z_min} mm) for \
                    \u03c6 = {current_positions['thor_azi']}\N{DEGREE SIGN} & \u03b8 = {current_positions['thor_polar']}\N{DEGREE SIGN}."
                    logger.warning(msg)
                else:
                    msg = f"Move not permitted. Target position z = {stage_target_position} mm for move to B = {b_target} G is in exclusion zone (z_max = {z_max} mm) for \
                    \u03c6 = {current_positions['thor_azi']}\N{DEGREE SIGN} & \u03b8 = {current_positions['thor_polar']}\N{DEGREE SIGN}."
                    logger.warning(msg)
                return False, msg
            else:
                msg = f"Move is safe. Moving to target position z = {stage_target_position} mm for B = {b_target} G..."
                logger.info(msg)
                return True, msg


        ### --- Check if move is within stage bounds --- ###
        if stage_target_position >= stage.lower_bound and stage_target_position <= stage.upper_bound:
            move_in_bounds = True
        else:
            logger.warning(f"Requested move to position {stage_target_position} for stage {stage_name} is out of bounds ({stage.lower_bound} to {stage.upper_bound}).")
        
        ### --- Check if move is safe within physical microscope space --- ###
        ### --- Note: exclusion zone returns True if zone is breached (i.e., move is not safe) and False if move appears safe --- ###
        move_in_space = not exclusion_zone_check(
            current_positions=current_positions,
            target_position=stage_target_position,
            stage_name=stage_name,
            exclusion_zones=None  # TODO: define exclusion zones based on magnet geometry and microscope layout
        )

        ### --- Check if both conditions met for move being safe --- ###
        if move_in_bounds and move_in_space:
            if stage_name == 'zaber':
                msg = f"Move is safe. Moving to target position z = {stage_target_position} mm..."
            elif stage_name == 'thor_polar':
                if move_type == "Absolute":
                    msg = f"Move is safe. Moving to target position \u03b8 = {stage_target_position}\N{DEGREE SIGN}..."
                else:
                    msg = f"Move is safe. Moving by {new_pos}\N{DEGREE SIGN} to new position \u03b8 = {stage_target_position}\N{DEGREE SIGN}..."
            else:
                if move_type == "Absolute":
                    msg = f"Move is safe. Moving to target position \u03c6 = {stage_target_position}\N{DEGREE SIGN}..."
                else:
                    msg = f"Move is safe. Moving by {new_pos}\N{DEGREE SIGN} to new position \u03c6 = {stage_target_position}\N{DEGREE SIGN}..."
            logger.info(msg)
            move_is_safe = True
        else:
            if not move_in_bounds:
                if stage_name == 'zaber':
                    msg = f"Move not permitted. Target position z = {stage_target_position} mm is out of bounds ({stage.lower_bound} to {stage.upper_bound} mm) for Zaber stage."
                elif stage_name == 'thor_polar':
                    if move_type == "Absolute":
                        msg = f"Move not permitted. Target position \u03b8 = {stage_target_position}\N{DEGREE SIGN} is out of bounds ({stage.lower_bound} to {stage.upper_bound} degrees) for Thorlabs polar stage."
                    else:
                        msg = f"Move not permitted. Target position \u03b8 = {stage_target_position}\N{DEGREE SIGN} (after move by {new_pos}\N{DEGREE SIGN}) is out of bounds ({stage.lower_bound} to {stage.upper_bound} degrees) for Thorlabs polar stage."
                else:
                    if move_type == "Absolute":
                        msg = f"Move not permitted. Target position \u03c6 = {stage_target_position}\N{DEGREE SIGN} is out of bounds ({stage.lower_bound} to {stage.upper_bound} degrees) for Thorlabs azimuthal stage."
                    else:
                        msg = f"Move not permitted. Target position \u03c6 = {stage_target_position}\N{DEGREE SIGN} (after move by {new_pos}\N{DEGREE SIGN}) is out of bounds ({stage.lower_bound} to {stage.upper_bound} degrees) for Thorlabs azimuthal stage."
            if not move_in_space:
                msg = f"Move not permitted. May breach exclusion zone & risk collison with microscope components."
            logger.warning(msg)
            move_is_safe = False

        return move_is_safe, msg




    