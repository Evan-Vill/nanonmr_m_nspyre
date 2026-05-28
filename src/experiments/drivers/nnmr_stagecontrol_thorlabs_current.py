'''
Python file for controlling Zaber equipment in NanoNMR experiments

Copyright (c) 2022, Evan Villafranca
All rights reserved.
'''

from ctypes import c_int, c_uint, c_short, c_long, c_bool, byref
from re import L
from thorlabs_kinesis import benchtop_stepper_motor as bsm
import numpy as np

import time

from thorlabs import BSC201

import logging
logger = logging.getLogger(__name__)


class NanoNMRThorlabs:
    '''
    NanoNMR Thorlabs stages motion control
    '''

    def __init__(self, stage_serial_no):
        
        # define stages object to map Zaber stages to their respective axes
        self.Tstage = BSC201(stage_serial_no)
        
        time.sleep(0.5) # short sleep to ensure stage object is fully initialized before proceeding with rest of init function

        self.Tstage.__enter__()

        self.Tstage_serial = stage_serial_no

        if self.Tstage_serial == '40179174':
            'Azimuthal stage parameters'
            self.lower_bound = 0  # angular lower & upper bounds (degrees)
            self.upper_bound = 100 
            self.standby_pos = 100  # standby position (degrees)

        elif self.Tstage_serial == '40251814': 
            'Polar stage parameters'
            self.lower_bound = -50  # angular lower & upper bounds (degrees)
            self.upper_bound = 50
            self.standby_pos = 0  # standby position (degrees)

        else: 
            print("Invalid Thorlabs serial no.")

        # set velocity parameters for movements
        self.current_position = None
        
        self.update_positions_callback() # record positions on instrument server startup
        self.set_vel_params(3, 7) # initialize acceleration = 3 deg/s/s and velocity = 7 deg/s
        
        self.set_zero_backlash(zero_backlash=True) # set zero backlash compensation for initial testing; can be set to False to keep current backlash compensation setting

    def convert_accl_units(self, accl, accl_type = None):
        if accl_type is not None:
            return float(accl)/(2.5/2065)
        else:
            return float(accl)*(2.5/2065)

    def convert_vel_units(self, vel, vel_type = None):
        if vel_type is not None:
            return vel/(6/24189280)
        else:
            return vel*(6/24189280)

    def set_vel_params(self, accl, vel):
        self.accl = int(self.convert_accl_units(accl, 'deg/s/s'))
        self.accl = c_int(self.accl)
        self.max_vel = int(self.convert_vel_units(vel, 'deg/s'))
        self.max_vel = c_int(self.max_vel)
        bsm.SBC_SetVelParams(self.Tstage.serial_no, self.Tstage.channel, self.accl, self.max_vel)

    def request_backlash(self):
        bsm.SBC_RequestBacklash(self.Tstage.serial_no, self.Tstage.channel)
        time.sleep(0.2)
        return bsm.SBC_GetBacklash(self.Tstage.serial_no, self.Tstage.channel)

    def set_zero_backlash(self, zero_backlash=True):
        if not zero_backlash:
            backlash = self.request_backlash()
            print(f"Zero backlash not set. Current backlash: {backlash} microsteps.")
            return

        before = self.request_backlash()
        print(f"Backlash before setting: {before} microsteps.")
        print(f"Backlash before setting: {self.microstep_2_deg(before)} deg.")

        err = bsm.SBC_SetBacklash(self.Tstage.serial_no, self.Tstage.channel, 0)
        print(f"SBC_SetBacklash return code: {err}")

        after = self.request_backlash()
        print(f"Backlash after setting: {after} microsteps.")
        print(f"Backlash after setting: {self.microstep_2_deg(after)} deg.")

    def deg_2_microstep(self, angle_d):
        return int(angle_d/13.3e-6)

    def microstep_2_deg(self, step):
        return step*13.3e-6

    def update_positions_callback(self):
        self.current_position = self.get_position()
        logger.info(f"Thorlabs stage {int(self.Tstage_serial)} updated position = {self.current_position} deg.")
        return round(self.current_position)
    
    def get_position(self):
        # print("TStage type: ", type(self.Tstage.serial_no))
        # print("SERIAL NO: ", self.Tstage.serial_no)
        # print(int(bsm.SBC_GetPosition(self.Tstage.serial_no, self.Tstage.channel)))
        # print(type(int(bsm.SBC_GetPosition(self.Tstage, self.Tstage.channel))))
        return self.microstep_2_deg(int(bsm.SBC_GetPosition(self.Tstage.serial_no, self.Tstage.channel)))

    def enable(self):
        bsm.SBC_EnableChannel(self.Tstage.serial_no, self.Tstage.channel)
        return 0

    def disable(self):
        bsm.SBC_DisableChannel(self.Tstage.serial_no, self.Tstage.channel)
        return 0

    def stop_motion(self):
        '''
        Stop motion of Thorlabs stage
        '''
        bsm.SBC_StopImmediate(self.Tstage.serial_no, self.Tstage.channel)

    def check_standby(self):
        '''
        Standby for Thorlabs stages is when:
        1) AZI = 100 deg position
        2) POLAR = 0 deg position
        
        This function checks whether the polar & azimuthal stages are at their 
        correct respective standby positions.
        '''
        logger.info(f"Checking if Thorlabs stage {int(self.Tstage_serial)} set in Standby Mode...")

        if self.current_position is None:
            self.current_position = self.get_position()
      
        if (round(self.current_position) == self.standby_pos):
            standby_condition_met = True
            logger.info(f"Checked: Thorlabs stage {int(self.Tstage_serial)} set in Standby Mode.")
        else:
            standby_condition_met = False
            logger.info(f"Checked: Thorlabs stage {int(self.Tstage_serial)} NOT set in Standby Mode.")

        return standby_condition_met

    def standby(self):
        '''
        Move Thorlabs stages to Standby Mode.
        '''

        logger.info(f"Moving Thorlabs stage {int(self.Tstage_serial)} to Standby Mode...")

        # move to standby positions
        bsm.SBC_MoveToPosition(self.Tstage.serial_no, self.Tstage.channel, c_int(self.deg_2_microstep(self.standby_pos)))
        logger.info(f"Thorlabs stage {int(self.Tstage_serial)} set in Standby Mode.")
    
    # def check_homed(self):
    #     homed = True
    #     if bsm.SBC_NeedsHoming(self.Tstage.serial_no, self.Tstage.channel):
    #         homed = False

    #     # for stage in self.Tstage:
    #     #     if bsm.SBC_NeedsHoming(self.Tstage[stage], self.Tstage.channel):
    #     #         all_homed = False
        
    #     return homed 

    def home(self):
        '''
        Home a Thorlabs stage. 

        CANNOT be done when stage is positioned at negative angle
        '''
        if self.current_position is None:
            self.current_position = self.get_position()

        condition = ["home", self.current_position]
        print("HOMING CONDITION: ", condition[1])
        if not self.is_move_safe(condition):
            logger.warning(f"WARNING: Homing from the current position {round(condition[1], 3)}\N{DEGREE SIGN} is NOT safe for Thorlabs stage {int(self.Tstage_serial)}. Moving to safe position to proceed with homing.")
            bsm.SBC_MoveToPosition(self.Tstage.serial_no, self.Tstage.channel, c_int(self.deg_2_microstep(0))) # moves stage to +1 deg. position for safe homing
            return 

        logger.info(f"Homing Thorlabs stage {int(self.Tstage_serial)}...")
        time.sleep(0.2)
        # setting the velocity in init function wasn't working
        bsm.SBC_SetHomingVelocity(self.Tstage.serial_no, self.Tstage.channel, c_uint(10000000))
        bsm.SBC_Home(self.Tstage.serial_no, self.Tstage.channel)

    def move(self, angle, abs = False):
        '''
        Move a Thorlabs rotation stage to either an absolute or relative new position.
        '''
        condition = ["move", None]
        if abs:
            condition[1] = angle
        else:
            if self.current_position is None:
                self.current_position = self.get_position()
            condition[1] = self.current_position + angle

        if not self.is_move_safe(condition):
            logger.warning(f"WARNING: The movement to {condition[1]}\N{DEGREE SIGN} is NOT safe (out of bounds for Thorlabs stage {int(self.Tstage_serial)})")
        else:
            if abs:
                logger.info(f"Moving Thorlabs stage {int(self.Tstage_serial)} to {angle} deg...")
                bsm.SBC_MoveToPosition(self.Tstage.serial_no, self.Tstage.channel, c_int(self.deg_2_microstep(angle)))
                logger.info(f"Thorlabs stage {int(self.Tstage_serial)} set to {angle} deg.")
            else:
                logger.info(f"Moving Thorlabs stage {int(self.Tstage_serial)} to {round(condition[1], 3)} deg...")
                bsm.SBC_MoveRelative(self.Tstage.serial_no, self.Tstage.channel, c_int(self.deg_2_microstep(angle)))
                logger.info(f"Thorlabs stage {int(self.Tstage_serial)} set to {round(condition[1], 3)} deg.")

        bsm.SBC_RequestBacklash(self.Tstage.serial_no, self.Tstage.channel) # request backlash compensation after move
        print(f"Backlash compensation requested for stage {int(self.Tstage_serial)}.")
        print(f"Current backlash setting: {bsm.SBC_GetBacklash(self.Tstage.serial_no, self.Tstage.channel)} microsteps.")
        print(f"Current backlash in degrees: {self.microstep_2_deg(bsm.SBC_GetBacklash(self.Tstage.serial_no, self.Tstage.channel))} deg.")
        
    def is_move_safe(self, condition = [None, None]):
        '''
        Check if requested movement command is safe 
        (i.e., stage stays w/in safe range).
        --> Works by comparing a condition (final angle destination of stage after movement)
        to the defined angular bounds for a given stage.

        For checking if homing is safe, relevant only for polar stage.
        --> Works by checking if stage is at a negative angle with respect to its defined zero.
        Forces movement to positive 1 degree before homing if starting from negative angle.
        '''


        is_safe = False # default assumes move is not safe for precaution

        if condition[0] == "move":
            logger.info(f"Checking if move requested for stage {int(self.Tstage_serial)} is safe...")
            if condition[1] is not None:
                if condition[1] >= self.lower_bound and condition[1] <= self.upper_bound:
                    is_safe = True
            else:
                raise ValueError(f"Safety check condition must be supplied. None given.")

        elif condition[0] == "home":
            logger.info(f"Checking if homing requested for stage {int(self.Tstage_serial)} is safe...")
            if condition[1] is not None:
                if condition[1] >= 0:
                    is_safe = True
                print("SAFE TO GO HOME? ", is_safe)
            else:
                raise ValueError(f"Safety check condition must be supplied. None given.")
        
        else: 
            raise ValueError(f"Invalid safety check condition. Acceptable options are either 'move' or 'home'.")

        return is_safe

    def close(self):
        self.Tstage.__exit__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

if __name__ == '__main__':
    # enable logging to console
    logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s', datefmt = '%m-%d-%Y %H:%M:%S')

    with NanoNMRThorlabs() as thorlabs:
        pass