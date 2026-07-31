"""
Driver for a Thorlabs KDC101 DC servo controller with an MTS50-Z8 stage.

Positions, distances, velocities, and accelerations in this wrapper are
expressed in real-world MTS50-Z8 units: mm, mm/s, and mm/s/s.
"""

from ctypes import byref, c_char_p, c_double, c_int, c_long, c_uint
import logging
import time

from thorlabs_kinesis import KCube_DC_Servo as kcube

logger = logging.getLogger(__name__)


class KDC101:
    """NanoNMR driver wrapper for a KDC101 controlling an MTS50-Z8 linear stage."""

    POSITION_UNIT = 0
    VELOCITY_UNIT = 1
    ACCELERATION_UNIT = 2

    MIN_POSITION_MM = 0.0
    MAX_POSITION_MM = 50.0
    DEFAULT_STANDBY_POSITION_MM = 0.0
    POSITION_TOLERANCE_MM = 0.01

    MOVING_STATUS_BITS = 0x00000010 | 0x00000020 | 0x00000040 | 0x00000080
    HOMING_STATUS_BIT = 0x00000200
    HOMED_STATUS_BIT = 0x00000400

    def __init__(
        self,
        serial_id,
        poll_interval_ms=100,
        lower_bound_mm=MIN_POSITION_MM,
        upper_bound_mm=MAX_POSITION_MM,
        standby_position_mm=DEFAULT_STANDBY_POSITION_MM,
        zero_backlash_on_start=True,
        enforce_limits=True,
    ):
        self.stage_serial_id = str(serial_id)
        self.serial_no = c_char_p(bytes(self.stage_serial_id, "utf-8"))
        self.poll_interval_ms = int(poll_interval_ms)
        self.lower_bound_mm = float(lower_bound_mm)
        self.upper_bound_mm = float(upper_bound_mm)
        self.standby_position_mm = float(standby_position_mm)
        self.enforce_limits = bool(enforce_limits)
        self.current_position = None
        self._is_open = False

        if self.lower_bound_mm >= self.upper_bound_mm:
            raise ValueError("lower_bound_mm must be less than upper_bound_mm")

        self._check_error("TLI_BuildDeviceList", kcube.TLI_BuildDeviceList())
        logger.debug(f"Found {kcube.TLI_GetDeviceListSize()} Thorlabs devices.")

        self._check_error("CC_Open", kcube.CC_Open(self.serial_no))
        self._is_open = True

        try:
            self.hw_info = self.get_hardware_info()
            self._check_true("CC_LoadSettings", kcube.CC_LoadSettings(self.serial_no))
            self._check_error("CC_RequestSettings", kcube.CC_RequestSettings(self.serial_no))
            self._check_error(
                "CC_SetMotorTravelMode",
                kcube.CC_SetMotorTravelMode(self.serial_no, kcube.MOT_Linear),
            )
            self._check_true(
                "CC_StartPolling",
                kcube.CC_StartPolling(self.serial_no, self.poll_interval_ms),
            )
            time.sleep(1)

            self.enable()
            time.sleep(1)

            if zero_backlash_on_start:
                self.set_zero_backlash()

            self.update_positions_callback()
            logger.debug(
                f"Initialized KDC101/MTS50-Z8 stage with serial no. "
                f"{int(self.stage_serial_id)}."
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check_error(label, error_code):
        if error_code != 0:
            raise RuntimeError(f"{label} failed with error code {error_code}")

    @staticmethod
    def _check_true(label, ok):
        if not ok:
            raise RuntimeError(f"{label} failed")

    def _assert_ready_for_motion(self):
        if self.needs_homing():
            raise RuntimeError(
                "KDC101 reports that the MTS50-Z8 must be homed before motion. "
                "Call home() first."
            )

    def _assert_position_safe(self, position_mm):
        if not self.enforce_limits:
            return

        if not self.is_position_safe(position_mm):
            raise ValueError(
                f"Requested position {position_mm} mm is outside the configured "
                f"MTS50-Z8 range [{self.lower_bound_mm}, {self.upper_bound_mm}] mm."
            )

    def _real_to_device_units(self, value, unit_type):
        device_units = c_int()
        self._check_error(
            "CC_GetDeviceUnitFromRealValue",
            kcube.CC_GetDeviceUnitFromRealValue(
                self.serial_no,
                c_double(float(value)),
                byref(device_units),
                unit_type,
            ),
        )
        return device_units.value

    def _device_units_to_real(self, device_units, unit_type):
        real_value = c_double()
        self._check_error(
            "CC_GetRealValueFromDeviceUnit",
            kcube.CC_GetRealValueFromDeviceUnit(
                self.serial_no,
                int(device_units),
                byref(real_value),
                unit_type,
            ),
        )
        return real_value.value

    def mm_to_device_units(self, position_mm):
        return self._real_to_device_units(position_mm, self.POSITION_UNIT)

    def device_units_to_mm(self, device_units):
        return self._device_units_to_real(device_units, self.POSITION_UNIT)

    def velocity_to_device_units(self, velocity_mm_s):
        return self._real_to_device_units(velocity_mm_s, self.VELOCITY_UNIT)

    def device_units_to_velocity(self, device_units):
        return self._device_units_to_real(device_units, self.VELOCITY_UNIT)

    def acceleration_to_device_units(self, acceleration_mm_s2):
        return self._real_to_device_units(acceleration_mm_s2, self.ACCELERATION_UNIT)

    def device_units_to_acceleration(self, device_units):
        return self._device_units_to_real(device_units, self.ACCELERATION_UNIT)

    def get_hardware_info(self):
        hw_info = kcube.TLI_HardwareInformation()
        self._check_error(
            "CC_GetHardwareInfoBlock",
            kcube.CC_GetHardwareInfoBlock(self.serial_no, byref(hw_info)),
        )
        return hw_info

    def identify(self):
        kcube.CC_Identify(self.serial_no)
        return 0

    def update_positions_callback(self):
        self.current_position = self.get_position()
        logger.info(
            f"KDC101/MTS50-Z8 stage {int(self.stage_serial_id)} updated "
            f"position = {self.current_position} mm."
        )
        return self.current_position

    def get_position(self):
        self._check_error("CC_RequestPosition", kcube.CC_RequestPosition(self.serial_no))
        time.sleep(0.1)
        return self.device_units_to_mm(kcube.CC_GetPosition(self.serial_no))

    def get_position_device_units(self):
        self._check_error("CC_RequestPosition", kcube.CC_RequestPosition(self.serial_no))
        time.sleep(0.1)
        return kcube.CC_GetPosition(self.serial_no)

    def get_encoder_counter(self):
        self._check_error(
            "CC_RequestEncoderCounter",
            kcube.CC_RequestEncoderCounter(self.serial_no),
        )
        time.sleep(0.1)
        return kcube.CC_GetEncoderCounter(self.serial_no)

    def zero_position_counter(self):
        self._check_error(
            "CC_SetPositionCounter",
            kcube.CC_SetPositionCounter(self.serial_no, c_long(0)),
        )
        self.update_positions_callback()
        return self.current_position

    def needs_homing(self):
        return bool(kcube.CC_NeedsHoming(self.serial_no))

    def can_home(self):
        return bool(kcube.CC_CanHome(self.serial_no))

    def can_move_without_homing(self):
        return bool(kcube.CC_CanMoveWithoutHomingFirst(self.serial_no))

    def get_status_bits(self):
        self._check_error(
            "CC_RequestStatusBits",
            kcube.CC_RequestStatusBits(self.serial_no),
        )
        time.sleep(0.1)
        return kcube.CC_GetStatusBits(self.serial_no)

    def is_moving(self):
        return bool(self.get_status_bits() & self.MOVING_STATUS_BITS)

    def is_homing(self):
        return bool(self.get_status_bits() & self.HOMING_STATUS_BIT)

    def is_homed(self):
        return bool(self.get_status_bits() & self.HOMED_STATUS_BIT)

    def is_position_safe(self, position_mm):
        return self.lower_bound_mm <= float(position_mm) <= self.upper_bound_mm

    def check_standby(self, tolerance_mm=0.01):
        position = self.update_positions_callback()
        return abs(position - self.standby_position_mm) <= tolerance_mm

    def get_stage_axis_limits(self):
        min_units = kcube.CC_GetStageAxisMinPos(self.serial_no)
        max_units = kcube.CC_GetStageAxisMaxPos(self.serial_no)
        return (
            self.device_units_to_mm(min_units),
            self.device_units_to_mm(max_units),
        )

    def set_stage_axis_limits(self, lower_mm=MIN_POSITION_MM, upper_mm=MAX_POSITION_MM):
        if float(lower_mm) >= float(upper_mm):
            raise ValueError("lower_mm must be less than upper_mm")

        lower_units = self.mm_to_device_units(lower_mm)
        upper_units = self.mm_to_device_units(upper_mm)
        self._check_error(
            "CC_SetStageAxisLimits",
            kcube.CC_SetStageAxisLimits(self.serial_no, lower_units, upper_units),
        )
        self.lower_bound_mm = float(lower_mm)
        self.upper_bound_mm = float(upper_mm)
        return self.get_stage_axis_limits()

    def get_motor_travel_limits(self):
        lower = c_double()
        upper = c_double()
        self._check_error(
            "CC_GetMotorTravelLimits",
            kcube.CC_GetMotorTravelLimits(self.serial_no, byref(lower), byref(upper)),
        )
        return lower.value, upper.value

    def set_motor_travel_limits(self, lower_mm=MIN_POSITION_MM, upper_mm=MAX_POSITION_MM):
        if float(lower_mm) >= float(upper_mm):
            raise ValueError("lower_mm must be less than upper_mm")

        self._check_error(
            "CC_SetMotorTravelLimits",
            kcube.CC_SetMotorTravelLimits(
                self.serial_no,
                c_double(float(lower_mm)),
                c_double(float(upper_mm)),
            ),
        )
        self.lower_bound_mm = float(lower_mm)
        self.upper_bound_mm = float(upper_mm)
        return self.get_motor_travel_limits()

    def request_backlash(self):
        self._check_error("CC_RequestBacklash", kcube.CC_RequestBacklash(self.serial_no))
        time.sleep(0.2)
        return kcube.CC_GetBacklash(self.serial_no)

    def get_backlash(self):
        device_units = self.request_backlash()
        return self.device_units_to_mm(device_units)

    def set_backlash(self, backlash_mm):
        device_units = self.mm_to_device_units(backlash_mm)
        self._check_error(
            "CC_SetBacklash",
            kcube.CC_SetBacklash(self.serial_no, c_long(device_units)),
        )
        return self.get_backlash()

    def set_zero_backlash(self):
        before = self.request_backlash()
        self._check_error(
            "CC_SetBacklash",
            kcube.CC_SetBacklash(self.serial_no, c_long(0)),
        )
        after = self.request_backlash()
        logger.info(
            f"KDC101/MTS50-Z8 stage {int(self.stage_serial_id)} backlash set "
            f"from {before} to {after} device units."
        )
        return after

    def get_velocity_params(self):
        self._check_error(
            "CC_RequestVelParams",
            kcube.CC_RequestVelParams(self.serial_no),
        )
        time.sleep(0.1)

        acceleration = c_int()
        max_velocity = c_int()
        self._check_error(
            "CC_GetVelParams",
            kcube.CC_GetVelParams(
                self.serial_no,
                byref(acceleration),
                byref(max_velocity),
            ),
        )
        return {
            "acceleration_mm_s2": self.device_units_to_acceleration(
                acceleration.value
            ),
            "max_velocity_mm_s": self.device_units_to_velocity(max_velocity.value),
            "acceleration_device_units": acceleration.value,
            "max_velocity_device_units": max_velocity.value,
        }

    def set_velocity_params(self, acceleration_mm_s2, max_velocity_mm_s):
        if acceleration_mm_s2 <= 0 or max_velocity_mm_s <= 0:
            raise ValueError("Velocity and acceleration must be positive")

        acceleration = self.acceleration_to_device_units(acceleration_mm_s2)
        max_velocity = self.velocity_to_device_units(max_velocity_mm_s)
        self._check_error(
            "CC_SetVelParams",
            kcube.CC_SetVelParams(self.serial_no, acceleration, max_velocity),
        )
        return self.get_velocity_params()

    def get_homing_velocity(self):
        self._check_error(
            "CC_RequestHomingParams",
            kcube.CC_RequestHomingParams(self.serial_no),
        )
        time.sleep(0.1)
        return self.device_units_to_velocity(kcube.CC_GetHomingVelocity(self.serial_no))

    def set_homing_velocity(self, velocity_mm_s):
        if velocity_mm_s <= 0:
            raise ValueError("Homing velocity must be positive")

        velocity = self.velocity_to_device_units(velocity_mm_s)
        self._check_error(
            "CC_SetHomingVelocity",
            kcube.CC_SetHomingVelocity(self.serial_no, c_uint(velocity)),
        )
        return self.get_homing_velocity()

    def enable(self):
        self._check_error("CC_EnableChannel", kcube.CC_EnableChannel(self.serial_no))
        return 0

    def disable(self):
        self._check_error("CC_DisableChannel", kcube.CC_DisableChannel(self.serial_no))
        return 0

    def home(self, wait=True, timeout_s=60):
        if not self.can_home():
            raise RuntimeError("KDC101 reports that this stage cannot home.")

        logger.info(f"Homing KDC101/MTS50-Z8 stage {int(self.stage_serial_id)}...")
        self._check_error("CC_Home", kcube.CC_Home(self.serial_no))
        if wait:
            self._wait_for_home(timeout_s=timeout_s)
            self.update_positions_callback()
        return 0

    def move_to_position(self, position_mm, wait=True, timeout_s=30):
        self._assert_ready_for_motion()
        self._assert_position_safe(position_mm)

        target = self.mm_to_device_units(position_mm)
        logger.info(
            f"Moving KDC101/MTS50-Z8 stage {int(self.stage_serial_id)} "
            f"to {position_mm} mm."
        )
        self._check_error(
            "CC_MoveToPosition",
            kcube.CC_MoveToPosition(self.serial_no, target),
        )

        if wait:
            self._wait_for_position(target, timeout_s=timeout_s)

        self.update_positions_callback()
        return self.current_position

    def move_to(self, position_mm, wait=True, timeout_s=30):
        return self.move_to_position(position_mm, wait=wait, timeout_s=timeout_s)

    def move_relative(self, distance_mm, wait=True, timeout_s=30):
        self._assert_ready_for_motion()

        start = self.get_position_device_units()
        delta = self.mm_to_device_units(distance_mm)
        target = start + delta
        target_mm = self.device_units_to_mm(target)
        self._assert_position_safe(target_mm)

        logger.info(
            f"Moving KDC101/MTS50-Z8 stage {int(self.stage_serial_id)} by "
            f"{distance_mm} mm to {target_mm} mm."
        )
        self._check_error(
            "CC_MoveRelative",
            kcube.CC_MoveRelative(self.serial_no, delta),
        )

        if wait:
            self._wait_for_position(target, timeout_s=timeout_s)

        self.update_positions_callback()
        return self.current_position

    def jog_position(self, distance_mm, wait=True, timeout_s=30):
        return self.move_relative(distance_mm, wait=wait, timeout_s=timeout_s)

    def move(self, position_or_distance_mm, absolute=False, wait=True, timeout_s=30):
        if absolute:
            return self.move_to_position(
                position_or_distance_mm,
                wait=wait,
                timeout_s=timeout_s,
            )
        return self.move_relative(
            position_or_distance_mm,
            wait=wait,
            timeout_s=timeout_s,
        )

    def standby(self, wait=True, timeout_s=30):
        return self.move_to_position(
            self.standby_position_mm,
            wait=wait,
            timeout_s=timeout_s,
        )

    def move_to_min(self, wait=True, timeout_s=30):
        return self.move_to_position(
            self.lower_bound_mm,
            wait=wait,
            timeout_s=timeout_s,
        )

    def move_to_max(self, wait=True, timeout_s=30):
        return self.move_to_position(
            self.upper_bound_mm,
            wait=wait,
            timeout_s=timeout_s,
        )

    def move_to_center(self, wait=True, timeout_s=30):
        center_mm = (self.lower_bound_mm + self.upper_bound_mm) / 2.0
        return self.move_to_position(center_mm, wait=wait, timeout_s=timeout_s)

    def move_at_velocity(self, direction):
        self._assert_ready_for_motion()
        direction_value = self._direction_to_kinesis_value(direction)
        self._check_error(
            "CC_MoveAtVelocity",
            kcube.CC_MoveAtVelocity(self.serial_no, direction_value),
        )
        return 0

    def jog_forward(self):
        return self.move_at_velocity("forward")

    def jog_reverse(self):
        return self.move_at_velocity("reverse")

    def stop_motion(self, profiled=False):
        if profiled:
            self._check_error("CC_StopProfiled", kcube.CC_StopProfiled(self.serial_no))
        else:
            self._check_error("CC_StopImmediate", kcube.CC_StopImmediate(self.serial_no))
        self.update_positions_callback()
        return 0

    def _direction_to_kinesis_value(self, direction):
        if isinstance(direction, str):
            cleaned = direction.strip().lower()
            if cleaned in ("forward", "forwards", "positive", "+", "fwd"):
                return kcube.MOT_Forwards
            if cleaned in ("reverse", "backward", "backwards", "negative", "-", "rev"):
                return kcube.MOT_Reverse

        if direction in (kcube.MOT_Forwards, kcube.MOT_Reverse):
            return direction

        raise ValueError("direction must be 'forward' or 'reverse'")

    def _wait_for_position(
        self,
        target_device_units,
        timeout_s=30,
        tolerance_mm=POSITION_TOLERANCE_MM,
    ):
        tolerance_units = max(1, abs(self.mm_to_device_units(tolerance_mm)))
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            position = self.get_position_device_units()
            if abs(position - target_device_units) <= tolerance_units:
                return position
            time.sleep(0.2)

        target_mm = self.device_units_to_mm(target_device_units)
        raise TimeoutError(
            f"Timed out waiting for KDC101/MTS50-Z8 stage "
            f"{int(self.stage_serial_id)} to reach {target_mm} mm "
            f"({target_device_units} device units)."
        )

    def _wait_for_home(self, timeout_s=60):
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            status_bits = self.get_status_bits()
            homing = bool(status_bits & self.HOMING_STATUS_BIT)
            homed = bool(status_bits & self.HOMED_STATUS_BIT)
            if not homing and (homed or not self.needs_homing()):
                return status_bits
            time.sleep(0.2)

        raise TimeoutError(
            f"Timed out waiting for KDC101/MTS50-Z8 stage "
            f"{int(self.stage_serial_id)} to finish homing."
        )

    def close(self):
        if self._is_open:
            kcube.CC_StopPolling(self.serial_no)
            kcube.CC_Close(self.serial_no)
            self._is_open = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s.%(msecs)03d [%(levelname)8s] %(message)s",
#         datefmt="%m-%d-%Y %H:%M:%S",
#     )
#
#     with KDC101("27250000") as stage:
#         print(f"Needs homing: {stage.needs_homing()}")
#         print(f"Current position: {stage.current_position} mm")
#         stage.home()
#         stage.move_to_position(25.0)
#         stage.move_relative(-5.0)
