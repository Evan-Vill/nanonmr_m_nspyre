"""
Copyright (c) 2021, Jacob Feder
All rights reserved.

This work is licensed under the terms of the 3-Clause BSD license.
For a copy, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

"""
Magnet Mount Motion and Magnet Alignment GUI
Evan Villafranca - 2022
"""

from importlib import reload
import logging
from multiprocessing import Queue

from pyqtgraph import ComboBox, SpinBox
from PyQt6.QtCore import Qt, QTimer, QSignalBlocker
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

import nnmr_magnet_motion as mag
from find_magnet_position import find_mag_pos

from nspyre import ParamsWidget, ProcessRunner, InstrumentManager

# --- Qt validity helper for PyQt6 ---
try:
    from shiboken6 import isValid as _pyside_is_valid

    def _alive(w):
        """Return True if widget is still valid (not deleted)."""
        return (w is not None) and _pyside_is_valid(w)

except Exception:

    def _alive(w):
        """Fallback: be conservative if validity check is unavailable."""
        return False if w is None else True


logger = logging.getLogger(__name__)


class InstWidget(QWidget):
    """Qt widget subclass that generates an interface for operating magnet mount."""

    QUEUE_CHECK_TIME = 100  # ms
    EQUIP_STATUS_CHECK_TIME = 500  # ms
    GUI_OWNER_PREFIX = "GUI_"

    def __init__(self):
        super().__init__()

        self._mgr = InstrumentManager()
        self._mgr.__enter__()  # open once, close on widget destruction

        self.setWindowTitle("NanoNMR Magnet Alignment")

        self._last_rpc_err = {}  # keep track of last RPC errors. Context -> msg

        # magnet status queue handling
        self.updateTimer = QTimer(self)  # timer to update widget from queue
        self.updateTimer.timeout.connect(self.check_queue_from_mag)
        self.updateTimer.start(self.QUEUE_CHECK_TIME)

        # hardware status update handling
        self.hwTimer = QTimer(self)
        self.hwTimer.timeout.connect(self.get_laser_status)
        self.hwTimer.timeout.connect(self.get_interlock_status)
        self.hwTimer.timeout.connect(self.check_ps_status)
        self.hwTimer.timeout.connect(self.check_laser_shutter_status)
        self.hwTimer.timeout.connect(self.check_laser_temp_status)
        self.hwTimer.start(self.EQUIP_STATUS_CHECK_TIME)

        self._gui_id = "GUI_Instruments"

        # current stage positions
        self.curr_r = 0
        self.curr_theta = 0
        self.curr_phi = 0
        self.r_offset = 0

        self.r_min = 0
        self.r_max = 100
        self.theta_min = -50
        self.theta_max = 50
        self.phi_min = 0
        self.phi_max = 100

        self.white_border = "border: 1px solid white"
        self.font = "Arial"

        self.q = Queue()

        # magnet stage widgets
        self.magnet_label = QLabel("Magnet Stage Control")
        self.magnet_label.setFixedHeight(30)
        self.magnet_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.magnet_label.setFont(QFont(self.font, 20))

        all_disable_button_style = """
        QPushButton {
        background-color: #333333;
        color: white;
        border: 2px solid white;
        border-radius: 5px;
        padding: 5px;
        }

        QPushButton:hover {
        background-color: #444444;
        border: 2px solid #ffffff;
        }

        QPushButton:pressed {
        background-color: #555555;
        border: 2px solid #ffffff;
        }"""
        self.all_disable_button = QPushButton("Enable All")
        self.all_disable_button.setStyleSheet(all_disable_button_style)
        self.all_disable_proc = ProcessRunner()
        self.all_disable_button.clicked.connect(self.all_disable_clicked)

        all_standby_button_style = """
        QPushButton {
        background-color: #474b00;
        color: white;
        border: 2px solid #d7d700;
        border-radius: 5px;
        padding: 5px;
        }
        
        QPushButton:hover {
        background-color: #3f4b00;
        border: 2px solid #00d7c9;
        }
        
        QPushButton:pressed {
        background-color: #4f5f00;
        border: 2px solid #00d7c9;
        }"""
        self.all_standby_button = QPushButton("Standby")
        self.all_standby_button.setStyleSheet(all_standby_button_style)
        self.all_standby_proc = ProcessRunner()
        self.all_standby_button.clicked.connect(self.all_standby_clicked)

        self.status_label = QLabel("Magnets status here")
        self.status_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.status_label.setFixedHeight(40)

        # initialize widgets
        self.init_b_widgets()
        self.init_r_widgets()
        self.init_polar_widgets()
        self.init_azi_widgets()
        self.init_sg396_widgets()
        self.init_laser_widgets()
        self.init_flipper_widgets()
        self.init_nd_filter_widgets()
        self.init_bpd_shutter_widgets()

        self.layouts()
        self.park_all()

        # one-time immediate refresh after UI is live
        QTimer.singleShot(0, self._initial_refresh)

    # for initial refresh and setting hardware state upon GUI open
    def _set_radio_safely(self, rb, checked: bool):
        blocker = QSignalBlocker(rb)
        rb.setChecked(checked)
        del blocker

    def _initial_refresh(self):
        # skip if widget is already closing
        if getattr(self, "_closing", False):
            return

        # get microscope excitation mode (flip mirror 1 state) upon startup
        self.call_daq(lambda daq: daq.open_do_task("flip mirror"))
        self.call_daq(lambda daq: daq.start_do_task())
        microscope_excitation_mode_is_tir = self.call_daq(lambda daq: daq.read_do_task())
        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())
        if microscope_excitation_mode_is_tir:
            self._set_radio_safely(self.optics_flipper_b2, True)
        else:
            self._set_radio_safely(self.optics_flipper_b1, True)

        # hardware + UI sync (ONE TIME)
        self.get_laser_status()
        self.get_interlock_status()
        self.check_ps_status()
        self.check_laser_shutter_status()
        self.check_laser_temp_status()

        # get microscope detector mode (flip mirror 2 state) upon startup
        self.call_daq(lambda daq: daq.open_do_task("flip mirror 2"))
        self.call_daq(lambda daq: daq.start_do_task())
        microscope_detector_mode_is_apd = self.call_daq(lambda daq: daq.read_do_task())
        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())
        if microscope_detector_mode_is_apd:
            self._set_radio_safely(self.flipper_b1, True)
        else:
            self._set_radio_safely(self.flipper_b2, True)

        # get BPD shutter state upon startup
        self.bpd_shutter_status_read()  # this will query hardware and update button + label accordingly

        # magnet state
        self.check_queue_from_mag()

    def _reset_mgr(self):
        # Don't do anything while closing
        if getattr(self, "_closing", False):
            return

        """Tear down and recreate the InstrumentManager connection."""
        try:
            self._mgr.__exit__(None, None, None)
        except Exception:
            pass
        self._mgr = InstrumentManager()
        self._mgr.__enter__()

    def _call(self, fn, *, context=""):
        """Safe RPC wrapper used by call_ps / call_laser / etc."""
        try:
            return fn()
        except Exception:
            # first failure: rebuild manager and retry once
            self._reset_mgr()
            try:
                return fn()
            except Exception as e2:
                # second failure: log once, return None, do NOT spam
                msg = f"{type(e2).__name__}: {e2}"
                tag = f"[RPC {context}]" if context else "[RPC]"
                if self._last_rpc_err.get(context) != msg:
                    self._last_rpc_err[context] = msg
                    print(f"{tag} {msg}")
                return None

    @property
    def ps(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.ps

    def call_ps(self, fn):
        return self._call(lambda: fn(self.ps), context="ps")

    @property
    def laser(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.laser

    def call_laser(self, fn):
        return self._call(lambda: fn(self.laser), context="laser")

    @property
    def sg(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.sg

    def call_sg(self, fn):
        return self._call(lambda: fn(self.sg), context="sg")

    @property
    def awg(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.awg

    def call_awg(self, fn):
        return self._call(lambda: fn(self.awg), context="awg")

    @property
    def daq(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.daq

    def call_daq(self, fn):
        return self._call(lambda: fn(self.daq), context="daq")

    @property
    def filter_wheel(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.filter_wheel

    def call_filter_wheel(self, fn):
        return self._call(lambda: fn(self.filter_wheel), context="filter wheel")

    @property
    def laser_shutter(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.laser_shutter

    def call_laser_shutter(self, fn):
        return self._call(lambda: fn(self.laser_shutter), context="laser shutter")

    def closeEvent(self, event):
        self._closing = True

        # stop timers first so no more slots run while widgets are being destroyed
        for tname in ("updateTimer", "hwTimer"):
            t = getattr(self, tname, None)
            if t is None:
                continue
            try:
                if t.isActive():
                    t.stop()
            except Exception:
                pass
            try:
                t.timeout.disconnect()
            except Exception:
                pass

        # close InstrumentManager last
        try:
            if getattr(self, "_mgr", None) is not None:
                self._mgr.__exit__(None, None, None)
        except Exception:
            pass

        super().closeEvent(event)

    def set_section_enabled(self, widget, enabled: bool, *, dim_opacity: float = 0.3):
        """
        widget: a QFrame/QWidget that contains the controls you want to gray out.
        enabled=False => disables all children and dims the whole section.
        """
        widget.setEnabled(enabled)

        eff = widget.graphicsEffect()
        if eff is None or not isinstance(eff, QGraphicsOpacityEffect):
            eff = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(eff)

        eff.setOpacity(1.0 if enabled else dim_opacity)

    # CREATE WIDGETS
    def init_b_widgets(self):
        """B field widgets"""
        self.b_label = QLabel("B Field (G)")
        self.b_label.setFixedHeight(30)
        self.b_label.setFont(QFont(self.font, 20))

        self.phi_label = QLabel("Set \u03c6: ")
        self.phi_label.setFixedHeight(30)
        self.phi_label.setFont(QFont(self.font, 20))

        self.phi_position = QLineEdit()
        self.phi_position.setStyleSheet("QLineEdit { border: 1px solid #666666; }")

        self.target_b_label = QLabel("B Field (G): ")
        self.target_b_label.setFixedHeight(30)
        self.target_b_label.setFont(QFont(self.font, 20))

        self.target_b_value = QLineEdit()
        self.target_b_value.setStyleSheet("QLineEdit { border: 1px solid #666666; }")

        self.b_execute_button = QPushButton("Go To B Field")
        self.b_execute_button_proc = ProcessRunner()
        self.b_execute_button.clicked.connect(
            lambda: self.single_move_clicked("B_zaber")
        )

    def init_r_widgets(self):
        """r stage widgets"""
        self.r_opacity_effects = []
        for _ in range(4):
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.3)
            self.r_opacity_effects.append(effect)

        self.r_label = QLabel("Step 3: R")
        self.r_label.setFixedHeight(30)
        self.r_label.setFont(QFont(self.font, 20))

        self.r_move_checkbox = QCheckBox("R Move Type: ")
        self.r_move_checkbox.setChecked(False)
        self.r_move_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.r_move_checkbox.stateChanged.connect(
            lambda: self.move_button_checked(self.r_move_checkbox)
        )

        self.r_move_types = QComboBox()
        self.r_move_types.addItems(["Abs.", "Rel.", "B"])
        self.r_move_types.currentIndexChanged.connect(self.r_move_type_changed)
        self.r_move_option = "Abs."

        self.r_move_position = QLineEdit()
        self.r_move_position_units = QLabel("")
        self.r_execute_move_button = QPushButton("Execute Move")
        self.r_execute_move_button_proc = ProcessRunner()
        self.r_execute_move_button.clicked.connect(
            lambda: self.single_move_clicked("zaber")
        )

        self.r_move_types.setEnabled(False)
        self.r_move_position.setEnabled(False)
        self.r_move_position_units.setEnabled(False)
        self.r_execute_move_button.setEnabled(False)

        self.r_move_types.setGraphicsEffect(self.r_opacity_effects[0])
        self.r_move_position.setGraphicsEffect(self.r_opacity_effects[1])
        self.r_move_position_units.setGraphicsEffect(self.r_opacity_effects[2])
        self.r_execute_move_button.setGraphicsEffect(self.r_opacity_effects[3])

        self.r_stop_button = QPushButton("Stop Motion")
        self.r_stop_button_proc = ProcessRunner()
        self.r_stop_button.clicked.connect(lambda: self.stop_button_clicked("zaber"))

    def init_polar_widgets(self):
        """polar stage widgets"""
        self.polar_opacity_effects = []
        for _ in range(4):
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.3)
            self.polar_opacity_effects.append(effect)

        self.polar_label = QLabel("Step 2: \u03b8")
        self.polar_label.setFixedHeight(30)
        self.polar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.polar_label.setFont(QFont(self.font, 20))

        self.polar_move_checkbox = QCheckBox("\u03b8 Move Type: ")
        self.polar_move_checkbox.setChecked(False)
        self.polar_move_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.polar_move_checkbox.stateChanged.connect(
            lambda: self.move_button_checked(self.polar_move_checkbox)
        )

        self.polar_move_types = QComboBox()
        self.polar_move_types.addItems(["Abs.", "Jog"])
        self.polar_move_position = QLineEdit()
        self.polar_move_position_units = QLabel("")
        self.polar_execute_move_button = QPushButton("Execute Move")
        self.polar_execute_move_button_proc = ProcessRunner()
        self.polar_execute_move_button.clicked.connect(
            lambda: self.single_move_clicked("thor_polar")
        )

        self.polar_move_types.setEnabled(False)
        self.polar_move_position.setEnabled(False)
        self.polar_move_position_units.setEnabled(False)
        self.polar_execute_move_button.setEnabled(False)

        self.polar_move_types.setGraphicsEffect(self.polar_opacity_effects[0])
        self.polar_move_position.setGraphicsEffect(self.polar_opacity_effects[1])
        self.polar_move_position_units.setGraphicsEffect(self.polar_opacity_effects[2])
        self.polar_execute_move_button.setGraphicsEffect(self.polar_opacity_effects[3])

        self.polar_stop_button = QPushButton("Stop Motion")
        self.polar_stop_button_proc = ProcessRunner()
        self.polar_stop_button.clicked.connect(
            lambda: self.stop_button_clicked("thor_polar")
        )

    def init_azi_widgets(self):
        """azimuthal stage widgets"""
        self.azi_opacity_effects = []
        for _ in range(4):
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.3)
            self.azi_opacity_effects.append(effect)

        self.azi_label = QLabel("Step 1: \u03c6")
        self.azi_label.setFixedHeight(30)
        self.azi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.azi_label.setFont(QFont(self.font, 20))

        self.azi_move_checkbox = QCheckBox("\u03c6 Move Type: ")
        self.azi_move_checkbox.setChecked(False)
        self.azi_move_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.azi_move_checkbox.stateChanged.connect(
            lambda: self.move_button_checked(self.azi_move_checkbox)
        )

        self.azi_move_types = QComboBox()
        self.azi_move_types.addItems(["Abs.", "Jog"])
        self.azi_move_position = QLineEdit()
        self.azi_move_position_units = QLabel("")
        self.azi_execute_move_button = QPushButton("Execute Move")
        self.azi_execute_move_button_proc = ProcessRunner()
        self.azi_execute_move_button.clicked.connect(
            lambda: self.single_move_clicked("thor_azi")
        )

        self.azi_move_types.setEnabled(False)
        self.azi_move_position.setEnabled(False)
        self.azi_move_position_units.setEnabled(False)
        self.azi_execute_move_button.setEnabled(False)

        self.azi_move_types.setGraphicsEffect(self.azi_opacity_effects[0])
        self.azi_move_position.setGraphicsEffect(self.azi_opacity_effects[1])
        self.azi_move_position_units.setGraphicsEffect(self.azi_opacity_effects[2])
        self.azi_execute_move_button.setGraphicsEffect(self.azi_opacity_effects[3])

        self.azi_stop_button = QPushButton("Stop Motion")
        self.azi_stop_button_proc = ProcessRunner()
        self.azi_stop_button.clicked.connect(
            lambda: self.stop_button_clicked("thor_azi")
        )

    def get_combobox_val(self, combobox):
        return str(combobox.value())
    
    def init_sg396_widgets(self):
        self.sg396_label = QLabel("Signal Generator Control")
        self.sg396_label.setFixedHeight(30)
        self.sg396_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sg396_label.setFont(QFont(self.font, 20))

        self.sg396_checkbox = QCheckBox("Enable SG396 output")
        self.sg396_checkbox.setChecked(False)
        self.sg396_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.sg396_checkbox.stateChanged.connect(
            lambda: self.sg396_checked(self.sg396_checkbox)
        )

        self.sg396_opacity_effects = [QGraphicsOpacityEffect() for _ in range(5)]
        for eff in self.sg396_opacity_effects:
            eff.setOpacity(0.3)

        self.sideband_opts = ["Lower", "Upper"]
        self.channel_opts = ["IQ", "DEER"]

        self.sg396_params_widget_1 = ParamsWidget(
            {
                "channel": {
                    "display_text": "AWG Channel: ",
                    "widget": ComboBox(items=self.channel_opts),
                },
                "RF_Frequency": {
                    "display_text": "MW Frequency: ",
                    "widget": SpinBox(
                        value=2.87e9,
                        suffix="Hz",
                        siPrefix=True,
                        bounds=(100e3, 6e9),
                        dec=True,
                    ),
                },
                "RF_Power": {
                    "display_text": "MW Power: ",
                    "widget": SpinBox(value=1e-9, suffix="W", siPrefix=True),
                },
                "deer_power": {
                    "display_text": "DEER RF Power: ",
                    "widget": SpinBox(value=0.2, suffix="V", siPrefix=True),
                },
            },
            get_param_value_funs={ComboBox: self.get_combobox_val},
        )

        self.sg396_params_widget_2 = ParamsWidget(
            {
                "sideband_freq": {
                    "display_text": "Sideband Mod. Freq.: ",
                    "widget": SpinBox(
                        value=30e6,
                        suffix="Hz",
                        siPrefix=True,
                        bounds=(100, 100e6),
                        dec=True,
                    ),
                },
                "sideband_power": {
                    "display_text": "Sideband Power: ",
                    "widget": SpinBox(value=0.45, suffix="V", siPrefix=True),
                },
                "sideband": {
                    "display_text": "Sideband: ",
                    "widget": ComboBox(items=self.sideband_opts),
                },
                "i_offset": {
                    "display_text": "I Offset: ",
                    "widget": SpinBox(value=-0.002, suffix="V", siPrefix=True),
                },
                "q_offset": {
                    "display_text": "Q Offset: ",
                    "widget": SpinBox(value=-0.004, suffix="V", siPrefix=True),
                },
            },
            get_param_value_funs={ComboBox: self.get_combobox_val},
        )

        self.sg396_status_label = QLabel("SRS SG396 status here")
        self.sg396_status_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.sg396_status_label.setFixedHeight(40)

        self.sg396_emit_button = QPushButton("Emit")
        self.sg396_emit_button_proc = ProcessRunner()
        self.sg396_emit_button.clicked.connect(self.sg396_emit_button_clicked)

        self.sg396_stop_button = QPushButton("Stop")
        self.sg396_stop_button_proc = ProcessRunner()
        self.sg396_stop_button.clicked.connect(self.sg396_stop_button_clicked)

        # default disabled
        self.sg396_params_widget_1.setEnabled(False)
        self.sg396_params_widget_2.setEnabled(False)
        self.sg396_status_label.setEnabled(False)
        self.sg396_emit_button.setEnabled(False)
        self.sg396_stop_button.setEnabled(False)

        self.sg396_params_widget_1.setGraphicsEffect(self.sg396_opacity_effects[0])
        self.sg396_params_widget_2.setGraphicsEffect(self.sg396_opacity_effects[1])
        self.sg396_status_label.setGraphicsEffect(self.sg396_opacity_effects[2])
        self.sg396_emit_button.setGraphicsEffect(self.sg396_opacity_effects[3])
        self.sg396_stop_button.setGraphicsEffect(self.sg396_opacity_effects[4])

    def init_laser_widgets(self):
        self.laser_label = QLabel("Laser")
        self.laser_label.setFixedHeight(40)
        self.laser_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.laser_label.setFont(QFont(self.font, 20))

        laser_style = """
        QRadioButton::indicator {
        width: 25px; /* Adjust size as needed */
        height: 15px;
        border-radius: 7px; /* Makes it circular */
        background-color: lightgray; /* Default color when unchecked */
        border: 1px solid black;}
                                        
        QRadioButton::indicator:checked {
            background-color: limegreen; /* Color when checked */
            border: 1px solid green;}

        QRadioButton {
            color: white; /* Default text color */}

        QRadioButton:checked {
            color: limegreen; /* Text color when checked */}"""

        self.laser_b1 = QRadioButton("CW ON")
        self.laser_b1.toggled.connect(lambda: self.toggle_laser(self.laser_b1))
        self.laser_b1.setStyleSheet(laser_style)

        self.laser_b2 = QRadioButton("CW OFF")
        self.laser_b2.setChecked(True)
        self.laser_b2.toggled.connect(lambda: self.toggle_laser(self.laser_b2))
        self.laser_b2.setStyleSheet(laser_style)

        self.laser_group = QButtonGroup(self)
        self.laser_group.setExclusive(True)
        self.laser_group.addButton(self.laser_b1)
        self.laser_group.addButton(self.laser_b2)

        self.optics_flipper_label = QLabel("Excitation Configuration:  ")
        self.optics_flipper_label.setFixedHeight(40)

        self.optics_flipper_b1 = QRadioButton("Epifluorescence")
        self.optics_flipper_b1.toggled.connect(
            lambda: self.toggle_optics_mode(self.optics_flipper_b1)
        )
        self.optics_flipper_b1.setStyleSheet(laser_style)

        self.optics_flipper_b2 = QRadioButton("TIR")
        self.optics_flipper_b2.toggled.connect(
            lambda: self.toggle_optics_mode(self.optics_flipper_b2)
        )
        self.optics_flipper_b2.setStyleSheet(laser_style)

        self.optics_flipper_group = QButtonGroup(self)
        self.optics_flipper_group.setExclusive(True)
        self.optics_flipper_group.addButton(self.optics_flipper_b1)
        self.optics_flipper_group.addButton(self.optics_flipper_b2)

        self.laser_current_label = QLabel("Diode Current (%):")
        self.laser_current_label.setFont(QFont("Sanserif", 15))

        self.laser_current_value = QLineEdit()

        laser_button_style = """
        QPushButton {
        background-color: #184B00;
        color: white;
        border: 2px solid #64D700;
        border-radius: 5px;
        padding: 5px;
        }
        
        QPushButton:hover {
        background-color: #004B31;
        border: 2px solid #0ED700;
        }
        
        QPushButton:pressed {
        background-color: #155F00;
        border: 2px solid #1DD700;
        }"""

        self.laser_power_set_button = QPushButton("Set Diode Current")
        self.laser_power_set_button.setStyleSheet(laser_button_style)
        self.laser_power_set_button.clicked.connect(self.laser_power_changed)

        self.laser_current_setpt_label = QLabel("CW Current Setpoint: 0%")
        self.laser_current_setpt_label.setFont(QFont("Sanserif", 15))
        self.laser_power_label = QLabel("CW Power Setpoint: 0 mW")
        self.laser_power_label.setFont(QFont("Sanserif", 15))

        self.laser_baseplate_temp_label = QLabel("Baseplate Temp: ---")
        self.laser_baseplate_temp_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.laser_baseplate_temp_label.setFixedHeight(40)

        self.laser_diode_temp_label = QLabel("Diode Temp: ---")
        self.laser_diode_temp_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.laser_diode_temp_label.setFixedHeight(40)

        self.laser_status_label = QLabel("Laser status")
        self.laser_status_label.setFixedHeight(40)

        self.laser_interlock_status_label = QLabel("Interlock status")
        self.laser_interlock_status_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.laser_interlock_status_label.setFixedHeight(40)

        self.laser_emit_status_label = QLabel("Emission status")
        self.laser_emit_status_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.laser_emit_status_label.setFixedHeight(40)
        self.laser_emit_status_label.setFixedWidth(145)

        self.laser_shutter_button = QPushButton("Open shutter")
        self.laser_shutter_button.setStyleSheet(laser_button_style)
        self.laser_shutter_button.clicked.connect(self.laser_shutter_status_changed)

        self.laser_shutter_status_label = QLabel("Laser Shutter status")
        self.laser_shutter_status_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.laser_shutter_status_label.setFixedHeight(40)
        self.laser_shutter_status_label.setFixedWidth(225)

        self.laser_alarm_reset_button = QPushButton("Reset Alarm")
        self.laser_alarm_reset_button.setStyleSheet(laser_button_style)
        self.laser_alarm_reset_button.clicked.connect(self.laser_alarm_reset_clicked)

        self.ps_status_label = QLabel("Pulse Streamer: CONSTANT")
        self.ps_status_label.setStyleSheet(
            "color: white; background-color: #222; border: 2px solid #444;"
        )
        self.ps_status_label.setFixedHeight(40)

    def init_flipper_widgets(self):
        self.flipper_label = QLabel("Detector")
        self.flipper_label.setFixedHeight(40)
        self.flipper_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flipper_label.setFont(QFont(self.font, 20))

        flipper_style = """
        QRadioButton::indicator {
        width: 25px; /* Adjust size as needed */
        height: 15px;
        border-radius: 7px; /* Makes it circular */
        background-color: lightgray; /* Default color when unchecked */
        border: 1px solid black;}
                                        
        QRadioButton::indicator:checked {
            background-color: #27CFF5; /* Color when checked */
            border: 1px solid blue;}

        QRadioButton {
            color: white; /* Default text color */}

        QRadioButton:checked {
            color: #27CFF5; /* Text color when checked */}"""

        self.flipper_b1 = QRadioButton("APD")
        self.flipper_b1.toggled.connect(
            lambda: self.toggle_detector_mode(self.flipper_b1)
        )
        self.flipper_b1.setStyleSheet(flipper_style)

        self.flipper_b2 = QRadioButton("BPD")
        self.flipper_b2.toggled.connect(
            lambda: self.toggle_detector_mode(self.flipper_b2)
        )
        self.flipper_b2.setStyleSheet(flipper_style)

    def init_nd_filter_widgets(self):
        self.nd_filter_label = QLabel("BPD ND Filter")
        self.nd_filter_label.setFixedHeight(20)
        self.nd_filter_label.setStyleSheet("font-weight: bold")

        nd_filter_stylesheet = """
        QComboBox {
            border: 3px solid lightblue;
            border-radius: 3px;
            padding: 1px 18px 1px 3px;
            min-width: 6em;
        }

        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left-width: 1px;
            border-left-color: black;
            border-left-style: solid;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
        }

        QComboBox QAbstractItemView {
            border: 1px solid lightblue;
            selection-background-color: lightblue;
        }
        """

        self.nd_filter_opts = QComboBox()
        self.nd_filter_opts.setStyleSheet(nd_filter_stylesheet)
        self.nd_filter_opts.addItems(["ND 0", "ND 0.5", "ND 1", "ND 2", "ND 3", "ND 4"])
        self.nd_filter_opts.currentIndexChanged.connect(self.nd_filter_changed)

    def init_bpd_shutter_widgets(self):
        self.bpd_shutter_label = QLabel("BPD Shutter")
        self.bpd_shutter_label.setFixedHeight(20)
        self.bpd_shutter_label.setStyleSheet("font-weight: bold")

        bpd_shutter_button_style = """
        QPushButton {
        background-color: #002E4B;
        color: white;
        border: 2px solid #0088D7;
        border-radius: 5px;
        padding: 5px;
        }
        
        QPushButton:hover {
        background-color: #00284B;
        border: 2px solid #005DD7;
        }
        
        QPushButton:pressed {
        background-color: #004A5F;
        border: 2px solid #0085D7;
        }"""
        self.bpd_shutter_button = QPushButton("Open BPD shutter")
        self.bpd_shutter_button.setStyleSheet(bpd_shutter_button_style)
        self.bpd_shutter_button.clicked.connect(self.bpd_shutter_status_changed)

        self.bpd_shutter_status_label = QLabel("BPD Shutter status")
        self.bpd_shutter_status_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.bpd_shutter_status_label.setFixedHeight(40)
        self.bpd_shutter_status_label.setFixedWidth(220)

    def layouts(self):
        """GUI Layout Structure"""
        self.gui_layout = QVBoxLayout()

        self.magnet_frame = QFrame(self)
        self.magnet_frame.setObjectName("magFrame")
        self.magnet_frame.setStyleSheet(
            "QFrame#magFrame {background-color: #2b2b2b; border: 2px solid #717171; "
            "border-radius: 5px;}"
        )
        self.magnet_layout = QGridLayout(self.magnet_frame)
        self.magnet_layout.setSpacing(0)
        self.magnet_layout.addWidget(self.magnet_label, 1, 1, 1, 3)
        self.magnet_layout.addWidget(
            self.r_label, 2, 1, 1, 1, Qt.AlignmentFlag.AlignCenter
        )
        self.magnet_layout.addWidget(
            self.polar_label, 2, 2, 1, 1, Qt.AlignmentFlag.AlignCenter
        )
        self.magnet_layout.addWidget(
            self.azi_label, 2, 3, 1, 1, Qt.AlignmentFlag.AlignCenter
        )

        # B frame
        self.b_frame = QFrame(self)
        self.b_frame.setObjectName("bFrame")
        self.b_frame.setStyleSheet(
            "QFrame#bFrame {background-color: #20373b; border: 2px solid #47747c; "
            "border-radius: 5px;}"
        )
        self.b_layout = QGridLayout(self.b_frame)
        self.b_layout.setSpacing(0)
        self.b_layout.addWidget(self.b_label, 1, 1, 1, 2, Qt.AlignmentFlag.AlignCenter)
        self.b_layout.addWidget(self.phi_label, 2, 1)
        self.b_layout.addWidget(self.phi_position, 2, 2)
        self.b_layout.addWidget(self.target_b_label, 3, 1)
        self.b_layout.addWidget(self.target_b_value, 3, 2)
        self.b_layout.addWidget(self.b_execute_button, 4, 1)

        # Zaber frame
        self.zaber_frame = QFrame(self)
        self.zaber_frame.setObjectName("zaberFrame")
        self.zaber_frame.setStyleSheet(
            "QFrame#zaberFrame {background-color: #331313; border: 2px solid #854141;"
            " border-radius: 5px;}"
        )
        self.r_layout = QGridLayout(self.zaber_frame)
        self.r_layout.setSpacing(0)
        self.r_layout.addWidget(self.r_label, 1, 1, 1, 2, Qt.AlignmentFlag.AlignCenter)
        self.r_layout.addWidget(self.r_move_checkbox, 2, 1)
        self.r_layout.addWidget(self.r_move_types, 2, 2)
        self.r_layout.addWidget(self.r_move_position, 3, 1)
        self.r_layout.addWidget(self.r_move_position_units, 3, 2)
        self.r_layout.addWidget(self.r_execute_move_button, 4, 1, 1, 2)
        self.r_layout.addWidget(self.r_stop_button, 5, 1, 1, 2)

        # Thor frames (polar/azi)
        self.thor1_frame = QFrame(self)
        self.thor1_frame.setObjectName("thor1Frame")
        self.thor1_frame.setStyleSheet(
            "QFrame#thor1Frame {background-color: #331313; border: 2px solid #854141;"
            " border-radius: 5px;}"
        )
        self.polar_layout = QGridLayout(self.thor1_frame)
        self.polar_layout.setSpacing(0)
        self.polar_layout.addWidget(self.polar_label, 1, 1, 1, 2)
        self.polar_layout.addWidget(self.polar_move_checkbox, 2, 1)
        self.polar_layout.addWidget(self.polar_move_types, 2, 2)
        self.polar_layout.addWidget(self.polar_move_position, 3, 1)
        self.polar_layout.addWidget(self.polar_move_position_units, 3, 2)
        self.polar_layout.addWidget(self.polar_execute_move_button, 4, 1, 1, 2)
        self.polar_layout.addWidget(self.polar_stop_button, 5, 1, 1, 2)

        self.thor2_frame = QFrame(self)
        self.thor2_frame.setObjectName("thor2Frame")
        self.thor2_frame.setStyleSheet(
            "QFrame#thor2Frame {background-color: #331313; border: 2px solid #854141;"
            " border-radius: 5px;}"
        )
        self.azi_layout = QGridLayout(self.thor2_frame)
        self.azi_layout.setSpacing(0)
        self.azi_layout.addWidget(self.azi_label, 1, 1, 1, 2)
        self.azi_layout.addWidget(self.azi_move_checkbox, 2, 1)
        self.azi_layout.addWidget(self.azi_move_types, 2, 2)
        self.azi_layout.addWidget(self.azi_move_position, 3, 1)
        self.azi_layout.addWidget(self.azi_move_position_units, 3, 2)
        self.azi_layout.addWidget(self.azi_execute_move_button, 4, 1, 1, 2)
        self.azi_layout.addWidget(self.azi_stop_button, 5, 1, 1, 2)

        # status frame
        self.status_frame = QFrame(self)
        self.status_frame.setObjectName("statusFrame")
        self.status_frame.setStyleSheet(
            "QFrame#statusFrame {background-color: #2b2b2b; border: 2px solid #717171;"
            " border-radius: 5px;}"
        )
        self.status_bar_layout = QGridLayout(self.status_frame)
        self.status_bar_layout.setSpacing(0)
        self.status_bar_layout.addWidget(self.all_disable_button, 1, 1, 1, 1)
        self.status_bar_layout.addWidget(self.all_standby_button, 1, 2, 1, 1)
        self.status_bar_layout.addWidget(self.status_label, 2, 1, 1, 2)

        self.individual_cmds_layout = QGridLayout()
        self.individual_cmds_layout.setSpacing(0)
        self.individual_cmds_layout.addWidget(self.magnet_frame, 1, 1, 1, 4)
        self.individual_cmds_layout.addWidget(self.thor2_frame, 2, 1, 1, 1)
        self.individual_cmds_layout.addWidget(self.thor1_frame, 2, 2, 1, 1)
        self.individual_cmds_layout.addWidget(self.zaber_frame, 2, 3, 1, 1)
        self.individual_cmds_layout.addWidget(self.b_frame, 2, 4, 1, 1)
        self.individual_cmds_layout.addWidget(self.status_frame, 3, 1, 1, 4)

        # signal generators
        self.sg396_frame = QFrame(self)
        self.sg396_frame.setObjectName("sgFrame")
        self.sg396_frame.setStyleSheet(
            "QFrame#sgFrame {background-color: #2b2b2b; border: 2px solid #717171;"
            " border-radius: 5px;}"
        )
        self.sg396_layout = QGridLayout(self.sg396_frame)
        self.sg396_layout.setSpacing(0)
        self.sg396_layout.addWidget(self.sg396_label, 1, 1, 1, 2)
        self.sg396_layout.addWidget(self.sg396_checkbox, 2, 1, 1, 2)
        self.sg396_layout.addWidget(self.sg396_params_widget_1, 3, 1, 1, 1)
        self.sg396_layout.addWidget(self.sg396_params_widget_2, 3, 2, 1, 1)
        self.sg396_layout.addWidget(self.sg396_status_label, 4, 1, 1, 2)
        self.sg396_layout.addWidget(self.sg396_emit_button, 5, 1, 1, 1)
        self.sg396_layout.addWidget(self.sg396_stop_button, 5, 2, 1, 1)

        self.sig_gens_layout = QGridLayout()
        self.sig_gens_layout.setSpacing(0)
        self.sig_gens_layout.addWidget(self.sg396_frame, 1, 1, 1, 1)

        # laser & detector
        self.laser_control_frame = QFrame(self)
        self.laser_control_frame.setObjectName("laserControlFrame")
        self.laser_control_frame.setStyleSheet(
            "QFrame#laserControlFrame {background-color: #003407; border: 2px solid limegreen;"
            " border-radius: 5px;}"
        )
        self.laser_controls_layout = QGridLayout(self.laser_control_frame)
        self.laser_controls_layout.setSpacing(0)
        self.laser_controls_layout.addWidget(self.laser_label, 1, 1, 1, 3)
        self.laser_controls_layout.addWidget(self.optics_flipper_label, 2, 1, 1, 1)
        self.laser_controls_layout.addWidget(self.optics_flipper_b1, 2, 2, 1, 1)
        self.laser_controls_layout.addWidget(self.optics_flipper_b2, 2, 3, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_current_label, 3, 1, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_current_value, 3, 2, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_power_set_button, 3, 3, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_b1, 4, 1, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_b2, 4, 2, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_shutter_button, 4, 3, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_alarm_reset_button, 5, 1, 1, 3)

        self.device_status_frame = QFrame(self)
        self.device_status_frame.setObjectName("deviceStatusFrame")
        self.device_status_frame.setStyleSheet(
            "QFrame#deviceStatusFrame {"
            "background-color: #3b1f0f;"      # dark brown background
            "border: 2px solid #c68642;"      # warm tan / copper border
            "border-radius: 5px;"
            "}"
        )
        self.device_status_layout = QGridLayout(self.device_status_frame)
        self.device_status_layout.setSpacing(0)
        self.device_status_layout.addWidget(self.laser_current_setpt_label, 1, 1, 1, 2)
        self.device_status_layout.addWidget(self.laser_power_label, 1, 3, 1, 1)
        self.device_status_layout.addWidget(self.laser_baseplate_temp_label, 2, 1, 1, 2)
        self.device_status_layout.addWidget(self.laser_diode_temp_label, 2, 3, 1, 1)
        self.device_status_layout.addWidget(self.laser_status_label, 3, 1, 1, 1)
        self.device_status_layout.addWidget(self.laser_interlock_status_label, 3, 2, 1, 1)
        self.device_status_layout.addWidget(self.laser_emit_status_label, 4, 1, 1, 1)
        self.device_status_layout.addWidget(self.laser_shutter_status_label, 4, 2, 1, 1)
        self.device_status_layout.addWidget(self.bpd_shutter_status_label, 4, 3, 1, 1)
        self.device_status_layout.addWidget(self.ps_status_label, 5, 1, 1, 3)

        self.detector_frame = QFrame(self)
        self.detector_frame.setObjectName("detectorFrame")
        self.detector_frame.setStyleSheet(
            "QFrame#detectorFrame {background-color: #0A0034; border: 2px solid blue;"
            " border-radius: 5px;}"
        )
        self.detector_layout = QGridLayout(self.detector_frame)
        self.detector_layout.setSpacing(0)
        self.detector_layout.addWidget(self.flipper_label, 1, 1, 1, 2)
        self.detector_layout.addWidget(self.flipper_b1, 2, 1, 1, 1)
        self.detector_layout.addWidget(self.flipper_b2, 2, 2, 1, 1)

        self.detector_layout.addWidget(self.nd_filter_label, 3, 1, 1, 2)
        self.detector_layout.addWidget(self.nd_filter_opts, 4, 1, 1, 2)
        self.detector_layout.addWidget(self.bpd_shutter_label, 5, 1, 1, 2)
        self.detector_layout.addWidget(self.bpd_shutter_button, 6, 1, 1, 2)

        self.other_widgets_layout = QGridLayout()
        self.other_widgets_layout.addWidget(self.laser_control_frame, 1, 1, 1, 1)
        self.other_widgets_layout.addWidget(self.detector_frame, 1, 2, 2, 1)
        self.other_widgets_layout.addWidget(self.device_status_frame, 2, 1, 1, 1)


        self.gui_layout.addLayout(self.individual_cmds_layout)
        self.gui_layout.addLayout(self.status_bar_layout)
        self.gui_layout.addLayout(self.sig_gens_layout)
        self.gui_layout.addLayout(self.other_widgets_layout)

        self.setLayout(self.gui_layout)

    def park_all(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        self.all_disable_proc.run(mag_control.disable_all)
        logger.info(
            "All magnet stages have been parked upon startup of magnet alignment widget."
        )

    # INTERACTIVE WIDGET FUNCTIONS
    def r_move_type_changed(self):
        # TODO: update this part
        self.r_move_option = self.r_move_types.currentText()

        match self.r_move_types.currentIndex():
            case 2:
                self.r_move_position_units.setText("G")
            case _:
                self.r_move_position_units.setText("mm")

    def move_button_checked(self, box):
        if box.text() == "R Move Type: ":
            if box.isChecked() is True:
                self.r_move_types.setEnabled(True)
                self.r_move_position.setEnabled(True)
                self.r_move_position_units.setEnabled(True)
                self.r_execute_move_button.setEnabled(True)
                [self.r_opacity_effects[i].setEnabled(False) for i in range(4)]

                match self.r_move_types.currentIndex():
                    case 2:
                        self.r_move_position_units.setText("G")
                    case _:
                        self.r_move_position_units.setText("mm")
            else:
                self.r_move_types.setEnabled(False)
                self.r_move_position.setEnabled(False)
                self.r_move_position_units.setEnabled(False)
                self.r_move_position_units.setText("")
                self.r_execute_move_button.setEnabled(False)
                [self.r_opacity_effects[i].setEnabled(True) for i in range(4)]

        elif box.text() == "\u03b8 Move Type: ":
            if box.isChecked() is True:
                self.polar_move_types.setEnabled(True)
                self.polar_move_position.setEnabled(True)
                self.polar_move_position_units.setEnabled(True)
                self.polar_move_position_units.setText("\N{DEGREE SIGN}")
                self.polar_execute_move_button.setEnabled(True)
                [self.polar_opacity_effects[i].setEnabled(False) for i in range(4)]
            else:
                self.polar_move_types.setEnabled(False)
                self.polar_move_position.setEnabled(False)
                self.polar_move_position_units.setEnabled(False)
                self.polar_move_position_units.setText("")
                self.polar_execute_move_button.setEnabled(False)
                [self.polar_opacity_effects[i].setEnabled(True) for i in range(4)]

        elif box.text() == "\u03c6 Move Type: ":
            if box.isChecked() is True:
                self.azi_move_types.setEnabled(True)
                self.azi_move_position.setEnabled(True)
                self.azi_move_position_units.setEnabled(True)
                self.azi_move_position_units.setText("\N{DEGREE SIGN}")
                self.azi_execute_move_button.setEnabled(True)
                [self.azi_opacity_effects[i].setEnabled(False) for i in range(4)]
            else:
                self.azi_move_types.setEnabled(False)
                self.azi_move_position.setEnabled(False)
                self.azi_move_position_units.setEnabled(False)
                self.azi_move_position_units.setText("")
                self.azi_execute_move_button.setEnabled(False)
                [self.azi_opacity_effects[i].setEnabled(True) for i in range(4)]

    def single_move_clicked(self, stage):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()

        if self.all_disable_button.text() == "Disable All":
            if stage == "zaber":
                type_idx = self.r_move_types.currentIndex()
                try:
                    new_position = float(self.r_move_position.text())
                    if type_idx == 0:
                        self.r_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_pos": new_position,
                                "queue": self.q,
                                "abs": True,
                            },
                        )
                    elif type_idx == 1:
                        self.r_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_pos": new_position,
                                "queue": self.q,
                                "abs": False,
                            },
                        )
                except ValueError:
                    logger.debug("Invalid entry for new position. Try again.")

            elif stage == "thor_polar":
                type_idx = self.polar_move_types.currentIndex()
                try:
                    new_position = float(self.polar_move_position.text())
                    if type_idx == 0:
                        self.polar_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_angle": new_position,
                                "queue": self.q,
                                "abs": True,
                            },
                        )
                    elif type_idx == 1:
                        self.polar_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_angle": new_position,
                                "queue": self.q,
                                "abs": False,
                            },
                        )
                except ValueError:
                    logger.debug("Invalid entry for new angle. Try again.")

            elif stage == "thor_azi":
                type_idx = self.azi_move_types.currentIndex()
                try:
                    new_position = float(self.azi_move_position.text())
                    if type_idx == 0:
                        self.azi_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_angle": new_position,
                                "queue": self.q,
                                "abs": True,
                            },
                        )
                    elif type_idx == 1:
                        self.azi_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_angle": new_position,
                                "queue": self.q,
                                "abs": False,
                            },
                        )
                except ValueError:
                    logger.debug("Invalid entry for new angle. Try again.")

            else:
                # Move Zaber (z) to target B field
                try:
                    phi_val = float(self.phi_position.text())
                    b_val = float(self.target_b_value.text())
                except (TypeError, ValueError):
                    pass
                else:
                    new_position = find_mag_pos(phi_val, b_val)

                    self.status_label.setStyleSheet(
                        "color: black; background-color: gold; border: 4px solid black;"
                    )
                    self.status_label.setText(
                        f"Magnet Status: MOVING ZABER TO {new_position}..."
                    )

                    self.b_execute_button_proc.run(
                        mag_control.move_to_orientation,
                        **{
                            "stage": "zaber",
                            "new_pos": new_position,
                            "queue": self.q,
                            "abs": True,
                        },
                    )
        else:
            self.status_label.setStyleSheet(
                "color: black; background-color: red; border: 4px solid black;"
            )
            self.status_label.setText("Magnet Status: CANNOT MOVE STAGE IS PARKED")

        return 0

    def stop_button_clicked(self, stage):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()

        if stage == "zaber":
            self.r_stop_button_proc.run(mag_control.stop_motion, stage)
        elif stage == "thor_polar":
            self.polar_stop_button_proc.run(mag_control.stop_motion, stage)
        elif stage == "thor_azi":
            self.azi_stop_button_proc.run(mag_control.stop_motion, stage)

        return 0

    def all_disable_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()

        if self.all_disable_button.text() == "Disable All":
            self.all_disable_proc.run(mag_control.disable_all)
            self.all_disable_button.setText("Enable All")
            self.status_label.setStyleSheet(
                "color: white; background-color: black; border: 4px solid black;"
            )
            self.status_label.setText("Magnet Status: STAGES DISABLED")
        else:
            self.all_disable_proc.run(mag_control.enable_all)
            self.all_disable_button.setText("Disable All")
            self.status_label.setStyleSheet(
                "color: black; background-color: white; border: 4px solid black;"
            )
            self.status_label.setText("Magnet Status: STAGES ENABLED")

    def all_standby_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()

        if self.all_disable_button.text() == "Disable All":
            self.all_standby_proc.run(mag_control.standby_all, **{"queue": self.q})
        else:
            self.status_label.setStyleSheet(
                "color: black; background-color: red; border: 4px solid black;"
            )
            self.status_label.setText("Magnet Status: CANNOT MOVE STAGE IS PARKED")

    def check_queue_from_mag(self):
        # queue checker to control progress bar display
        while not self.q.empty():  # if there is something in the queue
            queueText = self.q.get_nowait()

            if queueText[0] == "done":
                self.status_label.setStyleSheet(
                    "color: black; background-color: limegreen; border: 4px solid black;"
                )
                self.status_label.setText("Magnet Status: MOVEMENT COMPLETED")

                self.r_label.setText(f"Step 3: R = {queueText[1][0]}")
                self.polar_label.setText(
                    f"Step 2: \u03b8 = {round(queueText[2], 1)}\N{DEGREE SIGN}"
                )
                self.azi_label.setText(
                    f"Step 1: \u03c6 = {round(queueText[3], 1)}\N{DEGREE SIGN}"
                )

                self.r_label.setStyleSheet("color: white;")
                self.polar_label.setStyleSheet("color: white;")
                self.azi_label.setStyleSheet("color: white;")

            elif queueText[0] == "start standby":
                self.status_label.setStyleSheet(
                    "color: black; background-color: gold; border: 4px solid black;"
                )
                self.status_label.setText(
                    "Magnet Status: MOVING TO STANDBY POSITION..."
                )

            elif queueText[0] == "in motion to standby":
                self.status_label.setStyleSheet(
                    "color: black; background-color: gold; border: 4px solid black;"
                )
                self.status_label.setText(
                    "Magnet Status: MOVING TO STANDBY POSITION..."
                )

                self.r_label.setText(f"Step 3: R = {queueText[1][0]}")
                self.polar_label.setText(
                    f"Step 2: \u03b8 = {round(queueText[2], 1)}\N{DEGREE SIGN}"
                )
                self.azi_label.setText(
                    f"Step 1: \u03c6 = {round(queueText[3], 1)}\N{DEGREE SIGN}"
                )

                # Change color to yellow while moving
                self.r_label.setStyleSheet("color: yellow;")
                self.polar_label.setStyleSheet("color: yellow;")
                self.azi_label.setStyleSheet("color: yellow;")

            elif queueText[0] == "start z move":
                self.status_label.setStyleSheet(
                    "color: black; background-color: gold; border: 4px solid black;"
                )

                if queueText[6] is True:
                    self.status_label.setText(
                        f"Magnet Status: MOVING {queueText[4]} TO {queueText[5]}..."
                    )
                else:
                    self.status_label.setText(
                        f"Magnet Status: MOVING {queueText[4]} BY {queueText[5]}..."
                    )

                self.r_label.setText(f"Step 3: R = {queueText[1][0]}")
                self.r_label.setStyleSheet("color: yellow;")
                self.polar_label.setStyleSheet("color: white;")
                self.azi_label.setStyleSheet("color: white;")

            elif queueText[0] == "start rotation move":
                self.status_label.setStyleSheet(
                    "color: black; background-color: gold; border: 4px solid black;"
                )

                if queueText[6] is True:
                    self.status_label.setText(
                        f"Magnet Status: MOVING {queueText[4]} TO {queueText[5]}\N{DEGREE SIGN} ..."
                    )
                else:
                    self.status_label.setText(
                        f"Magnet Status: MOVING {queueText[4]} BY {queueText[5]}\N{DEGREE SIGN} ..."
                    )

            elif queueText[0] == "in rotation motion to target":
                self.status_label.setStyleSheet(
                    "color: black; background-color: gold; border: 4px solid black;"
                )

                if queueText[4] == "thor_polar":
                    if queueText[6] is True:
                        self.status_label.setText(
                            f"Magnet Status: MOVING \u03b8 TO {queueText[5]}\N{DEGREE SIGN} ..."
                        )
                    else:
                        self.status_label.setText(
                            f"Magnet Status: MOVING \u03b8 BY {queueText[5]}\N{DEGREE SIGN} ..."
                        )

                    self.polar_label.setText(
                        f"Step 2: \u03b8 = {round(queueText[2], 1)}\N{DEGREE SIGN}"
                    )
                    self.r_label.setStyleSheet("color: white;")
                    self.polar_label.setStyleSheet("color: yellow;")
                    self.azi_label.setStyleSheet("color: white;")

                elif queueText[4] == "thor_azi":
                    if queueText[6] is True:
                        self.status_label.setText(
                            f"Magnet Status: MOVING \u03c6 TO {queueText[5]}\N{DEGREE SIGN} ..."
                        )
                    else:
                        self.status_label.setText(
                            f"Magnet Status: MOVING \u03c6 BY {queueText[5]}\N{DEGREE SIGN} ..."
                        )

                    self.azi_label.setText(
                        f"Step 1: \u03c6 = {round(queueText[3], 1)}\N{DEGREE SIGN}"
                    )
                    self.r_label.setStyleSheet("color: white;")
                    self.polar_label.setStyleSheet("color: white;")
                    self.azi_label.setStyleSheet("color: yellow;")

            else:
                self.status_label.setStyleSheet(
                    "color: black; background-color: gold; border: 4px solid black;"
                )
                self.status_label.setText("Magnet Status: MOVEMENT IN PROGRESS...")

                self.r_label.setStyleSheet("color: white;")
                self.polar_label.setStyleSheet("color: white;")
                self.azi_label.setStyleSheet("color: white;")

        self.updateTimer.start(self.QUEUE_CHECK_TIME)

    # Duplicate `check_ps_status` removed: the working implementation remains
    # later in this file (the later definition is what Python will use).

    def _set_label_if_changed(self, lbl, text=None, style=None):
        if not _alive(lbl):
            return
        # Only set if different (cheap guard)
        if (text is not None) and (lbl.text() != text):
            lbl.setText(text)
        if (style is not None) and (lbl.styleSheet() != style):
            lbl.setStyleSheet(style)

    # --- Shutter ---
    def check_laser_temp_status(self):
        # don't do anything while tearing down
        if getattr(self, "_closing", False):
            return

        base_lbl = getattr(self, "laser_baseplate_temp_label", None)
        diode_lbl = getattr(self, "laser_diode_temp_label", None)

        try:
            base_temp = self.call_laser(lambda laser: laser.get_baseplate_temp())
            diode_temp = self.call_laser(lambda laser: laser.get_diode_temp())

            if base_temp is None or diode_temp is None:
                # show disconnected/error state and return (no exception spam)
                self._set_label_if_changed(
                    base_lbl, "Baseplate Temp: ---", "color: #bbb;"
                )
                self._set_label_if_changed(diode_lbl, "Diode Temp: ---", "color: #bbb;")
                return

            # TODO: add get laser status and interlock status
            self._set_label_if_changed(
                base_lbl, f"Baseplate Temp: {base_temp:.1f} °C", "color: white;"
            )
            self._set_label_if_changed(
                diode_lbl, f"Diode Temp: {diode_temp:.1f} °C", "color: white;"
            )

            self._temp_last_err = None  # clear last error on success

        except Exception as e:
            self._set_label_if_changed(
                base_lbl,
                "Baseplate Temp: ERROR",
                "color: white; background-color: #800; border: 2px solid #f00;",
            )
            self._set_label_if_changed(
                diode_lbl,
                "Diode Temp: ERROR",
                "color: white; background-color: #800; border: 2px solid #f00;",
            )

            msg = f"{type(e).__name__}: {e}"
            if getattr(self, "_temp_last_err", None) != msg:
                self._temp_last_err = msg
                print(f"[Laser Temperature status] {msg}")

    def check_laser_shutter_status(self):
        # don't do anything while tearing down
        if getattr(self, "_closing", False):
            return

        lbl = getattr(self, "laser_shutter_status_label", None)
        try:
            state = self.call_laser_shutter(
                lambda laser_shutter: laser_shutter.get_shutter_state()
            )
            if state is None:
                self._set_label_if_changed(lbl, "Laser Shutter: ---", "color: #bbb;")
                return

            if state == "Active":
                self._set_label_if_changed(
                    lbl,
                    "Laser Shutter: OPEN",
                    "color: black; background-color: white; border: 4px solid black;",
                )
            else:
                self._set_label_if_changed(
                    lbl,
                    "Laser Shutter: CLOSED",
                    "color: white; background-color: black; border: 4px solid black;",
                )
            self._sh_last_err = None
        except Exception as e:
            self._set_label_if_changed(
                lbl,
                "Laser Shutter: ERROR",
                "color: white; background-color: #800; border: 2px solid #f00;",
            )
            msg = f"{type(e).__name__}: {e}"
            if getattr(self, "_sh_last_err", None) != msg:
                self._sh_last_err = msg
                print(f"[Laser Shutter status] {msg}")

    # --- Pulse Streamer ---
    def check_ps_status(self):
        # don't do anything while tearing down
        if getattr(self, "_closing", False):
            return

        # get labels
        ps_lbl = getattr(self, "ps_status_label", None)
        emit_lbl = getattr(self, "laser_emit_status_label", None)
        pow_lbl = getattr(self, "laser_power_label", None)
        b1 = getattr(self, "laser_b1", None)

        try:
            st = self.call_ps(
                lambda ps: ps.get_status()
            )  # {"owner","mode","streaming"}
            if st is None:
                self._set_label_if_changed(
                    ps_lbl, "Pulse Streamer: ---", "color: #bbb;"
                )
                return
            owner = st.get("owner")
            mode = st.get("mode", "IDLE")

            # enable/disable CW controls if ownership is external
            external_owner = bool(
                owner and not str(owner).startswith(self.GUI_OWNER_PREFIX)
            )
            self._set_cw_controls_enabled(not external_owner)

            if external_owner:
                if _alive(ps_lbl):
                    self._set_label_if_changed(
                        ps_lbl,
                        f"Pulse Streamer: {mode} (owner: {owner})",
                        "color: black; background-color: gold; border: 2px solid black;",
                    )
                if _alive(emit_lbl):
                    self._set_label_if_changed(
                        emit_lbl,
                        "Emission: ON",
                        "color: black; background-color: limegreen; border: 4px solid black;",
                    )
                if _alive(pow_lbl):
                    self._set_label_if_changed(
                        pow_lbl, pow_lbl.text(), "color: limegreen;"
                    )

            else:
                cw_on = _alive(b1) and self.laser_b1.isChecked()

                if _alive(ps_lbl):
                    suffix = " (CW ON)" if cw_on else ""
                    self._set_label_if_changed(
                        ps_lbl,
                        f"Pulse Streamer: {mode}{suffix}",
                        "color: white; background-color: #222; border: 2px solid #444;",
                    )

                if cw_on:
                    if _alive(emit_lbl):
                        self._set_label_if_changed(
                            emit_lbl,
                            "Emission: ON",
                            "color: black; background-color: limegreen; border: 4px solid black;",
                        )
                    if _alive(pow_lbl):
                        self._set_label_if_changed(
                            pow_lbl, pow_lbl.text(), "color: limegreen;"
                        )
                else:
                    if _alive(emit_lbl):
                        self._set_label_if_changed(
                            emit_lbl,
                            "Emission: OFF",
                            "color: white; background-color: black; border: 4px solid black;",
                        )
                    if _alive(pow_lbl):
                        self._set_label_if_changed(
                            pow_lbl, pow_lbl.text(), "color: white;"
                        )

            self._ps_last_error = None

        except Exception as e:
            if getattr(self, "_closing", False):
                return

            if _alive(ps_lbl):
                self._set_label_if_changed(
                    ps_lbl,
                    "Pulse Streamer: STATUS ERROR",
                    "color: white; background-color: #800; border: 2px solid #f00;",
                )
            if _alive(emit_lbl):
                self._set_label_if_changed(
                    emit_lbl,
                    "Emission: ERROR",
                    "color: white; background-color: #800; border: 2px solid #f00;",
                )
            if _alive(pow_lbl):
                self._set_label_if_changed(pow_lbl, pow_lbl.text(), "color: white;")

            msg = f"{type(e).__name__}: {e}"
            if getattr(self, "_ps_last_error", None) != msg:
                self._ps_last_error = msg
                print(f"[PS status] {msg}")

    def sg396_checked(self, box):
        if box.isChecked() is True:
            self.sg396_params_widget_1.setEnabled(True)
            self.sg396_params_widget_2.setEnabled(True)
            self.sg396_status_label.setEnabled(True)
            self.sg396_emit_button.setEnabled(True)
            self.sg396_stop_button.setEnabled(True)
            [self.sg396_opacity_effects[i].setEnabled(False) for i in range(5)]

        else:
            self.sg396_params_widget_1.setEnabled(False)
            self.sg396_params_widget_2.setEnabled(False)
            self.sg396_status_label.setEnabled(False)
            self.sg396_emit_button.setEnabled(False)
            self.sg396_stop_button.setEnabled(False)
            [self.sg396_opacity_effects[i].setEnabled(True) for i in range(5)]

    def choose_sideband(self, opt, nv_freq, side_freq, pulse_axis="x"):
        match pulse_axis:
            case "y":
                delta = 90
            case _:
                delta = 0

        match opt:
            case "Upper":
                frequency = nv_freq - side_freq
                iq_phases = [delta + 90, delta + 0]
            case "Both":
                frequency = nv_freq - side_freq
                iq_phases = [delta + 0, delta + 90, delta + 90, delta + 0]
            case _:
                frequency = nv_freq + side_freq
                iq_phases = [delta + 0, delta + 90]

        return frequency, iq_phases

    def sg396_emit_button_clicked(self):
        fun_kwargs = dict(
            **self.sg396_params_widget_1.all_params(),
            **self.sg396_params_widget_2.all_params(),
        )
        output_freq, iq_phases = self.choose_sideband(
            fun_kwargs["sideband"],
            fun_kwargs["RF_Frequency"],
            fun_kwargs["sideband_freq"],
            "y",
        )

        self.call_sg(lambda sg: sg.set_frequency(output_freq))
        self.call_sg(lambda sg: sg.set_rf_amplitude(fun_kwargs["RF_Power"]))
        self.call_sg(lambda sg: sg.set_mod_type("QAM"))
        self.call_sg(lambda sg: sg.set_mod_function("external"))
        self.call_sg(lambda sg: sg.set_mod_toggle(1))

        self.call_awg(
            lambda awg: awg.set_sequence(
                **{
                    "seq": "Test",
                    "channel": fun_kwargs["channel"],
                    "RF_Frequency": fun_kwargs["RF_Frequency"],
                    "deer_power": fun_kwargs["deer_power"],
                    "i_offset": fun_kwargs["i_offset"],
                    "q_offset": fun_kwargs["q_offset"],
                    "sideband_power": fun_kwargs["sideband_power"],
                    "sideband_freq": fun_kwargs["sideband_freq"],
                    "iq_phases": iq_phases,
                }
            )
        )

        self.call_sg(lambda sg: sg.set_rf_toggle(1))

        self.sg396_status_label.setStyleSheet(
            "color: black; background-color: gold; border: 4px solid black;"
        )
        self.sg396_status_label.setText(
            f"SRS SG396 Status: ON ({output_freq*1e-9} GHz at {fun_kwargs['RF_Power']*1e6} uW)"
        )

    def sg396_stop_button_clicked(self):
        self.call_sg(lambda sg: sg.set_rf_toggle(0))
        self.call_sg(lambda sg: sg.set_mod_toggle(0))
        self.call_awg(lambda awg: awg.set_disabled())

        self.sg396_status_label.setStyleSheet(
            "color: white; background-color: black; border: 4px solid black;"
        )
        self.sg396_status_label.setText("SRS SG396 Status: OFF")

    def laser_temp_clicked(self, component):
        match component:
            case "diode":
                diode_temp = self.call_laser(lambda laser: laser.get_diode_temp())
                self.laser_diode_temp_label.setStyleSheet(
                    "color: white; background-color: black; border: 4px solid black;"
                )
                self.laser_diode_temp_label.setText(f"{diode_temp:.2f} °C")
            case "base":
                heat_sink_temp = self.call_laser(
                    lambda laser: laser.get_baseplate_temp()
                )
                self.laser_baseplate_temp_label.setStyleSheet(
                    "color: white; background-color: black; border: 4px solid black;"
                )
                self.laser_baseplate_temp_label.setText(f"{heat_sink_temp:.2f} °C")
            case _:
                pass

    def get_laser_status(self):
        if getattr(self, "_closing", False):
            return None

        lbl = getattr(self, "laser_status_label", None)

        laser_status = self.call_laser(lambda laser: laser.get_status())

        # If RPC failed, call_laser returns None
        if laser_status is None:
            if _alive(lbl):
                self._set_label_if_changed(
                    lbl,
                    "Status: ---",
                    "color: white; background-color: #800; border: 2px solid #f00;",
                )
            return None

        # Normal states
        if laser_status in ("Warm Up", "Standby", "Searching SLM point"):
            style = "color: black; background-color: orange; border: 4px solid black;"
        elif laser_status == "Laser ON":
            style = (
                "color: black; background-color: limegreen; border: 4px solid black;"
            )
        elif laser_status in ("Error", "Alarm"):
            style = "color: black; background-color: red; border: 4px solid black;"
        else:
            style = "color: black; background-color: white; border: 4px solid black;"

        if _alive(lbl):
            self._set_label_if_changed(lbl, f"{laser_status}", style)

        return laser_status

    def get_interlock_status(self):
        interlock_status = self.call_laser(lambda laser: laser.get_interlock_state())
        
        if isinstance(interlock_status, bytes):
            interlock_status = interlock_status.decode().strip()

        if interlock_status == '1':
            self.laser_interlock_status_label.setStyleSheet(
                "color: black; background-color: limegreen; border: 4px solid black;"
            )
            self.laser_interlock_status_label.setText("Interlock: CLOSED")
        else:
            self.laser_interlock_status_label.setStyleSheet(
                "color: black; background-color: red; border: 4px solid black;"
            )
            self.laser_interlock_status_label.setText("Interlock: OPEN")

        return interlock_status

    def laser_alarm_reset_clicked(self):
        self.call_laser(lambda laser: laser.reset_alarm())
        self.get_laser_status()
        self.get_interlock_status()

    def toggle_laser(self, b):
        match b.text():
            case "CW ON":
                if b.isChecked() is True:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_on(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: ON")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: limegreen; border: 4px solid black;"
                    )
                    self.laser_power_label.setStyleSheet("color: limegreen;")
                else:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_off(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: OFF")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: white; border: 4px solid black;"
                    )
                    self.laser_power_label.setStyleSheet("color: white;")

            case "CW OFF":
                if b.isChecked() is True:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_off(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: OFF")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: white; border: 4px solid black;"
                    )
                    self.laser_power_label.setStyleSheet("color: white;")
                else:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_on(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: ON")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: limegreen; border: 4px solid black;"
                    )
                    self.laser_power_label.setStyleSheet("color: limegreen;")

    def _set_cw_controls_enabled(self, enabled: bool):
        # skip if window is closing
        if getattr(self, "_closing", False):
            return

        # avoid redundant work (optional cache)
        if getattr(self, "_cw_enabled_last", None) == enabled:
            return
        self._cw_enabled_last = enabled

        # radio buttons (only touch if alive)
        for rb in (getattr(self, "laser_b1", None), getattr(self, "laser_b2", None)):
            if _alive(rb):
                rb.blockSignals(True)
                rb.setEnabled(enabled)
                rb.blockSignals(False)
        
        # laser power controls - set enabled/disabled if owner is GUI/external
        if not enabled:
            self.set_section_enabled(self.laser_control_frame, enabled=False)
            self.set_section_enabled(self.detector_frame, enabled=False)
        else:
            self.set_section_enabled(self.laser_control_frame, enabled=True)
            self.set_section_enabled(self.detector_frame, enabled=True)
            
    def toggle_optics_mode(self, b):
        self.call_daq(lambda daq: daq.open_do_task("flip mirror"))
        self.call_daq(lambda daq: daq.start_do_task())
        
        match b.text():
            case "TIR":
                self.call_daq(
                    lambda daq: daq.write_do_task("flip mirror", detector="tirf")
                )
            case _:
                self.call_daq(
                    lambda daq: daq.write_do_task("flip mirror", detector="apd")
                )

        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())

    def toggle_detector_mode(self, b):
        self.call_daq(lambda daq: daq.open_do_task("flip mirror 2"))
        self.call_daq(lambda daq: daq.start_do_task())

        match b.text():
            case "APD":
                self.call_daq(
                    lambda daq: daq.write_do_task("flip mirror 2", detector="apd")
                )
            case "BPD":
                self.call_daq(
                    lambda daq: daq.write_do_task("flip mirror 2", detector="bpd")
                )

        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())

    def nd_filter_changed(self):
        self.nd_filter_choice = self.nd_filter_opts.currentText()
        idx = self.nd_filter_opts.findText(self.nd_filter_choice)
        self.call_filter_wheel(lambda filter_wheel: filter_wheel.set_pos(idx + 1))

    def bpd_shutter_status_changed(self):
        self.call_daq(lambda daq: daq.open_do_task("shutter"))
        self.call_daq(lambda daq: daq.start_do_task())

        if self.bpd_shutter_button.text() == "Open BPD shutter":
            self.bpd_shutter_button.setText("Close BPD shutter")
            self.call_daq(
                lambda daq: daq.write_do_task("shutter", shutter_status="open")
            )
            self.bpd_shutter_status_label.setStyleSheet(
                "color: black; background-color: white; border: 4px solid black;"
            )
            self.bpd_shutter_status_label.setText("BPD Shutter: OPEN")
        else:
            self.bpd_shutter_button.setText("Open BPD shutter")
            self.call_daq(
                lambda daq: daq.write_do_task("shutter", shutter_status="close")
            )
            self.bpd_shutter_status_label.setStyleSheet(
                "color: white; background-color: black; border: 4px solid black;"
            )
            self.bpd_shutter_status_label.setText("BPD Shutter: CLOSED")

        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())

    def bpd_shutter_status_read(self):
        self.call_daq(lambda daq: daq.open_do_task("shutter"))
        self.call_daq(lambda daq: daq.start_do_task())
        bpd_shutter_mode_is_open = self.call_daq(lambda daq: daq.read_do_task())

        if bpd_shutter_mode_is_open:
            self.bpd_shutter_button.setText("Close BPD shutter")
            self.bpd_shutter_status_label.setStyleSheet(
                "color: black; background-color: white; border: 4px solid black;"
            )
            self.bpd_shutter_status_label.setText("BPD Shutter: OPEN")
        else:
            self.bpd_shutter_button.setText("Open BPD shutter")
            self.bpd_shutter_status_label.setStyleSheet(
                "color: white; background-color: black; border: 4px solid black;"
            )
            self.bpd_shutter_status_label.setText("BPD Shutter: CLOSED")

        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())

    def laser_power_changed(self):
        try:
            self.laser_power = float(self.laser_current_value.text())
        except ValueError:
            logger.debug("Invalid laser power value. Try again.")
        else:
            self.laser_power_label.setText(
                f"Power Setpoint: {str(round((self.laser_power), 1))} mW"
            )
            self.call_laser(
                lambda laser: laser.set_diode_current_realtime(self.laser_power)
            )

    def laser_shutter_status_changed(self):
        if self.laser_shutter_button.text() == "Open shutter":
            self.laser_shutter_button.setText("Close shutter")
            self.call_laser_shutter(lambda laser_shutter: laser_shutter.open_shutter())
        else:
            self.laser_shutter_button.setText("Open shutter")
            self.call_laser_shutter(lambda laser_shutter: laser_shutter.close_shutter())

    def kill_process(self):
        """Stop the run process."""
        self.run_proc.kill()
