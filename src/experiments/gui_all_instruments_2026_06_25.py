"""
Copyright (c) 2021, Jacob Feder
All rights reserved.

This work is licensed under the terms of the 3-Clause BSD license.
For a copy, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

"""
Magnet Mount Motion and Magnet Alignment GUI
Evan Villafranca - 2026
"""

from importlib import reload
import logging
from multiprocessing import Event, Queue
from queue import Empty
from turtle import left

from pyqtgraph import ComboBox, SpinBox
from PyQt6.QtCore import Qt, QTimer, QSignalBlocker
import PyQt6.QtGui as QtGui
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
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import nnmr_magnet_motion_2026_06_25 as mag
from find_magnet_position_2 import find_b_for_z_from_fits

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

    EXP_QUEUE_CHECK_TIME = 200  # ms
    QUEUE_CHECK_TIME = 100  # ms
    EQUIP_STATUS_CHECK_TIME = 500  # ms
    GUI_OWNER_PREFIX = "GUI_"

    def __init__(self, status_queue=None):
        super().__init__()

        self._mgr = InstrumentManager()
        self._mgr.__enter__()  # open once, close on widget destruction

        self.setWindowTitle("NanoNMR Magnet Alignment")

        self._last_rpc_err = {}  # keep track of last RPC errors. Context -> msg

        ### --- Instrument status from experiments queue handling --- ###
        self.expQueueTimer = QTimer(self)  # timer to check for messages from experiments that are relevant to instruments
        self.expQueueTimer.timeout.connect(self.check_queue_from_exp_inst)
        self.expQueueTimer.start(self.EXP_QUEUE_CHECK_TIME)

        ### --- Magnet status update handling --- ###
        self.updateTimer = QTimer(self)  # timer to update widget from queue
        self.updateTimer.timeout.connect(self.check_queue_from_mag)
        self.updateTimer.start(self.QUEUE_CHECK_TIME)

        ### --- Hardware status update handling --- ###
        self.hwTimer = QTimer(self)
        # self.hwTimer.timeout.connect(self.get_laser_status)
        # self.hwTimer.timeout.connect(self.get_interlock_status)
        self.hwTimer.timeout.connect(self.check_ps_status)
        self.hwTimer.timeout.connect(self.check_laser_shutter_status)
        # self.hwTimer.timeout.connect(self.check_laser_temp_status)
        self.hwTimer.start(self.EQUIP_STATUS_CHECK_TIME)

        self._gui_id = "GUI_Instruments"

        self.exp_inst_queue = status_queue  # queue for receiving status updates from experiments that are relevant to instruments 

        # current stage positions
        self.curr_z = 0
        self.curr_theta = 0
        self.curr_phi = 0
        self.z_offset = 0

        self.z_min = self._mgr.zaber.lower_bound
        self.z_max = self._mgr.zaber.upper_bound
        self.theta_min = self._mgr.thor_polar.lower_bound
        self.theta_max = self._mgr.thor_polar.upper_bound
        self.phi_min = self._mgr.thor_azi.lower_bound
        self.phi_max = self._mgr.thor_azi.upper_bound

        # Setup professional fonts and styling
        self._setup_professional_styles()

        self.q = Queue()
        self._mag_stop_event = Event()

        # initialize widgets
        self.init_mag_widgets()
        self.init_r_widgets()
        self.init_polar_widgets()
        self.init_azi_widgets()
        self.init_laser_widgets()
        self.init_inst_status_widgets()
        self.init_detector_widgets()

        self.layouts()
        self.park_all()

        # one-time immediate refresh after UI is live
        QTimer.singleShot(0, self._initial_refresh)

    def _set_font_recursive(self, widget, font):
        """Recursively set font on widget and all its children."""
        widget.setFont(font)
        for child in widget.findChildren(QWidget):
            child.setFont(font)

    def _setup_professional_styles(self):
        """Setup centralized professional styling for all widgets."""
        # Define fonts
        self.base_font = QtGui.QFont("Segoe UI", 12)
        self.label_font = QtGui.QFont("Segoe UI", 13)
        self.bold_label_font = QtGui.QFont("Segoe UI", 13)
        self.bold_label_font.setBold(True)
        self.magnet_label_font = QtGui.QFont("Segoe UI", 14)
        self.magnet_label_font.setBold(True)
        self.magnet_label_font.setItalic(True)
        self.larger_label_font = QtGui.QFont("Segoe UI", 13)
        self.largest_label_font = QtGui.QFont("Segoe UI", 20)
        self.largest_label_font.setBold(True)
        self.header_font = QtGui.QFont("Segoe UI", 18)
        self.header_font.setBold(True)

        # Frame header styles (colored bars at top of frames)
        self.magnet_header_style = """
            QLabel {
                background-color: rgba(133, 65, 65, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid #854141;
                border-radius: 6px;
            }
        """
        self.b_field_header_style = """
            QLabel {
                background-color: white;
                color: rgba(150, 80, 80, 1);
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid #CF6161;
                border-radius: 6px;
            }
        """
        self.thor_header_style = """
            QLabel {
                background-color: rgba(133, 65, 65, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid #854141;
                border-radius: 6px;
            }
        """
        self.thor_header_motion_style = """
            QLabel {
                background-color: rgba(88, 34, 122, 0.18);
                color: yellow;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid #6E3FAF;
                border-radius: 6px;
            }
        """

        self.status_header_style = """
            QLabel {
                background-color: rgba(113, 113, 113, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid #717171;
                border-radius: 6px;
            }
        """

        self.laser_header_style = """
            QLabel {
                background-color: rgba(50, 205, 50, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid limegreen;
                border-radius: 6px;
            }
        """

        self.inst_status_header_style = """
            QLabel {
                background-color: rgba(255, 165, 0, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid orange;
                border-radius: 6px;
            }
        """

        self.detector_header_style = """
            QLabel {
                background-color: rgba(255, 0, 0, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid red;
                border-radius: 6px;
            }
        """

        # Magnet control button: Dark Gray with white border
        self.magnet_move_button_style = """
            QPushButton {
                background-color: rgba(88, 34, 122, 0.18);
                color: white;
                border: 2px solid #6E3FAF;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(88, 34, 122, 0.28);
            }
            QPushButton:pressed {
                background-color: rgba(88, 34, 122, 0.38);
            }
        """
        
        self.magnet_stop_button_style = """
            QPushButton {
                background-color: rgba(130, 26, 7, 0.25);
                color: white;
                border: 2px solid #FF2500;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(130, 26, 7, 0.35);
            }
            QPushButton:pressed {
                background-color: rgba(130, 26, 7, 0.45);
            }
        """

        # Standby button: Yellow/Green
        self.standby_button_style = """
            QPushButton {
                background-color: #3b3f12;
                color: #f2f1d6;
                border: 2px solid #a9a65a;
                border-radius: 4px;
                padding: 6px 12px;
                font-style: italic;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4f1c;
            }
            QPushButton:pressed {
                background-color: #2f3310;
            }
        """

        self.magnet_stop_all_button_style = """
            QPushButton {
                background-color: rgba(130, 26, 7, 0.25);
                color: white;
                border: 2px solid #FF2500;
                border-radius: 4px;
                padding: 6px 12px;
                font-style: italic;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(130, 26, 7, 0.35);
            }
            QPushButton:pressed {
                background-color: rgba(130, 26, 7, 0.45);
            }
        """

        # All disable/enable button: Teal (complementary to standby yellow)
        self.all_disable_button_style_active = """
            QPushButton {
                background-color: rgba(0, 110, 20, 0.25);
                color: #CDEFD8;
                border: 2px solid #5CC78A;
                border-radius: 4px;
                padding: 6px 12px;
                font-style: italic;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 110, 20, 0.35);
                border-color: #6ED79A;
            }
            QPushButton:pressed {
                background-color: rgba(0, 110, 20, 0.45);
                border-color: #7CE6A8;
            }
        """
        self.all_disable_button_style_inactive = """
            QPushButton {
                background-color: rgba(130, 26, 7, 0.25);
                color: #FFD2B8;
                border: 2px solid #F08A5B;
                border-radius: 4px;
                padding: 6px 12px;
                font-style: italic;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(130, 26, 7, 0.35);
                border-color: #F59A6A;
            }
            QPushButton:pressed {
                background-color: rgba(130, 26, 7, 0.45);
                border-color: #FAB078;
            }
        """

        # Laser control button: Dark Green
        self.laser_button_style = """
            QPushButton {
                background-color: #184B00;
                color: white;
                border: 1px solid #64D700;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1F5C0F;
            }
            QPushButton:pressed {
                background-color: #0F3A00;
            }
        """

        # BPD shutter button: Dark Blue
        self.shutter_button_style = """
            QPushButton {
                background-color: #002E4B;
                color: white;
                border: 1px solid #0088D7;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #003A5E;
            }
            QPushButton:pressed {
                background-color: #001F35;
            }
        """

        # Combobox
        self.combobox_style = """
            QComboBox {
                background-color: #1a1a1a;
                color: white;
                border: 2px solid #FF8C00;
                border-radius: 6px;
                padding: 6px 8px;
                font-weight: 500;
            }
            QComboBox:hover {
                background-color: #252525;
                border: 2px solid #FFA500;
            }
            QComboBox:focus {
                background-color: #2a2a2a;
                border: 2px solid #FFB84D;
                padding: 6px 8px;
                outline: none;
            }
            QComboBox:on {
                background-color: #1f1f1f;
                border: 2px solid #FFB84D;
            }
            QComboBox::drop-down {
                border: none;
                background-color: transparent;
                border-left: 1px solid #FF8C00;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 12px;
                height: 12px;
            }
            QComboBox::down-arrow:on {
                top: 1px;
            }
            QComboBox QAbstractItemView {
                background-color: #1f1f1f;
                color: white;
                border: 2px solid #FF8C00;
                border-radius: 4px;
                outline: none;
                selection-background-color: #FF8C00;
                selection-color: white;
                padding: 3px 0px;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 8px;
                border-radius: 2px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #FFB84D;
                color: #1a1a1a;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #FF8C00;
                color: white;
            }
        """

        # ND Filter combobox
        self.nd_filter_combobox_style = """
            QComboBox {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox:focus {
                background-color: #323232;
                border: 2px solid #4DA6D8;
                padding: 3px;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #2a2a2a;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: white;
                selection-background-color: #4DA6D8;
                border: 1px solid #4DA6D8;
            }
        """

        # Text edit
        self.lineedit_style = """
            QLineEdit {
                background-color: #141414;
                color: white;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px;
            }
            QLineEdit:focus {
                background-color: #323232;
                border: 2px solid #4166F5;
                padding: 3px;
            }
        """

        # Status labels with colored left accent bars
        self.status_label_style = """
            QLabel {
                color: white;
                background-color: black;
                border-left: 4px solid #854141;
                border-top: 1px solid #555555;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-radius: 2px;
                padding-left: 8px;
            }
        """
        
        self.laser_status_label_style = """
            QLabel {
                color: white;
                background-color: black;
                border-left: 4px solid limegreen;
                border-top: 1px solid #555555;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-radius: 2px;
                padding-left: 8px;
            }
        """
        
        self.ps_status_label_style = """
            QLabel {
                color: white;
                background-color: #222;
                border-left: 4px solid #717171;
                border-top: 1px solid #555555;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-radius: 2px;
                padding-left: 8px;
            }
        """
        
        self.shutter_status_label_style = """
            QLabel {
                color: white;
                background-color: black;
                border-left: 4px solid #0088D7;
                border-top: 1px solid #555555;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-radius: 2px;
                padding-left: 8px;
            }
        """
        
        self.enabled_label_style_active = """
            QLabel {
                color: #00FF2E;
                background-color: rgba(0, 110, 20, 0.3);
                border: 1px solid #00FF2E;
                border-radius: 4px;
                padding: 4px 8px;
                font-style: italic;
            }
        """
        
        self.enabled_label_style_inactive = """
            QLabel {
                color: #FF5500;
                background-color: rgba(130, 26, 7, 0.3);
                border: 1px solid #FF5500;
                border-radius: 4px;
                padding: 4px 8px;
                font-style: italic;
            }
        """

    def set_experiment_inst_queue(self, q):
        self.exp_inst_queue = q

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
        # self.get_laser_status()
        # self.get_interlock_status()
        self.check_ps_status()
        self.check_laser_shutter_status()
        # self.check_laser_temp_status()

        # get microscope detector mode (flip mirror 2 state) upon startup
        self.call_daq(lambda daq: daq.open_do_task("flip mirror 2"))
        self.call_daq(lambda daq: daq.start_do_task())
        microscope_detector_mode_is_apd = self.call_daq(lambda daq: daq.read_do_task())
        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())
        if microscope_detector_mode_is_apd:
            self._set_radio_safely(self.apd_flipper_b1, True)
        else:
            self._set_radio_safely(self.apd_flipper_b2, True)

        # get BPD shutter state upon startup
        self.bpd_shutter_status_read()  # this will query hardware and update button + label accordingly

        # magnet state
        self._mgr.thor_azi.update_positions_callback()
        self._mgr.thor_polar.update_positions_callback()  
        self._mgr.zaber.update_positions_callback() # query current magnet stage positions and update labels

        azi_pos_init = self._mgr.thor_azi.current_position
        polar_pos_init = self._mgr.thor_polar.current_position
        z_pos_init = self._mgr.zaber.current_positions[0] # current stage positions

        self.azi_label.setText(
            f"\u03c6 = {azi_pos_init:.1f}\N{DEGREE SIGN}"
        )
        self.polar_label.setText(
            f"\u03b8 = {polar_pos_init:.1f}\N{DEGREE SIGN}"
        )
        self.z_label.setText(f"z = {z_pos_init:.1f} mm")
        self.d_label.setText(f"d = {self.d_from_z(z_pos_init):.1f} mm")
        print(f"Initial magnet position: z={z_pos_init:.1f} mm, polar={polar_pos_init:.1f} deg, azi={azi_pos_init:.1f} deg")
        b_field_current = find_b_for_z_from_fits(
                        z_target=z_pos_init,
                        azi=azi_pos_init,
                        polar=polar_pos_init
                    )
        self.b_label.setText(f"B \u2248 {b_field_current:.1f} G")

        self.z_label.setStyleSheet(self.thor_header_style)
        self.polar_label.setStyleSheet(self.thor_header_style)
        self.azi_label.setStyleSheet(self.thor_header_style)

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
    def zaber(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.zaber
    
    def call_zaber(self, fn):
        return self._call(lambda: fn(self.zaber), context="zaber")
    
    @property
    def thor_polar(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.thor_polar
    
    def call_thor_polar(self, fn):
        return self._call(lambda: fn(self.thor_polar), context="thor_polar")
    
    @property
    def thor_azi(self):
        if getattr(self, "_closing", False):
            raise RuntimeError("GUI is closing.")
        return self._mgr.thor_azi
    
    def call_thor_azi(self, fn):
        return self._call(lambda: fn(self.thor_azi), context="thor_azi")

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

    #TODO: add HWP and BPD ND filter control to GUI
    # @property
    # def hwp(self):
    #     if getattr(self, "_closing", False):
    #         raise RuntimeError("GUI is closing.")
    #     return self._mgr.hwp
    
    # def call_hwp(self, fn):
    #     return self._call(lambda: fn(self.hwp), context="hwp")
    
    # @property
    # def bpd_nd_filter(self):
    #     if getattr(self, "_closing", False):
    #         raise RuntimeError("GUI is closing.")
    #     return self._mgr.bpd_nd_filter

    # def call_bpd_nd_filter(self, fn):
    #     return self._call(lambda: fn(self.bpd_nd_filter), context="bpd_nd_filter")

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
    def init_mag_widgets(self):
        # magnet stage widgets
        self.magnet_label = QLabel("Magnet Stage Configuration")
        self.magnet_label.setFixedHeight(52)
        self.magnet_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.magnet_label.setFont(self.header_font)
        self.magnet_label.setStyleSheet(self.magnet_header_style)
        
        self.magnet_instruction_header_label = QLabel("Magnet Alignment Instructions:")
        self.magnet_instruction_header_label.setFont(self.magnet_label_font)
        self.magnet_instruction_header_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.magnet_instruction_header_label.setFixedHeight(28)
        self.magnet_instruction_header_label.setStyleSheet(
            "background-color: #1a1a1a;"
            "border-bottom: 1px solid #3d3d3d;"
            "padding: 0px 4px 0px 4px;"
        )
        
        self.magnet_instruction_text_label = QLabel(
            "1. Set azimuthal angle φ position (either absolute stage angle or jog relative to the current position).\n\n"
            "2. Set polar angle θ position (same options as above).\n\n"
            "3. Set Zaber linear stage position directly, or set a target B field and let the software calculate Z for you.\n\n"
        )
        magnet_instruction_font = QFont(self.magnet_label_font)
        magnet_instruction_font.setBold(False)
        magnet_instruction_font.setPointSize(max(1, self.label_font.pointSize() - 2))
        self.magnet_instruction_text_label.setFont(magnet_instruction_font)
        self.magnet_instruction_text_label.setStyleSheet(
            "background-color: #1a1a1a; padding: 2px 0px;"
        )
        self.magnet_instruction_text_label.setWordWrap(True)
        self.magnet_instruction_text_label.setMinimumWidth(260)
        self.magnet_instruction_text_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.magnet_instruction_scroll = QScrollArea()
        self.magnet_instruction_scroll.setWidget(self.magnet_instruction_text_label)
        self.magnet_instruction_scroll.setWidgetResizable(True)
        self.magnet_instruction_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.magnet_instruction_scroll.setMinimumWidth(260)
        self.magnet_instruction_scroll.setMaximumHeight(150)
        self.magnet_instruction_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.magnet_instruction_scroll.setStyleSheet(
            "QScrollArea { background-color: #1a1a1a; border: none; }"
            "QScrollBar:vertical { background-color: #202020; border: 1px solid #3d3d3d; width: 12px; }"
            "QScrollBar::handle:vertical { background-color: #7a7a7a; border-radius: 6px; min-height: 24px; }"
            "QScrollBar::handle:vertical:hover { background-color: #9a9a9a; }"
        )

        self.all_disable_button = QPushButton("ENABLE ALL STAGES")
        self.all_disable_button.setStyleSheet(self.all_disable_button_style_active)
        self.all_disable_button.setFont(self.label_font)
        self.all_disable_proc = ProcessRunner()
        self.all_disable_button.clicked.connect(self.all_disable_clicked)

        self.all_standby_button = QPushButton("MOVE TO STANDBY")
        self.all_standby_button.setStyleSheet(self.standby_button_style)
        self.all_standby_button.setFont(self.label_font)
        self.all_standby_proc = ProcessRunner()
        self.all_standby_button.clicked.connect(self.all_standby_clicked)

        self.stop_all_button = QPushButton("STOP ALL STAGES")
        self.stop_all_button.setStyleSheet(self.magnet_stop_all_button_style)
        self.stop_all_button.setFont(self.label_font)
        self.stop_all_proc = ProcessRunner()
        self.stop_all_button.clicked.connect(self.stop_all_button_clicked)

        self.magnet_status_bar_label = QLabel("Magnet Status:")
        self.magnet_status_bar_label.setFont(self.magnet_label_font)
        self.magnet_status_bar_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.magnet_status_bar_label.setFixedHeight(28)
        self.magnet_status_bar_label.setStyleSheet(
            "background-color: #1a1a1a;"
            "border-bottom: 1px solid #3d3d3d;"
            "padding: 0px 4px 0px 4px;"
        )

        self.status_label = QLabel("Status here...")
        self.status_label.setStyleSheet(
            f"{self.status_label_style} padding: 2px 0px;"
        )
        status_label_font = QFont(self.label_font)
        status_label_font.setBold(True)
        self.status_label.setFont(status_label_font)
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumHeight(120)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        
        self.status_scroll_area = QScrollArea()
        self.status_scroll_area.setWidget(self.status_label)
        self.status_scroll_area.setWidgetResizable(True)
        self.status_scroll_area.setMinimumWidth(260)
        self.status_scroll_area.setMaximumHeight(120)
        self.status_scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.status_scroll_area.setStyleSheet(
            "QScrollArea { background-color: #1a1a1a; border: none; }"
            "QScrollBar:vertical { background-color: #202020; border: 1px solid #3d3d3d; width: 12px; }"
            "QScrollBar::handle:vertical { background-color: #7a7a7a; border-radius: 6px; min-height: 24px; }"
            "QScrollBar::handle:vertical:hover { background-color: #9a9a9a; }"
        )

    def init_r_widgets(self):
        """r stage widgets"""
        self.r_opacity_effects = []
        for _ in range(5):
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.3)
            self.r_opacity_effects.append(effect)

        self.z_label = QLabel("z")
        self.z_label.setFixedHeight(40)
        self.z_label.setFixedWidth(200)
        self.z_label.setFont(self.largest_label_font)
        self.z_label.setStyleSheet(self.thor_header_style)

        self.r_move_checkbox = QCheckBox("Zaber Move Type: ")
        self.r_move_checkbox.setChecked(False)
        self.r_move_checkbox.setFont(self.label_font)
        self.r_move_checkbox.setFixedWidth(185)
        self.r_move_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.r_move_checkbox.stateChanged.connect(
            lambda: self.move_button_checked(self.r_move_checkbox)
        )

        self.r_move_types = QComboBox()
        self.r_move_types.addItems(["Absolute", "Jog", "B Field"])
        self.r_move_types.setFont(self.label_font)
        self.r_move_types.setStyleSheet(self.combobox_style)
        self.r_move_types.setFixedWidth(90)
        self.r_move_types.currentIndexChanged.connect(self.r_move_type_changed)
        
        self.r_enabled_label = QLabel("DISABLED")
        self.r_enabled_label.setFont(self.label_font)
        self.r_enabled_label.setFixedHeight(35)
        self.r_enabled_label.setStyleSheet(self.enabled_label_style_inactive)
        self.r_move_label = QLabel("Move Zaber (0 \u2013 100 mm) to: ")
        self.r_move_label.setFixedWidth(225)
        self.r_move_label.setFont(self.label_font)
        self.r_move_position = QLineEdit()
        self.r_move_position.setStyleSheet(self.lineedit_style)
        self.r_move_position.setFont(self.label_font)
        self.r_move_position.setFixedWidth(60)
        self.r_move_position_units = QLabel("")
        self.r_move_position_units.setFont(self.label_font)
        self.r_execute_move_button = QPushButton("MOVE z")
        self.r_execute_move_button.setFixedWidth(95)
        self.r_execute_move_button.setStyleSheet(self.magnet_move_button_style)
        self.r_execute_move_button.setFont(self.label_font)
        self.r_execute_move_button_proc = ProcessRunner()
        self.r_execute_move_button.clicked.connect(
            lambda: self.single_move_clicked("zaber")
        )

        self.r_move_types.setEnabled(False)
        self.r_move_position.setEnabled(False)
        self.r_move_position_units.setEnabled(False)
        self.r_execute_move_button.setEnabled(False)

        self.r_move_types.setGraphicsEffect(self.r_opacity_effects[0])
        self.r_move_label.setGraphicsEffect(self.r_opacity_effects[1])
        self.r_move_position.setGraphicsEffect(self.r_opacity_effects[2])
        self.r_move_position_units.setGraphicsEffect(self.r_opacity_effects[3])
        self.r_execute_move_button.setGraphicsEffect(self.r_opacity_effects[4])

        self.r_stop_button = QPushButton("STOP z")
        self.r_stop_button.setFixedWidth(95)
        self.r_stop_button.setStyleSheet(self.magnet_stop_button_style)
        self.r_stop_button.setFont(self.label_font)
        self.r_stop_button_proc = ProcessRunner()
        self.r_stop_button.clicked.connect(lambda: self.stop_button_clicked("zaber"))

        self.d_label = QLabel("d")
        self.d_label.setFixedHeight(40)
        self.d_label.setFixedWidth(200)
        self.d_label.setFont(self.header_font)
        self.d_label.setStyleSheet(self.b_field_header_style)
        self.d_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.b_label = QLabel("B \u2248 0 G")
        self.b_label.setFixedHeight(40)
        self.b_label.setFixedWidth(200)
        self.b_label.setFont(self.header_font)
        self.b_label.setStyleSheet(self.b_field_header_style)
        self.b_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def init_polar_widgets(self):
        """polar stage widgets"""
        self.polar_opacity_effects = []
        for _ in range(5):
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.3)
            self.polar_opacity_effects.append(effect)

        self.polar_label = QLabel("\u03b8")
        self.polar_label.setFixedHeight(40)
        self.polar_label.setFixedWidth(200)
        self.polar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.polar_label.setFont(self.largest_label_font)
        self.polar_label.setStyleSheet(self.thor_header_style)

        self.polar_move_checkbox = QCheckBox("Polar Move Type: ")
        self.polar_move_checkbox.setChecked(False)
        self.polar_move_checkbox.setFont(self.label_font)
        self.polar_move_checkbox.setFixedWidth(185)
        self.polar_move_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.polar_move_checkbox.stateChanged.connect(
            lambda: self.move_button_checked(self.polar_move_checkbox)
        )

        self.polar_move_types = QComboBox()
        self.polar_move_types.addItems(["Absolute", "Jog"])
        self.polar_move_types.currentIndexChanged.connect(self.polar_move_type_changed)
        self.polar_move_types.setFont(self.label_font)
        self.polar_move_types.setStyleSheet(self.combobox_style)
        self.polar_move_types.setFixedWidth(90)
        self.polar_enabled_label = QLabel("DISABLED")
        self.polar_enabled_label.setFont(self.label_font)
        self.polar_enabled_label.setFixedHeight(35)
        self.polar_enabled_label.setStyleSheet(self.enabled_label_style_inactive)
        self.polar_move_label = QLabel(f"Move \u03b8 ({self.theta_min}\N{DEGREE SIGN} \u2013 {self.theta_max}\N{DEGREE SIGN}) to: ")
        self.polar_move_label.setFixedWidth(225)
        self.polar_move_label.setFont(self.label_font)
        self.polar_move_position = QLineEdit()
        self.polar_move_position.setStyleSheet(self.lineedit_style)
        self.polar_move_position.setFont(self.label_font)
        self.polar_move_position.setFixedWidth(60)
        self.polar_move_position_units = QLabel("")
        self.polar_move_position_units.setFont(self.label_font)
        self.polar_execute_move_button = QPushButton("MOVE \u03b8")
        self.polar_execute_move_button.setFixedWidth(95)
        self.polar_execute_move_button.setStyleSheet(self.magnet_move_button_style)
        self.polar_execute_move_button.setFont(self.label_font)
        self.polar_execute_move_button_proc = ProcessRunner()
        self.polar_execute_move_button.clicked.connect(
            lambda: self.single_move_clicked("thor_polar")
        )

        self.polar_move_types.setEnabled(False)
        self.polar_move_position.setEnabled(False)
        self.polar_move_position_units.setEnabled(False)
        self.polar_execute_move_button.setEnabled(False)

        self.polar_move_types.setGraphicsEffect(self.polar_opacity_effects[0])
        self.polar_move_label.setGraphicsEffect(self.polar_opacity_effects[1])
        self.polar_move_position.setGraphicsEffect(self.polar_opacity_effects[2])
        self.polar_move_position_units.setGraphicsEffect(self.polar_opacity_effects[3])
        self.polar_execute_move_button.setGraphicsEffect(self.polar_opacity_effects[4])

        self.polar_stop_button = QPushButton("STOP \u03b8")
        self.polar_stop_button.setFixedWidth(95)
        self.polar_stop_button.setStyleSheet(self.magnet_stop_button_style)
        self.polar_stop_button.setFont(self.label_font)
        self.polar_stop_button_proc = ProcessRunner()
        self.polar_stop_button.clicked.connect(
            lambda: self.stop_button_clicked("thor_polar")
        )

    def init_azi_widgets(self):
        """azimuthal stage widgets"""
        self.azi_opacity_effects = []
        for _ in range(5):
            effect = QGraphicsOpacityEffect()
            effect.setOpacity(0.3)
            self.azi_opacity_effects.append(effect)

        self.azi_label = QLabel("\u03c6")
        self.azi_label.setFixedHeight(52)
        self.azi_label.setFixedWidth(200)
        self.azi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.azi_label.setFont(self.largest_label_font)
        self.azi_label.setStyleSheet(self.thor_header_style)

        self.azi_move_checkbox = QCheckBox("Azimuth. Move Type: ")
        self.azi_move_checkbox.setChecked(False)
        self.azi_move_checkbox.setFont(self.label_font)
        self.azi_move_checkbox.setFixedWidth(185)
        self.azi_move_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.azi_move_checkbox.stateChanged.connect(
            lambda: self.move_button_checked(self.azi_move_checkbox)
        )

        self.azi_move_types = QComboBox()
        self.azi_move_types.addItems(["Absolute", "Jog"])
        self.azi_move_types.currentIndexChanged.connect(self.azi_move_type_changed)
        self.azi_move_types.setFont(self.label_font)
        self.azi_move_types.setStyleSheet(self.combobox_style)
        self.azi_move_types.setFixedWidth(90)
        self.azi_enabled_label = QLabel("DISABLED")
        self.azi_enabled_label.setFont(self.label_font)
        self.azi_enabled_label.setFixedHeight(35)
        self.azi_enabled_label.setStyleSheet(self.enabled_label_style_inactive)
        self.azi_move_label = QLabel(f"Move \u03c6 ({self.phi_min}\N{DEGREE SIGN} \u2013 {self.phi_max}\N{DEGREE SIGN}) to: ")
        self.azi_move_label.setFixedWidth(225)
        self.azi_move_label.setFont(self.label_font)
        self.azi_move_position = QLineEdit()
        self.azi_move_position.setStyleSheet(self.lineedit_style)
        self.azi_move_position.setFont(self.label_font)
        self.azi_move_position.setFixedWidth(60)
        self.azi_move_position_units = QLabel("")
        self.azi_move_position_units.setFont(self.label_font)
        self.azi_execute_move_button = QPushButton("MOVE \u03c6")
        self.azi_execute_move_button.setStyleSheet(self.magnet_move_button_style)
        self.azi_execute_move_button.setFont(self.label_font)
        self.azi_execute_move_button.setFixedWidth(95)
        self.azi_execute_move_button_proc = ProcessRunner()
        self.azi_execute_move_button.clicked.connect(
            lambda: self.single_move_clicked("thor_azi")
        )

        self.azi_move_types.setEnabled(False)
        self.azi_move_position.setEnabled(False)
        self.azi_move_position_units.setEnabled(False)
        self.azi_execute_move_button.setEnabled(False)

        self.azi_move_types.setGraphicsEffect(self.azi_opacity_effects[0])
        self.azi_move_label.setGraphicsEffect(self.azi_opacity_effects[1])
        self.azi_move_position.setGraphicsEffect(self.azi_opacity_effects[2])
        self.azi_move_position_units.setGraphicsEffect(self.azi_opacity_effects[3])
        self.azi_execute_move_button.setGraphicsEffect(self.azi_opacity_effects[4])

        self.azi_stop_button = QPushButton("STOP \u03c6")
        self.azi_stop_button.setFixedWidth(95)
        self.azi_stop_button.setStyleSheet(self.magnet_stop_button_style)
        self.azi_stop_button.setFont(self.label_font)
        self.azi_stop_button_proc = ProcessRunner()
        self.azi_stop_button.clicked.connect(
            lambda: self.stop_button_clicked("thor_azi")
        )

    def get_combobox_val(self, combobox):
        return str(combobox.value())
    
    def init_laser_widgets(self):
        self.laser_label = QLabel("Laser")
        self.laser_label.setFixedHeight(40)
        self.laser_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.laser_label.setFont(self.header_font)
        self.laser_label.setStyleSheet(self.laser_header_style)

        self.optics_flipper_label = QLabel("Optical Beam Path:  ")
        self.optics_flipper_label.setFixedHeight(40)
        self.optics_flipper_label.setFont(self.larger_label_font)

        laser_radio_style = """
        QRadioButton {
        color: white;
        background-color: #2D3B7D;
        border: 2px solid #4A7FFF;
        border-radius: 6px;
        padding: 8px 12px;
        spacing: 10px;
        font-weight: bold;
        font-size: 14pt;
        }

        QRadioButton:hover {
        background-color: #3D4B8D;
        border: 2px solid #6A9FFF;
        }

        QRadioButton:checked {
        color: limegreen;
        background-color: #1D2B6D;
        border: 2px solid #5470FF;
        }

        QRadioButton::indicator {
        width: 25px;
        height: 15px;
        border: 2px solid #888;
        border-radius: 7px;
        background-color: #1a1a1a;
        }

        QRadioButton::indicator:hover {
        border: 2px solid #AAA;
        background-color: #2a2a2a;
        }

        QRadioButton::indicator:checked {
        background-color: #5470FF;
        border: 2px solid #5470FF;
        image: none;
        }
        """

        self.optics_flipper_b1 = QRadioButton("EPIFLUORESCENCE")
        self.optics_flipper_b1.toggled.connect(
            lambda: self.toggle_optics_mode(self.optics_flipper_b1)
        )
        self.optics_flipper_b1.setStyleSheet(laser_radio_style)
        self.optics_flipper_b1.setFont(self.bold_label_font)

        self.optics_flipper_b2 = QRadioButton("TIR")
        self.optics_flipper_b2.toggled.connect(
            lambda: self.toggle_optics_mode(self.optics_flipper_b2)
        )
        self.optics_flipper_b2.setStyleSheet(laser_radio_style)
        self.optics_flipper_b2.setFont(self.bold_label_font)

        self.optics_flipper_group = QButtonGroup(self)
        self.optics_flipper_group.setExclusive(True)
        self.optics_flipper_group.addButton(self.optics_flipper_b1)
        self.optics_flipper_group.addButton(self.optics_flipper_b2)


        self.laser_status_label = QLabel("Laser status")
        self.laser_status_label.setFixedHeight(40)
        self.laser_status_label.setFont(self.label_font)

        self.laser_emit_status_label = QLabel("Emission status")
        self.laser_emit_status_label.setStyleSheet(self.laser_status_label_style)
        self.laser_emit_status_label.setFixedHeight(40)
        # self.laser_emit_status_label.setFixedWidth(145)
        self.laser_emit_status_label.setFont(self.label_font)

        self.laser_interlock_status_label = QLabel("Interlock status")
        self.laser_interlock_status_label.setStyleSheet(self.laser_status_label_style)
        self.laser_interlock_status_label.setFixedHeight(40)
        self.laser_interlock_status_label.setFont(self.label_font)

        laser_radio_style_2 = """
        QRadioButton {
        color: white;
        background-color: #2D3B7D;
        border: 2px solid #4A7FFF;
        border-radius: 6px;
        padding: 8px 12px;
        spacing: 10px;
        font-weight: bold;
        font-size: 14pt;
        }

        QRadioButton:hover {
        background-color: #3D4B8D;
        border: 2px solid #6A9FFF;
        }

        QRadioButton:checked {
        color: limegreen;
        background-color: #1D2B6D;
        border: 2px solid #5470FF;
        }

        QRadioButton::indicator {
        width: 25px;
        height: 15px;
        border: 2px solid #888;
        border-radius: 7px;
        background-color: #1a1a1a;
        }

        QRadioButton::indicator:hover {
        border: 2px solid #AAA;
        background-color: #2a2a2a;
        }

        QRadioButton::indicator:checked {
        background-color: #5470FF;
        border: 2px solid #5470FF;
        image: none;
        }
        """
        
        self.laser_b1 = QRadioButton("CW ON")
        self.laser_b1.toggled.connect(lambda: self.toggle_laser(self.laser_b1))
        self.laser_b1.setStyleSheet(laser_radio_style_2)
        self.laser_b1.setFont(self.bold_label_font)

        self.laser_b2 = QRadioButton("STANDBY (OFF - DAILY USE)")
        self.laser_b2.setChecked(True)
        self.laser_b2.toggled.connect(lambda: self.toggle_laser(self.laser_b2))
        self.laser_b2.setStyleSheet(laser_radio_style_2)
        self.laser_b2.setFont(self.bold_label_font)

        self.laser_group = QButtonGroup(self)
        self.laser_group.setExclusive(True)
        self.laser_group.addButton(self.laser_b1)
        self.laser_group.addButton(self.laser_b2)

        self.laser_head_power_label = QLabel("Laser Head Power Output: 5 W")
        self.laser_head_power_label.setFixedHeight(55)
        # self.laser_head_power_label.setFixedWidth(200)
        self.laser_head_power_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.laser_head_power_label.setFont(self.largest_label_font)
        self.laser_head_power_label.setStyleSheet(self.thor_header_style)

        #TODO: add checkbox to disable laser head power setting by default - user should set power by waveplate angle
        self.laser_current_label = QLabel("Diode Current (%):")
        self.laser_current_label.setFont(self.larger_label_font)

        self.laser_current_value = QLineEdit()
        self.laser_current_value.setStyleSheet(self.lineedit_style)
        self.laser_current_value.setFont(self.label_font)

        self.laser_enclosure_output_power_label = QLabel("Laser Enclosure Output Power: 0 mW")
        self.laser_enclosure_output_power_label.setFixedHeight(55)
        # self.laser_enclosure_output_power_label.setFixedWidth(200)
        self.laser_enclosure_output_power_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.laser_enclosure_output_power_label.setFont(self.largest_label_font)
        self.laser_enclosure_output_power_label.setStyleSheet(self.thor_header_style)

        self.waveplate_angle_label = QLabel("\u03BB/2 Setpoint: 0\N{DEGREE SIGN}")
        self.waveplate_angle_label.setFixedHeight(55)
        # self.waveplate_angle_label.setFixedWidth(200)
        self.waveplate_angle_label.setFont(self.larger_label_font)
        self.waveplate_angle_label.setStyleSheet(self.thor_header_style)

        self.laser_power_set_checkbox = QCheckBox("Setpoint Type: ")
        self.laser_power_set_checkbox.setChecked(False)
        self.laser_power_set_checkbox.setFont(self.label_font)
        self.laser_power_set_checkbox.setFixedWidth(185)
        self.laser_power_set_checkbox.setStyleSheet(
            "QCheckBox::indicator:hover {background-color : yellow;}"
            "QCheckBox::indicator:pressed {background-color : lightgreen;}"
        )
        self.laser_power_set_checkbox.stateChanged.connect(
            lambda: self.move_button_checked(self.laser_power_set_checkbox)
        )

        self.laser_power_set_types = QComboBox()
        self.laser_power_set_types.addItems(["Power Level (mW)", "Percent (%)", "\u03BB/2 (\N{DEGREE SIGN})"])
        self.laser_power_set_types.setFont(self.label_font)
        self.laser_power_set_types.setStyleSheet(self.combobox_style)
        self.laser_power_set_types.setFixedWidth(90)
        self.laser_power_set_types.currentIndexChanged.connect(self.laser_power_set_type_changed)

        self.laser_power_set_button = QPushButton("Set Power")
        self.laser_power_set_button.setStyleSheet(self.laser_button_style)
        self.laser_power_set_button.setFont(self.bold_label_font)
        self.laser_power_set_button.clicked.connect(self.laser_power_changed)

        self.laser_power_label = QLabel("CW Power Setpoint: 0 mW")
        self.laser_power_label.setFont(self.label_font)

        self.laser_shutter_button = QPushButton("Open shutter")
        self.laser_shutter_button.setStyleSheet(self.laser_button_style)
        self.laser_shutter_button.setFont(self.label_font)
        self.laser_shutter_button.clicked.connect(self.laser_shutter_status_changed)

        self.lock_front_panel_button = QPushButton("Lock front panel")
        self.lock_front_panel_button.setStyleSheet(self.laser_button_style)
        self.lock_front_panel_button.setFont(self.label_font)
        # self.lock_front_panel_button.clicked.connect(self.lock_front_panel)

    def init_inst_status_widgets(self):
        self.inst_status_label = QLabel("Pulse Streamer Status")
        self.inst_status_label.setFixedHeight(40)
        self.inst_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inst_status_label.setFont(self.header_font)
        self.inst_status_label.setStyleSheet(self.inst_status_header_style)

        self.laser_baseplate_temp_label = QLabel("Baseplate Temp: ---")
        self.laser_baseplate_temp_label.setStyleSheet(self.laser_status_label_style)
        self.laser_baseplate_temp_label.setFixedHeight(40)
        self.laser_baseplate_temp_label.setFont(self.label_font)

        self.laser_diode_temp_label = QLabel("Diode Temp: ---")
        self.laser_diode_temp_label.setStyleSheet(self.laser_status_label_style)
        self.laser_diode_temp_label.setFixedHeight(40)
        self.laser_diode_temp_label.setFont(self.label_font)

        

        self.laser_shutter_status_label = QLabel("Laser Shutter status")
        self.laser_shutter_status_label.setStyleSheet(self.laser_status_label_style)
        self.laser_shutter_status_label.setFixedHeight(40)
        self.laser_shutter_status_label.setFixedWidth(225)
        self.laser_shutter_status_label.setFont(self.label_font)

        self.ps_status_label = QLabel("Pulse Streamer: CONSTANT")
        self.ps_status_label.setStyleSheet(self.ps_status_label_style)
        self.ps_status_label.setFixedHeight(40)
        self.ps_status_label.setFont(self.label_font)
    
    def init_detector_widgets(self):
        self.detector_label = QLabel("Detectors")
        self.detector_label.setFixedHeight(40)
        self.detector_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detector_label.setFont(self.header_font)
        self.detector_label.setStyleSheet(self.detector_header_style)
        
        detector_radio_style = """
        QRadioButton {
        color: white;
        background-color: #2D3B7D;
        border: 2px solid #4A7FFF;
        border-radius: 6px;
        padding: 8px 12px;
        spacing: 10px;
        font-weight: bold;
        font-size: 14pt;
        }

        QRadioButton:hover {
        background-color: #3D4B8D;
        border: 2px solid #6A9FFF;
        }

        QRadioButton:checked {
        color: limegreen;
        background-color: #1D2B6D;
        border: 2px solid #5470FF;
        }

        QRadioButton::indicator {
        width: 25px;
        height: 15px;
        border: 2px solid #888;
        border-radius: 7px;
        background-color: #1a1a1a;
        }

        QRadioButton::indicator:hover {
        border: 2px solid #AAA;
        background-color: #2a2a2a;
        }

        QRadioButton::indicator:checked {
        background-color: #5470FF;
        border: 2px solid #5470FF;
        image: none;
        }
        """

        self.apd_flipper_b1 = QRadioButton("APD")
        self.apd_flipper_b1.toggled.connect(
            lambda: self.toggle_detector_mode(self.apd_flipper_b1)
        )
        self.apd_flipper_b1.setStyleSheet(detector_radio_style)
        self.apd_flipper_b1.setFont(self.larger_label_font)

        self.apd_flipper_b2 = QRadioButton("BPD")
        self.apd_flipper_b2.toggled.connect(
            lambda: self.toggle_detector_mode(self.apd_flipper_b2)
        )
        self.apd_flipper_b2.setStyleSheet(detector_radio_style)
        self.apd_flipper_b2.setFont(self.larger_label_font)

        self.nd_filter_label = QLabel("BPD ND Filter")
        self.nd_filter_label.setFixedHeight(20)
        self.nd_filter_label.setFont(self.label_font)
        self.nd_filter_label.setStyleSheet("font-weight: bold")

        self.nd_filter_opts = QComboBox()
        self.nd_filter_opts.setFont(self.label_font)
        self.nd_filter_opts.setStyleSheet(self.nd_filter_combobox_style)
        self.nd_filter_opts.addItems(["ND 0", "ND 0.5", "ND 1", "ND 2", "ND 3", "ND 4"])
        self.nd_filter_opts.currentIndexChanged.connect(self.nd_filter_changed)

        self.bpd_shutter_label = QLabel("BPD Shutter")
        self.bpd_shutter_label.setFixedHeight(20)
        self.bpd_shutter_label.setFont(self.label_font)
        self.bpd_shutter_label.setStyleSheet("font-weight: bold")

        self.bpd_shutter_button = QPushButton("Open BPD shutter")
        self.bpd_shutter_button.setStyleSheet(self.shutter_button_style)
        self.bpd_shutter_button.setFont(self.label_font)
        self.bpd_shutter_button.clicked.connect(self.bpd_shutter_status_changed)

        self.bpd_shutter_status_label = QLabel("BPD Shutter status")
        self.bpd_shutter_status_label.setStyleSheet(self.shutter_status_label_style)
        self.bpd_shutter_status_label.setFixedHeight(40)
        self.bpd_shutter_status_label.setFixedWidth(220)
        self.bpd_shutter_status_label.setFont(self.label_font)
        
    def layouts(self):
        """GUI Layout Structure"""
        self.gui_layout = QVBoxLayout()

        self.magnet_frame = QFrame(self)
        self.magnet_frame.setObjectName("magFrame")
        self.magnet_frame.setStyleSheet(
            "QFrame#magFrame {background-color: #1a1a1a; border: 2px solid #854141; "
            "border-radius: 8px;}"
        )
        self.magnet_layout = QGridLayout(self.magnet_frame)
        self.magnet_layout.setSpacing(0)
        self.magnet_layout.addWidget(self.magnet_label, 1, 1, 1, 2)

        # Zaber frame
        self.zaber_frame = QFrame(self)
        self.zaber_frame.setObjectName("zaberFrame")
        self.zaber_frame.setStyleSheet(
            "QFrame#zaberFrame {background-color: #1a1a1a; border: 2px solid #854141;"
            " border-radius: 8px;}"
        )
        self.r_layout = QGridLayout(self.zaber_frame)
        self.r_layout.setSpacing(0)
        self.r_layout.addWidget(self.z_label, 1, 1, Qt.AlignmentFlag.AlignLeft)
        self.r_layout.addWidget(self.r_move_checkbox, 1, 2, 1, 2)
        self.r_layout.addWidget(self.r_move_types, 1, 4)
        self.r_layout.addWidget(self.r_enabled_label, 1, 5, 1, 2, Qt.AlignmentFlag.AlignRight)
        self.r_layout.addWidget(self.r_move_label, 2, 1)
        self.r_layout.addWidget(self.r_move_position, 2, 2)
        self.r_layout.addWidget(self.r_move_position_units, 2, 3)
        self.r_layout.addWidget(self.r_execute_move_button, 2, 5, 1, 1)
        self.r_layout.addWidget(self.r_stop_button, 2, 6, 1, 1)

        # Thor frames (polar/azi)
        self.thor_polar_frame = QFrame(self)
        self.thor_polar_frame.setObjectName("thorpolarFrame")
        self.thor_polar_frame.setStyleSheet(
            "QFrame#thorpolarFrame {background-color: #1a1a1a; border: 2px solid #854141;"
            " border-radius: 8px;}"
        )
        self.polar_layout = QGridLayout(self.thor_polar_frame)
        self.polar_layout.setSpacing(0)
        self.polar_layout.addWidget(self.polar_label, 1, 1, Qt.AlignmentFlag.AlignLeft)
        self.polar_layout.addWidget(self.polar_move_checkbox, 1, 2, 1, 2)
        self.polar_layout.addWidget(self.polar_move_types, 1, 4)
        self.polar_layout.addWidget(self.polar_enabled_label, 1, 5, 1, 2, Qt.AlignmentFlag.AlignRight)
        self.polar_layout.addWidget(self.polar_move_label, 2, 1)
        self.polar_layout.addWidget(self.polar_move_position, 2, 2)
        self.polar_layout.addWidget(self.polar_move_position_units, 2, 3)
        self.polar_layout.addWidget(self.polar_execute_move_button, 2, 5, 1, 1)
        self.polar_layout.addWidget(self.polar_stop_button, 2, 6, 1, 1)

        self.thor_azi_frame = QFrame(self)
        self.thor_azi_frame.setObjectName("thoraziFrame")
        self.thor_azi_frame.setStyleSheet(
            "QFrame#thoraziFrame {background-color: #1a1a1a; border: 2px solid #854141;"
            " border-radius: 8px;}"
        )
        self.azi_layout = QGridLayout(self.thor_azi_frame)
        self.azi_layout.setSpacing(0)
        self.azi_layout.addWidget(self.azi_label, 1, 1, Qt.AlignmentFlag.AlignLeft)
        self.azi_layout.addWidget(self.azi_move_checkbox, 1, 2, 1, 2)
        self.azi_layout.addWidget(self.azi_move_types, 1, 4)
        self.azi_layout.addWidget(self.azi_enabled_label, 1, 5, 1, 2, Qt.AlignmentFlag.AlignRight)
        self.azi_layout.addWidget(self.azi_move_label, 2, 1)
        self.azi_layout.addWidget(self.azi_move_position, 2, 2)
        self.azi_layout.addWidget(self.azi_move_position_units, 2, 3)
        self.azi_layout.addWidget(self.azi_execute_move_button, 2, 5, 1, 1)
        self.azi_layout.addWidget(self.azi_stop_button, 2, 6, 1, 1)

        self.b_field_frame = QFrame(self)
        self.b_field_frame.setObjectName("bFieldFrame")
        self.b_field_frame.setStyleSheet(
            "QFrame#bFieldFrame {background-color: #1a1a1a; border: 2px solid #854141;"
            " border-radius: 8px;}"
        )
        self.b_field_layout = QGridLayout(self.b_field_frame)
        self.b_field_layout.setSpacing(0)
        self.b_field_layout.addWidget(self.d_label, 1, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        self.b_field_layout.addWidget(self.b_label, 2, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        self.b_field_layout.addWidget(self.all_disable_button, 2, 2, 2, 1)
        self.b_field_layout.addWidget(self.all_standby_button, 1, 3, 1, 1)
        self.b_field_layout.addWidget(self.stop_all_button, 2, 3, 1, 1)

        # status frame
        self.status_frame = QFrame(self)
        self.status_frame.setObjectName("statusFrame")
        self.status_frame.setStyleSheet(
            "QFrame#statusFrame {background-color: #1a1a1a; border: 2px solid #717171;"
            " border-radius: 8px;}"
        )
        self.status_bar_layout = QGridLayout(self.status_frame)
        self.status_bar_layout.setSpacing(0)
        self.status_bar_layout.setColumnStretch(1, 1)
        self.status_bar_layout.setRowStretch(1, 0)
        self.status_bar_layout.setRowStretch(2, 1)
        self.status_bar_layout.setRowStretch(3, 1)
        self.status_bar_layout.addWidget(self.magnet_instruction_header_label, 1, 1)
        self.status_bar_layout.addWidget(self.magnet_instruction_scroll, 2, 1)
        self.status_bar_layout.addWidget(self.magnet_status_bar_label, 3, 1)
        self.status_bar_layout.addWidget(self.status_scroll_area, 4, 1)

        self.individual_cmds_layout = QGridLayout()
        self.individual_cmds_layout.setSpacing(0)
        self.magnet_layout.addWidget(self.thor_azi_frame, 2, 1, 1, 1)
        self.magnet_layout.addWidget(self.thor_polar_frame, 3, 1, 1, 1)
        self.magnet_layout.addWidget(self.zaber_frame, 4, 1, 1, 1)
        self.magnet_layout.addWidget(self.b_field_frame, 5, 1, 1, 1)
        self.magnet_layout.addWidget(self.status_frame, 2, 2, 4, 1)
        # Make status_frame expand horizontally when window is resized
        self.magnet_layout.setColumnStretch(1, 0)
        self.magnet_layout.setColumnStretch(2, 1)
        self.individual_cmds_layout.addWidget(self.magnet_frame, 1, 1, 1, 1)

        ### --- Laser Control Frame --- ###
        self.laser_control_frame = QFrame(self)
        self.laser_control_frame.setObjectName("laserControlFrame")
        self.laser_control_frame.setStyleSheet(
            "QFrame#laserControlFrame {background-color: #1a1a1a; border: 2px solid limegreen;"
            " border-radius: 8px;}"
        )
        self.laser_controls_layout = QGridLayout(self.laser_control_frame)
        self.laser_controls_layout.setSpacing(0)
        self.laser_controls_layout.addWidget(self.laser_label, 1, 1, 1, 3)
        self.laser_controls_layout.addWidget(self.optics_flipper_label, 2, 1, 1, 1)
        self.laser_controls_layout.addWidget(self.optics_flipper_b1, 2, 2, 1, 1)
        self.laser_controls_layout.addWidget(self.optics_flipper_b2, 2, 3, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_status_label, 3, 1, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_emit_status_label, 3, 2, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_interlock_status_label, 3, 3, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_head_power_label, 4, 1, 1, 3)
        self.laser_controls_layout.addWidget(self.laser_enclosure_output_power_label, 5, 1, 1, 3)
        self.laser_controls_layout.addWidget(self.waveplate_angle_label, 6, 1, 1, 3)
        self.laser_controls_layout.addWidget(self.laser_current_label, 7, 1, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_current_value, 7, 2, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_power_set_button, 7, 3, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_b1, 8, 1, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_b2, 8, 2, 1, 1)
        self.laser_controls_layout.addWidget(self.laser_shutter_button, 8, 3, 1, 1)

        self.device_status_frame = QFrame(self)
        self.device_status_frame.setObjectName("deviceStatusFrame")
        self.device_status_frame.setStyleSheet(
            "QFrame#deviceStatusFrame {"
            "background-color: #1a1a1a;"
            "border: 2px solid #c68642;"
            "border-radius: 8px;"
            "}"
        )
        self.device_status_layout = QGridLayout(self.device_status_frame)
        self.device_status_layout.setSpacing(0)
        self.device_status_layout.addWidget(self.inst_status_label, 1, 1, 1, 3)
        # self.device_status_layout.addWidget(self.laser_power_label, 2, 3, 1, 1)
        # self.device_status_layout.addWidget(self.laser_baseplate_temp_label, 3, 1, 1, 2)
        # self.device_status_layout.addWidget(self.laser_diode_temp_label, 3, 3, 1, 1)
        # self.device_status_layout.addWidget(self.laser_status_label, 4, 1, 1, 1)
        # self.device_status_layout.addWidget(self.laser_interlock_status_label, 4, 2, 1, 1)
        # self.device_status_layout.addWidget(self.laser_emit_status_label, 5, 1, 1, 1)
        # self.device_status_layout.addWidget(self.laser_shutter_status_label, 5, 2, 1, 1)
        self.device_status_layout.addWidget(self.ps_status_label, 2, 1, 1, 3)        

        self.detector_frame = QFrame(self)
        self.detector_frame.setObjectName("detectorFrame")
        self.detector_frame.setStyleSheet(
            "QFrame#detectorFrame {"
            "background-color: #1a1a1a;"
            "border: 2px solid blue;"
            "border-radius: 8px;"
            "}"
        )
        self.detector_layout = QGridLayout(self.detector_frame)
        self.detector_layout.setSpacing(0)
        self.detector_layout.addWidget(self.detector_label, 1, 1, 1, 2)
        self.detector_layout.addWidget(self.apd_flipper_b1, 2, 1, 1, 1)
        self.detector_layout.addWidget(self.apd_flipper_b2, 2, 2, 1, 1)
        self.detector_layout.addWidget(self.nd_filter_label, 3, 1, 1, 2)
        self.detector_layout.addWidget(self.nd_filter_opts, 4, 1, 1, 2)
        self.detector_layout.addWidget(self.bpd_shutter_label, 5, 1, 1, 2)
        self.detector_layout.addWidget(self.bpd_shutter_button, 6, 1, 1, 2)
        self.detector_layout.addWidget(self.bpd_shutter_status_label, 7, 1, 1, 2)

        self.other_widgets_layout = QGridLayout()
        self.other_widgets_layout.addWidget(self.laser_control_frame, 1, 1, 1, 1)
        self.other_widgets_layout.addWidget(self.detector_frame, 1, 2, 2, 1)
        self.other_widgets_layout.addWidget(self.device_status_frame, 2, 1, 1, 1)

        self.gui_layout.addLayout(self.individual_cmds_layout)
        self.gui_layout.addLayout(self.status_bar_layout)
        self.gui_layout.addLayout(self.other_widgets_layout)

        self.setLayout(self.gui_layout)

    def park_all(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        self.all_disable_proc.run(mag_control.disable_all)
        logger.info(
            "All magnet stages have been parked upon startup of magnet alignment widget."
        )

    def d_from_z(self, z_pos):
        """Calculate physical magnet separation 'd' in mm from Zaber stage position 'z_pos'."""
        return 235 - 2 * (100 - z_pos)
    
    # INTERACTIVE WIDGET FUNCTIONS
    def r_move_type_changed(self):
        match self.r_move_types.currentText():
            case 'Absolute':
                self.r_move_label.setText("Move Zaber (0 \u2013 100 mm) to: ")
                self.r_move_position_units.setText("mm")
            case 'Jog':
                self.r_move_label.setText("Jog Zaber stage by: ")
                self.r_move_position_units.setText("mm")
            case _:
                self.r_move_label.setText("Move Zaber to B field: ")
                self.r_move_position_units.setText("G")

    def azi_move_type_changed(self):
        match self.azi_move_types.currentText():
            case 'Jog':
                self.azi_move_label.setText("Jog \u03c6 by: ")
            case _:
                self.azi_move_label.setText(f"Move \u03c6 ({self.phi_min}\N{DEGREE SIGN} \u2013 {self.phi_max}\N{DEGREE SIGN}) to: ")

    def polar_move_type_changed(self):
        match self.polar_move_types.currentText():
            case 'Jog':
                self.polar_move_label.setText("Jog \u03b8 by: ")
            case _:
                self.polar_move_label.setText(f"Move \u03b8 ({self.theta_min}\N{DEGREE SIGN} \u2013 {self.theta_max}\N{DEGREE SIGN}) to: ")

    def move_button_checked(self, box):
        if box.text() == "Zaber Move Type: ":
            if box.isChecked() is True:
                self.r_move_types.setEnabled(True)
                self.r_move_position.setEnabled(True)
                self.r_move_position_units.setEnabled(True)
                self.r_execute_move_button.setEnabled(True)
                [self.r_opacity_effects[i].setEnabled(False) for i in range(5)]

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
                [self.r_opacity_effects[i].setEnabled(True) for i in range(5)]

        elif box.text() == "Polar Move Type: ":
            if box.isChecked() is True:
                self.polar_move_types.setEnabled(True)
                self.polar_move_position.setEnabled(True)
                self.polar_move_position_units.setEnabled(True)
                self.polar_move_position_units.setText("\N{DEGREE SIGN}")
                self.polar_execute_move_button.setEnabled(True)
                [self.polar_opacity_effects[i].setEnabled(False) for i in range(5)]
            else:
                self.polar_move_types.setEnabled(False)
                self.polar_move_position.setEnabled(False)
                self.polar_move_position_units.setEnabled(False)
                self.polar_move_position_units.setText("")
                self.polar_execute_move_button.setEnabled(False)
                [self.polar_opacity_effects[i].setEnabled(True) for i in range(5)]

        elif box.text() == "Azimuth. Move Type: ":
            if box.isChecked() is True:
                self.azi_move_types.setEnabled(True)
                self.azi_move_position.setEnabled(True)
                self.azi_move_position_units.setEnabled(True)
                self.azi_move_position_units.setText("\N{DEGREE SIGN}")
                self.azi_execute_move_button.setEnabled(True)
                [self.azi_opacity_effects[i].setEnabled(False) for i in range(5)]
            else:
                self.azi_move_types.setEnabled(False)
                self.azi_move_position.setEnabled(False)
                self.azi_move_position_units.setEnabled(False)
                self.azi_move_position_units.setText("")
                self.azi_execute_move_button.setEnabled(False)
                [self.azi_opacity_effects[i].setEnabled(True) for i in range(5)]

    def single_move_clicked(self, stage):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        if getattr(self, "_mag_stop_event", None) is not None:
            self._mag_stop_event.clear()

        if self.all_disable_button.text() == "DISABLE ALL STAGES":
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
                                "stop_event": self._mag_stop_event,
                                "abs": True,
                                "move_type": "Absolute"
                            },
                        )
                    elif type_idx == 1:
                        self.r_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_pos": new_position,
                                "queue": self.q,
                                "stop_event": self._mag_stop_event,
                                "abs": False,
                                "move_type": "Jog"
                            },
                        )
                    else:
                        self.r_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_pos": new_position,
                                "queue": self.q,
                                "stop_event": self._mag_stop_event,
                                "abs": False,
                                "move_type": "B Field Target"
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
                                "stop_event": self._mag_stop_event,
                                "abs": True,
                                "move_type": "Absolute"
                            },
                        )
                    elif type_idx == 1:
                        self.polar_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_angle": new_position,
                                "queue": self.q,
                                "stop_event": self._mag_stop_event,
                                "abs": False,
                                "move_type": "Jog"
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
                                "stop_event": self._mag_stop_event,
                                "abs": True,
                                "move_type": "Absolute"
                            },
                        )
                    elif type_idx == 1:
                        self.azi_execute_move_button_proc.run(
                            mag_control.move_to_orientation,
                            **{
                                "stage": stage,
                                "new_angle": new_position,
                                "queue": self.q,
                                "stop_event": self._mag_stop_event,
                                "abs": False,
                                "move_type": "Jog"
                            },
                        )
                except ValueError:
                    logger.debug("Invalid entry for new angle. Try again.")

            else:
                logger.debug("Invalid stage for move command.")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: red;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )
                self.status_label.setText("Invalid stage for move command.")
        else:
            self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: red;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )
            self.status_label.setText("Cannot move - stages are disabled.")

        return 0

    def stop_button_clicked(self, stage):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        if getattr(self, "_mag_stop_event", None) is not None:
            self._mag_stop_event.set()

        move_procs = (
            self.r_execute_move_button_proc,
            self.polar_execute_move_button_proc,
            self.azi_execute_move_button_proc,
        )

        for proc in move_procs:
            if proc is None:
                continue
            is_running = getattr(proc, "is_running", None)
            if callable(is_running):
                try:
                    active = bool(is_running())
                except Exception:
                    active = False
            else:
                active = bool(getattr(proc, "running", False))

            if not active:
                continue

            try:
                proc.kill()
            except Exception:
                stop_fn = getattr(proc, "stop", None)
                if callable(stop_fn):
                    try:
                        stop_fn()
                    except Exception:
                        pass

        if stage == "zaber":
            self.r_stop_button_proc.run(
                mag_control.stop_motion, 
                **{
                    "stage": stage, 
                    "queue": self.q
                }
            )
        elif stage == "thor_polar":
            self.polar_stop_button_proc.run(
                mag_control.stop_motion, 
                **{
                    "stage": stage, 
                    "queue": self.q
                }
            )
        elif stage == "thor_azi":
            self.azi_stop_button_proc.run(
                mag_control.stop_motion, 
                **{
                    "stage": stage, 
                    "queue": self.q
                }
            )

        return 0

    def stop_all_button_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        if getattr(self, "_mag_stop_event", None) is not None:
            self._mag_stop_event.set()

        move_procs = (
            self.r_execute_move_button_proc,
            self.polar_execute_move_button_proc,
            self.azi_execute_move_button_proc,
            self.all_standby_proc,
        )

        for proc in move_procs:
            if proc is None:
                continue
            is_running = getattr(proc, "is_running", None)
            if callable(is_running):
                try:
                    active = bool(is_running())
                except Exception:
                    active = False
            else:
                active = bool(getattr(proc, "running", False))

            if not active:
                continue

            try:
                proc.kill()
            except Exception:
                stop_fn = getattr(proc, "stop", None)
                if callable(stop_fn):
                    try:
                        stop_fn()
                    except Exception:
                        pass

        self.stop_all_proc.run(
            mag_control.stop_all_motion,
            **{
                "stop_event": self._mag_stop_event,
                "queue": self.q
            }
        )

    def all_disable_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()

        if self.all_disable_button.text() == "DISABLE ALL STAGES":
            self.all_disable_proc.run(mag_control.disable_all)
            self.all_disable_button.setText("ENABLE ALL STAGES")
            self.all_disable_button.setStyleSheet(
                self.all_disable_button_style_active
            )
            self.status_label.setStyleSheet("""
                    QLabel {
                        color: white;
                        background-color: black;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )
            self.status_label.setText("Stages disabled. Cannot move until enabled.")
            # Update enabled labels to show disabled state
            self.r_enabled_label.setText("DISABLED")
            self.r_enabled_label.setStyleSheet(self.enabled_label_style_inactive)
            self.polar_enabled_label.setText("DISABLED")
            self.polar_enabled_label.setStyleSheet(self.enabled_label_style_inactive)
            self.azi_enabled_label.setText("DISABLED")
            self.azi_enabled_label.setStyleSheet(self.enabled_label_style_inactive)
        else:
            self.all_disable_proc.run(mag_control.enable_all)
            self.all_disable_button.setText("DISABLE ALL STAGES")
            self.all_disable_button.setStyleSheet(self.all_disable_button_style_inactive)
            self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: white;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

            self.status_label.setText("Stages enabled.")
            # Update enabled labels to show enabled state
            self.r_enabled_label.setText("ENABLED")
            self.r_enabled_label.setStyleSheet(self.enabled_label_style_active)
            self.polar_enabled_label.setText("ENABLED")
            self.polar_enabled_label.setStyleSheet(self.enabled_label_style_active)
            self.azi_enabled_label.setText("ENABLED")
            self.azi_enabled_label.setStyleSheet(self.enabled_label_style_active)

    def all_standby_clicked(self):
        reload(mag)
        mag_control = mag.NanoNMRMagnetMotion()
        if getattr(self, "_mag_stop_event", None) is not None:
            self._mag_stop_event.clear()

        if self.all_disable_button.text() == "DISABLE ALL STAGES":
            self.all_standby_proc.run(
                mag_control.standby_all, 
                **{
                    "queue": self.q,
                    "stop_event": self._mag_stop_event,
                }
            )
        else:
            self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: red;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

            self.status_label.setText("Cannot move to standby - stages are disabled.")

    def check_queue_from_mag(self):
        # queue checker to control progress bar display
        while not self.q.empty():  # if there is something in the queue
            queueText = self.q.get_nowait()

            if queueText[0] == "done":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: limegreen;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

                self.status_label.setText("Move completed successfully.")

                self.z_label.setText(f"z = {queueText[1]} mm")
                self.d_label.setText(f"d = {round(self.d_from_z(queueText[1]), 1)} mm")
                self.polar_label.setText(
                    f"\u03b8 = {round(queueText[2], 1)}\N{DEGREE SIGN}"
                )
                self.azi_label.setText(
                    f"\u03c6 = {round(queueText[3], 1)}\N{DEGREE SIGN}"
                )

                self.b_label.setText(f"B \u2248 {round(queueText[4], 1)} G")

                self.z_label.setStyleSheet(self.thor_header_style)
                self.polar_label.setStyleSheet(self.thor_header_style)
                self.azi_label.setStyleSheet(self.thor_header_style)

            elif queueText[0] == "stopped":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: white;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )
                self.status_label.setText("Move stopped by user.")

                self.z_label.setText(f"z = {queueText[1]} mm")
                self.d_label.setText(f"d = {round(self.d_from_z(queueText[1]), 1)} mm")
                self.polar_label.setText(
                    f"\u03b8 = {round(queueText[2], 1)}\N{DEGREE SIGN}"
                )
                self.azi_label.setText(
                    f"\u03c6 = {round(queueText[3], 1)}\N{DEGREE SIGN}"
                )
                self.b_label.setText(f"B = --- G")

                self.z_label.setStyleSheet(self.thor_header_style)
                self.polar_label.setStyleSheet(self.thor_header_style)
                self.azi_label.setStyleSheet(self.thor_header_style)

            elif queueText[0] == "start standby":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: gold;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """)
                self.status_label.setText(
                    "Moving to standby position..."
                )

                # Change color to yellow while moving
                self.z_label.setStyleSheet(self.thor_header_motion_style)

            elif queueText[0] == "in motion to standby":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: gold;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

                self.status_label.setText(
                    "Moving to standby position..."
                )

                self.z_label.setText(f"z = {queueText[1]} mm")
                self.d_label.setText(f"d = {round(self.d_from_z(queueText[1]), 1)} mm")
                self.polar_label.setText(
                    f"\u03b8 = {round(queueText[2], 1)}\N{DEGREE SIGN}"
                )
                self.azi_label.setText(
                    f"\u03c6 = {round(queueText[3], 1)}\N{DEGREE SIGN}"
                )

                # Change color to yellow while moving
                self.z_label.setStyleSheet(self.thor_header_motion_style)
                self.polar_label.setStyleSheet(self.thor_header_motion_style)
                self.azi_label.setStyleSheet(self.thor_header_motion_style)

            elif queueText[0] == "unsafe z move":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: red;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

                self.status_label.setText(
                    f"{queueText[7]}"
                )

            elif queueText[0] == "start z move":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: gold;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

                if queueText[6] is True:
                    self.status_label.setText(
                        f"{queueText[7]}"
                    )
                else:
                    self.status_label.setText(
                        f"{queueText[7]}"
                    )

                self.z_label.setText(f"z = {queueText[1]} mm")
                self.d_label.setText(f"d = {round(self.d_from_z(queueText[1]), 1)} mm")
                self.z_label.setStyleSheet(self.thor_header_motion_style)
                self.polar_label.setStyleSheet(self.thor_header_style)
                self.azi_label.setStyleSheet(self.thor_header_style)

            elif queueText[0] == "unsafe rotation move":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: red;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

                self.status_label.setText(
                    f"{queueText[7]}"
                )

            elif queueText[0] == "start rotation move":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: gold;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

                if queueText[4] == "thor_polar":
                    if queueText[6] is True:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )
                    else:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )
                else:
                    if queueText[6] is True:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )
                    else:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )

            elif queueText[0] == "in rotation motion to target":
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: gold;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """
                )

                if queueText[4] == "thor_polar":
                    if queueText[6] is True:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )
                    else:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )

                    self.polar_label.setText(
                        f"\u03b8 = {round(queueText[2], 1)}\N{DEGREE SIGN}"
                    )
                    self.z_label.setStyleSheet(self.thor_header_style)
                    self.polar_label.setStyleSheet(self.thor_header_motion_style)
                    self.azi_label.setStyleSheet(self.thor_header_style)

                else:
                    if queueText[6] is True:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )
                    else:
                        self.status_label.setText(
                            f"{queueText[7]}"
                        )

                    self.azi_label.setText(
                        f"\u03c6 = {round(queueText[3], 1)}\N{DEGREE SIGN}"
                    )
                    self.z_label.setStyleSheet(self.thor_header_style)
                    self.polar_label.setStyleSheet(self.thor_header_style)
                    self.azi_label.setStyleSheet(self.thor_header_motion_style)

            else:
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: black;
                        background-color: gold;
                        border-left: 4px solid #854141;
                        border-top: 1px solid #555555;
                        border-right: 1px solid #555555;
                        border-bottom: 1px solid #555555;
                        border-radius: 2px;
                        padding-left: 8px;
                    }
                """)
                self.status_label.setText("Move in progress...")

                self.z_label.setStyleSheet(self.thor_header_style)
                self.polar_label.setStyleSheet(self.thor_header_style)
                self.azi_label.setStyleSheet(self.thor_header_style)

        self.updateTimer.start(self.QUEUE_CHECK_TIME)

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
                "color: white; background-color: #800; border: 1px solid #f00;",
            )
            self._set_label_if_changed(
                diode_lbl,
                "Diode Temp: ERROR",
                "color: white; background-color: #800; border: 1px solid #f00;",
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
                    "color: black; background-color: white; border: 1px solid #555555;",
                )
            else:
                self._set_label_if_changed(
                    lbl,
                    "Laser Shutter: CLOSED",
                    "color: white; background-color: black; border: 1px solid #555555;",
                )
            self._sh_last_err = None
        except Exception as e:
            self._set_label_if_changed(
                lbl,
                "Laser Shutter: ERROR",
                "color: white; background-color: #800; border: 1px solid #f00;",
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
                        "color: black; background-color: gold; border: 1px solid #555555;",
                    )
                if _alive(emit_lbl):
                    self._set_label_if_changed(
                        emit_lbl,
                        "Emission: ON",
                        "color: black; background-color: limegreen; border: 1px solid #555555;",
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
                        "color: white; background-color: #222; border: 1px solid #555555;",
                    )

                if cw_on:
                    if _alive(emit_lbl):
                        self._set_label_if_changed(
                            emit_lbl,
                            "Emission: ON",
                            "color: black; background-color: limegreen; border: 1px solid #555555;",
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
                            "color: white; background-color: black; border: 1px solid #555555;",
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
                    "color: white; background-color: #800; border: 1px solid #f00;",
                )
            if _alive(emit_lbl):
                self._set_label_if_changed(
                    emit_lbl,
                    "Emission: ERROR",
                    "color: white; background-color: #800; border: 1px solid #f00;",
                )
            if _alive(pow_lbl):
                self._set_label_if_changed(pow_lbl, pow_lbl.text(), "color: white;")

            msg = f"{type(e).__name__}: {e}"
            if getattr(self, "_ps_last_error", None) != msg:
                self._ps_last_error = msg
                print(f"[PS status] {msg}")

    def laser_temp_clicked(self, component):
        match component:
            case "diode":
                diode_temp = self.call_laser(lambda laser: laser.get_diode_temp())
                self.laser_diode_temp_label.setStyleSheet(
                    "color: white; background-color: black; border: 1px solid black;"
                )
                self.laser_diode_temp_label.setText(f"{diode_temp:.2f} °C")
            case "base":
                heat_sink_temp = self.call_laser(
                    lambda laser: laser.get_baseplate_temp()
                )
                self.laser_baseplate_temp_label.setStyleSheet(
                    "color: white; background-color: black; border: 1px solid black;"
                )
                self.laser_baseplate_temp_label.setText(f"{heat_sink_temp:.2f} °C")
            case _:
                pass

    def laser_power_set_type_changed(self, index):
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
                    "color: white; background-color: #800; border: 1px solid #f00;",
                )
            return None

        # Normal states
        if laser_status in ("Warm Up", "Standby", "Searching SLM point"):
            style = "color: black; background-color: orange; border: 1px solid black;"
        elif laser_status == "Laser ON":
            style = (
                "color: black; background-color: limegreen; border: 1px solid black;"
            )
        elif laser_status in ("Error", "Alarm"):
            style = "color: black; background-color: red; border: 1px solid black;"
        else:
            style = "color: black; background-color: white; border: 1px solid black;"

        if _alive(lbl):
            self._set_label_if_changed(lbl, f"{laser_status}", style)

        return laser_status

    def get_interlock_status(self):
        interlock_status = self.call_laser(lambda laser: laser.get_interlock_state())
        
        if isinstance(interlock_status, bytes):
            interlock_status = interlock_status.decode().strip()

        if interlock_status == '1':
            self.laser_interlock_status_label.setStyleSheet(
                "color: black; background-color: limegreen; border: 1px solid black;"
            )
            self.laser_interlock_status_label.setText("Interlock: CLOSED")
        else:
            self.laser_interlock_status_label.setStyleSheet(
                "color: black; background-color: red; border: 1px solid black;"
            )
            self.laser_interlock_status_label.setText("Interlock: OPEN")

        return interlock_status

    def toggle_laser(self, b):
        match b.text():
            case "CW ON":
                if b.isChecked() is True:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_on(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: ON")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: limegreen; border: 1px solid black;"
                    )
                    self.laser_power_label.setStyleSheet("color: limegreen;")
                else:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_off(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: OFF")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: white; border: 1px solid black;"
                    )
                    self.laser_power_label.setStyleSheet("color: white;")

            case "STANDBY (OFF - DAILY USE)":
                if b.isChecked() is True:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_off(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: OFF")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: white; border: 1px solid black;"
                    )
                    self.laser_power_label.setStyleSheet("color: white;")
                else:
                    self.call_ps(lambda ps: ps.laser_ttl_toggle_on(owner=self._gui_id))
                    self.laser_emit_status_label.setText("Emission: ON")
                    self.laser_emit_status_label.setStyleSheet(
                        "color: black; background-color: limegreen; border: 1px solid black;"
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

    def set_bpd_shutter_ui(self, is_open: bool):
        if is_open:
            self.bpd_shutter_button.setText("Close BPD shutter")
            self.bpd_shutter_status_label.setStyleSheet(
                "color: black; background-color: white; border: 1px solid black;"
            )
            self.bpd_shutter_status_label.setText("BPD Shutter: OPEN")
        else:
            self.bpd_shutter_button.setText("Open BPD shutter")
            self.bpd_shutter_status_label.setStyleSheet(
                "color: white; background-color: black; border: 1px solid black;"
            )
            self.bpd_shutter_status_label.setText("BPD Shutter: CLOSED")

    def check_queue_from_exp_inst(self):
        if self.exp_inst_queue is None:
            return

        try:
            while True:
                msg = self.exp_inst_queue.get_nowait()
                if msg is None:
                    continue

                if isinstance(msg, dict) and msg.get("type") == "bpd_shutter":
                    self.set_bpd_shutter_ui(bool(msg.get("open", False)))

        except Empty:
            pass
        except Exception as e:
            logger.warning(f"Error reading experiment->instrument queue: {e}")

    # TODO: switch this to use the SH05R shutter 
    def bpd_shutter_status_changed(self):
        self.call_daq(lambda daq: daq.open_do_task("shutter"))
        self.call_daq(lambda daq: daq.start_do_task())

        opening = self.bpd_shutter_button.text() == "Open BPD shutter"

        if opening:
            self.call_daq(lambda daq: daq.write_do_task("shutter", shutter_status="open"))
        else:
            self.call_daq(lambda daq: daq.write_do_task("shutter", shutter_status="close"))

        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())

        self.set_bpd_shutter_ui(opening)

    def bpd_shutter_status_read(self):
        self.call_daq(lambda daq: daq.open_do_task("shutter"))
        self.call_daq(lambda daq: daq.start_do_task())
        bpd_shutter_mode_is_open = self.call_daq(lambda daq: daq.read_do_task())
        self.call_daq(lambda daq: daq.stop_do_task())
        self.call_daq(lambda daq: daq.close_do_task())

        self.set_bpd_shutter_ui(bool(bpd_shutter_mode_is_open))

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
