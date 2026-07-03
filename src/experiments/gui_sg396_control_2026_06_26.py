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


class SigGenWidget(QWidget):
    """Qt widget subclass that generates an interface for operating magnet mount."""

    def __init__(self, status_queue=None):
        super().__init__()

        self._mgr = InstrumentManager()
        self._mgr.__enter__()  # open once, close on widget destruction

        self.setWindowTitle("NanoNMR Signal Generator Control")

        self._last_rpc_err = {}  # keep track of last RPC errors. Context -> msg

        # Setup professional fonts and styling
        self._setup_professional_styles()

        # initialize widgets
        self.init_sg396_widgets()
        self.init_awg_widgets()

        self.layouts()
        self.park_all()

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
        self.larger_label_font = QtGui.QFont("Segoe UI", 13)
        self.largest_label_font = QtGui.QFont("Segoe UI", 20)
        self.largest_label_font.setBold(True)
        self.header_font = QtGui.QFont("Segoe UI", 18)
        self.header_font.setBold(True)

        # Frame header styles (colored bars at top of frames)
        self.sg396_header_style = """
            QLabel {
                background-color: rgba(152, 59, 179, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid #983BB3;
                border-radius: 6px;
            }
        """
        self.awg_header_style = """
            QLabel {
                background-color: rgba(198, 134, 66, 0.10);
                color: white;
                padding: 8px;
                font-weight: bold;
                border-bottom: 2px solid #c68642;
                border-radius: 6px;
            }
        """

        # AWG control button: Dark Brown
        self.awg_button_style = """
            QPushButton {
                background-color: #4B3700;
                color: white;
                border: 1px solid #D78100;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A4410;
            }
            QPushButton:pressed {
                background-color: #3B2800;
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
        
        self.sg396_status_label_style = """
            QLabel {
                color: white;
                background-color: black;
                border-left: 4px solid #983BB3;
                border-top: 1px solid #555555;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-radius: 2px;
                padding-left: 8px;
            }
        """
        
        self.awg_status_label_style = """
            QLabel {
                color: white;
                background-color: black;
                border-left: 4px solid #c68642;
                border-top: 1px solid #555555;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-radius: 2px;
                padding-left: 8px;
            }
        """

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

    # CREATE WIDGETS
    def get_combobox_val(self, combobox):
        return str(combobox.value())
    
    def init_sg396_widgets(self):
        self.sg396_label = QLabel("SRS Signal Generator Control")
        self.sg396_label.setFixedHeight(50)
        self.sg396_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sg396_label.setFont(self.header_font)
        self.sg396_label.setStyleSheet(self.sg396_header_style)

        self.sg396_checkbox = QCheckBox("Enable SG396 output")
        self.sg396_checkbox.setFont(self.label_font)
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
        self._set_font_recursive(self.sg396_params_widget_1, self.label_font)

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
        self._set_font_recursive(self.sg396_params_widget_2, self.label_font)

        self.sg396_status_label = QLabel("SRS SG396 status here")
        self.sg396_status_label.setStyleSheet(self.sg396_status_label_style)
        self.sg396_status_label.setFixedHeight(40)
        self.sg396_status_label.setFont(self.label_font)

        self.sg396_emit_button = QPushButton("Emit MW")
        self.sg396_emit_button.setStyleSheet(self.magnet_move_button_style)
        self.sg396_emit_button.setFont(self.label_font)
        self.sg396_emit_button_proc = ProcessRunner()
        self.sg396_emit_button.clicked.connect(self.sg396_emit_button_clicked)

        self.sg396_stop_button = QPushButton("Stop MW")
        self.sg396_stop_button.setStyleSheet(self.magnet_stop_button_style)
        self.sg396_stop_button.setFont(self.label_font)
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

    def init_awg_widgets(self):
        self.awg_label = QLabel("AWG Control")
        self.awg_label.setFixedHeight(40)
        self.awg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.awg_label.setFont(self.header_font)
        self.awg_label.setStyleSheet(self.awg_header_style)

        self.awg_volt_range_ch3_label = QLabel("AWG Ch. 3 Volt. Range: ")
        self.awg_volt_range_ch4_label = QLabel("AWG Ch. 4 Volt. Range: ")
        self.awg_volt_range_ch4_label.setFont(self.label_font)
        self.awg_samp_rate_group0_label = QLabel("AWG Group 0 Samp. Rate: ")
        self.awg_samp_rate_group0_label.setFont(self.label_font)
        self.awg_samp_rate_group1_label = QLabel("AWG Group 1 Samp. Rate: ")
        self.awg_samp_rate_group1_label.setFont(self.label_font)

        self.awg_volt_range_ch3_opts = ["0.2 V", "0.4 V", "0.6 V", "0.8 V", "1 V", "2 V", "3 V", "4 V", "5 V"]
        self.awg_volt_range_ch4_opts = ["0.2 V", "0.4 V", "0.6 V", "0.8 V", "1 V", "2 V", "3 V", "4 V", "5 V"]
        self.awg_volt_range_ch3_combobox = QComboBox()
        self.awg_volt_range_ch3_combobox.setFont(self.label_font)
        self.awg_volt_range_ch3_combobox.setStyleSheet(self.combobox_style)
        self.awg_volt_range_ch3_combobox.addItems(self.awg_volt_range_ch3_opts)
        self.awg_volt_range_ch3_combobox.currentIndexChanged.connect(self.awg_volt_range_ch3_changed)
        self.awg_volt_range_ch4_combobox = QComboBox()
        self.awg_volt_range_ch4_combobox.setFont(self.label_font)
        self.awg_volt_range_ch4_combobox.setStyleSheet(self.combobox_style)
        self.awg_volt_range_ch4_combobox.addItems(self.awg_volt_range_ch4_opts)
        self.awg_volt_range_ch4_combobox.currentIndexChanged.connect(self.awg_volt_range_ch4_changed)

        self.awg_volt_range_ch3_str = self.awg_volt_range_ch3_combobox.currentText()
        self.awg_volt_range_ch4_str = self.awg_volt_range_ch4_combobox.currentText()

        self.apply_awg_volts_button = QPushButton("Apply Volt Ranges")
        self.apply_awg_volts_button.setStyleSheet(self.awg_button_style)
        self.apply_awg_volts_button.setFont(self.label_font)
        self.apply_awg_volts_button.clicked.connect(self.apply_awg_volt_ranges)

        self.awg_sampling_group0_opts = ["2.4 GHz", "1.2 GHz", "600 MHz", "300 MHz", "150 MHz", "75 MHz", "37.5 MHz", "18.75 MHz", "9.37 MHz", "4.68 MHz",
                                  "2.34 MHz", "1.17 MHz", "585.93 kHz", "292.96 kHz"]
        self.awg_sampling_group1_opts = ["2.4 GHz", "1.2 GHz", "600 MHz", "300 MHz", "150 MHz", "75 MHz", "37.5 MHz", "18.75 MHz", "9.37 MHz", "4.68 MHz",
                                  "2.34 MHz", "1.17 MHz", "585.93 kHz", "292.96 kHz"]
        self.awg_samp_rate_group_0_combobox = QComboBox()
        self.awg_samp_rate_group_0_combobox.setFont(self.label_font)
        self.awg_samp_rate_group_0_combobox.setStyleSheet(self.combobox_style)
        self.awg_samp_rate_group_0_combobox.addItems(self.awg_sampling_group0_opts)
        self.awg_samp_rate_group_0_combobox.currentIndexChanged.connect(self.awg_samp_rate_group0_changed)
        self.awg_samp_rate_group_1_combobox = QComboBox()
        self.awg_samp_rate_group_1_combobox.setFont(self.label_font)
        self.awg_samp_rate_group_1_combobox.setStyleSheet(self.combobox_style)
        self.awg_samp_rate_group_1_combobox.addItems(self.awg_sampling_group1_opts)
        self.awg_samp_rate_group_1_combobox.currentIndexChanged.connect(self.awg_samp_rate_group1_changed)

        self.awg_samp_rate_boxes = {
            0: self.awg_samp_rate_group_0_combobox,
            1: self.awg_samp_rate_group_1_combobox
        }

        self.awg_samp_rate_group_0_str = self.awg_samp_rate_boxes[0].currentText()
        self.awg_samp_rate_group_1_str = self.awg_samp_rate_boxes[1].currentText()

        self.apply_awg_rates_button = QPushButton("Apply Rates")
        self.apply_awg_rates_button.setStyleSheet(self.awg_button_style)
        self.apply_awg_rates_button.setFont(self.label_font)
        self.apply_awg_rates_button.clicked.connect(self.apply_awg_rates)

        self.awg_status_label = QLabel("AWG status here")
        self.awg_status_label.setStyleSheet(self.awg_status_label_style)
        self.awg_status_label.setFixedHeight(40)
        self.awg_status_label.setFont(self.label_font)

    def layouts(self):
        """GUI Layout Structure"""
        self.gui_layout = QVBoxLayout()

        # signal generators
        self.sg396_frame = QFrame(self)
        self.sg396_frame.setObjectName("sgFrame")
        self.sg396_frame.setStyleSheet(
            "QFrame#sgFrame {"
            "background-color: #1a1a1a;"
            "border: 2px solid #983BB3;"
            " border-radius: 8px;"
            "}"
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

        ### --- AWG Control Frame --- ### 
        self.awg_control_frame = QFrame(self)
        self.awg_control_frame.setObjectName("awgControlFrame")
        self.awg_control_frame.setStyleSheet(
            "QFrame#awgControlFrame {"
            "background-color: #1a1a1a;"
            "border: 2px solid #c68642;"
            "border-radius: 8px;"
            "}"
        )
        self.awg_control_layout = QGridLayout(self.awg_control_frame)
        self.awg_control_layout.setSpacing(0)
        self.awg_control_layout.addWidget(self.awg_label, 1, 1, 1, 4)
        self.awg_control_layout.addWidget(self.awg_volt_range_ch3_label, 2, 1, 1, 1)
        self.awg_control_layout.addWidget(self.awg_volt_range_ch3_combobox, 2, 2, 1, 1)
        self.awg_control_layout.addWidget(self.awg_samp_rate_group0_label, 2, 3, 1, 1)
        self.awg_control_layout.addWidget(self.awg_samp_rate_group_0_combobox, 2, 4, 1, 1)
        self.awg_control_layout.addWidget(self.awg_volt_range_ch4_label, 3, 1, 1, 1)
        self.awg_control_layout.addWidget(self.awg_volt_range_ch4_combobox, 3, 2, 1, 1)
        self.awg_control_layout.addWidget(self.awg_samp_rate_group1_label, 3, 3, 1, 1)
        self.awg_control_layout.addWidget(self.awg_samp_rate_group_1_combobox, 3, 4, 1, 1)
        self.awg_control_layout.addWidget(self.apply_awg_volts_button, 4, 1, 1, 2)
        self.awg_control_layout.addWidget(self.apply_awg_rates_button, 4, 3, 1, 2)
        self.awg_control_layout.addWidget(self.awg_status_label, 5, 1, 1, 4)
        
        self.awg_layout = QGridLayout()
        self.awg_layout.setSpacing(0)
        self.awg_layout.addWidget(self.awg_control_frame, 1, 1, 1, 1)

        self.gui_layout.addLayout(self.sig_gens_layout)
        self.gui_layout.addLayout(self.awg_layout)

        self.setLayout(self.gui_layout)

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
            "color: black; background-color: gold; border: 1px solid black;"
        )
        self.sg396_status_label.setText(
            f"SRS SG396 Status: ON ({output_freq*1e-9} GHz at {fun_kwargs['RF_Power']*1e6} uW)"
        )

    def sg396_stop_button_clicked(self):
        self.call_sg(lambda sg: sg.set_rf_toggle(0))
        self.call_sg(lambda sg: sg.set_mod_toggle(0))
        self.call_awg(lambda awg: awg.set_disabled())

        self.sg396_status_label.setStyleSheet(
            "color: white; background-color: black; border: 1px solid black;"
        )
        self.sg396_status_label.setText("SRS SG396 Status: OFF")

    def awg_volt_range_ch3_changed(self):
        self.awg_volt_range_ch3_str = self.awg_volt_range_ch3_combobox.currentText()

    def awg_volt_range_ch4_changed(self):
        self.awg_volt_range_ch4_str = self.awg_volt_range_ch4_combobox.currentText()

    def awg_samp_rate_group0_changed(self):
        self.awg_samp_rate_group_0_str = self.awg_samp_rate_boxes[0].currentText()

    def awg_samp_rate_group1_changed(self):
        self.awg_samp_rate_group_1_str = self.awg_samp_rate_boxes[1].currentText()

    def apply_awg_rates(self):
        if hasattr(self, "awg_samp_rate_group_0_str") and hasattr(self, "awg_samp_rate_group_1_str"):
            self.call_awg(lambda awg: awg.set_sampling_rate(0, self.awg_samp_rate_group_0_str))
            self.call_awg(lambda awg: awg.set_sampling_rate(1, self.awg_samp_rate_group_1_str))
            self.awg_status_label.setText(f"Applied AWG rates: Group 0 = {self.awg_samp_rate_group_0_str}, Group 1 = {self.awg_samp_rate_group_1_str}")
        else:
            self.awg_status_label.setText("Please select sampling rates for both groups before applying.")

    def apply_awg_volt_ranges(self):
        print("in apply awg volt range function")
        if hasattr(self, "awg_volt_range_ch3_str") and hasattr(self, "awg_volt_range_ch4_str"):
            print("applying AWG volt ranges now")
            self.call_awg(lambda awg: awg.set_voltage_range(2, self.awg_volt_range_ch3_str))
            self.call_awg(lambda awg: awg.set_voltage_range(3, self.awg_volt_range_ch4_str))
            self.awg_status_label.setText(f"Applied AWG volt ranges: CH3 = {self.awg_volt_range_ch3_str}, CH4 = {self.awg_volt_range_ch4_str}")
        else:
            self.awg_status_label.setText("Please select voltage ranges for both channels before applying.")
    
    def kill_process(self):
        """Stop the run process."""
        self.run_proc.kill()
