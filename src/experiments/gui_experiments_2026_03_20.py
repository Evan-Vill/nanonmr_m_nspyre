"""
Example GUI elements.
"""
import logging
from functools import partial
from importlib import reload
from multiprocessing import Queue
from queue import Empty

from pyqtgraph import ComboBox, SpinBox
from pyqtgraph.Qt import QtGui, QtWidgets
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QGraphicsOpacityEffect,
)

from nspyre import ParamsWidget
from nspyre.misc.misc import ProcessRunner, run_experiment
from styling.flex_line_plot_2026_03_24 import FlexLinePlotWidget
from styling.params_2026_03_05 import FitParamsWidget

import experiment_defaults
import nv_experiments_2026_04_16
import nv_experiments_daq

def ellipsize(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."

class ExpWidget(QWidget):

    QUEUE_CHECK_TIME = 50 # ms

    def __init__(self, status_queue=None):
        super().__init__()

        self.all_defaults = experiment_defaults.build_experiment_defaults()

        self.setWindowTitle('NV Experiments')
        app_font = QtGui.QFont("Segoe UI", 16)
        self.setFont(app_font)
        self.setStyleSheet("""
        QWidget {
            font-family: "Segoe UI";
            font-size: 16pt;
        }
        """)
        
        # Setup professional styling
        self._setup_professional_styles()
        
        self.updateTimer = QTimer() #create a timer that will try to update that widget with messages from the from_exp_queue
        self.updateTimer.timeout.connect(lambda: self.check_queue_from_exp())
        self.updateTimer.start(self.QUEUE_CHECK_TIME)

        self.exp_inst_queue = status_queue

        # experiment dictionary - associates experiment function, default parameter array, dataset, laser parameters and digitizer parameters to an experiment type
        self.exp_dict = {
            "Signal vs Time": [
                "sigvstime_scan",
                self.all_defaults['sigvstime_params_defaults'],
                self.all_defaults['sigvstime_mw_params_defaults'],
                'sigvstime',
                self.all_defaults['laser_params_sigvstime_defaults'],
                self.all_defaults['digitizer_defaults']
            ], # ODMR MW params hidden and serves as placeholder for sig vs time experiment in GUI

            "CW ODMR": [
                "odmr_scan",
                self.all_defaults['odmr_params_defaults'],
                self.all_defaults['odmr_mw_params_defaults'],
                'odmr',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "ODMR Smart Scan": [
                "odmr_smart_scan",
                self.all_defaults['odmr_smart_params_defaults'],
                self.all_defaults['odmr_smart_mw_params_defaults'],
                'odmr',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "Sig Laser": [
                None,
                self.all_defaults['laser_params_sigvstime_defaults']
            ],

            "Laser": [
                None,
                self.all_defaults['laser_params_defaults']
            ],

            "Digitizer": [
                None,
                self.all_defaults['digitizer_defaults']
            ],

            "Pulsed ODMR": [
                "pulsed_odmr_scan",
                self.all_defaults['pulsed_odmr_params_defaults'],
                self.all_defaults['pulsed_odmr_mw_params_defaults'],
                'odmr',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "RF Coil: Pulsed ODMR": [
                "pulsed_odmr_rf_scan",
                self.all_defaults['pulsed_odmr_rf_params_defaults'],
                self.all_defaults['pulsed_odmr_rf_mw_params_defaults'],
                'odmr rf',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "Rabi": [
                "rabi_scan",
                self.all_defaults['rabi_params_defaults'],
                self.all_defaults['rabi_mw_params_defaults'],
                'rabi',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "Optical T1": [
                "OPT_T1_scan",
                self.all_defaults['opt_t1_params_defaults'],
                self.all_defaults['opt_t1_mw_params_defaults'],
                't1',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "MW T1": [
                "MW_T1_scan",
                self.all_defaults['mw_t1_params_defaults'],
                self.all_defaults['mw_t1_mw_params_defaults'],
                't1',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "T2": [
                "T2_scan",
                self.all_defaults['t2_params_defaults'],
                self.all_defaults['t2_mw_params_defaults'],
                't2',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "RF Coil: T2": [
                "T2_rf_scan",
                self.all_defaults['t2_rf_params_defaults'],
                self.all_defaults['t2_rf_mw_params_defaults'],
                't2',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DQ Relaxation": [
                "DQ_scan",
                self.all_defaults['dq_params_defaults'],
                self.all_defaults['dq_mw_params_defaults'],
                'dq',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DEER": [
                "DEER_scan",
                self.all_defaults['deer_params_defaults'],
                self.all_defaults['deer_mw_params_defaults'],
                'deer',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DEER Rabi": [
                "DEER_rabi_scan",
                self.all_defaults['deer_rabi_params_defaults'],
                self.all_defaults['deer_rabi_mw_params_defaults'],
                'deer rabi',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DEER FID": [
                "DEER_FID_scan",
                self.all_defaults['deer_fid_params_defaults'],
                self.all_defaults['deer_fid_mw_params_defaults'],
                'fid',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DEER FID Continuous Drive": [
                "DEER_FID_CD_scan",
                self.all_defaults['deer_fid_cd_params_defaults'],
                self.all_defaults['deer_fid_cd_mw_params_defaults'],
                'fid cd',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DEER Correlation Rabi": [
                "DEER_corr_rabi_scan",
                self.all_defaults['deer_corr_rabi_params_defaults'],
                self.all_defaults['deer_corr_rabi_mw_params_defaults'],
                'corr rabi',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DEER T1": [
                "DEER_T1_scan",
                self.all_defaults['deer_corr_t1_params_defaults'],
                self.all_defaults['deer_corr_t1_mw_params_defaults'],
                'deer t1',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "DEER T2": [
                "DEER_T2_scan",
                self.all_defaults['deer_t2_params_defaults'],
                self.all_defaults['deer_t2_mw_params_defaults'],
                'deer t2',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "NMR Correlation Spectroscopy": [
                "Corr_Spec_scan",
                self.all_defaults['corr_spec_params_defaults'],
                self.all_defaults['corr_spec_mw_params_defaults'],
                'nmr',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],

            "NMR CASR": [
                "CASR_scan",
                self.all_defaults['casr_params_defaults'],
                self.all_defaults['casr_mw_params_defaults'],
                'casr',
                self.all_defaults['laser_params_defaults'],
                self.all_defaults['digitizer_defaults']
            ],
        }

        self.experiments = QComboBox()
        self.experiments.setFixedHeight(40)
        self.experiments.setStyleSheet("color: black; background-color: #C7C7C7; border: 1px solid #888888; padding: 2px; border-radius: 4px; font-weight: bold;")
        self.experiments.addItems([
            "Select experiment from list",
            "Signal vs Time",
            "CW ODMR",
            "ODMR Smart Scan",
            "Rabi",
            "Pulsed ODMR",
            "RF Coil: Pulsed ODMR",
            "RF Coil: T2",
            "T2",
            "Optical T1",
            "MW T1",
            "DQ Relaxation",
            "DEER",
            "DEER Rabi",
            "DEER FID",
            "DEER FID Continuous Drive",
            "DEER Correlation Rabi",
            "DEER T1",
            "DEER T2",
            "NMR Correlation Spectroscopy",
            "NMR CASR"
        ])

        self.experiments.currentIndexChanged.connect(lambda: self.exp_selector())

        # used to send to experiment process to determine extra actions to take
        self.to_save = False
        self.file_format = "json"
        self.to_fit = False
        self.to_fit_live = False
        self.to_override_fit = False
        self.extra_kwarg_params: dict() = {}

        self.dataset_label = QLabel("Data Set Name: ---")
        self.dataset_label.setStyleSheet("color: black; background-color: #C7C7C7; border: 1px solid #888888; padding: 2px; border-radius: 4px; font-weight: bold;")
        self.dataset_label.setFixedHeight(40)
        self.dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # experiment params label
        self.exp_label = QLabel("Experiment Settings")
        self.exp_label.setFixedHeight(28)
        self.exp_label.setStyleSheet("font-weight: bold; color: #CCCCCC;")

        self.params_widget = ParamsWidget(self.create_params_widget('CW ODMR', self.exp_dict['CW ODMR'][1]))

        self.mw_label = QLabel("Microwave Settings")
        self.mw_label.setFixedHeight(28)
        self.mw_label.setStyleSheet("font-weight: bold; color: #CCCCCC;")

        self.mw_params_widget = ParamsWidget(self.create_mw_params_widget('CW ODMR', self.exp_dict['CW ODMR'][2]))
        self.mw_overrides = {}

        self.opacity_effects = []
        for i in range(24): # total number of GUI elements that need to be faded when no experiment is selected
            self.opacity_effects.append(QGraphicsOpacityEffect())
            self.opacity_effects[i].setOpacity(0.3)

        self.params_widget.setEnabled(False)

        # save params button
        self.save_params = QPushButton("Save Experiment Parameters")
        self.save_params.setStyleSheet(self.neutral_button_stylesheet)
        self.save_params.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.save_params.clicked.connect(lambda: self.save_params_clicked())
        self.save_params.setEnabled(False)

        # mw params widget & label
        self.mw_params_widget.setEnabled(False)

        # laser params widget & label
        self.laser_params_widget = ParamsWidget(self.create_params_widget('Laser', self.exp_dict['Laser'][1]))
        self.laser_label = QLabel("Laser & AWG Settings")
        self.laser_label.setFixedHeight(28)
        self.laser_label.setStyleSheet("font-weight: bold; color: #CCCCCC;")
        self.laser_params_widget.setEnabled(False)

        self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]))
        self.dig_label = QLabel("Digitizer Settings")
        self.dig_label.setFixedHeight(28)
        self.dig_label.setStyleSheet("font-weight: bold; color: #CCCCCC;")
        self.dig_params_widget.setEnabled(False)

        radio_style = """
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
        color: #5470FF;
        background-color: #1D2B6D;
        border: 2px solid #5470FF;
        }

        QRadioButton::indicator {
        width: 15px;
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

        # data acquisition system selection radio buttons
        self.daq_b1 = QRadioButton("Digitizer")
        self.daq_b1.toggled.connect(lambda:self.toggle_daq(self.daq_b1))
        self.daq_b1.setStyleSheet(radio_style)
        self.daq_b1.setFixedHeight(45)
        self.daq_b1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.daq_b1.setEnabled(False)

        self.daq_b2 = QRadioButton("NI DAQ")
        self.daq_b2.toggled.connect(lambda:self.toggle_daq(self.daq_b2))
        self.daq_b2.setStyleSheet(radio_style)
        self.daq_b2.setFixedHeight(45)
        self.daq_b2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.daq_b2.setEnabled(False)

        self.daq_group = QButtonGroup()
        self.daq_group.setExclusive(True)
        self.daq_group.addButton(self.daq_b1)
        self.daq_group.addButton(self.daq_b2)

        # time elapsed label
        self.time_elapsed = QLabel("<i>Experiment Timer</i>: 00:00:00")
        self.time_elapsed.setStyleSheet("color: black; background-color: #66DBE8; border: 1px solid #4A9BA8; padding: 2px; border-radius: 4px; font-weight: bold;")
        self.time_elapsed.setFixedHeight(40)
        self.time_elapsed.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # auto save checkbox
        self.auto_save_checkbox = QCheckBox("Auto Save ")
        self.auto_save_checkbox.setFixedHeight(40)
        self.auto_save_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #5F3200;
                        border: 1px solid #B8660D;
                        padding: 2px 6px;
                        border-radius: 4px;
                        font-weight: bold;
                        spacing: 8px;
                }

                QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                        border: 2px solid #B8660D;
                        border-radius: 3px;
                        background-color: #3D1E00;
                }

                QCheckBox::indicator:hover {
                        background-color: #6F4420;
                        border: 2px solid #D48620;
                }

                QCheckBox::indicator:checked {
                        background-color: #D48620;
                        border: 2px solid #D48620;
                        image: none;
                }

                QCheckBox::indicator:checked:hover {
                        background-color: #6F4420;
                        border: 2px solid #E8A040;
                }

                QCheckBox::indicator:pressed {
                        background-color: #2D1410;
                        border: 2px solid #B87018;
                }""")
        self.auto_save_checkbox.setChecked(False)
        self.auto_save_checkbox.setEnabled(False)
        self.auto_save_checkbox.stateChanged.connect(lambda: self.auto_save_changed())

        # select directory button
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.setEnabled(False)
        self.select_dir_button.setFixedHeight(40)
        self.select_dir_button.setFixedWidth(180)
        self.select_dir_button.setStyleSheet(self.neutral_button_stylesheet)
        self.select_dir_button.clicked.connect(lambda: self.select_directory())

        # file type selection combobox for saving
        self.select_file_format_combobox = QComboBox()
        self.select_file_format_combobox.setFixedHeight(40)
        self.select_file_format_combobox.setStyleSheet(self.combobox_stylesheet)
        self.select_file_format_combobox.addItems(["Form: JSON", "Form: Pickle"])
        self.select_file_format_combobox.setCurrentIndex(0)
        self.select_file_format_combobox.setEnabled(False)
        self.select_file_format_combobox.currentIndexChanged.connect(lambda: self.file_format_selector())

        # selected directory display for saving
        self.chosen_dir = QLabel()
        self.chosen_dir.setStyleSheet("color: orange")

        self.filename_label = QLabel("Filename: ")
        self.filename_label.setFixedHeight(20)
        self.filename_label.setStyleSheet("font-weight: bold;")

        self.filename_lineedit = QLineEdit()
        self.filename_lineedit.setFixedHeight(30)
        self.filename_lineedit.setEnabled(False)

        # auto fit checkbox
        self.auto_fit_checkbox = QCheckBox("Auto Fit  ")
        self.auto_fit_checkbox.setFixedHeight(40)
        self.auto_fit_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #55005F;
                        border: 1px solid #9955BB;
                        padding: 2px 6px;
                        border-radius: 4px;
                        font-weight: bold;
                        spacing: 8px;
                }

                QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                        border: 2px solid #9955BB;
                        border-radius: 3px;
                        background-color: #330033;
                }

                QCheckBox::indicator:hover {
                        background-color: #663366;
                        border: 2px solid #BB77DD;
                }

                QCheckBox::indicator:checked {
                        background-color: #BB77DD;
                        border: 2px solid #BB77DD;
                        image: none;
                }

                QCheckBox::indicator:checked:hover {
                        background-color: #663366;
                        border: 2px solid #DD99FF;
                }

                QCheckBox::indicator:pressed {
                        background-color: #220022;
                        border: 2px solid #9955BB;
                }""")
        self.auto_fit_checkbox.setChecked(False)
        self.auto_fit_checkbox.setEnabled(False)
        self.auto_fit_checkbox.stateChanged.connect(lambda: self.auto_fit_changed())

        # fit type combobox
        self.fit_select = QComboBox()
        self.fit_select.setFixedHeight(40)
        self.fit_select.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.fit_select.setStyleSheet("background-color: #C2C2C2; color: black; border: 1px solid #888888; padding: 2px; border-radius: 4px; font-weight: bold;")
        self.fit_select.addItems(["Choose Fit Type",
                                 "Neg. Lorentz.",
                                 "Pos. Lorentz.",
                                 "Two Neg. Lorentz.",
                                 "Decaying Cos.",
                                 "Stretched Exp.",
                                 "Modulated Str. Exp.",
                                 "DEER T1 Str. Exp."])
        self.fit_select.currentIndexChanged.connect(self.fit_selector)
        self.fit_select.setEnabled(False)

        self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit ODMR', self.all_defaults['fit_neg_lorentz_defaults']))
        self.fit_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self.fit_params_widget.setEnabled(False)

        # live fitting checkbox
        self.live_fit_checkbox = QCheckBox("Live Fitting")
        self.live_fit_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #515151;
                        border: 1px solid #888888;
                        padding: 2px;
                        border-radius: 4px;
                        font-weight: bold;
                }

                QCheckBox::indicator:hover {
                        background-color: yellow;
                }

                QCheckBox::indicator:pressed {
                        background-color: lightgreen;
                }""")

        self.live_fit_checkbox.setChecked(False)
        self.live_fit_checkbox.stateChanged.connect(lambda: self.live_fit_changed())
        self.live_fit_checkbox.setEnabled(False)

        self.override_fit_checkbox = QCheckBox("Apply Fits to Settings")
        self.override_fit_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #3D3D3D;
                        border: 1px solid #888888;
                        padding: 2px;
                        border-radius: 4px;
                        font-weight: bold;
                }
                QCheckBox::indicator:hover {
                        background-color: yellow;
                }
                QCheckBox::indicator:pressed {
                        background-color: lightgreen;
                }""")
        self.override_fit_checkbox.setChecked(False)
        self.override_fit_checkbox.stateChanged.connect(lambda: self.override_fit_changed())
        self.override_fit_checkbox.setEnabled(False)

        # photodetector selection radio buttons
        detector_radio_style = """
        QRadioButton {
        color: white;
        background-color: #6B6B1F;
        border: 2px solid #A89933;
        border-radius: 6px;
        padding: 8px 12px;
        spacing: 10px;
        font-weight: bold;
        font-size: 14pt;
        }

        QRadioButton:hover {
        background-color: #7B7B2F;
        border: 2px solid #C4B443;
        }

        QRadioButton:checked {
        color: #C4C433;
        background-color: #5B5B0F;
        border: 2px solid #C4C433;
        }

        QRadioButton::indicator {
        width: 15px;
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
        background-color: #C4C433;
        border: 2px solid #C4C433;
        image: none;
        }
        """
        self.detector_b1 = QRadioButton("APD")
        self.detector_b1.setStyleSheet(detector_radio_style)
        self.detector_b1.setFixedHeight(45)
        self.detector_b1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.detector_b1.setEnabled(False)

        self.detector_b2 = QRadioButton("BPD")
        self.detector_b2.setStyleSheet(detector_radio_style)
        self.detector_b2.setFixedHeight(45)
        self.detector_b2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.detector_b2.setEnabled(False)

        self.detector_group = QButtonGroup()
        self.detector_group.setExclusive(True)
        self.detector_group.addButton(self.detector_b1)
        self.detector_group.addButton(self.detector_b2)

        self.detector_b1.setChecked(True) # default to APD

        # status label
        self.status = QLabel("Set parameters and press 'Run' to begin experiment.")
        self.status.setStyleSheet("color: black; background-color: #00b8ff; padding: 2px; font-weight: bold;")
        self.status.setFixedHeight(40)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # time estimate label
        self.time_estimate = QLabel("<i>Time Estimate</i>: --:--:--")
        self.time_estimate.setStyleSheet("color: black; background-color: #EDA855; border: 1px solid #D68812; padding: 2px; border-radius: 4px; font-weight: bold;")
        self.time_estimate.setFixedHeight(40)

        # time remaining label
        self.time_remaining = QLabel("<i>Time Remaining</i>: --:--:--")
        self.time_remaining.setStyleSheet("color: black; background-color: #EDD155; border: 1px solid #D4B835; padding: 2px; border-radius: 4px; font-weight: bold;")
        self.time_remaining.setFixedHeight(40)

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
        QProgressBar {
                border: 1px solid #888888;
                border-radius: 4px;
                background-color: #2a2a2a;
                color: white;
                text-align: center;
                font-weight: bold;
        }

        QProgressBar::chunk {
                background-color: #00C7BA;
        }
        """)

        # run button
        run_button = QPushButton('Run')
        run_button.setStyleSheet(self.run_button_stylesheet)
        self.run_proc = ProcessRunner()
        run_button.clicked.connect(self.run)

        self.queue_to_exp: Queue = Queue()
        """multiprocessing Queue to pass to the experiment subprocess and use
        for sending messages to the subprocess."""
        self.queue_from_exp: Queue = Queue()
        """multiprocessing Queue to pass to the experiment subprocess and use
        for receiving messages from the subprocess."""

        # stop button
        stop_button = QPushButton('Stop')
        stop_button.setStyleSheet(self.stop_button_stylesheet)
        stop_button.clicked.connect(self.stop)
        # use a partial because the stop function may already be destroyed by the time
        # this is called
        self.destroyed.connect(partial(self.stop, log=False))

        # kill button
        # this is used to kill the experiment process if it is stuck
        kill_button = QPushButton('Kill')
        kill_button.setStyleSheet(self.kill_button_stylesheet)
        kill_button.clicked.connect(self.kill)

        ### --- Set graphics effects for elements that should be faded when no experiment is selected --- ###
        self.exp_label.setGraphicsEffect(self.opacity_effects[0])
        self.params_widget.setGraphicsEffect(self.opacity_effects[1])
        self.save_params.setGraphicsEffect(self.opacity_effects[2])
        self.mw_label.setGraphicsEffect(self.opacity_effects[3])
        self.mw_params_widget.setGraphicsEffect(self.opacity_effects[4])
        self.laser_label.setGraphicsEffect(self.opacity_effects[5])
        self.laser_params_widget.setGraphicsEffect(self.opacity_effects[6])
        self.dig_label.setGraphicsEffect(self.opacity_effects[7])
        self.dig_params_widget.setGraphicsEffect(self.opacity_effects[8])
        self.daq_b1.setGraphicsEffect(self.opacity_effects[9])
        self.daq_b2.setGraphicsEffect(self.opacity_effects[10])
        self.auto_save_checkbox.setGraphicsEffect(self.opacity_effects[11])
        self.select_dir_button.setGraphicsEffect(self.opacity_effects[12])
        self.chosen_dir.setGraphicsEffect(self.opacity_effects[13])
        self.filename_label.setGraphicsEffect(self.opacity_effects[14])
        self.filename_lineedit.setGraphicsEffect(self.opacity_effects[15])
        self.auto_fit_checkbox.setGraphicsEffect(self.opacity_effects[16])
        self.fit_select.setGraphicsEffect(self.opacity_effects[17])
        self.fit_params_widget.setGraphicsEffect(self.opacity_effects[18])
        self.live_fit_checkbox.setGraphicsEffect(self.opacity_effects[19])
        self.override_fit_checkbox.setGraphicsEffect(self.opacity_effects[20])
        self.detector_b1.setGraphicsEffect(self.opacity_effects[21])
        self.detector_b2.setGraphicsEffect(self.opacity_effects[22])
        self.select_file_format_combobox.setGraphicsEffect(self.opacity_effects[23])

        self.gui_layout = QVBoxLayout()

        # frame styling
        self.top_frame = QFrame(self)
        self.top_frame.setObjectName("topFrame")
        self.top_frame.setStyleSheet("QFrame#topFrame {background-color: #1e1e1e; border: 1px solid #666666; border-radius: 4px;}")
        self.top_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.top_layout = QHBoxLayout(self.top_frame)
        self.top_layout.setSpacing(0)
        self.top_layout.addWidget(self.experiments)
        self.top_layout.addWidget(self.dataset_label)
        self.top_layout.addWidget(self.time_elapsed)

        self.exp_frame = QFrame(self)
        self.exp_frame.setObjectName("expFrame")
        self.exp_frame.setStyleSheet("QFrame#expFrame {background-color: #4b0000; border: 1px solid #8B3333; border-radius: 4px;}")
        self.exp_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.exp_params_layout = QVBoxLayout(self.exp_frame)
        self.exp_params_layout.setContentsMargins(6,6,6,6)
        self.exp_params_layout.setSpacing(0)
        self.exp_params_layout.addWidget(self.exp_label)
        self.exp_params_layout.addWidget(self.params_widget)

        self.exp_scroll = QScrollArea(self)
        self.exp_scroll.setWidgetResizable(True)
        self.exp_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.exp_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.exp_scroll.setFrameShape(QFrame.Shape.NoFrame)  # cleaner
        # self.exp_scroll.setMinimumWidth(400)
        self.exp_scroll.setWidget(self.exp_frame)

        # size policies: scroll area takes space, inner frame stays minimal
        self.exp_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.mw_frame = QFrame(self)
        self.mw_frame.setObjectName("mwFrame")
        self.mw_frame.setStyleSheet("QFrame#mwFrame {background-color: #474b00; border: 1px solid #B8A600; border-radius: 4px;}")
        self.mw_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.mw_params_layout = QVBoxLayout(self.mw_frame)
        self.mw_params_layout.setContentsMargins(6,6,6,6)
        self.mw_params_layout.setSpacing(0)
        self.mw_params_layout.addWidget(self.mw_label)
        self.mw_params_layout.addWidget(self.mw_params_widget)

        self.mw_scroll = QScrollArea(self)
        self.mw_scroll.setWidgetResizable(True)
        self.mw_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mw_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mw_scroll.setFrameShape(QFrame.Shape.NoFrame)  # cleaner
        # self.mw_scroll.setMinimumWidth(400)
        self.mw_scroll.setWidget(self.mw_frame)

        # size policies: scroll area takes space, inner frame stays minimal
        self.mw_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.detector_frame = QFrame(self)
        self.detector_frame.setObjectName("detectorFrame")
        self.detector_frame.setStyleSheet("QFrame#detectorFrame {background-color: #262626; border: 1px solid #888888; border-radius: 4px;}")
        self.detector_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.detector_layout = QGridLayout(self.detector_frame)
        self.detector_layout.setSpacing(0)
        self.detector_layout.setContentsMargins(6, 6, 6, 6)
        self.detector_layout.setColumnStretch(0, 1)  # make column expand
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_params)
        button_row = QHBoxLayout()
        button_row.setSpacing(0)
        button_row.addWidget(self.daq_b1)
        button_row.addWidget(self.daq_b2)
        button_row.addWidget(self.detector_b1)
        button_row.addWidget(self.detector_b2)
        self.detector_layout.addLayout(save_row,1,0)
        self.detector_layout.addLayout(button_row,2,0)

        self.save_frame = QFrame(self)
        self.save_frame.setObjectName("saveFrame")
        self.save_frame.setStyleSheet("QFrame#saveFrame {background-color: #1e1e1e; border: 1px solid #666666; border-radius: 4px;}")
        self.save_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.save_layout = QGridLayout(self.save_frame)
        self.save_layout.setSpacing(0)
        self.save_layout.addWidget(self.auto_save_checkbox,1,1,1,1)
        self.save_layout.addWidget(self.select_dir_button,1,2,1,1, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.save_layout.addWidget(self.select_file_format_combobox,1,3,1,1)
        self.save_layout.addWidget(self.chosen_dir,2,1,1,3)
        self.save_layout.addWidget(self.filename_label,3,1,1,1)
        self.save_layout.addWidget(self.filename_lineedit,3,2,1,2)

        self.fit_frame = QFrame(self)
        self.fit_frame.setObjectName("fitFrame")
        self.fit_frame.setStyleSheet("QFrame#fitFrame {background-color: #1e1e1e; border: 1px solid #666666; border-radius: 4px;}")
        self.fit_layout = QGridLayout(self.fit_frame)
        self.fit_layout.setContentsMargins(8, 8, 8, 8)
        self.fit_layout.setHorizontalSpacing(10)
        self.fit_layout.setVerticalSpacing(8)
        # let the right side expand more
        self.fit_layout.setColumnStretch(0, 0)   # left checkbox column
        self.fit_layout.setColumnStretch(1, 1)   # right combobox / controls column

        self.fit_layout.addWidget(self.auto_fit_checkbox, 0, 0, 1, 1)
        self.fit_layout.addWidget(self.fit_select,         0, 1, 1, 1)
        self.fit_layout.addWidget(self.fit_params_widget,  1, 0, 1, 2) # fit params gets the whole row
        self.fit_layout.addWidget(self.live_fit_checkbox,     2, 0, 1, 1)
        self.fit_layout.addWidget(self.override_fit_checkbox, 2, 1, 1, 1)

        self.fit_scroll = QScrollArea(self)
        self.fit_scroll.setWidgetResizable(True)
        self.fit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.fit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.fit_scroll.setFrameShape(QFrame.Shape.NoFrame)  # cleaner
        self.fit_scroll.setMinimumWidth(500)
        self.fit_scroll.setWidget(self.fit_frame)

        # size policies: scroll area takes space, inner frame stays minimal
        self.fit_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fit_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.bottom_frame = QFrame(self)
        self.bottom_frame.setObjectName("bottomFrame")
        self.bottom_frame.setStyleSheet("QFrame#bottomFrame {background-color: #1e1e1e; border: 1px solid #666666; border-radius: 4px;}")
        self.bottom_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.bottom_layout = QGridLayout(self.bottom_frame)
        self.bottom_layout.setSpacing(0)
        self.bottom_layout.addWidget(self.status,1,1,1,2)
        self.bottom_layout.addWidget(self.time_estimate,1,3,1,1)
        self.bottom_layout.addWidget(self.progress_bar,2,1,1,2)
        self.bottom_layout.addWidget(self.time_remaining,2,3,1,1)
        self.bottom_layout.addWidget(run_button,3,1,1,1)
        self.bottom_layout.addWidget(stop_button,3,2,1,1)
        self.bottom_layout.addWidget(kill_button,3,3,1,1)

        self.laser_frame = QFrame(self)
        self.laser_frame.setObjectName("laserFrame")
        self.laser_frame.setStyleSheet("QFrame#laserFrame {background-color: #004b47; border: 1px solid #009999; border-radius: 4px;}")
        self.laser_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.laser_params_layout = QGridLayout(self.laser_frame)
        self.laser_params_layout.setContentsMargins(6,6,6,6)
        self.laser_params_layout.setSpacing(0)
        self.laser_params_layout.addWidget(self.laser_label,1,1,1,2)
        self.laser_params_layout.addWidget(self.laser_params_widget,2,1,1,2)

        self.laser_scroll = QScrollArea(self)
        self.laser_scroll.setWidgetResizable(True)
        self.laser_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.laser_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.laser_scroll.setFrameShape(QFrame.Shape.NoFrame)  # cleaner
        # self.laser_scroll.setMinimumWidth(400)
        self.laser_scroll.setWidget(self.laser_frame)

        # size policies: scroll area takes space, inner frame stays minimal
        self.laser_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.dig_frame = QFrame(self)
        self.dig_frame.setObjectName("digFrame")
        self.dig_frame.setStyleSheet("QFrame#digFrame {background-color: #000b4b; border: 1px solid #1155AA; border-radius: 4px;}")
        self.dig_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.dig_params_layout = QVBoxLayout(self.dig_frame)
        self.dig_params_layout.setContentsMargins(6,6,6,6)
        self.dig_params_layout.setSpacing(0)
        self.dig_params_layout.addWidget(self.dig_label)
        self.dig_params_layout.addWidget(self.dig_params_widget)

        self.dig_scroll = QScrollArea(self)
        self.dig_scroll.setWidgetResizable(True)
        self.dig_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.dig_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.dig_scroll.setFrameShape(QFrame.Shape.NoFrame)  # cleaner
        # self.dig_scroll.setMinimumWidth(400)
        self.dig_scroll.setWidget(self.dig_frame)

        # size policies: scroll area takes space, inner frame stays minimal
        self.dig_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.top_widgets_layout = QVBoxLayout()
        self.top_widgets_layout.addWidget(self.top_frame)

        self.exp_widgets_layout = QHBoxLayout()
        self.exp_widgets_layout.addWidget(self.exp_scroll)
        self.exp_widgets_layout.addWidget(self.mw_scroll)

        self.laser_widgets_layout = QHBoxLayout()
        self.laser_widgets_layout.addWidget(self.laser_scroll)
        self.laser_widgets_layout.addWidget(self.dig_scroll)

        self.save_widgets_layout = QGridLayout()
        self.save_widgets_layout.addWidget(self.detector_frame,1,1,1,1)
        self.save_widgets_layout.addWidget(self.save_frame,2,1,1,1)
        self.save_widgets_layout.addWidget(self.fit_scroll,1,2,2,1)

        self.bottom_widgets_layout = QVBoxLayout()
        self.bottom_widgets_layout.addWidget(self.bottom_frame)

        self.gui_layout.addLayout(self.top_widgets_layout)
        self.gui_layout.addLayout(self.exp_widgets_layout)
        self.gui_layout.addLayout(self.laser_widgets_layout)
        self.gui_layout.addLayout(self.save_widgets_layout)
        self.gui_layout.addLayout(self.bottom_widgets_layout)

        self.setLayout(self.gui_layout)

    def _setup_professional_styles(self):
        """Setup centralized professional styling for all widgets."""
        # Define reusable stylesheets based on existing color scheme
        
        # Primary button: Green (Run)
        self.run_button_stylesheet = """
            QPushButton {
                background-color: #005C3D;
                color: white;
                border: 1px solid #00A652;
                border-radius: 4px;
                padding: 8px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #007A52;
            }
            QPushButton:pressed {
                background-color: #003D28;
            }
        """
        
        # Secondary button: Gray (Stop)
        self.stop_button_stylesheet = """
            QPushButton {
                background-color: #4A4A4A;
                color: white;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 8px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QPushButton:pressed {
                background-color: #3A3A3A;
            }
        """
        
        # Danger button: Red (Kill)
        self.kill_button_stylesheet = """
            QPushButton {
                background-color: #5C1F1F;
                color: white;
                border: 1px solid #8B3333;
                border-radius: 4px;
                padding: 8px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7A2828;
            }
            QPushButton:pressed {
                background-color: #3D1515;
            }
        """
        
        # Neutral button: Brown/Gray (Save, Select) - lighter color
        self.neutral_button_stylesheet = """
            QPushButton {
                background-color: #A0A0A0;
                color: black;
                border: 1px solid #888888;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #B8B8B8;
            }
            QPushButton:pressed {
                background-color: #888888;
            }
        """
        
        # Standard combobox
        self.combobox_stylesheet = """
            QComboBox {
                background-color: #C7C7C7;
                color: black;
                border: 1px solid #888888;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
            }
            QComboBox:focus {
                border: 2px solid #4166F5;
            }
            QComboBox QAbstractItemView {
                background-color: #C7C7C7;
                color: black;
                selection-background-color: #808080;
                border: 1px solid #888888;
            }
        """

    def create_pl_widgets(self, num_pts_widget, defaults_dict, pl_pt_key):
        """
        Create PL trace checkbox + pl_pt widget with dynamic bounds.

        Args:
            num_pts_widget: SpinBox controlling number of points
            defaults_dict: dict of defaults
            pl_pt_key: key for pl_pt in defaults_dict

        Returns:
            enable_pl_widget, pl_pt_widget
        """

        # --- safe default ---
        pl_pt_default = defaults_dict.get(pl_pt_key, 0)

        pl_pt_widget = SpinBox(value=pl_pt_default, int=True, bounds=(0, None), dec=True)
        enable_pl_widget = QCheckBox()
        enable_pl_widget.setChecked(False)

        # --- update bounds ---
        def update_pl_pt_bounds():
            num_pts = max(1, int(num_pts_widget.value()))
            pl_pt_widget.setOpts(bounds=(0, num_pts - 1))

            if pl_pt_widget.value() >= num_pts:
                pl_pt_widget.setValue(num_pts - 1)

        # --- toggle enable ---
        def toggle_pl_widgets():
            pl_pt_widget.setEnabled(enable_pl_widget.isChecked())

        # --- connect ---
        num_pts_widget.sigValueChanged.connect(update_pl_pt_bounds)
        enable_pl_widget.toggled.connect(toggle_pl_widgets)

        # --- initialize ---
        update_pl_pt_bounds()
        toggle_pl_widgets()

        return enable_pl_widget, pl_pt_widget

    def create_params_widget(self, tag, defaults):
        match tag:
            case 'Sig Laser':
                params = {
                        'laser_power': {'display_text': 'Laser Power (%): ',
                                'widget': SpinBox(value = defaults['laser_power'], int = True, bounds=(0, 100), dec = True)}}
            case 'Laser':
                params = {
                        'laser_power': {'display_text': 'Laser Power (%): ',
                                'widget': SpinBox(value = defaults['laser_power'], int = True, bounds=(0, 95), dec = True)},
                        'laser_init': {'display_text': 'Initialize Time (pulsed): ',
                                'widget': SpinBox(value = defaults['laser_init'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'laser_readout': {'display_text': 'Read Time (pulsed): ',
                                'widget': SpinBox(value = defaults['laser_readout'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'sideband_freq': {'display_text': 'MW Sideband Mod. Freq.: ',
                                        'widget': SpinBox(value = defaults['sideband_freq'], suffix = 'Hz', siPrefix = True, bounds = (100, 100e6), dec = True)},
                        'sideband_power': {'display_text': 'MW Sideband Power: ',
                                        'widget': SpinBox(value = defaults['sideband_power'], suffix = 'V', siPrefix = True, bounds = (0, 0.45))},
                        'sideband': {'display_text': 'MW Sideband: ',
                                        'widget': ComboBox(items = defaults['sideband'])},
                        'i_offset': {'display_text': 'MW I Offset: ',
                                        'widget': SpinBox(value = defaults['i_offset'], suffix = 'V', siPrefix = True)},
                        'q_offset': {'display_text': 'MW Q Offset: ',
                                        'widget': SpinBox(value = defaults['q_offset'], suffix = 'V', siPrefix = True)}}
            case 'Digitizer':
                params = {
                        'segment_size': {'display_text': '# Samples (seg. size): ',
                                'widget': SpinBox(value = defaults['segment_size'], int = True, bounds=(0, 1e9), dec = True)},
                        'dig_sampling_freq': {'display_text': 'Sampling Frequency: ',
                                'widget': SpinBox(value = defaults['dig_sampling_freq'], suffix = 'Hz', siPrefix = True, bounds = (100, 500e6), dec = True)},
                        'dig_amplitude': {'display_text': 'Amplitude: ',
                                'widget': SpinBox(value = defaults['dig_amplitude'], suffix = 'V', siPrefix = True)},
                        'read_channel': {'display_text': 'Readout Channel: ',
                                'widget': ComboBox(items = defaults['read_channel'])},
                        'both_channels': {'display_text': 'Both Channels: ',
                                'widget': QCheckBox()},
                        'dig_coupling': {'display_text': 'Coupling: ',
                                'widget': ComboBox(items = defaults['dig_coupling'])},
                        'dig_termination': {'display_text': 'Termination (\u03A9): ',
                                'widget': ComboBox(items = defaults['dig_termination'])},
                        'pretrig_size': {'display_text': '# Pretrig. Samples: ',
                                'widget': SpinBox(value = defaults['pretrig_size'], int = True, bounds=(0, 1024), dec = True)},
                        'dig_timeout': {'display_text': 'Card Timeout: ',
                                'widget': SpinBox(value = defaults['dig_timeout'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
                params['both_channels']['widget'].setChecked(defaults['both_channels'])
            case 'Signal vs Time':
                params = {
                'exp_sampling_rate': {'display_text': 'Exp. Sampling Rate: ',
                        'widget': SpinBox(value = defaults['exp_sampling_rate'], suffix = 'Hz', siPrefix = True, bounds = (10, 1e6), dec = True)}}
            case 'CW ODMR':
                num_pts_widget = SpinBox(value=defaults['num_pts'], int=True, bounds=(1, None), dec=True)

                enable_pl_widget, pl_pt_widget = self.create_pl_widgets(
                    num_pts_widget, defaults, pl_pt_key='pl_pt'
                )

                params = {
                    'runs': {
                        'display_text': '# Averages per Iteration: ',
                        'widget': SpinBox(value=defaults['runs'], int=True, bounds=(1, None))
                    },
                    'iters': {
                        'display_text': '# Experiment Iterations: ',
                        'widget': SpinBox(value=defaults['iters'], int=True, bounds=(1, None))
                    },
                    'num_pts': {
                        'display_text': '# Frequencies: ',
                        'widget': num_pts_widget
                    },
                    'enable_pl_trace': {
                        'display_text': 'Enable PL Trace: ',
                        'widget': enable_pl_widget
                    },
                    'pl_pt': {
                        'display_text': 'Data Pt. for PL Trace: ',
                        'widget': pl_pt_widget
                    }
                }
            case 'ODMR Smart Scan':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'start_angle': {'display_text': 'Start Angle: ',
                                'widget': SpinBox(value = defaults['start_angle'], bounds=(1, 115), dec = True)},
                        'stop_angle': {'display_text': 'Stop Angle: ',
                                'widget': SpinBox(value = defaults['stop_angle'], bounds=(1, 115), dec = True)},
                        'iters': {'display_text': '# Angles to Sweep: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, 100))}}
            case 'Pulsed ODMR':
                num_pts_widget = SpinBox(value=defaults['num_pts'], int=True, bounds=(1, None), dec=True)

                enable_pl_widget, pl_pt_widget = self.create_pl_widgets(
                    num_pts_widget, defaults, pl_pt_key='pl_pt'
                )

                params = {
                    'runs': {'display_text': '# Averages per Iteration: ',
                            'widget': SpinBox(value=defaults['runs'], int=True, bounds=(1, None))},
                    'iters': {'display_text': '# Experiment Iterations: ',
                            'widget': SpinBox(value=defaults['iters'], int=True, bounds=(1, None))},
                    'num_pts': {'display_text': '# Frequencies: ',
                                'widget': num_pts_widget},
                    'enable_pl_trace': {'display_text': 'Enable PL Trace: ',
                                        'widget': enable_pl_widget},
                    'pl_pt': {'display_text': 'Data Pt. for PL Trace: ',
                            'widget': pl_pt_widget},
                }
            case 'RF Coil: Pulsed ODMR':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)}}
            case 'Rabi':
                num_pts_widget = SpinBox(value=defaults['num_pts'], int=True, bounds=(1, None), dec=True)

                enable_pl_widget, pl_pt_widget = self.create_pl_widgets(
                    num_pts_widget, defaults, pl_pt_key='pl_pt'
                )

                params = {
                    'runs': {'display_text': '# Averages per Iteration: ',
                            'widget': SpinBox(value=defaults['runs'], int=True, bounds=(1, None))},
                    'iters': {'display_text': '# Experiment Iterations: ',
                            'widget': SpinBox(value=defaults['iters'], int=True, bounds=(1, None))},
                    'start': {'display_text': 'Start MW Pulse Time: ',
                            'widget': SpinBox(value=defaults['start'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'stop': {'display_text': 'End MW Pulse Time: ',
                            'widget': SpinBox(value=defaults['stop'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': num_pts_widget},
                    'enable_pl_trace': {'display_text': 'Enable PL Trace: ',
                                        'widget': enable_pl_widget},
                    'pl_pt': {'display_text': 'Data Pt. for PL Trace: ',
                            'widget': pl_pt_widget},
                }
            case 'Optical T1':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])}}
            case 'MW T1':
                num_pts_widget = SpinBox(value=defaults['num_pts'], int=True, bounds=(1, None), dec=True)

                enable_pl_widget, pl_pt_widget = self.create_pl_widgets(
                    num_pts_widget, defaults, pl_pt_key='pl_pt'
                )

                params = {
                    'runs': {'display_text': '# Averages per Iteration: ',
                            'widget': SpinBox(value=defaults['runs'], int=True, bounds=(1, None))},
                    'iters': {'display_text': '# Experiment Iterations: ',
                            'widget': SpinBox(value=defaults['iters'], int=True, bounds=(1, None))},
                    'start': {'display_text': 'Start \u03C4 Time: ',
                            'widget': SpinBox(value=defaults['start'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'stop': {'display_text': 'Stop \u03C4 Time: ',
                            'widget': SpinBox(value=defaults['stop'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'num_pts': {'display_text': '# \u03C4: ',
                                'widget': num_pts_widget},
                    'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items=defaults['array_type'])},
                    'enable_pl_trace': {'display_text': 'Enable PL Trace: ',
                                        'widget': enable_pl_widget},
                    'pl_pt': {'display_text': 'Data Pt. for PL Trace: ',
                            'widget': pl_pt_widget},
                }
            case 'RF Coil: T2':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])}}
            case 'T2':
                num_pts_widget = SpinBox(value=defaults['num_pts'], int=True, bounds=(1, None), dec=True)

                enable_pl_widget, pl_pt_widget = self.create_pl_widgets(
                    num_pts_widget, defaults, pl_pt_key='pl_pt'
                )

                params = {
                    'runs': {'display_text': '# Averages per Iteration: ',
                            'widget': SpinBox(value=defaults['runs'], int=True, bounds=(1, None))},
                    'iters': {'display_text': '# Experiment Iterations: ',
                            'widget': SpinBox(value=defaults['iters'], int=True, bounds=(1, None))},
                    'start': {'display_text': 'Start \u03C4 Time: ',
                            'widget': SpinBox(value=defaults['start'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'stop': {'display_text': 'Stop \u03C4 Time: ',
                            'widget': SpinBox(value=defaults['stop'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'num_pts': {'display_text': '# \u03C4: ',
                                'widget': num_pts_widget},
                    'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items=defaults['array_type'])},
                    'enable_pl_trace': {'display_text': 'Enable PL Trace: ',
                                        'widget': enable_pl_widget},
                    'pl_pt': {'display_text': 'Data Pt. for PL Trace: ',
                            'widget': pl_pt_widget},
                }
            case 'DQ Relaxation':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])}}
            case 'DEER':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start Frequency: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'stop': {'display_text': 'End Frequency: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults['tau'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
            case 'DEER Rabi':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start AWG Pulse Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'stop': {'display_text': 'End AWG Pulse Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults['tau'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
            case 'DEER FID':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])}}
            case 'DEER FID Continuous Drive':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])}}
            case 'DEER Correlation Rabi':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start AWG Corr. Pulse Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'stop': {'display_text': 'End AWG Corr. Pulse Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults['tau'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        't_corr': {'display_text': 'Correlation Time \u03C4_c: ',
                                'widget': SpinBox(value = defaults['t_corr'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
            case 'DEER T1':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults['tau'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
            case 'DEER T2':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start t Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop t Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# t Points: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults['tau'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'deer_t2_buffer': {'display_text': 't Buffer Time: ',
                                'widget': SpinBox(value = defaults['deer_t2_buffer'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
            case 'NMR Correlation Spectroscopy':
                num_pts_widget = SpinBox(value=defaults['num_pts'], int=True, bounds=(1, None), dec=True)

                enable_pl_widget, pl_pt_widget = self.create_pl_widgets(
                    num_pts_widget, defaults, pl_pt_key='pl_pt'
                )

                params = {
                    'runs': {'display_text': '# Averages per Iteration: ',
                            'widget': SpinBox(value=defaults['runs'], int=True, bounds=(1, None))},
                    'iters': {'display_text': '# Experiment Iterations: ',
                            'widget': SpinBox(value=defaults['iters'], int=True, bounds=(1, None))},
                    'start': {'display_text': 'Start \u03C4 Time: ',
                            'widget': SpinBox(value=defaults['start'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'stop': {'display_text': 'Stop \u03C4 Time: ',
                            'widget': SpinBox(value=defaults['stop'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'num_pts': {'display_text': '# Frequencies: ',
                                'widget': num_pts_widget},
                    'tau': {'display_text': 'Free Precession Interval (\u03C4): ',
                            'widget': SpinBox(value=defaults['tau'], suffix='s', siPrefix=True, bounds=(0, None), dec=True)},
                    'sig_opt': {'display_text': 'Signal Source: ',
                                'widget': ComboBox(items=defaults['sig_opt'])},
                    'enable_pl_trace': {'display_text': 'Enable PL Trace: ',
                                        'widget': enable_pl_widget},
                    'pl_pt': {'display_text': 'Data Pt. for PL Trace: ',
                            'widget': pl_pt_widget},
                }
            case 'NMR CASR':
                params = {
                        'runs': {'display_text': 'Runs (avgs. per iteration): ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': 'n_R (# synch. readout pts.): ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'tau = 1/(2f_0): ',
                                'widget': SpinBox(value = defaults['tau'], suffix = 's', siPrefix = True, bounds = (0, 1e-3), dec = True)},
                    'sig_opt': {'display_text': 'Signal Source: ',
                                'widget': ComboBox(items=defaults['sig_opt'])},
                    'dnp': {'display_text': 'Hyperpolarization: ',
                            'widget': ComboBox(items=defaults['dnp'])}}

        return params

    def create_mw_params_widget(self, tag, defaults, overrides=None):
        overrides = {} if overrides is None else overrides

        match tag:
            case 'Signal vs Time': # not used - hidden
                params = {
                'exp_sampling_rate_0': {'display_text': 'Sampling Rate: ',
                        'widget': SpinBox(value = defaults['exp_sampling_rate_0'], suffix = 'Hz', siPrefix = True, bounds = (10, 1e6), dec = True)}}
            case 'CW ODMR':
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults['center_freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults['half_span_sideband_freq'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults['rf_power'], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'probe': {'display_text': 'MW Probe Time: ',
                                'widget': SpinBox(value = defaults['probe'], suffix = 's', siPrefix = True, bounds = (10e-9, None))}}
            case 'ODMR Smart Scan':
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults['center_freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults['half_span_sideband_freq'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults['rf_power'], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'probe': {'display_text': 'MW Probe Time: ',
                                'widget': SpinBox(value = defaults['probe'], suffix = 's', siPrefix = True, bounds = (10e-9, None))}}
            case 'Pulsed ODMR':
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults['center_freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults['half_span_sideband_freq'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)}}
            case 'RF Coil: Pulsed ODMR':
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults['center_freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults['half_span_sideband_freq'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'rf_pulse_freq': {'display_text': 'RF Frequency: ',
                                'widget': SpinBox(value = defaults['rf_pulse_freq'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_pulse_power': {'display_text': 'RF Power: ',
                                'widget': SpinBox(value = defaults['rf_pulse_power'], suffix = 'V', siPrefix = True)},
                        'rf_pulse_phase': {'display_text': 'RF Phase (deg.): ',
                                'widget': SpinBox(value = defaults['rf_pulse_phase'], int = True, bounds=(0, 360))}}
            case 'Rabi':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults['rf_power'], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': ComboBox(items = defaults['pulse_axis'])}}
            case 'Optical T1': # not used - hidden
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults['runs'], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults['iters'], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['start'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults['stop'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults['num_pts'], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults['array_type'])}}
            case 'MW T1':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])}}
            case 'T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        't2_seq': {'display_text': 'Sequence: ',
                                'widget': ComboBox(items = defaults['t2_seq'])},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults['n'], int = True, bounds=(1, None))}}
            case 'RF Coil: T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'rf_pulse_freq': {'display_text': 'RF Frequency: ',
                                'widget': SpinBox(value = defaults['rf_pulse_freq'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'rf_pulse_power': {'display_text': 'RF Power: ',
                                'widget': SpinBox(value = defaults['rf_pulse_power'], suffix = 'V', siPrefix = True)},
                        'rf_pulse_phase': {'display_text': 'RF Phase (deg.): ',
                                'widget': SpinBox(value = defaults['rf_pulse_phase'], int = True, bounds=(0, 360))},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults['n'], int = True, bounds=(1, None))}}
            case 'DQ Relaxation':
                params = {
                        'freq_minus': {'display_text': 'NV Frequency |-1>: ',
                                'widget': SpinBox(value = defaults['freq_minus'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power_minus': {'display_text': 'NV MW Power |-1>: ',
                                'widget': SpinBox(value = defaults['rf_power_minus'], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi_minus': {'display_text': '\u03C0 Pulse |-1>: ',
                                'widget': SpinBox(value = defaults['pi_minus'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq_plus': {'display_text': 'NV Frequency |+1>: ',
                                'widget': SpinBox(value = defaults['freq_plus'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power_plus': {'display_text': 'NV MW Power |+1>: ',
                                'widget': SpinBox(value = defaults['rf_power_plus'], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi_plus': {'display_text': '\u03C0 Pulse |+1>: ',
                                'widget': SpinBox(value = defaults['pi_plus'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])}}
            case 'DEER':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = overrides.get('awg_power', defaults['awg_power']), suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('dark_pi', defaults['dark_pi']), suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'drive_type': {'display_text': 'Dark MW Driving',
                                'widget': ComboBox(items = defaults['drive_type'])}}
            case 'DEER Rabi':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = overrides.get('dark_freq', defaults['dark_freq']), suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = overrides.get('awg_power', defaults['awg_power']), suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'DEER FID':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = overrides.get('dark_freq', defaults['dark_freq']), suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = overrides.get('awg_power', defaults['awg_power']), suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('dark_pi', defaults['dark_pi']), suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults['n'], int = True, bounds=(1, None))}}
            case 'DEER FID Continuous Drive':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = overrides.get('dark_freq', defaults['dark_freq']), suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = overrides.get('awg_power', defaults['awg_power']), suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('dark_pi', defaults['dark_pi']), suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_cd_power': {'display_text': 'Continuous Drive (AWG) Power: ',
                                'widget': SpinBox(value = defaults['awg_cd_power'], suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults['n'], int = True, bounds=(1, None))}}
            case 'DEER Correlation Rabi':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = overrides.get('dark_freq', defaults['dark_freq']), suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('dark_pi', defaults['dark_pi']), suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = overrides.get('awg_power', defaults['awg_power']), suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'DEER T1':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = overrides.get('dark_freq', defaults['dark_freq']), suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('dark_pi', defaults['dark_pi']), suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = overrides.get('awg_power', defaults['awg_power']), suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'DEER T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = overrides.get('dark_freq', defaults['dark_freq']), suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('dark_pi', defaults['dark_pi']), suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = overrides.get('awg_power', defaults['awg_power']), suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'NMR Correlation Spectroscopy':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults['pulse_axis'])},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults['n'], int = True, bounds=(1, None))}}
            case 'NMR CASR':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults['freq']), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults['rf_power']), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults['pi']), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'rf_pulse_freq': {'display_text': 'RF Frequency: ',
                                'widget': SpinBox(value = defaults['rf_pulse_freq'], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_pulse_power': {'display_text': 'RF Power: ',
                                'widget': SpinBox(value = defaults['rf_pulse_power'], suffix = 'V', siPrefix = True, bounds = (0, 2))},
                        'rf_pi_half': {'display_text': 'RF \u03C0/2 Pulse: ',
                                'widget': SpinBox(value = defaults['rf_pi_half'], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'rf_pulse_phase': {'display_text': 'RF Phase (deg.): ',
                                'widget': SpinBox(value = defaults['rf_pulse_phase'], int = True, bounds=(0, 360))},
                        'n': {'display_text': 'n (# XY8-n repetitions): ',
                                'widget': SpinBox(value = defaults['n'], int = True, bounds=(1, None))}}

        return params

    def create_fit_params_widget(self, tag, defaults):
        match tag:
            case 'Fit Neg Lorentz': # -A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults.get('A', 0))},
                        'x0': {'display_text': 'x0: ',
                                'widget': SpinBox(value = defaults.get('x0', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'gamma': {'display_text': '\u03B3: ',
                                'widget': SpinBox(value = defaults.get('gamma', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults.get('c', 0))}}
            case 'Fit Decaying Cos': # A * exp(-x / t_decay) * cos(2 * pi * x / T + phi) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults.get('A', 0))},
                        't_decay': {'display_text': 't<sub>decay</sub>: ',
                                'widget': SpinBox(value = defaults.get('t_decay', 0), suffix = 's', siPrefix = True, dec = True)},
                        'T': {'display_text': 'T: ',
                                'widget': SpinBox(value = defaults.get('T', 0), suffix = 's', siPrefix = True, dec = True)},
                        'phi': {'display_text': '\u03C6: ',
                                'widget': SpinBox(value = defaults.get('phi', 0))},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults.get('c', 0))}}
            case 'Fit Two Neg Lorentz': # -A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults.get('A', 0))},
                        'x0': {'display_text': 'x0: ',
                                'widget': SpinBox(value = defaults.get('x0', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'gamma': {'display_text': '\u03B3: ',
                                'widget': SpinBox(value = defaults.get('gamma', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults.get('c', 0))},
                        'A_rf': {'display_text': 'A_rf: ',
                                'widget': SpinBox(value = defaults.get('A_rf', 0))},
                        'x0_rf': {'display_text': 'x0_rf: ',
                                'widget': SpinBox(value = defaults.get('x0_rf', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'gamma_rf': {'display_text': '\u03B3_rf: ',
                                'widget': SpinBox(value = defaults.get('gamma_rf', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'c_rf': {'display_text': 'c_rf: ',
                                'widget': SpinBox(value = defaults.get('c_rf', 0))}}
            case 'Fit Stretched Exp': # A * exp(-(t / T1)**n) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults.get('A', 0))},
                        'T1': {'display_text': 'T: ',
                                'widget': SpinBox(value = defaults.get('T1', 0), suffix = 's', siPrefix = True, dec = True)},
                        'n': {'display_text': 'n: ',
                                'widget': SpinBox(value = defaults.get('n', 0))},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults.get('c', 0))}}
            case 'Fit Modulated Str Exp':
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults.get('A', 0))},
                        'T2': {'display_text': 'T2: ',
                                'widget': SpinBox(value = defaults.get('T2', 0), suffix = 's', siPrefix = True, dec = True)},
                        'n': {'display_text': 'n: ',
                                'widget': SpinBox(value = defaults.get('n', 0))},
                        'a1': {'display_text': 'a1: ',
                                'widget': SpinBox(value = defaults.get('a1', 0))},
                        'f1': {'display_text': 'f1: ',
                                'widget': SpinBox(value = defaults.get('f1', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'phi1': {'display_text': '\u03C61: ',
                                'widget': SpinBox(value = defaults.get('phi1', 0))},
                        'a2': {'display_text': 'a2: ',
                                'widget': SpinBox(value = defaults.get('a2', 0))},
                        'f2': {'display_text': 'f2: ',
                                'widget': SpinBox(value = defaults.get('f2', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'phi2': {'display_text': '\u03C62: ',
                                'widget': SpinBox(value = defaults.get('phi2', 0))}}
            case 'Fit DEER T1 Str Exp': # A * exp(-(t / T1_nv)**n_nv - (t / T1_e)**n_e) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults.get('A', 0))},
                        'T1_nv': {'display_text': 'T1: ',
                                'widget': SpinBox(value = defaults.get('T1_nv', 0), suffix = 's', siPrefix = True, dec = True)},
                        'n_nv': {'display_text': 'n_nv: ',
                                'widget': SpinBox(value = defaults.get('n_nv', 0))},
                        'T1_e': {'display_text': 'T1_e: ',
                                'widget': SpinBox(value = defaults.get('T1_e', 0), suffix = 's', siPrefix = True, dec = True)},
                        'n_e': {'display_text': 'n_e: ',
                                'widget': SpinBox(value = defaults.get('n_e', 0))},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults.get('c', 0))}}
            case 'Fit Pos Lorentz': # A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults.get('A', 0))},
                        'x0': {'display_text': 'x0: ',
                                'widget': SpinBox(value = defaults.get('x0', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'gamma': {'display_text': '\u03B3: ',
                                'widget': SpinBox(value = defaults.get('gamma', 0), suffix = 'Hz', siPrefix = True, dec = True)},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults.get('c', 0))}}
            case _:
                params = {
                        'A': {'display_text': 'Fit params here',
                                'widget': SpinBox(value = defaults.get('A', 0))}}

        return params

    def _apply_fit_overrides(self, fit_value):
        if not self.to_override_fit or fit_value is None:
            return

        exp_name = self.experiments.currentText()

        if exp_name in ("CW ODMR", "Pulsed ODMR") and len(fit_value) > 1:
            odmr_freq_hz = fit_value[1] * 1e9
            if 100e3 <= odmr_freq_hz <= 6e9:
                self.mw_overrides['center_freq'] = odmr_freq_hz
                self.mw_overrides['freq'] = odmr_freq_hz
            else:
                print(f"Fitted ODMR frequency {odmr_freq_hz} Hz is out of bounds.")

        elif exp_name == "Rabi" and len(fit_value) > 2:
            rabi_pi_pulse_s = fit_value[2] * 1e-9
            if 1e-9 <= rabi_pi_pulse_s <= 10e-6:
                self.mw_overrides['pi'] = rabi_pi_pulse_s
                mw_params = dict(self.mw_params_widget.all_params())
                mw_power = mw_params.get('rf_power', None)
                if mw_power is not None:
                    self.mw_overrides['rf_power'] = mw_power
            else:
                print(f"Fitted Rabi π pulse {rabi_pi_pulse_s} s is out of bounds.")

        elif exp_name == "DEER" and len(fit_value) > 1:
            deer_freq_hz = fit_value[1] * 1e6   # if fit output is MHz
            print(f"Fitted DEER frequency: {deer_freq_hz} Hz")
            if 350e6 <= deer_freq_hz <= 750e6:
                self.mw_overrides['dark_freq'] = deer_freq_hz
                mw_params = dict(self.mw_params_widget.all_params())
                awg_power = mw_params.get('awg_power', None)
                if awg_power is not None:
                    self.mw_overrides['awg_power'] = awg_power
            else:
                print(f"Fitted DEER frequency {deer_freq_hz} Hz is out of bounds.")

        elif exp_name == "DEER Rabi" and len(fit_value) > 2:
            deer_rabi_pi_pulse_s = fit_value[2] * 1e-9
            if 1e-9 <= deer_rabi_pi_pulse_s <= 10e-6:
                self.mw_overrides['dark_pi'] = deer_rabi_pi_pulse_s
                mw_params = dict(self.mw_params_widget.all_params())
                awg_power = mw_params.get('awg_power', None)
                if awg_power is not None:
                    self.mw_overrides['awg_power'] = awg_power
            else:
                print(f"Fitted DEER Rabi π pulse {deer_rabi_pi_pulse_s} s is out of bounds.")

    def _handle_exp_message(self, msg):
        percent = int(msg.get("percent", 0))
        status = msg.get("status", "")
        fit_value = msg.get("fit_value", None)
        fit_error = msg.get("fit_error", None)
        fit_units = msg.get("fit_units", None)
        fit_value2 = msg.get("fit_value2", None)
        fit_error2 = msg.get("fit_error2", None)
        fit_units2 = msg.get("fit_units2", None)
        fit_val_list = []
        fit_err_list = []
        fit_unit_list = []

        if fit_value is not None and fit_error is not None:
            fit_val_list.extend(fit_value)
            fit_err_list.extend(fit_error)
            if fit_units is not None:
                fit_unit_list.extend(fit_units)
        if fit_value2 is not None and fit_error2 is not None:
            fit_val_list.extend(fit_value2)
            fit_err_list.extend(fit_error2)
            if fit_units2 is not None:
                fit_unit_list.extend(fit_units2)

        if fit_value is not None and fit_error is not None:
            self.fit_params_widget.set_fit_labels([fit_val_list, fit_err_list], fit_unit_list if fit_unit_list else None)

        # Optional timers (strings are easiest for labels)
        elapsed_str = msg.get("elapsed_str", None)
        remaining_str = msg.get("remaining_str", None)
        est_total_str = msg.get("est_total_str", None)

        # Optional exception text if you add it for 'failed'
        exc_text = msg.get("exception", None)

        self.progress_bar.setValue(percent)

        if status == 'in progress':
            self.status.setStyleSheet("color: black; background-color: gold; padding: 2px; font-weight: bold;")
            self.status.setText(f"{self.experiments.currentText()} scan in progress...")
            self.experiments.setEnabled(False)
            self.params_widget.setEnabled(False)
            self.save_params.setEnabled(False)
            self.laser_params_widget.setEnabled(False)
            self.dig_params_widget.setEnabled(False)

        elif status == 'complete':
            self.status.setStyleSheet("color: black; background-color: limegreen; padding: 2px; font-weight: bold;")
            self.status.setText(f"{self.experiments.currentText()} scan complete.")
            self.experiments.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)

            if (
                self.to_override_fit
                and self.experiments.currentText() in ("CW ODMR", "Pulsed ODMR", "Rabi", "DEER", "DEER Rabi")
                and fit_value is not None
                and len(fit_value) > 1
            ):
                self._apply_fit_overrides(fit_value)

        elif status == 'failed':
            self.status.setStyleSheet("color: black; background-color: red; padding: 2px; font-weight: bold;")
            if exc_text is not None:
                # Extract just the exception type (concise display in label)
                exc_type = exc_text.split(':')[0] if ':' in exc_text else exc_text
                self.status.setText(f"{self.experiments.currentText()} scan failed. Exception: {exc_type}")
                # Add full exception as tooltip for hover display
                self.status.setToolTip(f"Full error:\n{exc_text}\n\nCheck terminal output for full traceback.")
            else:
                self.status.setText(f"{self.experiments.currentText()} scan failed.")
                self.status.setToolTip("")
            self.experiments.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)

        else:
            self.status.setStyleSheet("color: black; background-color: white; padding: 2px; font-weight: bold;")
            self.status.setText(f"{self.experiments.currentText()} scan stopped.")
            self.experiments.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)

            if (
                self.to_override_fit
                and self.experiments.currentText() in ("CW ODMR", "Pulsed ODMR", "Rabi", "DEER", "DEER Rabi")
                and fit_value is not None
                and len(fit_value) > 1
            ):
                self._apply_fit_overrides(fit_value)

        # ----------------------------
        # Optional: timer labels if you have them
        # (won't error if these widgets don't exist)
        # ----------------------------
        if elapsed_str is not None and hasattr(self, "time_elapsed"):
            self.time_elapsed.setText(f"<i>Experiment Time Elapsed:</i> {elapsed_str}")
        if remaining_str is not None and hasattr(self, "time_remaining"):
            self.time_remaining.setText(f"<i>ETA:</i> {remaining_str}")
        if est_total_str is not None and hasattr(self, "time_estimate"):
            self.time_estimate.setText(f"<i>Est. Total Time:</i> {est_total_str}")

    def check_queue_from_exp(self):
        # queue checker to control progress bar display
        try:
            q = self.queue_from_exp  # Local reference for thread safety
            if q is None:
                print("Queue is None. Stopping queue checks.")
                self.updateTimer.stop()
                return

            while True:  # If there is something in the queue
                try:
                    msg = q.get_nowait()  # Get it
                    if msg is None:
                        continue
                except Empty:
                    break  # Queue is empty, exit function safely

                except (OSError, EOFError) as e:
                    # WinError 6 = invalid handle (queue pipe closed)
                    # WinError 109/232 also indicate broken pipe issues on Windows
                    winerr = getattr(e, "winerror", None)
                    if winerr in (6, 109, 232) or isinstance(e, EOFError):
                        print(f"Queue connection error: {e}. Stopping queue checks.")
                        self.queue_from_exp = None  # Mark queue as invalid
                        self.updateTimer.stop()  # Stop timer to prevent future checks
                        return

                    # Any other unexpected OSError: also stop
                    print(f"Queue OSError: {e}. Stopping queue checks.")
                    self.queue_from_exp = None
                    self.updateTimer.stop()
                    return

                 # ----------------------------
                # Support BOTH formats:
                #   New: dict
                #   Old: [percent, status, [fit_value, fit_error], ...]
                # ----------------------------
                self._handle_exp_message(msg)

        except Exception as e:
            print(f"Unexpected error in check_queue_from_exp: {e}")
            self.queue_from_exp = None  # If any unexpected error occurs, stop checking queue
            self.updateTimer.stop()
            return

    def _new_run_queues(self):
        # If you want, you can try to close old queues to release handles earlier
        for q in (
            getattr(self, "queue_from_exp", None),
            getattr(self, "queue_to_exp", None),
        ):
            try:
                if q is not None:
                    q.close()
                    q.join_thread()
            except Exception:
                pass

        self.queue_to_exp = Queue()
        self.queue_from_exp = Queue()

    def toggle_daq(self, b):
        match b.text():
            case 'Digitizer':
                if b.isChecked() == True:
                    self.dig_params_widget.setEnabled(True)
                    for i in [7,8]:
                        self.opacity_effects[i].setEnabled(False)
                else:
                    self.dig_params_widget.setEnabled(False)
                    for i in [7,8]:
                        self.opacity_effects[i].setEnabled(True)
            case 'NI DAQ':
                if b.isChecked() == True:
                    self.dig_params_widget.setEnabled(False)
                    for i in [7,8]:
                        self.opacity_effects[i].setEnabled(True)
                else:
                    self.dig_params_widget.setEnabled(True)
                    for i in [7,8]:
                        self.opacity_effects[i].setEnabled(False)

    def get_selected_detector(self) -> str:
        btn = self.detector_group.checkedButton()
        return btn.text() if btn is not None else "APD"  # safe fallback

    def auto_save_changed(self):
        if self.auto_save_checkbox.isChecked() == True:
            self.to_save = True
            self.select_dir_button.setEnabled(True)
            self.select_file_format_combobox.setEnabled(True)
            self.filename_lineedit.setEnabled(True)
            # self.opacity_effects[6].setEnabled(False) # change to 6? laser params widget
            self.opacity_effects[12].setEnabled(False)
            self.opacity_effects[13].setEnabled(False)
            self.opacity_effects[14].setEnabled(False)
            self.opacity_effects[15].setEnabled(False)
            self.opacity_effects[23].setEnabled(False)

        else:
            self.to_save = False
            self.select_dir_button.setEnabled(False)
            self.select_file_format_combobox.setEnabled(False)
            self.filename_lineedit.setEnabled(False)
            # self.opacity_effects[6].setEnabled(True) # change to 6?
            self.opacity_effects[12].setEnabled(True)
            self.opacity_effects[13].setEnabled(True)
            self.opacity_effects[14].setEnabled(True)
            self.opacity_effects[15].setEnabled(True)
            self.opacity_effects[23].setEnabled(True)

    def file_format_selector(self):
        match self.select_file_format_combobox.currentText():
            case 'Form: JSON':
                self.file_format = "json"
            case 'Form: Pickle':
                self.file_format = "pickle"

    def save_params_clicked(self):
        params = dict(**self.params_widget.all_params())
        mw_params = dict(**self.mw_params_widget.all_params())
        laser_params = dict(**self.laser_params_widget.all_params())
        dig_params = dict(**self.dig_params_widget.all_params())

        saved_params = dict(params) # set saved params for next time the experiment is selected
        saved_mw_params = dict(mw_params) # set saved params for next time the experiment is selected
        saved_laser_params = dict(laser_params)
        saved_dig_params = dict(dig_params)

        if self.experiments.currentText() != 'Signal vs Time':
            # update laser param comboboxes
            self.all_defaults['sideband_opts'].insert(
                0,
                self.all_defaults['sideband_opts'].pop(self.all_defaults['sideband_opts'].index(saved_laser_params['sideband']))
            )
            saved_laser_params['sideband'] = self.all_defaults['sideband_opts']

            # update digitizer param comboboxes
            self.all_defaults['dig_ro_chan_opts'].insert(
                0,
                self.all_defaults['dig_ro_chan_opts'].pop(self.all_defaults['dig_ro_chan_opts'].index(saved_dig_params['read_channel']))
            )
            saved_dig_params['read_channel'] = self.all_defaults['dig_ro_chan_opts']

            self.all_defaults['dig_coupling_opts'].insert(
                0,
                self.all_defaults['dig_coupling_opts'].pop(self.all_defaults['dig_coupling_opts'].index(saved_dig_params['dig_coupling']))
            )
            saved_dig_params['dig_coupling'] = self.all_defaults['dig_coupling_opts']

            self.all_defaults['dig_termination_opts'].insert(
                0,
                self.all_defaults['dig_termination_opts'].pop(self.all_defaults['dig_termination_opts'].index(saved_dig_params['dig_termination']))
            )
            saved_dig_params['dig_termination'] = self.all_defaults['dig_termination_opts']

        # TODO: update which params widget each condition goes under
        # take chosen combobox parameter and place it first in the updated combobox item list
        match self.experiments.currentText():
            case 'Rabi':
                self.all_defaults['rabi_axis_opts'].insert(
                    0,
                    self.all_defaults['rabi_axis_opts'].pop(self.all_defaults['rabi_axis_opts'].index(saved_mw_params['pulse_axis']))
                )
                saved_mw_params['pulse_axis'] = self.all_defaults['rabi_axis_opts']

            case 'Optical T1':
                self.all_defaults['opt_t1_array_opts'].insert(
                    0,
                    self.all_defaults['opt_t1_array_opts'].pop(self.all_defaults['opt_t1_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['opt_t1_array_opts']

            case 'MW T1':
                self.all_defaults['mw_t1_array_opts'].insert(
                    0,
                    self.all_defaults['mw_t1_array_opts'].pop(self.all_defaults['mw_t1_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['mw_t1_array_opts']

            case 'DQ Relaxation':
                self.all_defaults['dq_array_opts'].insert(
                    0,
                    self.all_defaults['dq_array_opts'].pop(self.all_defaults['dq_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['dq_array_opts']

            case 'RF Coil: T2':
                self.all_defaults['t2_rf_array_opts'].insert(
                    0,
                    self.all_defaults['t2_rf_array_opts'].pop(self.all_defaults['t2_rf_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['t2_rf_array_opts']

            case 'T2':
                self.all_defaults['t2_array_opts'].insert(
                    0,
                    self.all_defaults['t2_array_opts'].pop(self.all_defaults['t2_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['t2_array_opts']

                self.all_defaults['t2_seq_opts'].insert(
                    0,
                    self.all_defaults['t2_seq_opts'].pop(self.all_defaults['t2_seq_opts'].index(saved_mw_params['t2_seq']))
                )
                saved_mw_params['t2_seq'] = self.all_defaults['t2_seq_opts']

            case 'DEER':
                self.all_defaults['deer_drive_opts'].insert(
                    0,
                    self.all_defaults['deer_drive_opts'].pop(self.all_defaults['deer_drive_opts'].index(saved_mw_params['drive_type']))
                )
                saved_mw_params['drive_type'] = self.all_defaults['deer_drive_opts']

            case 'DEER FID':
                self.all_defaults['fid_array_opts'].insert(
                    0,
                    self.all_defaults['fid_array_opts'].pop(self.all_defaults['fid_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['fid_array_opts']

            case 'DEER FID Continuous Drive':
                self.all_defaults['fid_cd_array_opts'].insert(
                    0,
                    self.all_defaults['fid_cd_array_opts'].pop(self.all_defaults['fid_cd_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['fid_cd_array_opts']

            case 'DEER T1':
                self.all_defaults['corr_t1_array_opts'].insert(
                    0,
                    self.all_defaults['corr_t1_array_opts'].pop(self.all_defaults['corr_t1_array_opts'].index(saved_params['array_type']))
                )
                saved_params['array_type'] = self.all_defaults['corr_t1_array_opts']

            case 'NMR Correlation Spectroscopy':
                self.all_defaults['corr_spec_sig_opts'].insert(
                    0,
                    self.all_defaults['corr_spec_sig_opts'].pop(self.all_defaults['corr_spec_sig_opts'].index(saved_params['sig_opt']))
                )
                saved_params['sig_opt'] = self.all_defaults['corr_spec_sig_opts']

            case 'NMR CASR':
                self.all_defaults['casr_sig_opts'].insert(
                    0,
                    self.all_defaults['casr_sig_opts'].pop(self.all_defaults['casr_sig_opts'].index(saved_params['sig_opt']))
                )
                saved_params['sig_opt'] = self.all_defaults['casr_sig_opts']

                self.all_defaults['dnp_opts'].insert(
                    0,
                    self.all_defaults['dnp_opts'].pop(self.all_defaults['dnp_opts'].index(saved_params['dnp']))
                )
                saved_params['dnp'] = self.all_defaults['dnp_opts']

        self.exp_dict[self.experiments.currentText()][1] = saved_params
        if self.experiments.currentText() != 'Signal vs Time':
            self.exp_dict[self.experiments.currentText()][2] = saved_mw_params
        self.exp_dict[self.experiments.currentText()][4] = saved_laser_params
        self.exp_dict[self.experiments.currentText()][5] = saved_dig_params

    def auto_fit_changed(self):
        if self.auto_fit_checkbox.isChecked() == True:
            self.to_fit = True
            self.fit_select.setEnabled(True)
            self.fit_params_widget.setEnabled(True)
            self.live_fit_checkbox.setEnabled(True)
            self.override_fit_checkbox.setEnabled(True)
            self.opacity_effects[17].setEnabled(False)
            self.opacity_effects[18].setEnabled(False)
            self.opacity_effects[19].setEnabled(False)
            self.opacity_effects[20].setEnabled(False)
        else:
            self.to_fit = False
            self.fit_select.setEnabled(False)
            self.fit_params_widget.setEnabled(False)
            self.live_fit_checkbox.setEnabled(False)
            self.live_fit_checkbox.setChecked(False)
            self.override_fit_checkbox.setChecked(False)
            self.to_fit_live = False
            self.opacity_effects[17].setEnabled(True)
            self.opacity_effects[18].setEnabled(True)
            self.opacity_effects[19].setEnabled(True)
            self.opacity_effects[20].setEnabled(True)

    def live_fit_changed(self):
        if self.live_fit_checkbox.isChecked() == True:
            self.to_fit_live = True
        else:
            self.to_fit_live = False

    def override_fit_changed(self):
        if self.override_fit_checkbox.isChecked() == True:
            self.to_override_fit = True
        else:
            self.to_override_fit = False

    def select_directory(self):
        response = QFileDialog.getExistingDirectory(self, caption = "Select a folder")
        if not response:
            return

        full_path = str(response)

        # store the real path in the widget
        self.chosen_dir.setProperty("full_path", full_path)

        # display ellipsized path
        self.chosen_dir.setText(ellipsize(full_path, max_chars=50))

    def get_lineedit_val(self, lineedit):
        return lineedit.text()

    def get_combobox_val(self, combobox):
        return str(combobox.value())

    def exp_selector(self):
        # reset params widgets each time new experiment selected
        self.status.setStyleSheet("color: black; background-color: #00b8ff; padding: 2px; font-weight: bold;")
        self.status.setText("Set parameters and press 'Run' to begin experiment.")
        self.progress_bar.setValue(0)

        # self.daq_b1.hide()
        # self.daq_b2.hide()
        self.exp_label.hide()
        self.params_widget.hide()
        self.mw_label.hide()
        self.mw_params_widget.hide()
        self.save_params.hide()
        self.laser_label.hide()
        self.laser_params_widget.hide()
        self.dig_label.hide()
        self.dig_params_widget.hide()
        self.fit_params_widget.hide()

        # get up to date mw param defaults with any saved overrides for this experiment (except for signal vs time, which has its own unique mw params that don't follow the same pattern as the other experiments)
        overrides = self.mw_overrides

        try:
            self.params_widget = ParamsWidget(self.create_params_widget(self.experiments.currentText(), self.exp_dict[self.experiments.currentText()][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.mw_params_widget = ParamsWidget(self.create_mw_params_widget(self.experiments.currentText(), self.exp_dict[self.experiments.currentText()][2], overrides=overrides), get_param_value_funs = {ComboBox: self.get_combobox_val})

        except KeyError:
            # if "Select dropdown option" is selected, populate GUI with disabled ODMR widgets as filler
            self.params_widget = ParamsWidget(self.create_params_widget('CW ODMR', self.exp_dict['CW ODMR'][1]))
            self.params_widget.setGraphicsEffect(self.opacity_effects[1]) # reset opacity effects
            self.mw_params_widget = ParamsWidget(self.create_mw_params_widget('CW ODMR', self.exp_dict['CW ODMR'][2]))
            self.mw_params_widget.setGraphicsEffect(self.opacity_effects[4]) # reset opacity effects

            self.laser_params_widget = ParamsWidget(self.create_params_widget('Laser', self.exp_dict['Laser'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.laser_params_widget.setGraphicsEffect(self.opacity_effects[6]) # reset opacity effects
            self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.dig_params_widget.setGraphicsEffect(self.opacity_effects[8]) # reset opacity effects
            self.fit_select.blockSignals(True)
            self.fit_select.setCurrentText("Choose Fit Type")
            self.fit_select.blockSignals(False)
            self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit None', self.fit_none_default), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.fit_params_widget.setGraphicsEffect(self.opacity_effects[18]) # reset opacity effects
            self.fit_params_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred
            )
            for i in range(24):
                self.opacity_effects[i].setEnabled(True)

            self.params_widget.setEnabled(False)
            self.dataset_label.setText("Data Set Name: N/A")
            self.save_params.setText("Save Experiment Parameters")
            self.save_params.setEnabled(False)
            self.mw_params_widget.setEnabled(False)

            self.laser_params_widget.setEnabled(False)
            self.daq_b1.setEnabled(False)
            self.daq_b2.setEnabled(False)
            self.detector_b1.setEnabled(False)
            self.detector_b2.setEnabled(False)
            self.dig_params_widget.setEnabled(False)
            self.fit_params_widget.setEnabled(False)
            self.auto_save_checkbox.setEnabled(False)
            self.auto_fit_checkbox.setEnabled(False)
            self.auto_fit_checkbox.setChecked(False)

        else:
            self.dataset_label.setText(f"Data Set Name: '{self.exp_dict[self.experiments.currentText()][3]}'")
            self.daq_b1.setEnabled(True)
            self.daq_b2.setEnabled(True)
            self.detector_b1.setEnabled(True)
            self.detector_b2.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.save_params.setText("Save Settings for " + ellipsize(self.experiments.currentText(), max_chars=20) + "")
            self.mw_params_widget.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.auto_save_checkbox.setEnabled(True)
            self.auto_fit_checkbox.setEnabled(True)

            for i in range(12):
                if i == 7 or i == 8:
                    if self.daq_b1.isChecked(): # digitizer settings
                        self.opacity_effects[i].setEnabled(False)
                        self.dig_params_widget.setEnabled(True)
                    elif self.daq_b2.isChecked(): # NI DAQ settings
                        self.opacity_effects[i].setEnabled(True)
                        self.dig_params_widget.setEnabled(False)
                else:
                    self.opacity_effects[i].setEnabled(False)
            self.opacity_effects[16].setEnabled(False) # auto fit checkbox
            self.opacity_effects[21].setEnabled(False) # APD radio button
            self.opacity_effects[22].setEnabled(False) # BPD radio button

            if self.experiments.currentText() == 'Signal vs Time': # use specific signal vs time laser parameters (just power)
                self.laser_params_widget = ParamsWidget(self.create_params_widget('Sig Laser', self.exp_dict['Sig Laser'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            else:
                self.laser_params_widget = ParamsWidget(self.create_params_widget('Laser', self.exp_dict['Laser'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.laser_params_widget.setGraphicsEffect(self.opacity_effects[6]) # reset opacity effects

            self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.dig_params_widget.setGraphicsEffect(self.opacity_effects[8]) # reset opacity effects
            self.dig_params_widget.setEnabled(False) # TODO: check if this is how i want dig params widget to actually behave when new experiment selected

            self.fit_select.blockSignals(True)
            if self.experiments.currentText() in ("CW ODMR", "Pulsed ODMR", "DEER"):
                self.fit_select.setCurrentText("Neg. Lorentz.")
                self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Neg Lorentz', self.all_defaults['fit_neg_lorentz_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
            elif self.experiments.currentText() in ("Rabi", "DEER Rabi"): # TODO: add correlation Rabi
                self.fit_select.setCurrentText("Decaying Cos.")
                self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Decaying Cos', self.all_defaults['fit_decaying_cosine_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
            elif self.experiments.currentText() == "RF Coil: Pulsed ODMR":
                self.fit_select.setCurrentText("Two Neg. Lorentz.")
                self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Two Neg Lorentz', self.all_defaults['fit_two_neg_lorentz_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
            elif self.experiments.currentText() in ("T2", "Optical T1", "MW T1"):
                self.fit_select.setCurrentText("Stretched Exp.")
                self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Stretched Exp', self.all_defaults['fit_str_exp_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
            elif self.experiments.currentText() == "DEER T1":
                self.fit_select.setCurrentText("DEER T1 Str. Exp.")
                self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit DEER T1 Str Exp', self.all_defaults['fit_deer_t1_str_exp_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
            elif self.experiments.currentText() in ("NMR Correlation Spectroscopy", "NMR CASR"):
                self.fit_select.setCurrentText("Pos. Lorentz.")
                self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Pos Lorentz', self.all_defaults['fit_pos_lorentz_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
            else:
                self.fit_select.setCurrentText("Choose Fit Type")
                self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit None', self.all_defaults['fit_none_default']), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.fit_select.blockSignals(False)
            self.fit_params_widget.setGraphicsEffect(self.opacity_effects[18]) # reset opacity effects
            self.fit_params_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred
            )
            self.fit_params_widget.setEnabled(False)

        finally:
            # reinstate GUI widgets
            self.exp_params_layout.insertWidget(0, self.exp_label)
            self.exp_params_layout.insertWidget(1, self.params_widget)
            self.mw_params_layout.insertWidget(0, self.mw_label)
            self.mw_params_layout.insertWidget(1, self.mw_params_widget)
            self.laser_params_layout.addWidget(self.laser_label,1,1,1,2)
            self.laser_params_layout.addWidget(self.laser_params_widget,2,1,1,2)
        #     self.laser_params_layout.addWidget(self.daq_b1,3,1,1,1)
        #     self.laser_params_layout.addWidget(self.daq_b2,3,2,1,1)
            self.dig_params_layout.insertWidget(0, self.dig_label)
            self.dig_params_layout.insertWidget(1, self.dig_params_widget)

            save_row = QHBoxLayout()
            save_row.addWidget(self.save_params)
            self.detector_layout.addLayout(save_row,1,0)
            self.fit_layout.addWidget(self.auto_fit_checkbox,1,1,1,1)
            self.fit_layout.addWidget(self.fit_select,1,2,1,2)
            self.fit_layout.addWidget(self.fit_params_widget,2,1,1,2)
            self.fit_layout.addWidget(self.live_fit_checkbox,3,1,1,1)
            self.fit_layout.addWidget(self.override_fit_checkbox,3,2,1,2)

            self.exp_label.show()
            self.save_params.show()
            self.params_widget.show()

            if self.experiments.currentText() == 'Signal vs Time': # hide MW, laser and digitizer params widgets
                self.mw_label.hide()
                self.mw_params_widget.hide()
                self.laser_label.show()
                self.laser_params_widget.show()
                # self.daq_b1.show()
                # self.daq_b2.show()
                self.dig_label.show()
                self.dig_params_widget.show()
                self.auto_fit_checkbox.hide()
                self.fit_select.hide()
                self.fit_params_widget.hide()
                self.live_fit_checkbox.hide()
                self.override_fit_checkbox.hide()

            elif self.experiments.currentText() == 'Optical T1': # only hide MW params widget
                self.mw_label.hide()
                self.mw_params_widget.hide()
                self.laser_label.show()
                self.laser_params_widget.show()
                # self.daq_b1.show()
                # self.daq_b2.show()
                self.dig_label.show()
                self.dig_params_widget.show()
                self.auto_fit_checkbox.show()
                self.fit_select.show()
                self.fit_params_widget.show()
                self.live_fit_checkbox.show()
                self.override_fit_checkbox.show()

            else: # rest of experiments don't hide any widgets
                self.mw_label.show()
                self.mw_params_widget.show()
                self.laser_label.show()
                self.laser_params_widget.show()
                # self.daq_b1.show()
                # self.daq_b2.show()
                self.dig_label.show()
                self.dig_params_widget.show()
                self.auto_fit_checkbox.show()
                self.fit_select.show()
                self.fit_params_widget.show()
                self.live_fit_checkbox.show()
                self.override_fit_checkbox.show()

            if self.daq_b1.isChecked() == True:
                self.dig_params_widget.setEnabled(True)
            else:
                self.dig_params_widget.setEnabled(False)

            if self.auto_fit_checkbox.isChecked() == True:
                self.fit_select.setEnabled(True)
                self.fit_params_widget.setEnabled(True)
                self.live_fit_checkbox.setEnabled(True)
                self.override_fit_checkbox.setEnabled(True)
            else:
                self.fit_select.setEnabled(False)
                self.fit_params_widget.setEnabled(False)
                self.live_fit_checkbox.setEnabled(False)
                self.live_fit_checkbox.setChecked(False)
                self.override_fit_checkbox.setEnabled(False)
                self.override_fit_checkbox.setChecked(False)

            self.live_fit_changed() # to set live fit value based on current state of live fit checkbox
            self.override_fit_changed() # to set override fit value based on current state of override fit checkbox

    def fit_selector(self, index=None):
        # Choose Fit Type",
        #  "Neg. Lorentz.",
        #  "Pos. Lorentz.",
        #  "Two Neg. Lorentz."
        #  "Decaying Cos.",
        #  "Stretched Exp.",
        #  "Modulated Str. Exp.",
        #  "DEER T1 Str. Exp."
        self.fit_params_widget.hide()

        match self.fit_select.currentText():
                case 'Neg. Lorentz.':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Neg Lorentz', self.all_defaults['fit_neg_lorentz_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Pos. Lorentz.':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Pos Lorentz', self.all_defaults['fit_pos_lorentz_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Two Neg. Lorentz.':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Two Neg Lorentz', self.all_defaults['fit_two_neg_lorentz_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Decaying Cos.':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Decaying Cos', self.all_defaults['fit_decaying_cosine_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Stretched Exp.':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Stretched Exp', self.all_defaults['fit_str_exp_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Modulated Str. Exp.':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Modulated Str Exp', self.all_defaults['fit_mod_str_exp_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'DEER T1 Str. Exp.':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit DEER T1 Str Exp', self.all_defaults['fit_deer_t1_str_exp_defaults']), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case _:
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit None', self.all_defaults['fit_none_default']), get_param_value_funs = {ComboBox: self.get_combobox_val})

        self.fit_params_widget.setGraphicsEffect(self.opacity_effects[18]) # reset opacity effects
        self.fit_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self.fit_layout.addWidget(self.fit_params_widget,2,1,1,2)
        self.fit_params_widget.show()

        if self.auto_fit_checkbox.isChecked() == True:
            self.fit_params_widget.setEnabled(True)
        else:
            self.fit_params_widget.setEnabled(False)

    def run(self):
        """Run the experiment function in a subprocess."""

        if self.run_proc.running():
            logging.info(
                'Not starting the experiment process because it is still running.'
            )

            return

        self.status.setStyleSheet("color: black; background-color: gold; padding: 2px; font-weight: bold;")
        self.status.setText(f"{self.experiments.currentText()} scan in progress...")

        self.extra_kwarg_params['save'] = self.to_save
        self.extra_kwarg_params['file_format'] = self.file_format

        try:
            self.extra_kwarg_params['dataset'] = self.exp_dict[self.experiments.currentText()][3]
        except KeyError as e:
            self.status.setStyleSheet("color: black; background-color: red; padding: 2px; font-weight: bold;")
            self.status.setText(f"No experiment selected: {e}")
            return
        else:
            self.extra_kwarg_params['filename'] = self.filename_lineedit.text()
            full_dir = self.chosen_dir.property("full_path") or ""
        #     if not full_dir:
        #         # fallback: user didn't choose a directory yet
        #         full_dir = r"E:\Data"
            self.extra_kwarg_params["directory"] = full_dir
        #     self.extra_kwarg_params['seq'] = self.experiments.currentText()
            self.extra_kwarg_params['fit'] = self.to_fit
            self.extra_kwarg_params['fit_live'] = self.to_fit_live
            if self.fit_select.currentIndex() == 0:
                self.extra_kwarg_params['fit_type'] = "None"
            else:
                self.extra_kwarg_params['fit_type'] = self.fit_select.currentText()
            self.extra_kwarg_params['fit_params'] = list(self.fit_params_widget.all_params().values()) # send a list of fit parameters to experiment process

            detector = self.get_selected_detector()
            self.extra_kwarg_params['detector'] = detector

            # unpack all keyword arg parameters to send to experiment process
            fun_kwargs = dict(
                **self.params_widget.all_params(),
                **self.mw_params_widget.all_params(),
                **self.laser_params_widget.all_params(),
                **self.dig_params_widget.all_params(),
                **self.extra_kwarg_params
            )
            fun_kwargs.setdefault('enable_pl_trace', False) # if enable_pl_trace not an experiment parameter, set to False

            self._new_run_queues() # create new queues for new experiment run
            self.queue_to_exp.put('start') # start queue after creation of new queues to avoid any potential race conditions with old queues from previous runs

            # reload the module at runtime in case any changes were made to the code
            if self.daq_b1.isChecked(): # digitizer settings
                reload(nv_experiments_2026_04_16)
                # call the function in a new process
                self.run_proc.run(
                    run_experiment,
                    exp_cls = nv_experiments_2026_04_16.SpinMeasurements,
                    fun_name = self.exp_dict[self.experiments.currentText()][0],
                    constructor_args = list(),
                    constructor_kwargs=dict(queue_to_inst=self.exp_inst_queue),
                    queue_to_exp = self.queue_to_exp,
                    queue_from_exp = self.queue_from_exp,
                    fun_args = list(),
                    fun_kwargs = fun_kwargs
                )

            elif self.daq_b2.isChecked(): # NI DAQ settings
                reload(nv_experiments_daq)
                # call the function in a new process
                self.run_proc.run(
                    run_experiment,
                    exp_cls = nv_experiments_daq.SpinMeasurements,
                    fun_name = self.exp_dict[self.experiments.currentText()][0],
                    constructor_args = list(),
                    constructor_kwargs = dict(),
                    queue_to_exp = self.queue_to_exp,
                    queue_from_exp = self.queue_from_exp,
                    fun_args = list(),
                    fun_kwargs = fun_kwargs
                )

            else:
                self.status.setStyleSheet("color: black; background-color: red; padding: 2px; font-weight: bold;")
                self.status.setText(f"No acquisition mode selected. Choose 'Digitizer' or 'NI DAQ'.")
                raise ValueError(f"{self.experiments.currentText()} scan couldn't start because no data acquisition mode selected. Choose either 'Digitizer' or 'NI DAQ'.")

    def stop(self, log: bool = True):
        """Request the experiment subprocess to stop by sending the string :code:`stop`
        to :code:`queue_to_exp`.

        Args:
            log: if True, log when stop is called but the process isn't running.
        """

        if self.run_proc.running():
            self.queue_to_exp.put('stop')
        else:
            if log:
                logging.info(
                    'Not stopping the experiment process because it is not running.'
                )

        self.experiments.setEnabled(True)
        self.params_widget.setEnabled(True)
        self.mw_params_widget.setEnabled(True)
        self.laser_params_widget.setEnabled(True)
        self.dig_params_widget.setEnabled(True)
        self.save_params.setEnabled(True)
        self.daq_b1.setEnabled(True)
        self.daq_b2.setEnabled(True)
        self.detector_b1.setEnabled(True)
        self.detector_b2.setEnabled(True)

    def kill(self, log: bool = True):
        """Request the experiment subprocess to stop by sending the string :code:`stop`
        to :code:`queue_to_exp`.

        Args:
            log: if True, log when stop is called but the process isn't running.
        """

        if self.run_proc.running():
            self.run_proc.kill()
            logging.info('Processed killed.')
            self.status.setStyleSheet("color: black; background-color: red; padding: 2px; font-weight: bold;")
            self.status.setText(f"{self.experiments.currentText()} scan killed.")
            self.experiments.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.mw_params_widget.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.daq_b1.setEnabled(True)
            self.daq_b2.setEnabled(True)
            self.detector_b1.setEnabled(True)
            self.detector_b2.setEnabled(True)
        else:
            if log:
                logging.info(
                    'Not killing the experiment process because it is not running.'
                )

class FlexLinePlotWidgetAllDefaults(FlexLinePlotWidget):
    """Add some default settings to the FlexSinkLinePlotWidget."""
    def __init__(self):
        super().__init__()

        # manually set the XY range
        self.line_plot.plot_item().setXRange(3.0, 4.0)
        self.line_plot.plot_item().setYRange(-3000, 4500)

        # retrieve legend object
        legend = self.line_plot.plot_widget.addLegend()
        # set the legend location
        legend.setOffset((-10, -50))

        self.datasource_lineedit.setText('odmr')

#Val accessor fxns
def getQCheckBoxVal(QCheckBoxObj):
    return(QCheckBoxObj.isChecked())

def getComboBoxVal(ComboBoxObj):
    return(ComboBoxObj.currentText())

def getLineEditVal(lineEditObj):
    return(lineEditObj.text())



