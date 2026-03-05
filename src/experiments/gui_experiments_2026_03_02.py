"""
Example GUI elements.
"""
from tracemalloc import stop
import numpy as np
from numpy.fft import fft, ifft
import logging
import errno
from functools import partial
from importlib import reload
from multiprocessing import Queue
from queue import Empty  # Import Empty exception from queue module
from typing import Optional
from pathlib import Path

from inspect import signature
from scipy.optimize import curve_fit
from rpyc.utils.classic import obtain

# from nspyre import FlexLinePlotWidget, LinePlotWidget
from nspyre.gui.widgets.flex_line_plot_4 import FlexLinePlotWidget
from nspyre import DataSink
from pyqtgraph import SpinBox, ComboBox
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtWidgets

from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QRadioButton, QFileDialog, QProgressBar, QButtonGroup
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QSizePolicy, QScrollArea
from PyQt6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

from nspyre.misc.misc import ProcessRunner
from nspyre.misc.misc import run_experiment
from nspyre import ParamsWidget
from nspyre import FitParamsWidget
from nspyre import experiment_widget_process_queue
from nspyre import InstrumentManager

from gui_test import Communicate 

import nv_experiments_organized_2026_03_03
import nv_experiments_daq

def ellipsize(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."

class ExpWidget(QWidget):
    
    QUEUE_CHECK_TIME = 50 # ms

    def __init__(self):
        super().__init__()

        self.setWindowTitle('NV Experiments')

        self.updateTimer = QTimer() #create a timer that will try to update that widget with messages from the from_exp_queue
        self.updateTimer.timeout.connect(lambda: self.check_queue_from_exp())
        self.updateTimer.start(self.QUEUE_CHECK_TIME)
            
        # parameter defaults for different experiments
        self.sideband_opts = ["Lower", "Upper"]
        self.sideband_cw_opts = ["Lower", "Upper"]
        self.awg_sampling_group1_opts = ["2.4 GHz", "1.2 GHz", "600 MHz", "300 MHz", "150 MHz", "75 MHz", "37.5 MHz", "18.75 MHz", "9.37 MHz", "4.68 MHz",
                                  "2.34 MHz", "1.17 MHz", "585.93 kHz", "292.96 kHz"]
        self.awg_sampling_group2_opts = ["2.4 GHz", "1.2 GHz", "600 MHz", "300 MHz", "150 MHz", "75 MHz", "37.5 MHz", "18.75 MHz", "9.37 MHz", "4.68 MHz",
                                  "2.34 MHz", "1.17 MHz", "585.93 kHz", "292.96 kHz"]
        self.detector_opts = ["APD", "BPD"]

        self.dig_ro_chan_opts = ["0", "1"]
        self.dig_coupling_opts = ["DC", "AC"]
        self.dig_termination_opts = ["1M", "50"]
        
        self.sigvstime_detector_opts = ["APD", "BPD"]
        self.sigvstime_mw_detector_opts = ["APD", "BPD"]

        self.rabi_axis_opts = ["y", "x"]
        self.opt_t1_array_opts = ["geomspace", "linspace"]
        self.mw_t1_array_opts = ["geomspace", "linspace"]
        self.t2_array_opts = ["geomspace", "linspace"]
        self.t2_rf_array_opts = ["geomspace", "linspace"]
        self.dq_array_opts = ["geomspace", "linspace"]
        self.fid_array_opts = ["geomspace", "linspace"]
        self.fid_cd_array_opts = ["geomspace", "linspace"]
        self.t2_seq_opts = ["Ramsey", "Echo", "XY4", "YY4", "XY8", "YY8", "CPMG", "PulsePol"]
        self.deer_drive_opts = ["Pulsed", "Continuous"]
        self.fid_drive_opts = ["Pulsed", "Continuous"]
        self.corr_t1_array_opts = ["geomspace", "linspace"]
        
        self.casr_sig_opts = ["Sample", "Coil"]
        self.dnp_opts = ["Off", "Overhauser"]

        self.sigvstime_params_defaults = [1e3, self.sigvstime_detector_opts]
        self.sigvstime_mw_params_defaults = [1e3, self.sigvstime_mw_detector_opts] # not needed - hidden in GUI

        # self.laser_cw_params_defaults = [0, 15e-6, 2.5e-6, 30e6, 0.15, self.sideband_opts, -0.002, -0.004, self.awg_sampling_group1_opts, self.awg_sampling_group2_opts, self.detector_opts]
        self.laser_params_defaults = [0, 15e-6, 2.5e-6, 30e6, 0.45, self.sideband_opts, -0.002, -0.004, self.awg_sampling_group1_opts, self.awg_sampling_group2_opts, self.detector_opts]
        self.laser_params_sigvstime_defaults = [0]
        self.digitizer_defaults = [1024, 500e6, 1, self.dig_ro_chan_opts, self.dig_coupling_opts, self.dig_termination_opts, 32, 5]

        # split up defaults into the non-MW and MW settings to save vertical space in GUI
        self.odmr_params_defaults = [120, 10, 50]
        self.odmr_mw_params_defaults = [2.87e9, 100e6, 1e-9, 25e-6]

        self.odmr_smart_params_defaults = [120, 50, 45, 60, 20]
        self.odmr_smart_mw_params_defaults = [2.87e9, 100e6, 1e-9, 25e-6]

        self.rabi_params_defaults = [120, 10, 0, 500e-9, 50]   
        self.rabi_mw_params_defaults = [2.87e9, 1e-9, self.rabi_axis_opts]

        self.pulsed_odmr_params_defaults = [120, 10, 50]
        self.pulsed_odmr_mw_params_defaults = [2.87e9, 100e6, 1e-9, 100e-9]
        
        self.pulsed_odmr_rf_params_defaults = [120, 10, 50]
        self.pulsed_odmr_rf_mw_params_defaults = [2.87e9, 100e6, 1e-9, 100e-9, 500e3, 0.3, 0]

        self.opt_t1_params_defaults = [120, 10, 50e-9, 100e-6, 50, self.opt_t1_array_opts]
        self.opt_t1_mw_params_defaults = [12, 10, 50e-9, 100e-6, 50, self.opt_t1_array_opts] # not needed - hidden in GUI

        self.mw_t1_params_defaults = [120, 10, 50e-9, 100e-6, 50, self.mw_t1_array_opts]
        self.mw_t1_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y']

        self.t2_params_defaults = [120, 10, 50e-9, 20e-6, 50, self.t2_array_opts]
        self.t2_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', self.t2_seq_opts, 1]

        self.t2_rf_params_defaults = [120, 10, 50e-9, 20e-6, 50, self.t2_rf_array_opts]
        self.t2_rf_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 1e6, 0.1, 0, 1]

        self.dq_params_defaults = [120, 10, 50e-9, 100e-6, 50, self.dq_array_opts]
        self.dq_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 2.87e9, 1e-9, 20e-9, 'y']

        self.deer_params_defaults = [120, 10, 350e6, 750e6, 100, 800e-9]      
        self.deer_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 0.2, 40e-9, self.deer_drive_opts] 

        self.deer_rabi_params_defaults = [120, 10, 3e-9, 100e-9, 100, 800e-9]     
        self.deer_rabi_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 560e6, 0.2]     

        self.deer_fid_params_defaults = [120, 10, 50e-9, 20e-6, 100, self.fid_array_opts]    
        self.deer_fid_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 560e6, 0.2, 40e-9, 1]    

        self.deer_fid_cd_params_defaults = [120, 10, 50e-9, 20e-6, 100, self.fid_cd_array_opts]        
        self.deer_fid_cd_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 560e6, 0.2, 40e-9, 0.1, 1]

        self.deer_corr_rabi_params_defaults = [120, 10, 3e-9, 100e-9, 100, 800e-9, 1e-6]
        self.deer_corr_rabi_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 560e6, 40e-9, 0.2, 300]

        self.deer_corr_t1_params_defaults = [120, 10, 50e-9, 1e-6, 100, self.corr_t1_array_opts, 800e-9]
        self.deer_corr_t1_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 560e6, 40e-9, 0.2]

        self.deer_t2_params_defaults = [120, 10, 50e-9, 1e-6, 100, self.corr_t1_array_opts, 800e-9, 500e-9]
        self.deer_t2_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 560e6, 40e-9, 0.2] 

        self.nmr_params_defaults = [120, 10, 50e-9, 100e-6, 100, 1e-6]
        self.nmr_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 1]

        self.casr_params_defaults = [120, 10, 10, 200e-9, self.casr_sig_opts, self.dnp_opts]
        self.casr_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 1e6, 0.1, 1e-6, 0, 1]

        self.fit_none_default = [0]
        self.fit_odmr_defaults = [0.01, 2.31, 6e-3, 1] # defaults = 1% contrast, 2 GHz central freq, 6 MHz linewidth, 1 vertical offset
        self.fit_rabi_defaults = [0.02, 0.001, 200, 0, 1] # defaults = 2% contrast, 0.001 decay rate, 200 ns period, 0 phase, 1 vertical offset
        self.fit_odmr_rf_defaults = [0.01, 1, 6e-3, 1, 0.01, 1, 6e-3, 1] # defaults = 1% contrast, 1 GHz central freq, 6 MHz linewidth, 1 vertical offset, 1% contrast, 1 GHz central freq, 6 MHz linewidth, 1 vertical offset for the two overlapping Lorentzians
        self.fit_t1_defaults = [0.01, 1, 1, 0] # defaults = 0.01 amplitude, 1 ms T1, 1 stretching factor, 0 vertical offset
        self.fit_t2_defaults = [0.1, 2, 1, 1, 0.2, 0, 1, 0.2, 0] # defaults = 0.1 amplitude, 2 us T2, 1 stretching factor, 1 amp first sine wave, 0.2 MHz first sine wave, 0 phase first sine wave, 1 amp second sine wave, 0.2 MHz second sine wave, 0 phase second sine wave
        self.fit_deer_defaults = [0.1, 560, 10, 1] # defaults = 10% contrast, 560 MHz central freq, 10 MHz linewidth, 1 vertical offset
        self.fit_deer_rabi_defaults = [0.1, 0.001, 100, 0, 1] # defaults = 10% contrast, 0.001 decay rate, 100 ns period, 0 phase, 1 vertical offset
        self.fit_deer_t1_defaults = [0.1, 1, 1, 1, 1, 0] # defaults = 0.1 amplitude, 1 ms T1_NV, 1 n_NV, 1 ms T1_e, 1 n_e, 0 vertical offset
        self.fit_nmr_correlation_defaults = [0.01, 2, 0.1, 1] # defaults = 1% contrast, 2 MHz central freq, 100 kHz linewidth, 1 vertical offset
        self.fit_nmr_casr_defaults = [0.01, 2, 0.1, 1] # defaults = 1% contrast, 2 kHz central freq, 100 Hz linewidth, 1 vertical offset

        # experiment dictionary - associates experiment function, default parameter array, dataset, laser parameters and digitizer parameters to an experiment type
        self.exp_dict = {"Signal vs Time": ["sigvstime_scan", self.sigvstime_params_defaults, self.sigvstime_mw_params_defaults, 'sigvstime', self.laser_params_sigvstime_defaults, self.digitizer_defaults], # ODMR MW params hidden and serves as placeholder for sig vs time experiment in GUI
                    "CW ODMR": ["odmr_scan", self.odmr_params_defaults, self.odmr_mw_params_defaults, 'odmr', self.laser_params_defaults, self.digitizer_defaults],
                    "ODMR Smart Scan": ["odmr_smart_scan", self.odmr_smart_params_defaults, self.odmr_smart_mw_params_defaults, 'odmr', self.laser_params_defaults, self.digitizer_defaults],
                    "Sig Laser": [None, self.laser_params_sigvstime_defaults],
                    "Laser": [None, self.laser_params_defaults],
                    "Digitizer": [None, self.digitizer_defaults],
                    "Pulsed ODMR": ["pulsed_odmr_scan", self.pulsed_odmr_params_defaults, self.pulsed_odmr_mw_params_defaults, 'odmr', self.laser_params_defaults, self.digitizer_defaults],
                    "RF Coil: Pulsed ODMR": ["pulsed_odmr_rf_scan", self.pulsed_odmr_rf_params_defaults, self.pulsed_odmr_rf_mw_params_defaults, 'odmr rf', self.laser_params_defaults, self.digitizer_defaults],
                    "Rabi": ["rabi_scan", self.rabi_params_defaults, self.rabi_mw_params_defaults, 'rabi', self.laser_params_defaults, self.digitizer_defaults],
                    "Optical T1": ["OPT_T1_scan", self.opt_t1_params_defaults, self.opt_t1_mw_params_defaults, 't1', self.laser_params_defaults, self.digitizer_defaults],
                    "MW T1": ["MW_T1_scan", self.mw_t1_params_defaults, self.mw_t1_mw_params_defaults, 't1', self.laser_params_defaults, self.digitizer_defaults],
                    "T2": ["T2_scan", self.t2_params_defaults, self.t2_mw_params_defaults, 't2', self.laser_params_defaults, self.digitizer_defaults],
                    "RF Coil: T2": ["T2_rf_scan", self.t2_rf_params_defaults, self.t2_rf_mw_params_defaults, 't2', self.laser_params_defaults, self.digitizer_defaults],
                    "DQ Relaxation": ["DQ_scan", self.dq_params_defaults, self.dq_mw_params_defaults, 'dq', self.laser_params_defaults, self.digitizer_defaults],
                    "DEER": ["DEER_scan", self.deer_params_defaults, self.deer_mw_params_defaults, 'deer', self.laser_params_defaults, self.digitizer_defaults],
                    "DEER Rabi": ["DEER_rabi_scan", self.deer_rabi_params_defaults, self.deer_rabi_mw_params_defaults, 'deer rabi', self.laser_params_defaults, self.digitizer_defaults],
                    "DEER FID": ["DEER_FID_scan", self.deer_fid_params_defaults, self.deer_fid_mw_params_defaults, 'fid', self.laser_params_defaults, self.digitizer_defaults],
                    "DEER FID Continuous Drive": ["DEER_FID_CD_scan", self.deer_fid_cd_params_defaults, self.deer_fid_cd_mw_params_defaults, 'fid cd', self.laser_params_defaults, self.digitizer_defaults],
                    "DEER Correlation Rabi": ["DEER_corr_rabi_scan", self.deer_corr_rabi_params_defaults, self.deer_corr_rabi_mw_params_defaults, 'corr rabi', self.laser_params_defaults, self.digitizer_defaults],
                    "DEER T1": ["DEER_T1_scan", self.deer_corr_t1_params_defaults, self.deer_corr_t1_mw_params_defaults, 'deer t1', self.laser_params_defaults, self.digitizer_defaults],
                    "DEER T2": ["DEER_T2_scan", self.deer_t2_params_defaults, self.deer_t2_mw_params_defaults, 'deer t2', self.laser_params_defaults, self.digitizer_defaults],
                    "NMR Correlation Spectroscopy": ["Corr_Spec_scan", self.nmr_params_defaults, self.nmr_mw_params_defaults, 'nmr', self.laser_params_defaults, self.digitizer_defaults],
                    "NMR CASR": ["CASR_scan", self.casr_params_defaults, self.casr_mw_params_defaults, 'casr', self.laser_params_defaults, self.digitizer_defaults]}
        
        self.experiments = QComboBox()
        self.experiments.setFixedHeight(40)
        self.experiments.setStyleSheet("color: black; background-color: #C7C7C7; border: 4px solid black; padding: 2px; border-radius: 5px;")
        self.experiments.addItems(["Select experiment from list", 
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
                                 "NMR CASR"])
        
        self.experiments.currentIndexChanged.connect(lambda: self.exp_selector())

        # used to send to experiment process to determine extra actions to take 
        self.to_save = False 
        self.to_fit = False
        self.to_fit_live = False
        self.to_override_fit = False
        self.extra_kwarg_params: dict() = {}

        self.dataset_label = QLabel("<i>Data Set Name: ---</i>")
        self.dataset_label.setStyleSheet("color: black; background-color: #C7C7C7; border: 4px solid black; padding: 2px; border-radius: 5px;")
        self.dataset_label.setFixedHeight(40)
        self.dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # experiment params label
        self.exp_label = QLabel("Experiment Settings")
        self.exp_label.setFixedHeight(24)
        self.exp_label.setStyleSheet("font-weight: bold")
        
        self.params_widget = ParamsWidget(self.create_params_widget('CW ODMR', self.exp_dict['CW ODMR'][1]))

        self.mw_label = QLabel("Microwave Settings")
        self.mw_label.setFixedHeight(24)
        self.mw_label.setStyleSheet("font-weight: bold")

        self.mw_params_widget = ParamsWidget(self.create_mw_params_widget('CW ODMR', self.exp_dict['CW ODMR'][2]))
        self.mw_overrides = {}

        self.opacity_effects = []
        for i in range(23): # total number of GUI elements that need to be faded when no experiment is selected
            self.opacity_effects.append(QGraphicsOpacityEffect())
            self.opacity_effects[i].setOpacity(0.3)

        self.exp_label.setGraphicsEffect(self.opacity_effects[0])
        self.params_widget.setGraphicsEffect(self.opacity_effects[1])
        self.params_widget.setEnabled(False)
        
        # save params button
        save_button_style = """
        QPushButton {
        background-color: #E0E0E0;
        color: black;
        border: 2px solid #000000;
        border-radius: 5px;
        padding: 5px;
        }
        
        QPushButton:hover {
        background-color: #CCCCCC;
        border: 2px solid #636363;
        }
        
        QPushButton:pressed {
        background-color: #A5D4B0;
        border: 2px solid #636363;
        }"""
        self.save_params = QPushButton("Save Experiment Parameters")
        self.save_params.setStyleSheet(save_button_style)
        self.save_params.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.save_params.setMinimumWidth(440)
        self.save_params.clicked.connect(lambda: self.save_params_clicked())
        self.save_params.setGraphicsEffect(self.opacity_effects[2])
        self.save_params.setEnabled(False)

        # mw params widget & label
        self.mw_label.setGraphicsEffect(self.opacity_effects[3])        
        self.mw_params_widget.setGraphicsEffect(self.opacity_effects[4])
        self.mw_params_widget.setEnabled(False)

        # laser params widget & label
        self.laser_params_widget = ParamsWidget(self.create_params_widget('Laser', self.exp_dict['Laser'][1]))
        self.laser_label = QLabel("Laser & AWG Settings")
        self.laser_label.setFixedHeight(24)
        self.laser_label.setStyleSheet("font-weight: bold")
        self.laser_label.setGraphicsEffect(self.opacity_effects[5])
        self.laser_params_widget.setGraphicsEffect(self.opacity_effects[6])
        self.laser_params_widget.setEnabled(False)

        self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]))
        self.dig_label = QLabel("Digitizer Settings")
        self.dig_label.setFixedHeight(24)
        self.dig_label.setStyleSheet("font-weight: bold")
        self.dig_label.setGraphicsEffect(self.opacity_effects[7])
        self.dig_params_widget.setGraphicsEffect(self.opacity_effects[8])
        self.dig_params_widget.setEnabled(False)

        radio_style = """
        QRadioButton {
        color: white;
        background-color: black;
        border: 2px solid #5470FF;
        border-radius: 6px;
        padding: 4px 8px;
        spacing: 12px;
        }

        QRadioButton::indicator {
        width: 12px;
        height: 12px;
        border: 1px solid #ccc;
        border-radius: 6px;
        background: #222;
        }

        QRadioButton:checked {
        color: #5470FF;  
        }

        QRadioButton::indicator:checked {
        background-color: #5470FF;
        border: 1px solid #5470FF;
        }
        """

        # data acquisition system selection radio buttons
        self.daq_b1 = QRadioButton("Digitizer")
        self.daq_b1.toggled.connect(lambda:self.toggle_daq(self.daq_b1))
        self.daq_b1.setStyleSheet(radio_style)
        self.daq_b1.setFixedHeight(40)
        self.daq_b1.setFixedWidth(120)
        self.daq_b1.setGraphicsEffect(self.opacity_effects[9])
        self.daq_b1.setEnabled(False)

        self.daq_b2 = QRadioButton("NI DAQ")
        self.daq_b2.toggled.connect(lambda:self.toggle_daq(self.daq_b2))
        self.daq_b2.setStyleSheet(radio_style)
        self.daq_b2.setFixedHeight(40)
        self.daq_b2.setFixedWidth(120)
        self.daq_b2.setGraphicsEffect(self.opacity_effects[10])
        self.daq_b2.setEnabled(False)

        self.daq_group = QButtonGroup()
        self.daq_group.setExclusive(True)
        self.daq_group.addButton(self.daq_b1)
        self.daq_group.addButton(self.daq_b2)

        # time elapsed label
        self.time_elapsed = QLabel("<i>Experiment Timer</i>: 00:00:00")
        self.time_elapsed.setStyleSheet("color: black; background-color: #66DBE8; border: 4px solid black; padding: 2px; border-radius: 5px;")
        self.time_elapsed.setFixedHeight(40)
        self.time_elapsed.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # auto save checkbox
        self.auto_save_checkbox = QCheckBox("Auto Save ")
        self.auto_save_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #5F3200;
                        border: 2px solid orange;
                        padding: 2px;
                        border-radius: 5px;
                }

                QCheckBox::indicator:hover {
                        background-color: yellow;
                }

                QCheckBox::indicator:pressed {
                        background-color: lightgreen;
                }""")
        self.auto_save_checkbox.setChecked(False)
        self.auto_save_checkbox.setEnabled(False)
        self.auto_save_checkbox.stateChanged.connect(lambda: self.auto_save_changed())
        self.auto_save_checkbox.setGraphicsEffect(self.opacity_effects[11])

        # select directory button
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.setEnabled(False)
        self.select_dir_button.setStyleSheet("color: white; background-color: #7A7A7A; border: 2px solid #964900; padding: 2px; border-radius: 5px;")
        self.select_dir_button.setFixedWidth(250)
        self.select_dir_button.clicked.connect(lambda: self.select_directory())
        self.select_dir_button.setGraphicsEffect(self.opacity_effects[12])

        # selected directory display for saving
        self.chosen_dir = QLabel()
        self.chosen_dir.setGraphicsEffect(self.opacity_effects[13])
        self.chosen_dir.setStyleSheet("color: orange")

        self.filename_label = QLabel("Filename: ")
        self.filename_label.setGraphicsEffect(self.opacity_effects[14])
        self.filename_label.setFixedHeight(20)

        self.filename_lineedit = QLineEdit()
        self.filename_lineedit.setGraphicsEffect(self.opacity_effects[15])
        self.filename_lineedit.setFixedHeight(30)
        self.filename_lineedit.setEnabled(False)

        # auto fit checkbox
        self.auto_fit_checkbox = QCheckBox("Auto Fit  ")
        self.auto_fit_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #55005F;
                        border: 2px solid #D98BCB;
                        padding: 2px;
                        border-radius: 5px;
                }

                QCheckBox::indicator:hover {
                        background-color: yellow;
                }

                QCheckBox::indicator:pressed {
                        background-color: lightgreen;
                }""")
        self.auto_fit_checkbox.setChecked(False)
        self.auto_fit_checkbox.setEnabled(False)
        self.auto_fit_checkbox.stateChanged.connect(lambda: self.auto_fit_changed())
        self.auto_fit_checkbox.setGraphicsEffect(self.opacity_effects[16]) 
        
        # fit type label
        self.fit_label = QLabel("Fit Type")
        self.fit_label.setStyleSheet("color: #55005F; background-color: #C2C2C2; border: 2px solid #55005F; padding: 2px; border-radius: 5px;")
        self.fit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fit_label.setGraphicsEffect(self.opacity_effects[17]) 

        self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit ODMR', self.fit_odmr_defaults))
        self.fit_params_widget.setGraphicsEffect(self.opacity_effects[18])
        self.fit_params_widget.setEnabled(False)

        # live fitting checkbox
        self.live_fit_checkbox = QCheckBox("Live Fitting")
        self.live_fit_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #515151;
                        border: 2px solid white;
                        padding: 2px;
                        border-radius: 5px;
                }

                QCheckBox::indicator:hover {
                        background-color: yellow;
                }

                QCheckBox::indicator:pressed {
                        background-color: lightgreen;
                }""")
        
        self.live_fit_checkbox.setChecked(False)
        self.live_fit_checkbox.stateChanged.connect(lambda: self.live_fit_changed())
        self.live_fit_checkbox.setGraphicsEffect(self.opacity_effects[19])
        self.live_fit_checkbox.setEnabled(False)

        self.override_fit_checkbox = QCheckBox("Apply Fits to Settings")
        self.override_fit_checkbox.setStyleSheet("""
                QCheckBox {
                        color: white;
                        background-color: #3D3D3D;
                        border: 2px solid #A9A9A9;
                        padding: 2px;
                        border-radius: 5px;
                }
                QCheckBox::indicator:hover {
                        background-color: yellow;
                }
                QCheckBox::indicator:pressed {
                        background-color: lightgreen;
                }""")
        self.override_fit_checkbox.setChecked(False)
        self.override_fit_checkbox.stateChanged.connect(lambda: self.override_fit_changed())
        self.override_fit_checkbox.setGraphicsEffect(self.opacity_effects[20])
        self.override_fit_checkbox.setEnabled(False)

        # photodetector selection radio buttons
        detector_radio_style = """
        QRadioButton {
        color: white;
        background-color: black;
        border: 2px solid #C4C433;
        border-radius: 6px;
        padding: 4px 8px;
        spacing: 12px;
        }

        QRadioButton::indicator {
        width: 12px;
        height: 12px;
        border: 1px solid #ccc;
        border-radius: 6px;
        background: #222;
        }

        QRadioButton:checked {
        color: #C4C433;  
        }

        QRadioButton::indicator:checked {
        background-color: #C4C433;
        border: 1px solid #C4C433;
        }
        """
        self.detector_b1 = QRadioButton("APD")
        self.detector_b1.setStyleSheet(detector_radio_style)
        self.detector_b1.setFixedHeight(40)
        self.detector_b1.setFixedWidth(90)
        self.detector_b1.setGraphicsEffect(self.opacity_effects[21])
        self.detector_b1.setEnabled(False)

        self.detector_b2 = QRadioButton("BPD")
        self.detector_b2.setStyleSheet(detector_radio_style)
        self.detector_b2.setFixedHeight(40)
        self.detector_b2.setFixedWidth(90)
        self.detector_b2.setGraphicsEffect(self.opacity_effects[22])
        self.detector_b2.setEnabled(False)

        self.detector_group = QButtonGroup()
        self.detector_group.setExclusive(True)
        self.detector_group.addButton(self.detector_b1)
        self.detector_group.addButton(self.detector_b2)
        
        self.detector_b1.setChecked(True) # default to APD

        # status label
        self.status = QLabel("Set parameters and press 'Run' to begin experiment.")
        self.status.setStyleSheet("color: black; background-color: #00b8ff; border: 4px solid black; padding: 2px; border-radius: 5px;")
        self.status.setFixedHeight(40)
        
        # time estimate label
        self.time_estimate = QLabel("<i>Time Estimate</i>: --:--:--")
        self.time_estimate.setStyleSheet("color: black; background-color: #EDA855; border: 4px solid black; padding: 2px; border-radius: 5px;")
        self.time_estimate.setFixedHeight(40)

        # time remaining label
        self.time_remaining = QLabel("<i>Time Remaining</i>: --:--:--")
        self.time_remaining.setStyleSheet("color: black; background-color: #EDD155; border: 4px solid black; padding: 2px; border-radius: 5px;")
        self.time_remaining.setFixedHeight(40)

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
        QProgressBar {
                border: 2px solid black;
                border-radius: 4px;
                background-color: #2a2a2a;
                color: white;
                text-align: center;
        }

        QProgressBar::chunk {
                background-color: #00C7BA;
        }
        """)


        # run button
        run_button_style = """
        QPushButton {
        background-color: #003407;
        color: white;
        border: 2px solid limegreen;
        border-radius: 5px;
        padding: 5px;
        }
        
        QPushButton:hover {
        background-color: #004b47;
        border: 2px solid #00d7c9;
        }
        
        QPushButton:pressed {
        background-color: #005f5f;
        border: 2px solid #00d7c9;
        }"""
        run_button = QPushButton('Run')
        run_button.setStyleSheet(run_button_style)
        self.run_proc = ProcessRunner()
        run_button.clicked.connect(self.run)

        self.queue_to_exp: Queue = Queue()
        """multiprocessing Queue to pass to the experiment subprocess and use
        for sending messages to the subprocess."""
        self.queue_from_exp: Queue = Queue()
        """multiprocessing Queue to pass to the experiment subprocess and use
        for receiving messages from the subprocess."""

        # stop button
        stop_button_style = """
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
        stop_button = QPushButton('Stop')
        stop_button.setStyleSheet(stop_button_style)
        stop_button.clicked.connect(self.stop)
        # use a partial because the stop function may already be destroyed by the time
        # this is called
        self.destroyed.connect(partial(self.stop, log=False))
        
        # kill button
        # this is used to kill the experiment process if it is stuck
        kill_button_style = """
        QPushButton {
        background-color: #350000;
        color: white;
        border: 2px solid red;
        border-radius: 5px;
        padding: 5px;
        }
                
        QPushButton:hover {
        background-color: #4b0000;
        border: 2px solid #ff0000;
        }
        
        QPushButton:pressed {
        background-color: #610000;
        border: 2px solid #ff4d4d;
        }"""        
        kill_button = QPushButton('Kill')
        kill_button.setStyleSheet(kill_button_style)
        kill_button.clicked.connect(self.kill)

        self.gui_layout = QVBoxLayout()
        
        self.top_frame = QFrame(self)
        self.top_frame.setObjectName("topFrame")
        self.top_frame.setStyleSheet("QFrame#topFrame {background-color: #1e1e1e; border: 2px solid #717171; border-radius: 5px;}")
        self.top_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.top_layout = QHBoxLayout(self.top_frame)
        self.top_layout.setSpacing(0)
        self.top_layout.addWidget(self.experiments)
        self.top_layout.addWidget(self.dataset_label) 
        self.top_layout.addWidget(self.time_elapsed) 

        self.exp_frame = QFrame(self)
        self.exp_frame.setObjectName("expFrame")
        self.exp_frame.setStyleSheet("QFrame#expFrame {background-color: #4b0000; border: 2px solid #d70000; border-radius: 5px;}")
        self.exp_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.exp_params_layout = QVBoxLayout(self.exp_frame)
        self.exp_params_layout.setContentsMargins(6,6,6,6)
        self.exp_params_layout.setSpacing(0)
        self.exp_params_layout.addWidget(self.exp_label)
        self.exp_params_layout.addWidget(self.params_widget)

        self.mw_frame = QFrame(self)
        self.mw_frame.setObjectName("mwFrame")
        self.mw_frame.setStyleSheet("QFrame#mwFrame {background-color: #474b00; border: 2px solid #d7d700; border-radius: 5px;}")
        self.mw_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.mw_params_layout = QVBoxLayout(self.mw_frame)
        self.mw_params_layout.setContentsMargins(6,6,6,6)
        self.mw_params_layout.setSpacing(0)
        self.mw_params_layout.addWidget(self.mw_label)
        self.mw_params_layout.addWidget(self.mw_params_widget)

        self.detector_frame = QFrame(self)
        self.detector_frame.setObjectName("detectorFrame")
        self.detector_frame.setStyleSheet("QFrame#detectorFrame {background-color: #262626; border: 2px solid #FFFFFF; border-radius: 5px;}")
        self.detector_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.detector_layout = QGridLayout(self.detector_frame)
        self.detector_layout.setSpacing(0)
        save_row = QHBoxLayout()
        save_row.addStretch()
        save_row.addWidget(self.save_params)
        save_row.addStretch()
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.daq_b1)
        button_row.addWidget(self.daq_b2)
        button_row.addSpacing(20)
        button_row.addWidget(self.detector_b1)
        button_row.addWidget(self.detector_b2)
        button_row.addStretch()
        self.detector_layout.addLayout(save_row,1,0) 
        self.detector_layout.addLayout(button_row,2,0)

        self.save_frame = QFrame(self)
        self.save_frame.setObjectName("saveFrame")
        self.save_frame.setStyleSheet("QFrame#saveFrame {background-color: #1e1e1e; border: 2px solid #717171; border-radius: 5px;}")
        self.save_layout = QGridLayout(self.save_frame)
        self.save_layout.setSpacing(0)
        self.save_layout.addWidget(self.auto_save_checkbox,1,1,1,1)
        self.save_layout.addWidget(self.select_dir_button,1,2,1,1, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.save_layout.addWidget(self.chosen_dir,2,1,1,2)
        self.save_layout.addWidget(self.filename_label,3,1,1,1)
        self.save_layout.addWidget(self.filename_lineedit,3,2,1,1)
        
        self.fit_frame = QFrame(self)
        self.fit_frame.setObjectName("fitFrame")
        self.fit_frame.setStyleSheet("QFrame#fitFrame {background-color: #1e1e1e; border: 2px solid #717171; border-radius: 5px;}")
        self.fit_layout = QGridLayout(self.fit_frame)
        self.fit_layout.setSpacing(0)
        self.fit_layout.addWidget(self.auto_fit_checkbox,1,1,1,1)
        self.fit_layout.addWidget(self.fit_label,1,2,1,2)
        self.fit_layout.addWidget(self.fit_params_widget,2,1,1,2)
        self.fit_layout.addWidget(self.live_fit_checkbox,3,1,1,1)
        self.fit_layout.addWidget(self.override_fit_checkbox,3,2,1,2)
        
        self.fit_scroll = QScrollArea(self)
        self.fit_scroll.setWidgetResizable(True)
        self.fit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.fit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.fit_scroll.setFrameShape(QFrame.Shape.NoFrame)  # cleaner
        self.fit_scroll.setMinimumWidth(400)
        self.fit_scroll.setWidget(self.fit_frame)

        # size policies: scroll area takes space, inner frame stays minimal
        self.fit_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fit_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_frame = QFrame(self)
        self.bottom_frame.setObjectName("bottomFrame")
        self.bottom_frame.setStyleSheet("QFrame#bottomFrame {background-color: #1e1e1e; border: 2px solid #717171; border-radius: 5px;}")
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
        self.laser_frame.setStyleSheet("QFrame#laserFrame {background-color: #004b47; border: 2px solid #00d7c9; border-radius: 5px;}")
        self.laser_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.laser_params_layout = QGridLayout(self.laser_frame)
        self.laser_params_layout.setContentsMargins(6,6,6,6)
        self.laser_params_layout.setSpacing(0)
        self.laser_params_layout.addWidget(self.laser_label,1,1,1,2)
        self.laser_params_layout.addWidget(self.laser_params_widget,2,1,1,2)     

        self.dig_frame = QFrame(self)
        self.dig_frame.setObjectName("digFrame")
        self.dig_frame.setStyleSheet("QFrame#digFrame {background-color: #000b4b; border: 2px solid #1739FF; border-radius: 5px;}")
        # self.dig_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.dig_params_layout = QVBoxLayout(self.dig_frame)
        self.dig_params_layout.setContentsMargins(6,6,6,6)
        self.dig_params_layout.setSpacing(0)
        self.dig_params_layout.addWidget(self.dig_label)
        self.dig_params_layout.addWidget(self.dig_params_widget)
        
        self.top_widgets_layout = QVBoxLayout()
        self.top_widgets_layout.addWidget(self.top_frame)

        self.exp_widgets_layout = QHBoxLayout()
        self.exp_widgets_layout.addWidget(self.exp_frame)
        self.exp_widgets_layout.addWidget(self.mw_frame)

        self.laser_widgets_layout = QHBoxLayout()
        self.laser_widgets_layout.addWidget(self.laser_frame)
        self.laser_widgets_layout.addWidget(self.dig_frame)

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

    def create_params_widget(self, tag, defaults): 
        match tag:
            case 'Sig Laser':    
                params = {
                        'laser_power': {'display_text': 'Laser Power (%): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(0, 100), dec = True)}}
            case 'Laser':    
                params = {
                        'laser_power': {'display_text': 'Laser Power (%): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(0, 95), dec = True)},
                        'laser_init': {'display_text': 'Initialize Time (pulsed): ',
                                'widget': SpinBox(value = defaults[1], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'laser_readout': {'display_text': 'Read Time (pulsed): ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'sideband_freq': {'display_text': 'MW Sideband Mod. Freq.: ',
                                        'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100, 100e6), dec = True)},
                        'sideband_power': {'display_text': 'MW Sideband Power: ',
                                        'widget': SpinBox(value = defaults[4], suffix = 'V', siPrefix = True, bounds = (0, 0.45))},
                        'sideband': {'display_text': 'MW Sideband: ',
                                        'widget': ComboBox(items = defaults[5])},
                        'i_offset': {'display_text': 'MW I Offset: ',
                                        'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True)},
                        'q_offset': {'display_text': 'MW Q Offset: ',
                                        'widget': SpinBox(value = defaults[7], suffix = 'V', siPrefix = True)},
                        'awg_samp_rate_1': {'display_text': 'AWG Group 1 Samp. Rate: ',
                                        'widget': ComboBox(items = defaults[8])},
                        'awg_samp_rate_2': {'display_text': 'AWG Group 2 Samp. Rate: ',
                                        'widget': ComboBox(items = defaults[9])}}                
            case 'Digitizer':
                params = {
                        'segment_size': {'display_text': '# Samples (seg. size): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(0, 1e9), dec = True)},
                        'dig_sampling_freq': {'display_text': 'Sampling Frequency: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100, 500e6), dec = True)}, 
                        'dig_amplitude': {'display_text': 'Amplitude: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'V', siPrefix = True)},
                        'read_channel': {'display_text': 'Readout Channel: ',
                                'widget': ComboBox(items = defaults[3])},
                        'dig_coupling': {'display_text': 'Coupling: ',
                                'widget': ComboBox(items = defaults[4])},
                        'dig_termination': {'display_text': 'Termination (\u03A9): ',
                                'widget': ComboBox(items = defaults[5])},        
                        'pretrig_size': {'display_text': '# Pretrig. Samples: ',
                                'widget': SpinBox(value = defaults[6], int = True, bounds=(0, 1024), dec = True)},
                        'dig_timeout': {'display_text': 'Card Timeout: ',
                                'widget': SpinBox(value = defaults[7], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
            case 'Signal vs Time':
                params = {
                'exp_sampling_rate': {'display_text': 'Exp. Sampling Rate: ',
                        'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (10, 1e6), dec = True)}}
            
            case 'CW ODMR':    
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)}}
            case 'ODMR Smart Scan':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None), dec = True)},
                        'start_angle': {'display_text': 'Start Angle: ',
                                'widget': SpinBox(value = defaults[2], bounds=(1, 115), dec = True)},
                        'stop_angle': {'display_text': 'Stop Angle: ',
                                'widget': SpinBox(value = defaults[3], bounds=(1, 115), dec = True)},
                        'iters': {'display_text': '# Angles to Sweep: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, 100))}}                    
            case 'Pulsed ODMR':    
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)}}            
            case 'RF Coil: Pulsed ODMR':    
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)}}         
            case 'Rabi':    
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start MW Pulse Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'End MW Pulse Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)}}                
            case 'Optical T1':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}}
            case 'MW T1':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}}
            case 'RF Coil: T2':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}}   
            case 'T2':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}}    
            case 'DQ Relaxation':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}} 
            case 'DEER':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start Frequency: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'stop': {'display_text': 'End Frequency: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}           
            case 'DEER Rabi':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start AWG Pulse Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'stop': {'display_text': 'End AWG Pulse Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}           
            case 'DEER FID':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}}             
            case 'DEER FID Continuous Drive':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}}                          
            case 'DEER Correlation Rabi':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start AWG Corr. Pulse Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'stop': {'display_text': 'End AWG Corr. Pulse Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (3e-9, None), dec = True)},
                        'num_pts': {'display_text': '# Pulse Durations: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        't_corr': {'display_text': 'Correlation Time \u03C4_c: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}            
            case 'DEER T1':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4_corr Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4 Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}           
            case 'DEER T2':
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start t Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop t Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# t Points: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])},
                        'tau': {'display_text': 'Free Precession \u03C4: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'deer_t2_buffer': {'display_text': 't Buffer Time: ',
                                'widget': SpinBox(value = defaults[7], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}       
            case 'NMR Correlation Spectroscopy':    
                params = {
                'runs': {'display_text': '# Averages per Iteration: ',
                        'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                'iters': {'display_text': '# Experiment Iterations: ',
                        'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                'start': {'display_text': 'Start \u03C4 Time: ',
                        'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'stop': {'display_text': 'Stop \u03C4 Time: ',
                        'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                'num_pts': {'display_text': '# Frequencies: ',
                        'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                'tau': {'display_text': 'Free Precession Interval (\u03C4): ',
                        'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}           
            case 'NMR CASR':
                params = {
                'runs': {'display_text': 'Runs (avgs. per iteration): ',
                        'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                'iters': {'display_text': '# Experiment Iterations: ',
                        'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                'num_pts': {'display_text': 'n_R (# synch. readout pts.): ',
                        'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)},
                'tau': {'display_text': 'tau = 1/(2f_0): ',
                        'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, 1e-3), dec = True)},
                'sig_opt': {'display_text': 'CASR Signal Source: ',
                                'widget': ComboBox(items = defaults[4])},
                'dnp': {'display_text': 'Hyperpolarization: ',
                                'widget': ComboBox(items = defaults[5])}}
                
        return params

    def create_mw_params_widget(self, tag, defaults, overrides=None):
        overrides = {} if overrides is None else overrides

        match tag:
            case 'Signal vs Time': # not used - hidden
                params = {
                'exp_sampling_rate_0': {'display_text': 'Sampling Rate: ',
                        'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (10, 1e6), dec = True)}}
            case 'CW ODMR':
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'probe': {'display_text': 'MW Probe Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (10e-9, None))}}    
            case 'ODMR Smart Scan':
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'probe': {'display_text': 'MW Probe Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (10e-9, None))}}
            case 'Pulsed ODMR':    
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[2]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[3]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)}}
            case 'RF Coil: Pulsed ODMR':    
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = overrides.get('center_freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[2]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[3]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'rf_pulse_freq': {'display_text': 'RF Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_pulse_power': {'display_text': 'RF Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)},
                        'rf_pulse_phase': {'display_text': 'RF Phase (deg.): ',
                                'widget': SpinBox(value = defaults[6], int = True, bounds=(0, 360))}}
            case 'Rabi':    
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': ComboBox(items = defaults[2])}}
            case 'Optical T1': # not used - hidden
                params = {
                        'runs': {'display_text': '# Averages per Iteration: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'start': {'display_text': 'Start \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'stop': {'display_text': 'Stop \u03C4 Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'num_pts': {'display_text': '# \u03C4: ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None), dec = True)},
                        'array_type': {'display_text': 'Array Type: ',
                                'widget': ComboBox(items = defaults[5])}}
            case 'MW T1':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])}}
            case 'T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        't2_seq': {'display_text': 'Sequence: ',
                                'widget': ComboBox(items = defaults[4])},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[5], int = True, bounds=(1, None))}}
            case 'RF Coil: T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'rf_pulse_freq': {'display_text': 'RF Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 750e6), dec = True)},
                        'rf_pulse_power': {'display_text': 'RF Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)},
                        'rf_pulse_phase': {'display_text': 'RF Phase (deg.): ',
                                'widget': SpinBox(value = defaults[6], int = True, bounds=(0, 360))},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[7], int = True, bounds=(1, None))}}
            case 'DQ Relaxation':
                params = {
                        'freq_minus': {'display_text': 'NV Frequency |-1>: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power_minus': {'display_text': 'NV MW Power |-1>: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi_minus': {'display_text': '\u03C0 Pulse |-1>: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq_plus': {'display_text': 'NV Frequency |+1>: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power_plus': {'display_text': 'NV MW Power |+1>: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi_plus': {'display_text': '\u03C0 Pulse |+1>: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[6])}}
            case 'DEER':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'drive_type': {'display_text': 'Dark MW Driving',
                                'widget': ComboBox(items = defaults[6])}}
            case 'DEER Rabi':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'DEER FID':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[7], int = True, bounds=(1, None))}}      
            case 'DEER FID Continuous Drive':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_cd_power': {'display_text': 'Continuous Drive (AWG) Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'V', siPrefix = True, bounds = (0, 0.8))},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[8], int = True, bounds=(1, None))}}              
            case 'DEER Correlation Rabi':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'DEER T1':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'DEER T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (350e6, 750e6), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True, bounds = (0, 0.8))}}
            case 'NMR Correlation Spectroscopy':    
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None))}}    
            case 'NMR CASR':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = overrides.get('freq', defaults[0]), suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = overrides.get('rf_power', defaults[1]), suffix = 'W', siPrefix = True, bounds = (1e-14, 3e-3))},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = overrides.get('pi', defaults[2]), suffix = 's', siPrefix = True, bounds = (1e-9, 10e-6), dec = True)},
                        'rf_pulse_freq': {'display_text': 'RF Frequency: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_pulse_power': {'display_text': 'RF Power: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'V', siPrefix = True, bounds = (0, 2))},
                        'rf_pi_half': {'display_text': 'RF \u03C0/2 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},        
                        'rf_pulse_phase': {'display_text': 'RF Phase (deg.): ',
                                'widget': SpinBox(value = defaults[6], int = True, bounds=(0, 360))},
                        'n': {'display_text': 'n (# XY8-n repetitions): ',
                                'widget': SpinBox(value = defaults[7], int = True, bounds=(1, None))}}
                
        return params

    def create_fit_params_widget(self, tag, defaults): 
        match tag:
            case 'Fit ODMR': # -A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'x0': {'display_text': 'x0 (GHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'gamma': {'display_text': '\u03B3 (GHz): ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])}}
            case 'Fit Rabi': # A * exp(-gamma * x) * cos(2 * pi * x / T + phi) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'gamma': {'display_text': '\u03B3 (GHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'T': {'display_text': 'T (ns): ',
                                'widget': SpinBox(value = defaults[2])},
                        'phi': {'display_text': '\u03C6: ',
                                'widget': SpinBox(value = defaults[3])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[4])}}
            case 'Fit ODMR RF': # -A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ', 
                                'widget': SpinBox(value = defaults[0])},
                        'x0': {'display_text': 'x0 (GHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'gamma': {'display_text': '\u03B3 (GHz): ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])},
                        'A_rf': {'display_text': 'A_rf: ',
                                'widget': SpinBox(value = defaults[4])},
                        'x0_rf': {'display_text': 'x0_rf (GHz): ',
                                'widget': SpinBox(value = defaults[5])},
                        'gamma_rf': {'display_text': '\u03B3_rf (GHz): ',
                                'widget': SpinBox(value = defaults[6])},
                        'c_rf': {'display_text': 'c_rf: ',
                                'widget': SpinBox(value = defaults[7])}}
            case 'Fit T1': # A * exp(-(t / T1)**n) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'T1': {'display_text': 'T1 (ms): ',
                                'widget': SpinBox(value = defaults[1])},
                        'n': {'display_text': 'n: ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])}}
            case 'Fit T2':
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'T2': {'display_text': 'T2 (\u03BCs): ',
                                'widget': SpinBox(value = defaults[1])},
                        'n': {'display_text': 'n: ',
                                'widget': SpinBox(value = defaults[2])},
                        'a1': {'display_text': 'a1: ',
                                'widget': SpinBox(value = defaults[3])},
                        'f1': {'display_text': 'f1 (MHz): ',
                                'widget': SpinBox(value = defaults[4])},
                        'phi1': {'display_text': '\u03C61: ',
                                'widget': SpinBox(value = defaults[5])},
                        'a2': {'display_text': 'a2: ',
                                'widget': SpinBox(value = defaults[6])}, 
                        'f2': {'display_text': 'f2 (MHz): ',
                                'widget': SpinBox(value = defaults[7])},
                        'phi2': {'display_text': '\u03C62: ',
                                'widget': SpinBox(value = defaults[8])}}
            case 'Fit DEER': # -A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'x0': {'display_text': 'x0 (GHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'gamma': {'display_text': '\u03B3 (GHz): ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])}}
            case 'Fit DEER Rabi': # A * exp(-gamma * x) * cos(2 pi x / T + phi) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'gamma': {'display_text': '\u03B3 (GHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'T': {'display_text': 'T (ns): ',
                                'widget': SpinBox(value = defaults[2])},
                        'phi': {'display_text': '\u03C6: ',
                                'widget': SpinBox(value = defaults[3])},
                        'C': {'display_text': 'C: ',
                                'widget': SpinBox(value = defaults[4])}}
            case 'Fit DEER T1': # A * exp(-(t / T1_NV)**n_NV - (t / T1_e)**n_e) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'T1_NV': {'display_text': 'T1 (ms): ',
                                'widget': SpinBox(value = defaults[1])},
                        'n_NV': {'display_text': 'n_NV: ',
                                'widget': SpinBox(value = defaults[2])},
                        'T1_e': {'display_text': 'T1_e (ms): ',
                                'widget': SpinBox(value = defaults[3])},
                        'n_e': {'display_text': 'n_e: ',
                                'widget': SpinBox(value = defaults[4])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[5])}}
            case 'Fit NMR Correlation Spectroscopy': # A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'x0': {'display_text': 'x0 (MHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'gamma': {'display_text': '\u03B3 (kHz): ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])}}                    
            case 'Fit NMR CASR': # A / (1 + ((x - x0) / gamma)**2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'x0': {'display_text': 'x0 (kHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'gamma': {'display_text': '\u03B3 (kHz): ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])}}
            case _:
                params = {
                        'A': {'display_text': 'Fit params here',
                                'widget': SpinBox(value = defaults[0])}}
                
        return params

    def _handle_exp_message(self, msg):
        percent = int(msg.get("percent", 0))
        status = msg.get("status", "")
        fit_value = msg.get("fit_value", None)
        fit_error = msg.get("fit_error", None)
        fit_value2 = msg.get("fit_value2", None)
        fit_error2 = msg.get("fit_error2", None)
        fit_val_list = []
        fit_err_list = []

        if fit_value is not None and fit_error is not None:
            fit_val_list.extend(fit_value)
            fit_err_list.extend(fit_error)
        if fit_value2 is not None and fit_error2 is not None:
            fit_val_list.extend(fit_value2)
            fit_err_list.extend(fit_error2)

        if fit_value is not None and fit_error is not None:
            self.fit_params_widget.set_fit_labels([fit_val_list, fit_err_list])

        # Optional timers (strings are easiest for labels)
        elapsed_str = msg.get("elapsed_str", None)
        remaining_str = msg.get("remaining_str", None)
        est_total_str = msg.get("est_total_str", None)

        # Optional exception text if you add it for 'failed'
        exc_text = msg.get("exception", None)

        self.progress_bar.setValue(percent)

        if status == 'in progress':
            self.status.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
            self.status.setText(f"{self.experiments.currentText()} scan in progress...")
            self.experiments.setEnabled(False)
            self.params_widget.setEnabled(False)
            self.save_params.setEnabled(False)
            self.laser_params_widget.setEnabled(False)
            self.dig_params_widget.setEnabled(False)

        elif status == 'complete':
            self.status.setStyleSheet("color: black; background-color: limegreen; border: 4px solid black;")
            self.status.setText(f"{self.experiments.currentText()} scan complete.")
            self.experiments.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)
            
            # if desired, populate fitted ODMR resonance frequency or Rabi period back into the params for all relevant experiments
            if (
                self.to_override_fit
                and self.experiments.currentText() in ("CW ODMR", "Pulsed ODMR")
                and fit_value is not None
                and len(fit_value) > 1
            ):
                odmr_freq_hz = fit_value[1] * 1e9  # Convert GHz back to Hz

                if odmr_freq_hz < 100e3 or odmr_freq_hz > 6e9:
                    print(f"Fitted ODMR frequency {odmr_freq_hz} Hz is out of bounds. Not applying override.")
                else:
                    # override in all relevant experiments' defaults
                    self.mw_overrides['center_freq'] = odmr_freq_hz  # Store override for future use
                    self.mw_overrides['freq'] = odmr_freq_hz  # Store override for future use

            if (
                self.to_override_fit 
                and self.experiments.currentText() == "Rabi"
                and fit_value is not None
                and len(fit_value) > 2
            ):
                rabi_pi_pulse_s = fit_value[2] * 1e-9  # Convert ns back to s

                if rabi_pi_pulse_s < 1e-9 or rabi_pi_pulse_s > 10e-6:
                    print(f"Fitted Rabi \u03C0 pulse {rabi_pi_pulse_s} s is out of bounds. Not applying override.")
                else:
                    # Update pi pulse duration for all relevant experiments
                    self.mw_overrides['pi'] = rabi_pi_pulse_s  # Store override for future use

                    mw_params = dict(**self.mw_params_widget.all_params())
                    mw_power = mw_params.get('rf_power', None)

                    if mw_power is not None:
                        self.mw_overrides['rf_power'] = mw_power  # Also override power based on current setting
            
            if (
                self.to_override_fit 
                and self.experiments.currentText() == "DEER"
                and fit_value is not None
                and len(fit_value) > 1
            ):
                deer_freq_hz = fit_value[1] * 1e6  # Convert GHz back to Hz

                if deer_freq_hz < 350e6 or deer_freq_hz > 750e6:
                    print(f"Fitted DEER frequency {deer_freq_hz} Hz is out of bounds. Not applying override.")
                else:
                    # override in all relevant experiments' defaults
                    self.mw_overrides['dark_freq'] = deer_freq_hz  # Store override for future use

                    mw_params = dict(**self.mw_params_widget.all_params())
                    awg_power = mw_params.get('awg_power', None)

                    if mw_power is not None:
                        self.mw_overrides['awg_power'] = awg_power  # Also override power based on current setting
                        
            if (
                self.to_override_fit 
                and self.experiments.currentText() == "DEER Rabi"
                and fit_value is not None
                and len(fit_value) > 2
            ):
                deer_rabi_pi_pulse_s = fit_value[2] * 1e-9  # Convert ns back to s

                if deer_rabi_pi_pulse_s < 1e-9 or deer_rabi_pi_pulse_s > 10e-6:
                    print(f"Fitted DEER Rabi \u03C0 pulse {deer_rabi_pi_pulse_s} s is out of bounds. Not applying override.")
                else:
                    # Update pi pulse duration for all relevant experiments
                    self.mw_overrides['dark_pi'] = deer_rabi_pi_pulse_s  # Store override for future use

                    mw_params = dict(**self.mw_params_widget.all_params())
                    awg_power = mw_params.get('awg_power', None)

                    if mw_power is not None:
                        self.mw_overrides['awg_power'] = awg_power  # Also override power based on current setting


        elif status == 'failed':
            self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
            if exc_text is not None:
                self.status.setText(f"{self.experiments.currentText()} scan failed. Exception: '{exc_text}'.")
            else:
                self.status.setText(f"{self.experiments.currentText()} scan failed.")
            self.experiments.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)

        else:
            self.status.setStyleSheet("color: black; background-color: white; border: 4px solid black;")
            self.status.setText(f"{self.experiments.currentText()} scan stopped.")
            self.experiments.setEnabled(True)
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.dig_params_widget.setEnabled(True)

            # if desired, populate fitted ODMR resonance frequency or Rabi period back into the params for all relevant experiments
            if (
                self.to_override_fit
                and self.experiments.currentText() in ("CW ODMR", "Pulsed ODMR")
                and fit_value is not None
                and len(fit_value) > 1
            ):
                odmr_freq_hz = fit_value[1] * 1e9  # Convert GHz back to Hz

                if odmr_freq_hz < 100e3 or odmr_freq_hz > 6e9:
                    print(f"Fitted ODMR frequency {odmr_freq_hz} Hz is out of bounds. Not applying override.")
                else:
                    # override in all relevant experiments' defaults
                    self.mw_overrides['center_freq'] = odmr_freq_hz  # Store override for future use
                    self.mw_overrides['freq'] = odmr_freq_hz  # Store override for future use

            if (
                self.to_override_fit 
                and self.experiments.currentText() == "Rabi"
                and fit_value is not None
                and len(fit_value) > 2
            ):
                rabi_pi_pulse_s = fit_value[2] * 1e-9  # Convert ns back to s

                if rabi_pi_pulse_s < 1e-9 or rabi_pi_pulse_s > 10e-6:
                    print(f"Fitted Rabi \u03C0 pulse {rabi_pi_pulse_s} s is out of bounds. Not applying override.")
                else:
                    # Update pi pulse duration for all relevant experiments
                    self.mw_overrides['pi'] = rabi_pi_pulse_s  # Store override for future use

                    mw_params = dict(**self.mw_params_widget.all_params())
                    mw_power = mw_params.get('rf_power', None)

                    if mw_power is not None:
                        self.mw_overrides['rf_power'] = mw_power  # Also override power based on current setting

            if (
                self.to_override_fit 
                and self.experiments.currentText() == "DEER"
                and fit_value is not None
                and len(fit_value) > 1
            ):
                deer_freq_hz = fit_value[1] * 1e6  # Convert GHz back to Hz

                if deer_freq_hz < 350e6 or deer_freq_hz > 750e6:
                    print(f"Fitted DEER frequency {deer_freq_hz} Hz is out of bounds. Not applying override.")
                else:
                    # override in all relevant experiments' defaults
                    self.mw_overrides['dark_freq'] = deer_freq_hz  # Store override for future use

                    mw_params = dict(**self.mw_params_widget.all_params())
                    awg_power = mw_params.get('awg_power', None)

                    if mw_power is not None:
                        self.mw_overrides['awg_power'] = awg_power  # Also override power based on current setting

            if (
                self.to_override_fit 
                and self.experiments.currentText() == "DEER Rabi"
                and fit_value is not None
                and len(fit_value) > 2
            ):
                deer_rabi_pi_pulse_s = fit_value[2] * 1e-9  # Convert ns back to s

                if deer_rabi_pi_pulse_s < 1e-9 or deer_rabi_pi_pulse_s > 10e-6:
                    print(f"Fitted DEER Rabi \u03C0 pulse {deer_rabi_pi_pulse_s} s is out of bounds. Not applying override.")
                else:
                    # Update pi pulse duration for all relevant experiments
                    self.mw_overrides['dark_pi'] = deer_rabi_pi_pulse_s  # Store override for future use

                    mw_params = dict(**self.mw_params_widget.all_params())
                    awg_power = mw_params.get('awg_power', None)

                    if mw_power is not None:
                        self.mw_overrides['awg_power'] = awg_power  # Also override power based on current setting

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
        for q in (getattr(self, "queue_from_exp", None), getattr(self, "queue_to_exp", None)):
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
            self.filename_lineedit.setEnabled(True)
            # self.opacity_effects[6].setEnabled(False) # change to 6? laser params widget
            self.opacity_effects[12].setEnabled(False)
            self.opacity_effects[13].setEnabled(False)
            self.opacity_effects[14].setEnabled(False)
            self.opacity_effects[15].setEnabled(False)

        else:
            self.to_save = False
            self.select_dir_button.setEnabled(False)
            self.filename_lineedit.setEnabled(False)
            # self.opacity_effects[6].setEnabled(True) # change to 6?
            self.opacity_effects[12].setEnabled(True)
            self.opacity_effects[13].setEnabled(True)
            self.opacity_effects[14].setEnabled(True)
            self.opacity_effects[15].setEnabled(True)

    def save_params_clicked(self):
        params = dict(**self.params_widget.all_params())
        mw_params = dict(**self.mw_params_widget.all_params())
        laser_params = dict(**self.laser_params_widget.all_params())
        dig_params = dict(**self.dig_params_widget.all_params())

        saved_params = list(params.values()) # set saved params for next time the experiment is selected
        saved_mw_params = list(mw_params.values()) # set saved params for next time the experiment is selected
        saved_laser_params = list(laser_params.values())
        saved_dig_params = list(dig_params.values())

        if self.experiments.currentText() != 'Signal vs Time':
            # update laser param comboboxes
            self.sideband_opts.insert(0, self.sideband_opts.pop(self.sideband_opts.index(saved_laser_params[5])))
            saved_laser_params[5] = self.sideband_opts
            self.awg_sampling_group1_opts.insert(0, self.awg_sampling_group1_opts.pop(self.awg_sampling_group1_opts.index(saved_laser_params[8])))
            saved_laser_params[8] = self.awg_sampling_group1_opts
            self.awg_sampling_group2_opts.insert(0, self.awg_sampling_group2_opts.pop(self.awg_sampling_group2_opts.index(saved_laser_params[9])))
            saved_laser_params[9] = self.awg_sampling_group2_opts

            # update digitizer param comboboxes
            self.dig_ro_chan_opts.insert(0, self.dig_ro_chan_opts.pop(self.dig_ro_chan_opts.index(saved_dig_params[3])))
            saved_dig_params[3] = self.dig_ro_chan_opts
            self.dig_coupling_opts.insert(0, self.dig_coupling_opts.pop(self.dig_coupling_opts.index(saved_dig_params[4])))
            saved_dig_params[4] = self.dig_coupling_opts
            self.dig_termination_opts.insert(0, self.dig_termination_opts.pop(self.dig_termination_opts.index(saved_dig_params[5])))
            saved_dig_params[5] = self.dig_termination_opts

        # TODO: update which params widget each condition goes under
        # take chosen combobox parameter and place it first in the updated combobox item list
        match self.experiments.currentText():
            case 'Rabi':
                self.rabi_axis_opts.insert(0, self.rabi_axis_opts.pop(self.rabi_axis_opts.index(saved_mw_params[2])))
                saved_mw_params[2] = self.rabi_axis_opts   
                # self.rabi_device_opts.insert(0, self.rabi_device_opts.pop(self.rabi_device_opts.index(saved_params[8])))
                # saved_params[8] = self.rabi_device_opts                
            case 'Optical T1':
                self.opt_t1_array_opts.insert(0, self.opt_t1_array_opts.pop(self.opt_t1_array_opts.index(saved_params[5])))
                saved_params[5] = self.opt_t1_array_opts
            case 'MW T1':
                self.mw_t1_array_opts.insert(0, self.mw_t1_array_opts.pop(self.mw_t1_array_opts.index(saved_params[5])))
                saved_params[5] = self.mw_t1_array_opts
            case 'DQ Relaxation':
                self.dq_array_opts.insert(0, self.dq_array_opts.pop(self.dq_array_opts.index(saved_params[5])))
                saved_params[5] = self.dq_array_opts   
            case 'RF Coil: T2':
                self.t2_rf_array_opts.insert(0, self.t2_rf_array_opts.pop(self.t2_rf_array_opts.index(saved_params[5])))
                saved_params[5] = self.t2_rf_array_opts
            case 'T2':
                self.t2_array_opts.insert(0, self.t2_array_opts.pop(self.t2_array_opts.index(saved_params[5])))
                saved_params[5] = self.t2_array_opts
                self.t2_seq_opts.insert(0, self.t2_seq_opts.pop(self.t2_seq_opts.index(saved_mw_params[4])))
                saved_mw_params[4] = self.t2_seq_opts
            case 'DEER':
                self.deer_drive_opts.insert(0, self.deer_drive_opts.pop(self.deer_drive_opts.index(saved_mw_params[6])))
                saved_mw_params[6] = self.deer_drive_opts
            case 'DEER FID':
                self.fid_array_opts.insert(0, self.fid_array_opts.pop(self.fid_array_opts.index(saved_params[5])))
                saved_params[5] = self.fid_array_opts
            case 'DEER FID Continuous Drive':
                self.fid_cd_array_opts.insert(0, self.fid_cd_array_opts.pop(self.fid_cd_array_opts.index(saved_params[5])))
                saved_params[5] = self.fid_cd_array_opts
            case 'DEER T1':
                self.corr_t1_array_opts.insert(0, self.corr_t1_array_opts.pop(self.corr_t1_array_opts.index(saved_params[5])))
                saved_params[5] = self.corr_t1_array_opts
            case 'NMR CASR':
                self.casr_sig_opts.insert(0, self.casr_sig_opts.pop(self.casr_sig_opts.index(saved_params[4])))
                saved_params[4] = self.casr_sig_opts
                self.dnp_opts.insert(0, self.dnp_opts.pop(self.dnp_opts.index(saved_params[5])))
                saved_params[5] = self.dnp_opts

        self.exp_dict[self.experiments.currentText()][1] = saved_params
        if self.experiments.currentText() != 'Signal vs Time':
            self.exp_dict[self.experiments.currentText()][2] = saved_mw_params
        self.exp_dict[self.experiments.currentText()][4] = saved_laser_params
        self.exp_dict[self.experiments.currentText()][5] = saved_dig_params
        
    def auto_fit_changed(self):
        if self.auto_fit_checkbox.isChecked() == True:
            self.to_fit = True
            self.fit_params_widget.setEnabled(True)
            self.live_fit_checkbox.setEnabled(True)
            self.override_fit_checkbox.setEnabled(True)
            self.opacity_effects[17].setEnabled(False)
            self.opacity_effects[18].setEnabled(False)
            self.opacity_effects[19].setEnabled(False)
            self.opacity_effects[20].setEnabled(False)
        else:
            self.to_fit = False
            self.fit_params_widget.setEnabled(False)
            self.live_fit_checkbox.setEnabled(False)
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
        self.status.setStyleSheet("color: black; background-color: #00b8ff; border: 4px solid black;")
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
            self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit None', self.fit_none_default), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.fit_params_widget.setGraphicsEffect(self.opacity_effects[18]) # reset opacity effects
            for i in range(23):
                self.opacity_effects[i].setEnabled(True)

            self.params_widget.setEnabled(False)
            self.dataset_label.setText("<i>Data Set: N/A</i>")
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

        else:
            self.dataset_label.setText(f"<i>Data Set: '{self.exp_dict[self.experiments.currentText()][3]}'</i>")
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

            match self.experiments.currentText():
                case 'CW ODMR' | 'Pulsed ODMR':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit ODMR', self.fit_odmr_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Rabi':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit Rabi', self.fit_rabi_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'RF Coil: Pulsed ODMR':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit ODMR RF', self.fit_odmr_rf_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Optical T1' | 'MW T1':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit T1', self.fit_t1_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})    
                case 'T2':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit T2', self.fit_t2_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})  
                case 'DEER':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit DEER', self.fit_deer_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})    
                case 'DEER Rabi':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit DEER Rabi', self.fit_deer_rabi_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val}) 
                case 'DEER T1':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit DEER T1', self.fit_deer_t1_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'NMR Correlation Spectroscopy':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit NMR Correlation Spectroscopy', self.fit_nmr_correlation_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val}) 
                case 'NMR CASR':
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit NMR CASR', self.fit_nmr_casr_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})  
                case _:
                    self.fit_params_widget = FitParamsWidget(self.create_fit_params_widget('Fit None', self.fit_none_default), get_param_value_funs = {ComboBox: self.get_combobox_val})

            self.fit_params_widget.setGraphicsEffect(self.opacity_effects[18]) # reset opacity effects
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
            save_row.addStretch()
            save_row.addWidget(self.save_params)
            save_row.addStretch()
            self.detector_layout.addLayout(save_row,1,0) 
            self.fit_layout.addWidget(self.auto_fit_checkbox,1,1,1,1)
            self.fit_layout.addWidget(self.fit_label,1,2,1,2)
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
                self.fit_label.hide()
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
                self.fit_label.show()
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
                self.fit_label.show()
                self.fit_params_widget.show()
                self.live_fit_checkbox.show()
                self.override_fit_checkbox.show()
            
            if self.daq_b1.isChecked() == True:
                self.dig_params_widget.setEnabled(True)
            else:
                self.dig_params_widget.setEnabled(False)
            
            if self.auto_fit_checkbox.isChecked() == True:
                self.fit_params_widget.setEnabled(True)
            else:
                self.fit_params_widget.setEnabled(False)

            match self.experiments.currentText():
                case 'CW ODMR' | 'Pulsed ODMR':
                    self.fit_label.setText("\"Lorentzian\"")
                case 'Rabi':
                    self.fit_label.setText("\"Decaying Cosine\"")
                case 'RF Coil: Pulsed ODMR':
                    self.fit_label.setText("\"Lorentzian\"")
                case 'Optical T1' | 'MW T1':
                    self.fit_label.setText("\"Stretched Exponential\"")
                case 'T2':
                    self.fit_label.setText("\"Mod. Stretched Exponential\"")
                case 'DEER':
                    self.fit_label.setText("\"Lorentzian\"")
                case 'DEER Rabi':
                    self.fit_label.setText("\"Decaying Cosine\"")
                case 'DEER T1':
                    self.fit_label.setText("\"Stretched Exponential\"")
                case 'NMR Correlation Spectroscopy':
                    self.fit_label.setText("\"Lorentzian\"")
                case 'NMR CASR':
                    self.fit_label.setText("\"Lorentzian\"")
                case 'Select experiment from list':
                    self.fit_label.setText("Fit type")
                case _:
                    self.fit_label.setText("Fit type not implemented.")

    def run(self):
        """Run the experiment function in a subprocess."""
        
        if self.run_proc.running():
            logging.info(
                'Not starting the experiment process because it is still running.'
            )
            
            return
        
        self.status.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
        self.status.setText(f"{self.experiments.currentText()} scan in progress...")

        # self.communicator = Communicate()
        # self.communicator_params = self.communicator.speak.connect(self.retrieve_exp_params)
        
        self.extra_kwarg_params['save'] = self.to_save
        try:
            self.extra_kwarg_params['dataset'] = self.exp_dict[self.experiments.currentText()][3]
        except KeyError as e:
            self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
            self.status.setText(f"No experiment selected: {e}")
            return 
        else:
            self.extra_kwarg_params['filename'] = self.filename_lineedit.text()
            full_dir = self.chosen_dir.property("full_path") or ""
        #     if not full_dir:
        #         # fallback: user didn't choose a directory yet
        #         full_dir = r"E:\Data"
            self.extra_kwarg_params["directory"] = full_dir
            self.extra_kwarg_params['seq'] = self.experiments.currentText()
            self.extra_kwarg_params['fit'] = self.to_fit
            self.extra_kwarg_params['fit_live'] = self.to_fit_live
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

            self._new_run_queues() # create new queues for new experiment run
            self.queue_to_exp.put('start') # start queue after creation of new queues to avoid any potential race conditions with old queues from previous runs

            # reload the module at runtime in case any changes were made to the code
            if self.daq_b1.isChecked(): # digitizer settings
                reload(nv_experiments_organized_2026_03_03)
                # call the function in a new process
                self.run_proc.run(
                    run_experiment,
                    exp_cls = nv_experiments_organized_2026_03_03.SpinMeasurements,
                    fun_name = self.exp_dict[self.experiments.currentText()][0],
                    constructor_args = list(),
                    constructor_kwargs = dict(),
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
                self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
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
            self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
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

