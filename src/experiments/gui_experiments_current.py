"""
Example GUI elements.
"""
# from tkinter.tix import ComboBox
from tracemalloc import stop
import numpy as np
from numpy.fft import fft, ifft
import logging
from functools import partial
from importlib import reload
from multiprocessing import Queue
from queue import Empty  # Import Empty exception from queue module
from typing import Optional

from inspect import signature
from scipy.optimize import curve_fit
from rpyc.utils.classic import obtain

# from nspyre import FlexLinePlotWidget, LinePlotWidget
from nspyre.gui.widgets.flex_line_plot_3 import FlexLinePlotWidget
from nspyre import DataSink
from pyqtgraph import SpinBox, ComboBox
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtWidgets

from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QRadioButton, QFileDialog, QProgressBar
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout
from PyQt6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

from nspyre.misc.misc import ProcessRunner
from nspyre.misc.misc import run_experiment
from nspyre import ParamsWidget
from nspyre import experiment_widget_process_queue
from nspyre import InstrumentManager

from gui_test import Communicate 

import nv_experiments_current
import nv_experiments_daq

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
        self.detector_opts = ["APD", "BPD", "PMT"]

        self.dig_ro_chan_opts = ["0", "1"]
        self.dig_coupling_opts = ["DC", "AC"]
        self.dig_termination_opts = ["1M", "50"]
        
        self.sigvstime_detector_opts = ["APD", "BPD", "PMT"]
        self.sigvstime_mw_detector_opts = ["APD", "BPD", "PMT"]

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
        
        self.sigvstime_params_defaults = [1e3, self.sigvstime_detector_opts]
        self.sigvstime_mw_params_defaults = [1e3, self.sigvstime_mw_detector_opts] # not needed - hidden in GUI

        self.laser_params_defaults = [30, 30e-6, 2.5e-6, 30e6, 0.45, self.sideband_opts, -0.002, -0.004, self.detector_opts]
        self.digitizer_defaults = [256, 100e6, 1, self.dig_ro_chan_opts, self.dig_coupling_opts, self.dig_termination_opts, 32, 5]

        # split up defaults into the non-MW and MW settings to save vertical space in GUI
        self.odmr_params_defaults = [120, 10, 50]
        self.odmr_mw_params_defaults = [2.87e9, 100e6, 1e-9, 25e-6]

        self.rabi_params_defaults = [120, 10, 0, 500e-9, 100]   
        self.rabi_mw_params_defaults = [2.87e9, 1e-9, self.rabi_axis_opts]

        self.pulsed_odmr_params_defaults = [120, 10, 50]
        self.pulsed_odmr_mw_params_defaults = [2.87e9, 100e6, 1e-9, 100e-9]
        
        self.pulsed_odmr_rf_params_defaults = [120, 10, 50]
        self.pulsed_odmr_rf_mw_params_defaults = [2.87e9, 100e6, 1e-9, 100e-9, 500e3, 0.3, 0]

        self.opt_t1_params_defaults = [120, 10, 50e-9, 100e-6, 100, self.opt_t1_array_opts]
        self.opt_t1_mw_params_defaults = [12, 10, 50e-9, 100e-6, 100, self.opt_t1_array_opts] # not needed - hidden in GUI

        self.mw_t1_params_defaults = [120, 10, 50e-9, 100e-6, 100, self.mw_t1_array_opts]
        self.mw_t1_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y']

        self.t2_params_defaults = [120, 10, 50e-9, 20e-6, 100, self.t2_array_opts]
        self.t2_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', self.t2_seq_opts, 1]

        self.t2_rf_params_defaults = [120, 10, 50e-9, 20e-6, 100, self.t2_rf_array_opts]
        self.t2_rf_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 'y', 1e6, 0.1, 0, 1]

        self.dq_params_defaults = [120, 10, 50e-9, 100e-6, 100, self.dq_array_opts]
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

        self.casr_params_defaults = [120, 10, 10, 1e-6]
        self.casr_mw_params_defaults = [2.87e9, 1e-9, 20e-9, 1]

        self.fit_none_default = [0]
        self.fit_odmr_defaults = [0.01, 2, 6e-3, 1] # defaults = 1% contrast, 2 GHz central freq, 6 MHz linewidth, 1 vertical offset
        self.fit_rabi_defaults = [0.02, 0.001, 200, 0, 1] # defaults = 2% contrast, 0.001 decay rate, 200 ns period, 0 phase, 1 vertical offset
        self.fit_t1_defaults = [0.01, 1, 1, 0] # defaults = 0.01 amplitude, 1 ms T1, 1 stretching factor, 0 vertical offset
        self.fit_t2_defaults = [0.1, 2, 1, 1, 0.2, 0, 1, 0.2, 0] # defaults = 0.1 amplitude, 2 us T2, 1 stretching factor, 1 amp first sine wave, 0.2 MHz first sine wave, 0 phase first sine wave, 1 amp second sine wave, 0.2 MHz second sine wave, 0 phase second sine wave
        self.fit_deer_defaults = [0.1, 560, 10, 1] # defaults = 10% contrast, 560 MHz central freq, 10 MHz linewidth, 1 vertical offset
        self.fit_deer_rabi_defaults = [0.1, 0.001, 100, 0, 1]

        # experiment dictionary - associates experiment function, default parameter array, dataset, laser parameters and digitizer parameters to an experiment type
        self.exp_dict = {"Signal vs Time": ["sigvstime_scan", self.sigvstime_params_defaults, self.sigvstime_mw_params_defaults, 'sigvstime', self.laser_params_defaults, self.digitizer_defaults], # ODMR MW params hidden and serves as placeholder for sig vs time experiment in GUI
                    "CW ODMR": ["odmr_scan", self.odmr_params_defaults, self.odmr_mw_params_defaults, 'odmr', self.laser_params_defaults, self.digitizer_defaults],
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
                    "NMR: Correlation Spectroscopy": ["Corr_Spec_scan", self.nmr_params_defaults, self.nmr_mw_params_defaults, 'nmr', self.laser_params_defaults, self.digitizer_defaults],
                    "NMR: CASR": ["CASR_scan", self.casr_params_defaults, self.casr_mw_params_defaults, 'nmr', self.laser_params_defaults, self.digitizer_defaults]}
        
        self.experiments = QComboBox()
        self.experiments.setFixedHeight(30)
        self.experiments.addItems(["Select an experiment from dropdown menu", 
                                 "Signal vs Time", 
                                 "CW ODMR", 
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
                                 "NMR: Correlation Spectroscopy",
                                 "NMR: CASR"])
        
        self.experiments.currentIndexChanged.connect(lambda: self.exp_selector())

        # used to send to experiment process to determine extra actions to take 
        self.to_save = False 
        self.to_fit = False
        self.to_fit_live = False
        self.extra_kwarg_params: dict() = {}

        # experiment params label
        self.exp_label = QLabel("Experiment Settings")
        self.exp_label.setFixedHeight(24)
        self.exp_label.setStyleSheet("font-weight: bold")
        
        self.params_widget = ParamsWidget(self.create_params_widget('CW ODMR', self.exp_dict['CW ODMR'][1]))

        self.mw_label = QLabel("Microwave Settings")
        self.mw_label.setFixedHeight(24)
        self.mw_label.setStyleSheet("font-weight: bold")

        self.mw_params_widget = ParamsWidget(self.create_mw_params_widget('CW ODMR', self.exp_dict['CW ODMR'][2]))

        self.opacity_effects = []
        for i in range(20):
            self.opacity_effects.append(QGraphicsOpacityEffect())
            self.opacity_effects[i].setOpacity(0.3)

        self.exp_label.setGraphicsEffect(self.opacity_effects[0])
        self.params_widget.setGraphicsEffect(self.opacity_effects[1])
        self.params_widget.setEnabled(False)
        
        # save params button
        self.save_params = QPushButton("Save Parameters")
        self.save_params.clicked.connect(lambda: self.save_params_clicked())
        self.save_params.setGraphicsEffect(self.opacity_effects[2])
        self.save_params.setEnabled(False)

        # mw params widget & label
        self.mw_label.setGraphicsEffect(self.opacity_effects[3])        
        self.mw_params_widget.setGraphicsEffect(self.opacity_effects[4])
        self.mw_params_widget.setEnabled(False)

        # laser params widget & label
        self.laser_params_widget = ParamsWidget(self.create_params_widget('Laser', self.exp_dict['Laser'][1]))
        self.laser_label = QLabel("Laser & IQ Settings")
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

        self.daq_b1 = QRadioButton("Digitizer")
        self.daq_b1.toggled.connect(lambda:self.toggle_daq(self.daq_b1))
        self.daq_b1.setGraphicsEffect(self.opacity_effects[9])
        self.daq_b1.setEnabled(False)

        self.daq_b2 = QRadioButton("NI DAQ")
        self.daq_b2.toggled.connect(lambda:self.toggle_daq(self.daq_b2))
        self.daq_b2.setGraphicsEffect(self.opacity_effects[10])
        self.daq_b2.setEnabled(False)

        # auto save checkbox
        self.auto_save_checkbox = QCheckBox("Auto Save")
        self.auto_save_checkbox.setChecked(False)
        self.auto_save_checkbox.stateChanged.connect(lambda: self.auto_save_changed())

        # select directory button
        self.select_dir_button = QPushButton("Select Directory")
        self.select_dir_button.setEnabled(False)
        self.select_dir_button.clicked.connect(lambda: self.select_directory())
        self.select_dir_button.setGraphicsEffect(self.opacity_effects[11])

        # selected directory display for saving
        self.chosen_dir = QLabel()
        self.chosen_dir.setGraphicsEffect(self.opacity_effects[12])
        self.chosen_dir.setStyleSheet("color: #ffa500")

        self.filename_label = QLabel("Filename: ")
        self.filename_label.setGraphicsEffect(self.opacity_effects[13])
        self.filename_label.setFixedHeight(20)

        self.filename_lineedit = QLineEdit()
        self.filename_lineedit.setGraphicsEffect(self.opacity_effects[14])
        self.filename_lineedit.setFixedHeight(30)
        self.filename_lineedit.setEnabled(False)

        # auto fit checkbox
        self.auto_fit_checkbox = QCheckBox("Auto Fit  ")
        self.auto_fit_checkbox.setChecked(False)
        self.auto_fit_checkbox.stateChanged.connect(lambda: self.auto_fit_changed())

        # fit type label
        self.fit_label = QLabel("Fit Type: ")
        self.fit_label.setGraphicsEffect(self.opacity_effects[15]) 
        self.fit_label.setFixedHeight(20)

        self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit ODMR', self.fit_odmr_defaults))
        self.fit_params_widget.setGraphicsEffect(self.opacity_effects[16])
        self.fit_params_widget.setEnabled(False)

        # fit value labels
        self.fit_val_label_1 = QLabel("Fitted Value:")
        self.fit_val_label_1.setGraphicsEffect(self.opacity_effects[17]) 
        self.fit_val_label_1.setFixedHeight(20)
        self.fit_val_label_2 = QLabel()
        self.fit_val_label_2.setGraphicsEffect(self.opacity_effects[18]) 
        self.fit_val_label_2.setFixedHeight(20)
        
        # live fitting checkbox
        self.live_fit_checkbox = QCheckBox("Live Fitting")
        self.live_fit_checkbox.setChecked(False)
        self.live_fit_checkbox.stateChanged.connect(lambda: self.live_fit_changed())
        self.live_fit_checkbox.setGraphicsEffect(self.opacity_effects[19])
        self.live_fit_checkbox.setEnabled(False)

        # status label
        self.status = QLabel("Select parameters and press 'Run' to begin experiment.")
        self.status.setStyleSheet("color: black; background-color: #00b8ff; border: 4px solid black;")
        self.status.setFixedHeight(40)
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # run button
        run_button = QPushButton('Run')
        run_button.setStyleSheet("border: 2px solid limegreen")
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
        stop_button.setStyleSheet("border: 2px solid red")
        stop_button.clicked.connect(self.stop)
        # use a partial because the stop function may already be destroyed by the time
        # this is called
        self.destroyed.connect(partial(self.stop, log=False))

        self.gui_layout = QVBoxLayout()
        
        self.top_frame = QFrame(self)
        self.top_frame.setStyleSheet("background-color: #1e1e1e")
        self.top_layout = QVBoxLayout(self.top_frame)
        self.top_layout.setSpacing(0)
        self.top_layout.addWidget(self.experiments)

        self.exp_frame = QFrame(self)
        self.exp_frame.setStyleSheet("background-color: #4b0000")
        self.exp_params_layout = QVBoxLayout(self.exp_frame)
        self.exp_params_layout.setSpacing(0)
        self.exp_params_layout.addWidget(self.exp_label)
        self.exp_params_layout.addWidget(self.params_widget)
        self.exp_params_layout.addWidget(self.save_params)

        self.mw_frame = QFrame(self)
        self.mw_frame.setStyleSheet("background-color: #474b00")
        self.mw_params_layout = QVBoxLayout(self.mw_frame)
        self.mw_params_layout.setSpacing(0)
        self.mw_params_layout.addWidget(self.mw_label)
        self.mw_params_layout.addWidget(self.mw_params_widget)

        self.save_frame = QFrame(self)
        self.save_frame.setStyleSheet("background-color: #1e1e1e")
        self.save_layout = QGridLayout(self.save_frame)
        self.save_layout.setSpacing(0)
        self.save_layout.addWidget(self.auto_save_checkbox,1,1,1,1)
        self.save_layout.addWidget(self.select_dir_button,1,2,1,1)
        self.save_layout.addWidget(self.chosen_dir,2,1,1,2)
        self.save_layout.addWidget(self.filename_label,3,1,1,1)
        self.save_layout.addWidget(self.filename_lineedit,3,2,1,1)
        
        self.fit_frame = QFrame(self)
        self.fit_frame.setStyleSheet("background-color: #1e1e1e")
        self.fit_layout = QGridLayout(self.fit_frame)
        self.fit_layout.setSpacing(0)
        self.fit_layout.addWidget(self.auto_fit_checkbox,1,1,1,1)
        self.fit_layout.addWidget(self.fit_label,1,2,1,2)
        self.fit_layout.addWidget(self.fit_params_widget,2,1,1,3)
        self.fit_layout.addWidget(self.fit_val_label_1,3,1,1,1)
        self.fit_layout.addWidget(self.fit_val_label_2,3,2,1,1)
        self.fit_layout.addWidget(self.live_fit_checkbox,3,3,1,1)

        self.bottom_frame = QFrame(self)
        self.bottom_frame.setStyleSheet("background-color: black")
        self.bottom_layout = QGridLayout(self.bottom_frame)
        self.bottom_layout.setSpacing(0)
        self.bottom_layout.addWidget(self.status,1,1,1,2)
        self.bottom_layout.addWidget(self.progress_bar,2,1,1,2)
        self.bottom_layout.addWidget(run_button,3,1,1,1)
        self.bottom_layout.addWidget(stop_button,3,2,1,1)

        self.laser_frame = QFrame(self)
        self.laser_frame.setStyleSheet("background-color: #004b47")
        self.laser_params_layout = QGridLayout(self.laser_frame)
        self.laser_params_layout.setSpacing(0)
        self.laser_params_layout.addWidget(self.laser_label,1,1,1,2)
        self.laser_params_layout.addWidget(self.laser_params_widget,2,1,1,2)
        self.laser_params_layout.addWidget(self.daq_b1,3,1,1,1)
        self.laser_params_layout.addWidget(self.daq_b2,3,2,1,1)        

        self.dig_frame = QFrame(self)
        self.dig_frame.setStyleSheet("background-color: #000b4b")
        self.dig_params_layout = QVBoxLayout(self.dig_frame)
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

        self.save_widgets_layout = QHBoxLayout()
        self.save_widgets_layout.addWidget(self.save_frame)
        self.save_widgets_layout.addWidget(self.fit_frame)

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
            case 'Laser':    
                params = {
                        'laser_power': {'display_text': 'Power (%): ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(0, 110), dec = True)},
                        'laser_init': {'display_text': 'Initialization Time (pulsed): ',
                                'widget': SpinBox(value = defaults[1], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'laser_readout': {'display_text': 'Readout Time (pulsed): ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'sideband_freq': {'display_text': 'Sideband Mod. Frequency: ',
                                        'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100, 100e6), dec = True)},
                        'sideband_power': {'display_text': 'Sideband Power: ',
                                        'widget': SpinBox(value = defaults[4], suffix = 'V', siPrefix = True)},
                        'sideband': {'display_text': 'Sideband: ',
                                        'widget': ComboBox(items = defaults[5])},
                        'i_offset': {'display_text': 'I Offset: ',
                                        'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True)},
                        'q_offset': {'display_text': 'Q Offset: ',
                                        'widget': SpinBox(value = defaults[7], suffix = 'V', siPrefix = True)},
                        'detector': {'display_text': 'Detector: ',
                                        'widget': ComboBox(items = defaults[8])}}                
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
                        'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (10, 1e6), dec = True)},
                'sigvstime_detector': {'display_text': 'Detector: ',
                                        'widget': ComboBox(items = defaults[1])}}
            
            case 'CW ODMR':    
                params = {
                        'runs': {'display_text': '# Averages: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)}}                    
            case 'Pulsed ODMR':    
                params = {
                        'runs': {'display_text': '# Averages: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)}}            
            case 'RF Coil: Pulsed ODMR':    
                params = {
                        'runs': {'display_text': '# Averages: ',
                                'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                        'iters': {'display_text': '# Experiment Iterations: ',
                                'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                        'num_pts': {'display_text': '# Frequencies: ',
                                'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)}}         
            case 'Rabi':    
                params = {
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
                        'runs': {'display_text': '# Averages: ',
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
            case 'NMR: Correlation Spectroscopy':    
                params = {
                'runs': {'display_text': '# Averages: ',
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
            case 'NMR: CASR':
                params = {
                'runs': {'display_text': 'Runs (avgs. per iteration): ',
                        'widget': SpinBox(value = defaults[0], int = True, bounds=(1, None))},
                'iters': {'display_text': '# Experiment Iterations: ',
                        'widget': SpinBox(value = defaults[1], int = True, bounds=(1, None))},
                'num_pts': {'display_text': 'n_sr (# synch. readout pts.): ',
                        'widget': SpinBox(value = defaults[2], int = True, bounds=(1, None), dec = True)},
                'central_freq': {'display_text': 'f_0 Central Frequency: ',
                        'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (1e3, 6e9), dec = True)},
                'tau': {'display_text': 'Free Precession Interval (\u03C4): ',
                        'widget': SpinBox(value = defaults[4], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}

            case 'Fit None':
                params = {
                        'A': {'display_text': 'Fit params here',
                                'widget': SpinBox(value = defaults[0])}}
            case 'Fit ODMR': # -A / (1 + ((x - x0) / gamma) ** 2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'x0': {'display_text': 'x0 (GHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'gamma': {'display_text': '\u03B3 (GHz): ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])}}
            case 'Fit Rabi':
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
            case 'Fit T1':
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
            case 'Fit DEER': # -A / (1 + ((x - x0) / gamma) ** 2) + c
                params = {
                        'A': {'display_text': 'A: ',
                                'widget': SpinBox(value = defaults[0])},
                        'x0': {'display_text': 'x0 (GHz): ',
                                'widget': SpinBox(value = defaults[1])},
                        'gamma': {'display_text': '\u03B3 (GHz): ',
                                'widget': SpinBox(value = defaults[2])},
                        'c': {'display_text': 'c: ',
                                'widget': SpinBox(value = defaults[3])}}
            case 'Fit DEER Rabi':
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
        return params

    def create_mw_params_widget(self, tag, defaults):
        match tag:
            case 'Signal vs Time': # not used - hidden
                params = {
                'sampling_rate_0': {'display_text': 'Sampling Rate: ',
                        'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (10, 1e6), dec = True)},
                'sigvstime_detector_0': {'display_text': 'Detector: ',
                                        'widget': ComboBox(items = defaults[1])}}
            case 'CW ODMR':
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'W', siPrefix = True)},
                        'probe': {'display_text': 'MW Probe Time: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (10e-9, None))}}    
            case 'Pulsed ODMR':    
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)}}
            case 'RF Coil: Pulsed ODMR':    
                params = {
                        'center_freq': {'display_text': 'Center Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'half_span_sideband_freq': {'display_text': 'Half Frequency Span: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[2], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[3], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'rf_pulse_freq': {'display_text': 'RF Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 100e6), dec = True)},
                        'rf_pulse_power': {'display_text': 'RF Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)},
                        'rf_pulse_phase': {'display_text': 'RF Phase (deg.): ',
                                'widget': SpinBox(value = defaults[6], int = True, bounds=(0, 360))}}
            case 'Rabi':    
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': ComboBox(items = defaults[2])}}
            case 'Optical T1': # not used - hidden
                params = {
                        'runs': {'display_text': '# Averages: ',
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
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])}}
            case 'T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        't2_seq': {'display_text': 'Sequence: ',
                                'widget': ComboBox(items = defaults[4])},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[5], int = True, bounds=(1, None))}}
            case 'RF Coil: T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
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
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi_minus': {'display_text': '\u03C0 Pulse |-1>: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'freq_plus': {'display_text': 'NV Frequency |+1>: ',
                                'widget': SpinBox(value = defaults[3], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power_plus': {'display_text': 'NV MW Power |+1>: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'W', siPrefix = True)},
                        'pi_plus': {'display_text': '\u03C0 Pulse |+1>: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[6])}}
            case 'DEER':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'V', siPrefix = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'drive_type': {'display_text': 'Dark MW Driving',
                                'widget': ComboBox(items = defaults[6])}}
            case 'DEER Rabi':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)}}
            case 'DEER FID':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[7], int = True, bounds=(1, None))}}      
            case 'DEER FID Continuous Drive':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[5], suffix = 'V', siPrefix = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[6], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_cd_power': {'display_text': 'Continuous Drive (AWG) Power: ',
                                'widget': SpinBox(value = defaults[7], suffix = 'V', siPrefix = True)},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[8], int = True, bounds=(1, None))}}              
            case 'DEER Correlation Rabi':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True)}}
            case 'DEER T1':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True)}}
            case 'DEER T2':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV (SRS) Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': 'NV \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'NV Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'dark_freq': {'display_text': 'Dark Frequency: ',
                                'widget': SpinBox(value = defaults[4], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'dark_pi': {'display_text': 'Dark \u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[5], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'awg_power': {'display_text': 'Dark (AWG) Power: ',
                                'widget': SpinBox(value = defaults[6], suffix = 'V', siPrefix = True)}}
            case 'NMR: Correlation Spectroscopy':    
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'pulse_axis': {'display_text': 'Pulse Axis',
                                'widget': QtWidgets.QLineEdit(defaults[3])},
                        'n': {'display_text': '# Seqs. (n): ',
                                'widget': SpinBox(value = defaults[4], int = True, bounds=(1, None))}}    
            case 'NMR: CASR':
                params = {
                        'freq': {'display_text': 'NV Frequency: ',
                                'widget': SpinBox(value = defaults[0], suffix = 'Hz', siPrefix = True, bounds = (100e3, 6e9), dec = True)},
                        'rf_power': {'display_text': 'NV MW Power: ',
                                'widget': SpinBox(value = defaults[1], suffix = 'W', siPrefix = True)},
                        'pi': {'display_text': '\u03C0 Pulse: ',
                                'widget': SpinBox(value = defaults[2], suffix = 's', siPrefix = True, bounds = (0, None), dec = True)},
                        'n': {'display_text': 'XY8-N (# \u03C0 pulses): ',
                                'widget': SpinBox(value = defaults[3], int = True, bounds=(1, None))}}

        return params

    def check_queue_from_exp(self):
        # queue checker to control progress bar display
        try:
            if self.queue_from_exp is None:
                print("Queue is None. Stopping queue checks.")
                return

            while not self.queue_from_exp.empty():  # If there is something in the queue
                try:
                    queueText = self.queue_from_exp.get_nowait()  # Get it
                except (OSError, EOFError) as e:
                    print(f"Queue connection error: {e}. Stopping queue checks.")
                    self.queue_from_exp = None  # Mark queue as invalid
                    return
                except Empty:
                    return  # Queue is empty, exit function safely

                self.progress_bar.setValue(int(queueText[0]))

                if queueText[1] == 'in progress':
                    self.status.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
                    self.status.setText(f"{self.experiments.currentText()} scan in progress...")
                    self.experiments.setEnabled(False)
                    self.params_widget.setEnabled(False)
                    self.save_params.setEnabled(False)
                    self.laser_params_widget.setEnabled(False)
                    self.dig_params_widget.setEnabled(False)
                elif queueText[1] == 'complete':
                    self.status.setStyleSheet("color: black; background-color: limegreen; border: 4px solid black;")
                    self.status.setText(f"{self.experiments.currentText()} scan complete.")
                    self.experiments.setEnabled(True)
                    self.params_widget.setEnabled(True)
                    self.save_params.setEnabled(True)
                    self.laser_params_widget.setEnabled(True)
                    self.dig_params_widget.setEnabled(True)
                elif queueText[1] == 'failed':
                    self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
                    self.status.setText(f"{self.experiments.currentText()} scan failed. Exception type '{queueText[3]}'.")
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

                if queueText[2] is not None:
                    self.fit_val_label_2.setText(f"{queueText[2]}")

        except Exception as e:
            print(f"Unexpected error in check_queue_from_exp: {e}")
            self.queue_from_exp = None  # If any unexpected error occurs, stop checking queue

        # Restart the timer only if the queue is still valid
        if self.queue_from_exp is not None:
            self.updateTimer.start(self.QUEUE_CHECK_TIME)

#     def check_queue_from_exp(self):
#         # queue checker to control progress bar display
#         while not self.queue_from_exp.empty(): #if there is something in the queue
#             queueText = self.queue_from_exp.get_nowait() #get it

#             self.progress_bar.setValue(int(queueText[0]))
            
#             if queueText[1] == 'in progress':
#                 self.status.setStyleSheet("color: black; background-color: gold; border: 4px solid black;")
#                 # self.status.setText(f"{self.experiments.currentText()} scan in progress... ({round(float(queueText[1]),2)} s/it)")
#                 self.status.setText(f"{self.experiments.currentText()} scan in progress...")
#                 self.experiments.setEnabled(False) # disable widgets during experiment
#                 self.params_widget.setEnabled(False)
#                 self.save_params.setEnabled(False)
#                 self.laser_params_widget.setEnabled(False)
#                 self.dig_params_widget.setEnabled(False)
#             elif queueText[1] == 'complete':
#                     self.status.setStyleSheet("color: black; background-color: limegreen; border: 4px solid black;")
#                     self.status.setText(f"{self.experiments.currentText()} scan complete.")
#                     self.experiments.setEnabled(True)
#                     self.params_widget.setEnabled(True)
#                     self.save_params.setEnabled(True)
#                     self.laser_params_widget.setEnabled(True)
#                     self.dig_params_widget.setEnabled(True)
#             elif queueText[1] == 'failed':
#                     self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
#                     self.status.setText(f"{self.experiments.currentText()} scan failed. Exception type '{queueText[3]}'.")
#                     self.experiments.setEnabled(True)
#                     self.params_widget.setEnabled(True)
#                     self.save_params.setEnabled(True)
#                     self.laser_params_widget.setEnabled(True)
#                     self.dig_params_widget.setEnabled(True)
#             else:
#                 self.status.setStyleSheet("color: black; background-color: white; border: 4px solid black;")
#                 self.status.setText(f"{self.experiments.currentText()} scan stopped.")
#                 self.experiments.setEnabled(True)
#                 self.params_widget.setEnabled(True)
#                 self.save_params.setEnabled(True)
#                 self.laser_params_widget.setEnabled(True)
#                 self.dig_params_widget.setEnabled(True)

#             if queueText[2] is not None:
#                 self.fit_val_label_2.setText(f"{queueText[2]}")

#         self.updateTimer.start(self.QUEUE_CHECK_TIME)

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

    def auto_save_changed(self):
        if self.auto_save_checkbox.isChecked() == True:
            self.to_save = True
            self.select_dir_button.setEnabled(True)
            self.filename_lineedit.setEnabled(True)
            # self.opacity_effects[6].setEnabled(False) # change to 6? laser params widget
            self.opacity_effects[11].setEnabled(False)
            self.opacity_effects[12].setEnabled(False)
            self.opacity_effects[13].setEnabled(False)
            self.opacity_effects[14].setEnabled(False)

        else:
            self.to_save = False
            self.select_dir_button.setEnabled(False)
            self.filename_lineedit.setEnabled(False)
            # self.opacity_effects[6].setEnabled(True) # change to 6?
            self.opacity_effects[11].setEnabled(True)
            self.opacity_effects[12].setEnabled(True)
            self.opacity_effects[13].setEnabled(True)
            self.opacity_effects[14].setEnabled(True)

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
            self.detector_opts.insert(0, self.detector_opts.pop(self.detector_opts.index(saved_laser_params[8])))
            saved_laser_params[8] = self.detector_opts

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
            case 'Signal vs Time':
                self.sigvstime_detector_opts.insert(0, self.sigvstime_detector_opts.pop(self.sigvstime_detector_opts.index(saved_params[1])))
                saved_params[1] = self.sigvstime_detector_opts
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
        #     case 'NMR':
        #         self.nmr_seq_opts.insert(0, self.nmr_seq_opts.pop(self.nmr_seq_opts.index(saved_params[10])))
        #         saved_params[10] = self.nmr_seq_opts

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
            self.opacity_effects[15].setEnabled(False)
            self.opacity_effects[16].setEnabled(False)
            self.opacity_effects[17].setEnabled(False)
            self.opacity_effects[18].setEnabled(False)
            self.opacity_effects[19].setEnabled(False)
        else:
            self.to_fit = False
            self.fit_params_widget.setEnabled(False)
            self.live_fit_checkbox.setEnabled(False)
            self.live_fit_checkbox.setChecked(False)
            self.to_fit_live = False
            self.opacity_effects[15].setEnabled(True)
            self.opacity_effects[16].setEnabled(True)
            self.opacity_effects[17].setEnabled(True)
            self.opacity_effects[18].setEnabled(True)
            self.opacity_effects[19].setEnabled(True)

    def live_fit_changed(self):
        if self.live_fit_checkbox.isChecked() == True:
            self.to_fit_live = True
        else:
            self.to_fit_live = False

    def select_directory(self):
        response = QFileDialog.getExistingDirectory(self, caption = "Select a folder")
        
        self.chosen_dir.setText(str(response) + "/")

    def get_lineedit_val(self, lineedit):
        return lineedit.text()
    
    def get_combobox_val(self, combobox):
        return str(combobox.value())
    
    def exp_selector(self):
        # reset params widgets each time new experiment selected
        self.status.setStyleSheet("color: black; background-color: #00b8ff; border: 4px solid black;")
        self.status.setText("Select parameters and press 'Run' to begin experiment.")
        self.progress_bar.setValue(0)

        self.exp_label.hide()
        self.params_widget.hide()
        self.mw_label.hide()
        self.mw_params_widget.hide()
        self.save_params.hide()
        self.laser_label.hide()
        self.laser_params_widget.hide() 
        self.daq_b1.hide()
        self.daq_b2.hide()
        self.dig_label.hide()
        self.dig_params_widget.hide()
        self.fit_params_widget.hide()

        try:
            self.params_widget = ParamsWidget(self.create_params_widget(self.experiments.currentText(), self.exp_dict[self.experiments.currentText()][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.mw_params_widget = ParamsWidget(self.create_mw_params_widget(self.experiments.currentText(), self.exp_dict[self.experiments.currentText()][2]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            
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
            self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit None', self.fit_none_default), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.fit_params_widget.setGraphicsEffect(self.opacity_effects[16]) # reset opacity effects
            for i in range(11):
                self.opacity_effects[i].setEnabled(True)

            self.params_widget.setEnabled(False)
            self.save_params.setEnabled(False)
            self.mw_params_widget.setEnabled(False)

            self.laser_params_widget.setEnabled(False)
            self.daq_b1.setEnabled(False)
            self.daq_b2.setEnabled(False)
            self.dig_params_widget.setEnabled(False)
            self.fit_params_widget.setEnabled(False)
        else:
            self.params_widget.setEnabled(True)
            self.save_params.setEnabled(True)
            self.mw_params_widget.setEnabled(True)
            self.laser_params_widget.setEnabled(True)
            self.daq_b1.setEnabled(True)
            self.daq_b2.setEnabled(True)

            for i in range(11):
                if i == 7 or i == 8:
                    if self.daq_b1.isChecked(): # digitizer settings
                        self.opacity_effects[i].setEnabled(False)
                        self.dig_params_widget.setEnabled(True)
                    elif self.daq_b2.isChecked(): # NI DAQ settings
                        self.opacity_effects[i].setEnabled(True)
                        self.dig_params_widget.setEnabled(False)
                else:
                    self.opacity_effects[i].setEnabled(False)

            self.laser_params_widget = ParamsWidget(self.create_params_widget('Laser', self.exp_dict['Laser'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.laser_params_widget.setGraphicsEffect(self.opacity_effects[6]) # reset opacity effects
            self.dig_params_widget = ParamsWidget(self.create_params_widget('Digitizer', self.exp_dict['Digitizer'][1]), get_param_value_funs = {ComboBox: self.get_combobox_val})
            self.dig_params_widget.setGraphicsEffect(self.opacity_effects[8]) # reset opacity effects
            self.dig_params_widget.setEnabled(False) # TODO: check if this is how i want dig params widget to actually behave when new experiment selected

            match self.experiments.currentText():
                case 'CW ODMR' | 'Pulsed ODMR':
                    self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit ODMR', self.fit_odmr_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Rabi':
                    self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit Rabi', self.fit_rabi_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})
                case 'Optical T1' | 'MW T1':
                    self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit T1', self.fit_t1_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})    
                case 'T2':
                    self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit T2', self.fit_t2_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})  
                case 'DEER':
                    self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit DEER', self.fit_deer_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val})    
                case 'DEER Rabi':
                    self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit DEER Rabi', self.fit_deer_rabi_defaults), get_param_value_funs = {ComboBox: self.get_combobox_val}) 
                case _:
                    self.fit_params_widget = ParamsWidget(self.create_params_widget('Fit None', self.fit_none_default), get_param_value_funs = {ComboBox: self.get_combobox_val})

            self.fit_params_widget.setGraphicsEffect(self.opacity_effects[16]) # reset opacity effects
            self.fit_params_widget.setEnabled(False)
            
        finally:
            # reinstate GUI widgets
            self.exp_params_layout.insertWidget(0, self.exp_label)
            self.exp_params_layout.insertWidget(1, self.params_widget)
            self.exp_params_layout.insertWidget(2, self.save_params)
            self.mw_params_layout.insertWidget(0, self.mw_label)
            self.mw_params_layout.insertWidget(1, self.mw_params_widget)
            self.laser_params_layout.addWidget(self.laser_label,1,1,1,2)
            self.laser_params_layout.addWidget(self.laser_params_widget,2,1,1,2)
            self.laser_params_layout.addWidget(self.daq_b1,3,1,1,1)
            self.laser_params_layout.addWidget(self.daq_b2,3,2,1,1)
            self.dig_params_layout.insertWidget(0, self.dig_label)
            self.dig_params_layout.insertWidget(1, self.dig_params_widget)
            self.fit_layout.addWidget(self.auto_fit_checkbox,1,1,1,1)
            self.fit_layout.addWidget(self.fit_label,1,2,1,2)
            self.fit_layout.addWidget(self.fit_params_widget,2,1,1,3)
            self.fit_layout.addWidget(self.fit_val_label_1,3,1,1,1)
            self.fit_layout.addWidget(self.fit_val_label_2,3,2,1,1)
            self.fit_layout.addWidget(self.live_fit_checkbox,3,3,1,1)

            self.exp_label.show()
            self.save_params.show()
            self.params_widget.show()

            if self.experiments.currentText() == 'Signal vs Time': # hide MW, laser and digitizer params widgets
                self.mw_label.hide()
                self.mw_params_widget.hide()
                self.laser_label.hide()
                self.laser_params_widget.hide()
                self.daq_b1.show()
                self.daq_b2.show()
                self.dig_label.hide()
                self.dig_params_widget.hide()
                self.auto_fit_checkbox.hide()
                self.fit_label.hide()
                self.fit_params_widget.hide()
                self.fit_val_label_1.hide()
                self.fit_val_label_2.hide()
                self.live_fit_checkbox.hide()

            elif self.experiments.currentText() == 'Optical T1': # only hide MW params widget
                self.mw_label.hide()
                self.mw_params_widget.hide()
                self.laser_label.show()
                self.laser_params_widget.show()
                self.daq_b1.show()
                self.daq_b2.show()
                self.dig_label.show()
                self.dig_params_widget.show()
                self.auto_fit_checkbox.show()
                self.fit_label.show()
                self.fit_params_widget.show()
                self.fit_val_label_1.show()
                self.fit_val_label_2.show()
                self.live_fit_checkbox.show()

            else: # rest of experiments don't hide any widgets
                self.mw_label.show()
                self.mw_params_widget.show()
                self.laser_label.show()
                self.laser_params_widget.show()
                self.daq_b1.show()
                self.daq_b2.show()
                self.dig_label.show()
                self.dig_params_widget.show()
                self.auto_fit_checkbox.show()
                self.fit_label.show()
                self.fit_params_widget.show()
                self.fit_val_label_1.show()
                self.fit_val_label_2.show()
                self.live_fit_checkbox.show()
            
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
                    self.fit_label.setText("-A/(1 + ((x - x0)/\u03B3)^2) + c")
                    self.fit_val_label_1.setText("Fitted ODMR (GHz):  ")
                case 'Rabi':
                    self.fit_label.setText("A*exp(-\u03B3 t)*cos(2\u03C0 t/T + \u03C6) + c")
                    self.fit_val_label_1.setText("Fitted Rabi (ns):  ")
                case 'Optical T1' | 'MW T1':
                    self.fit_label.setText("A*exp(-(t/T)^n) + c")
                    self.fit_val_label_1.setText("Fitted T1 (ms):  ")
                case 'T2':
                    self.fit_label.setText("A*exp(-(t/T)^n)*(1-a1*sin(2\u03C0 f1/4 + \u03C61)^2)*(1-a2*sin(2\u03C0 f2/4 + \u03C62)^2)")
                    self.fit_val_label_1.setText("Fitted T2 (\u03BCs):  ")
                case 'DEER':
                    self.fit_label.setText("-A/(1 + ((x - x0)/\u03B3)^2 + c)")
                    self.fit_val_label_1.setText("Fitted DEER (MHz):  ")
                case 'DEER Rabi':
                    self.fit_label.setText("A*exp(-\u03B3 t)*cos(2\u03C0 t/T + \u03C6) + c")
                    self.fit_val_label_1.setText("Fitted DEER Rabi (ns):  ")
                case _:
                    self.fit_label.setText("Fit type not implemented.")
                    self.fit_val_label_1.setText("Fitted Value: None")
                    self.fit_val_label_2.setText(" ")

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
        self.extra_kwarg_params['dataset'] = self.exp_dict[self.experiments.currentText()][3]
        self.extra_kwarg_params['filename'] = self.filename_lineedit.text()
        self.extra_kwarg_params['directory'] = self.chosen_dir.text()
        self.extra_kwarg_params['seq'] = self.experiments.currentText()
        self.extra_kwarg_params['fit'] = self.to_fit
        self.extra_kwarg_params['fit_live'] = self.to_fit_live
        self.extra_kwarg_params['fit_params'] = list(self.fit_params_widget.all_params().values()) # send a list of fit parameters to experiment process 

        # unpack all keyword arg parameters to send to experiment process
        fun_kwargs = dict(**self.params_widget.all_params(), **self.mw_params_widget.all_params(), 
                          **self.laser_params_widget.all_params(), **self.dig_params_widget.all_params(), **self.extra_kwarg_params)

        self.queue_to_exp.put('start')

        # **Restart the queue if it was invalid**
        if self.queue_from_exp is None:
            print("Reinitializing queue for new experiment.")
            self.queue_from_exp: Queue = Queue() # Recreate the queue

        # reload the module at runtime in case any changes were made to the code
        if self.daq_b1.isChecked(): # digitizer settings
            reload(nv_experiments_current)
            # call the function in a new process
            self.run_proc.run(
            run_experiment,
            exp_cls = nv_experiments_current.SpinMeasurements,
            fun_name = self.exp_dict[self.experiments.currentText()][0],
            constructor_args = list(),
            constructor_kwargs = dict(),
            queue_to_exp = self.queue_to_exp,
            queue_from_exp = self.queue_from_exp,
            fun_args = list(),
            fun_kwargs = fun_kwargs)
        
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
            fun_kwargs = fun_kwargs)

        else:
            self.status.setStyleSheet("color: black; background-color: red; border: 4px solid black;")
            self.status.setText(f"{self.experiments.currentText()} scan couldn't start because no acquisition mode selected. Choose either 'Digitizer' or 'NI DAQ'.")
            raise ValueError(f"{self.experiments.currentText()} scan couldn't start because no data acquisition mode selected. Choose either 'Digitizer' or 'NI DAQ'.") 
        
        # **Restart the timer if the queue is valid**
        if self.queue_from_exp is not None:
            print("Restarting updateTimer to check the queue.")
            self.updateTimer.start(self.QUEUE_CHECK_TIME)

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

