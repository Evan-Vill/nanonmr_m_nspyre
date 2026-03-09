import logging
import time
import random
from typing import Callable
from typing import Optional
from typing import List
from typing import Tuple

import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfft, rfftfreq
from scipy.optimize import curve_fit

import pyqtgraph as pg
from pyqtgraph import InfiniteLine, SignalProxy
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtGui
from pyqtgraph.Qt import QtWidgets
from PyQt6.QtGui import QColor

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout
from PyQt6.QtWidgets import QWidget, QComboBox, QPushButton, QLabel, QInputDialog, QLineEdit, QTableWidget

from nspyre.data.sink import DataSink
from nspyre.gui.threadsafe import QThreadSafeObject
from nspyre.gui.widgets.layout import tree_layout
from line_plot_2026_03_05 import LinePlotWidget
from nspyre_colors_2026_03_05 import cyclic_colors

_logger = logging.getLogger(__name__)

def boxcar_complex_smooth(y: np.ndarray, width: int) -> np.ndarray:
    """Boxcar smooth a complex 1D array in frequency space."""
    y = np.asarray(y)
    if width <= 1:
        return y.copy()
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(y, kernel, mode='same')

class FitCurveDialog(QWidget):
    
    fit_parameters = pyqtSignal(list)

    def __init__(self, parent = None):
        super(FitCurveDialog, self).__init__(parent)
        
        self.layout = QFormLayout()

        self.fit_type_button = QPushButton("Choose fit type")
        self.fit_type_button.clicked.connect(self.get_fit_type)
        self.fit_choice = QLineEdit()
        self.layout.addRow(self.fit_type_button, self.fit_choice)

        self.func_label = QLabel("Function:")
        self.func_form_label = QLabel()
        self.layout.addRow(self.func_label, self.func_form_label)

        self.setLayout(self.layout)

    def fit(self):
        if self.fit_choice.text() == "Exponential":
            self.fit_parameters.emit([self.fit_choice.text(), self.A_value.text(), self.T_value.text()])

        elif self.fit_choice.text() == "Exponential (stretched)":
            self.fit_parameters.emit([self.fit_choice.text(), self.A_value.text(), self.T_value.text(), self.N_value.text()])

        elif self.fit_choice.text() == "Linear":
            self.fit_parameters.emit([self.fit_choice.text(), self.m_value.text(), self.b_value.text()])

        elif self.fit_choice.text() == "Trig":
            self.fit_parameters.emit([self.fit_choice.text(), self.A_value.text(), self.w_value.text(), self.phi_value.text(), self.y0_value.text()])

        elif self.fit_choice.text() == "Trig Decay":
            self.fit_parameters.emit([self.fit_choice.text(), self.A_value.text(), self.lamb_value.text(), self.w_value.text(), self.phi_value.text(), self.y0_value.text()])

        elif self.fit_choice.text() == "Lorentzian":
            self.fit_parameters.emit([self.fit_choice.text(), self.A_value.text(), self.x0_value.text(), self.gamma_value.text(), self.y0_value.text()])

        elif self.fit_choice.text() == "Gaussian":
            self.fit_parameters.emit([self.fit_choice.text(), self.A_value.text(), self.x0_value.text(), self.sigma_value.text(), self.y0_value.text()])
        self.close()

    def refresh_rows(self):
        row_count = self.layout.rowCount()
        while row_count > 3:
            self.layout.removeRow(row_count - 1)
            row_count -= 1

    def get_fit_type(self):
        options = ("Exponential", "Exponential (stretched)", "Linear", "Trig", "Trig Decay", "Lorentzian", "Gaussian")

        opt, ok = QInputDialog.getItem(self, "Select Fit Type", "List of Fits", options, 0, False)

        if opt and ok:
            self.fit_choice.setText(opt)  

        if self.fit_choice.text() == "Exponential":
            self.refresh_rows()

            self.func_form_label.setText("A exp(-t/T)")
            
            self.A_label = QLabel("A = ")
            self.A_value = QLineEdit()
            self.layout.addRow(self.A_label, self.A_value)

            self.T_label = QLabel("T = ")
            self.T_value = QLineEdit()
            self.layout.addRow(self.T_label, self.T_value)
            
            self.fit_execute_button = QPushButton("Fit")
            self.fit_execute_button.clicked.connect(self.fit)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)
            self.layout.addRow(self.fit_execute_button, self.cancel_button)

        elif self.fit_choice.text() == "Exponential (stretched)":
            self.refresh_rows()

            self.func_form_label.setText("A exp((-t/T)^N)")
            
            self.A_label = QLabel("A = ")
            self.A_value = QLineEdit()
            self.layout.addRow(self.A_label, self.A_value)

            self.T_label = QLabel("T = ")
            self.T_value = QLineEdit()
            self.layout.addRow(self.T_label, self.T_value)

            self.N_label = QLabel("N = ")
            self.N_value = QLineEdit()
            self.layout.addRow(self.N_label, self.N_value)

            self.fit_execute_button = QPushButton("Fit")
            self.fit_execute_button.clicked.connect(self.fit)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)
            self.layout.addRow(self.fit_execute_button, self.cancel_button)

        elif self.fit_choice.text() == "Linear":
            self.refresh_rows()

            self.func_form_label.setText("mx + b")
            
            self.m_label = QLabel("m = ")
            self.m_value = QLineEdit()
            self.layout.addRow(self.m_label, self.m_value)

            self.b_label = QLabel("b = ")
            self.b_value = QLineEdit()
            self.layout.addRow(self.b_label, self.b_value)

            self.fit_execute_button = QPushButton("Fit")
            self.fit_execute_button.clicked.connect(self.fit)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)
            self.layout.addRow(self.fit_execute_button, self.cancel_button)
        
        elif self.fit_choice.text() == "Trig":
            self.refresh_rows()

            self.func_form_label.setText("A exp(-\u03BBt) (cos(\u03C9t + \u03C6) + y0")
            
            self.A_label = QLabel("A = ")
            self.A_value = QLineEdit()
            self.layout.addRow(self.A_label, self.A_value)

            self.l_label = QLabel("\u03BB = ")
            self.l_value = QLineEdit()
            self.layout.addRow(self.l_label, self.l_value)

            self.w_label = QLabel("\u03C9 = ")
            self.w_value = QLineEdit()
            self.layout.addRow(self.w_label, self.w_value)

            self.phi_label = QLabel("\u03C6 = ")
            self.phi_value = QLineEdit()
            self.layout.addRow(self.phi_label, self.phi_value)

            self.y0_label = QLabel("y0 = ")
            self.y0_value = QLineEdit()
            self.layout.addRow(self.y0_label, self.y0_value)

            self.fit_execute_button = QPushButton("Fit")
            self.fit_execute_button.clicked.connect(self.fit)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)
            self.layout.addRow(self.fit_execute_button, self.cancel_button)

        elif self.fit_choice.text() == "Trig Decay":
            self.refresh_rows()

            self.func_form_label.setText("A exp(-\u03BBt) cos(\u03C9t + \u03C6) + y0")
            
            self.A_label = QLabel("A = ")
            self.A_value = QLineEdit()
            self.layout.addRow(self.A_label, self.A_value)

            self.lamb_label = QLabel("\u03BB = ")
            self.lamb_value = QLineEdit()
            self.layout.addRow(self.lamb_label, self.lamb_value)

            self.w_label = QLabel("\u03C9 = ")
            self.w_value = QLineEdit()
            self.layout.addRow(self.w_label, self.w_value)

            self.phi_label = QLabel("\u03C6 = ")
            self.phi_value = QLineEdit()
            self.layout.addRow(self.phi_label, self.phi_value)

            self.y0_label = QLabel("y0 = ")
            self.y0_value = QLineEdit()
            self.layout.addRow(self.y0_label, self.y0_value)

            self.fit_execute_button = QPushButton("Fit")
            self.fit_execute_button.clicked.connect(self.fit)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)
            self.layout.addRow(self.fit_execute_button, self.cancel_button)

        elif self.fit_choice.text() == "Lorentzian":
            self.refresh_rows()

            self.func_form_label.setText("A/(1 + ((xs - x0)/\u03B3)**2) + y0")
            
            self.A_label = QLabel("A = ")
            self.A_value = QLineEdit()
            self.layout.addRow(self.A_label, self.A_value)

            self.x0_label = QLabel("x0 (res. freq.) = ")
            self.x0_value = QLineEdit()
            self.layout.addRow(self.x0_label, self.x0_value)

            self.gamma_label = QLabel("\u03B3 = ")
            self.gamma_value = QLineEdit()
            self.layout.addRow(self.gamma_label, self.gamma_value)

            self.y0_label = QLabel("y0 = ")
            self.y0_value = QLineEdit()
            self.layout.addRow(self.y0_label, self.y0_value)

            self.fit_execute_button = QPushButton("Fit")
            self.fit_execute_button.clicked.connect(self.fit)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)
            self.layout.addRow(self.fit_execute_button, self.cancel_button)

        elif self.fit_choice.text() == "Gaussian":
            self.refresh_rows()

            self.func_form_label.setText("A exp(-0.5*((xs-x0)/\u03C3)**2) + y0")
            
            self.A_label = QLabel("A = ")
            self.A_value = QLineEdit()
            self.layout.addRow(self.A_label, self.A_value)

            self.x0_label = QLabel("x0 (res. freq.) = ")
            self.x0_value = QLineEdit()
            self.layout.addRow(self.x0_label, self.x0_value)

            self.sigma_label = QLabel("\u03C3 = ")
            self.sigma_value = QLineEdit()
            self.layout.addRow(self.sigma_label, self.sigma_value)

            self.y0_label = QLabel("y0 = ")
            self.y0_value = QLineEdit()
            self.layout.addRow(self.y0_label, self.y0_value)

            self.fit_execute_button = QPushButton("Fit")
            self.fit_execute_button.clicked.connect(self.fit)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)
            self.layout.addRow(self.fit_execute_button, self.cancel_button)

class ViewFitDialog(QWidget):
    
    fit_to_delete = pyqtSignal(int)
    fit_to_save = pyqtSignal(int)

    def __init__(self, fits, parent = None):
        super(ViewFitDialog, self).__init__(parent)
        
        try:
            assert isinstance(fits, dict)
        except AssertionError:
            pass
        else:
            self.fits = fits

            self.layout = QVBoxLayout()
            
            table_layout = QVBoxLayout()

            self.fits_table = QTableWidget()
            self.table_row_num = len(self.fits) + 1
            self.fits_table.setRowCount(self.table_row_num)
            self.fits_table.setColumnCount(3)
            self.fits_table.setHorizontalHeaderLabels(["Name", "Dataset", "Fit Params"])
            header = self.fits_table.horizontalHeader()       
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
            table_layout.addWidget(self.fits_table)

            hbox1_layout = QHBoxLayout()

            print("list: ", map(str, list(range(1,self.table_row_num,1))))

            self.delete_fit_button = QPushButton("Delete Fit #")
            self.delete_fit_button.clicked.connect(lambda: self.delete_fit_clicked())
            self.delete_fit_choice = QComboBox()
            self.delete_fit_choice.addItems(map(str, list(range(1,self.table_row_num,1))))
            hbox1_layout.addWidget(self.delete_fit_button)
            hbox1_layout.addWidget(self.delete_fit_choice)

            hbox2_layout = QHBoxLayout()

            self.save_fit_button = QPushButton("Save Fit #")
            self.save_fit_button.clicked.connect(lambda: self.save_fit_clicked())
            self.save_fit_choice = QComboBox()
            self.save_fit_choice.addItems(map(str, list(range(1,self.table_row_num,1))))

            hbox2_layout.addWidget(self.save_fit_button)
            hbox2_layout.addWidget(self.save_fit_choice)
            
            self.layout.addLayout(table_layout)
            self.layout.addLayout(hbox1_layout)
            self.layout.addLayout(hbox2_layout)

            self.setLayout(self.layout)
            
            self.add_fits()

    def add_fits(self):
        for i, name in enumerate(self.fits.keys()):
            dataset = self.fits[name]['dataset']
            params = list(self.fits[name]['params'].items())
            param_string = '\n'.join(map(str, params))

            self.fits_table.setItem(i,0, QtWidgets.QTableWidgetItem(name))
            self.fits_table.setItem(i,1, QtWidgets.QTableWidgetItem(dataset))
            self.fits_table.setItem(i,2, QtWidgets.QTableWidgetItem(param_string))

    def delete_fit_clicked(self):
        try:
            fit_num = int(self.delete_fit_choice.currentText())
        except TypeError:
            print("Enter in ")
        self.fit_to_delete.emit(fit_num) # tell flex line plot widget which fit to delete

    def save_fit_clicked(self):
        try:
            fit_num = int(self.save_fit_choice.currentText())
        except TypeError:
            print("Enter in ")
        self.fit_to_save.emit(fit_num) # tell flex line plot widget which fit to delete

class _FlexLinePlotSeriesSettings:
    """Contain the settings for a single plot."""

    def __init__(
        self, 
        series: str, 
        scan_i: str, 
        scan_j: str, 
        processing: str, 
        hidden: bool,
        boxcar_width: int = 1,
    ):
        self.series: str = series
        self.scan_i: str = scan_i
        self.scan_j: str = scan_j
        self.processing: str = processing
        self.hidden = hidden
        self.boxcar_width = boxcar_width

class _FlexLinePlotSettings(QThreadSafeObject):
    """Container class to hold the plot settings for a _FlexLinePlotWidget."""

    def __init__(self):
        self.series_settings = {}
        # DataSink for pulling plot data from the data server
        self.sink = None
        # protect access to the sink
        self.sink_mutex = QtCore.QMutex()
        # flag indicating that the plots should be updated
        self.force_update = False
        super().__init__()

    def get_settings(self, name: str, callback=None):
        with QtCore.QMutexLocker(self.mutex):
            settings = self.series_settings[name]
            if callback is not None:
                self.run_main(callback, name, settings, blocking=True)

    def add_plot(
        self,
        name: str,
        series: str,
        scan_i: str,
        scan_j: str,
        processing: str,
        hidden: bool,
        boxcar_width: int = 1,
        callback: Optional[Callable] = None,
    ):
        with QtCore.QMutexLocker(self.mutex):
            if name in self.series_settings:
                _logger.info(
                    f'A plot with the name [{name}] already exists. Ignoring add_plot '
                    'request.'
                )
                return
            self.series_settings[name] = _FlexLinePlotSeriesSettings(
                series=series,
                scan_i=scan_i,
                scan_j=scan_j,
                processing=processing,
                hidden=hidden,
                boxcar_width=boxcar_width,
            )
            self.force_update = True
            if callback is not None:
                self.run_main(callback, name, blocking=True)

    def remove_plot(self, name, callback=None):
        with QtCore.QMutexLocker(self.mutex):
            if name not in self.series_settings:
                _logger.info(
                    f'A plot with the name [{name}] does not exist. Ignoring '
                    'remove_plot request.'
                )
                return

            if callback is not None:
                self.run_main(callback, name, blocking=True)

            del self.series_settings[name]

    def hide_plot(self, name, callback=None):
        with QtCore.QMutexLocker(self.mutex):
            if name not in self.series_settings:
                _logger.info(
                    f'A plot with the name [{name}] does not exist. Ignoring '
                    'hide_plot request.'
                )
                return
            if self.series_settings[name].hidden:
                _logger.info(
                    f'The plot [{name}] is already hidden. Ignoring hide_plot request.'
                )
                return
            self.series_settings[name].hidden = True
            self.force_update = True
            if callback is not None:
                self.run_main(callback, name, blocking=True)

    def show_plot(self, name, callback=None):
        with QtCore.QMutexLocker(self.mutex):
            if name not in self.series_settings:
                _logger.info(
                    f'A plot with the name [{name}] does not exist. Ignoring show_plot '
                    'request.'
                )
                return
            if not self.series_settings[name].hidden:
                _logger.info(
                    f'The plot [{name}] is already shown. Ignoring show_plot request.'
                )
                return
            self.series_settings[name].hidden = False
            self.force_update = True
            if callback is not None:
                self.run_main(callback, name, blocking=True)

    def update_settings(self, name, series, scan_i, scan_j, processing, boxcar_width):
        with QtCore.QMutexLocker(self.mutex):
            if name not in self.series_settings:
                _logger.info(
                    f'A plot with the name [{name}] does not exist. Ignoring '
                    'update_settings request.'
                )
                return
            self.series_settings[name].series = series
            self.series_settings[name].scan_i = scan_i
            self.series_settings[name].scan_j = scan_j
            self.series_settings[name].processing = processing
            self.series_settings[name].boxcar_width = boxcar_width
            self.force_update = True

class FlexLinePlotWidget(QtWidgets.QWidget):
    """Qt widget for flexible plotting of 1D user data.
    It connects to an arbitrary data set stored in the
    :py:class:`~nspyre.data.server.DataServer`, collects and processes the data, and
    offers a variety of user-controlled plotting options.

    The user should push a dictionary containing the following key/value pairs
    to the corresponding :py:class:`~nspyre.data.source.DataSource`
    object sourcing data to the :py:class:`~nspyre.data.server.DataServer`:

    - key: :code:`title`, value: Plot title string
    - key: :code:`xlabel`, value: X label string
    - key: :code:`ylabel`, value: Y label string
    - key: :code:`datasets`, value: Dictionary where keys are a data series \
        name, and values are data as a list of 2D numpy arrays of shape (2, n). \
        The two rows represent the x and y axes, respectively, of the plot, and \
        the n columns each represent a data point.

    You may use np.NaN values in the data arrays to represent invalid entries,
    which won't contribute to the data averaging. An example is given below:

    .. code-block:: python

        from nspyre import DataSource, StreamingList

        with DataSource('my_dataset') as ds:
            channel_1_data = StreamingList([np.array([[1, 2, 3], [12, 12.5, 12.25]]), \
np.array([[4, 5, 6], [12.6, 13, 11.2]])])
            channel_2_data = StreamingList([np.array([[1, 2, 3], [3, 3.3, 3.1]]), \
np.array([[4, 5, 6], [3.4, 3.6, 3.5]])])
            my_plot_data = {
                'title': 'MyVoltagePlot',
                'xlabel': 'Time (s)',
                'ylabel': 'Amplitude (V)',
                'datasets': {
                    'channel_1': channel_1_data
                    'channel_2': channel_2_data
                }
            }
            ds.push(my_plot_data)

    """
    
    new_data = pyqtSignal(str)
    names_list = pyqtSignal(list)

    def __init__(self, 
                 timeout: float = 1,
                 color_flip: bool = True):
        """
        Args:
            timeout: Timeout for :py:meth:`~nspyre.data.sink.DataSink.pop`.
        """
        super().__init__()

        self.line_plot = _FlexLinePlotWidget(timeout=timeout)
        """Underlying LinePlotWidget."""

        self.current_exp_type = None
        self.fits = dict()
        self.fit_counter = 1

        # data source lineedit
        self.datasource_lineedit = QtWidgets.QLineEdit()

        # data source connect button
        connect_button = QtWidgets.QPushButton('Connect')
        connect_button.clicked.connect(self._update_source_clicked)

        # plot settings label
        plot_settings_label = QtWidgets.QLabel('Plot Settings')
        plot_settings_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        # plot name lineedit
        self.plot_name_lineedit = QtWidgets.QLineEdit('avg')

        # data series lineedit
        self.plot_series_lineedit = QtWidgets.QLineEdit('series1')

        # scan indices lineedits
        self.add_plot_scan_i_textbox = QtWidgets.QLineEdit()
        self.add_plot_scan_j_textbox = QtWidgets.QLineEdit()

        # avg/append label
        plot_processing_label = QtWidgets.QLabel('Processing: ')
        plot_processing_label.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
            )
        )

        # avg/append dropdown
        self.plot_processing_dropdown = QtWidgets.QComboBox()
        self.plot_processing_dropdown.addItem('Average')  # index 0
        self.plot_processing_dropdown.addItem('Append')  # index 1
        # default to average
        self.plot_processing_dropdown.setCurrentIndex(0)

        plot_boxcar_label = QtWidgets.QLabel('Boxcar FFT Width (Bins)')
        plot_boxcar_label.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
            )
        )

        self.plot_boxcar_spinbox = QtWidgets.QSpinBox()
        self.plot_boxcar_spinbox.setMinimum(1)
        self.plot_boxcar_spinbox.setMaximum(100000)
        self.plot_boxcar_spinbox.setValue(1)
        self.plot_boxcar_spinbox.setSingleStep(1)

        # show button
        show_button = QtWidgets.QPushButton('Show')
        show_button.clicked.connect(self._show_plot_clicked)

        # hide button
        hide_button = QtWidgets.QPushButton('Hide')
        hide_button.clicked.connect(self._hide_plot_clicked)

        # update button
        update_plot_button = QtWidgets.QPushButton('Update')
        update_plot_button.clicked.connect(self._update_plot_clicked)

        # add button
        add_plot_button = QtWidgets.QPushButton('Add')
        add_plot_button.clicked.connect(self._add_plot_clicked)

        # del button
        remove_button = QtWidgets.QPushButton('Remove')
        remove_button.clicked.connect(self._remove_plot_clicked)

        # plots label
        plots_label = QtWidgets.QLabel('Plots')
        plots_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        # list of plots
        self.plots_list_widget = QtWidgets.QListWidget()
        self.plots_list_widget.currentItemChanged.connect(self._plot_selection_changed)

        # cursor
        self.cursor_button = QPushButton("Cursor")
        self.cursor_button.clicked.connect(self._cursor_clicked)
                
        self.v_line = InfiniteLine(angle = 90, label = 'x={value:0.2f}', pen = (241,196,15), labelOpts={'position': 0.1, 'color': (241,196,15), 'fill': (154,125,10,50), 'movable': False})
        self.h_line = InfiniteLine(angle = 0, label = 'y={value:0.2f}', pen = (46,204,113), labelOpts={'position': 0.1, 'color': (46,204,113), 'fill': (29,131,72,50), 'movable': False})
        self.vb = self.line_plot.plot_widget.plotItem.vb
 
        # fit window
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self._fit_clicked)
        self.curvefit_lineedit = QtWidgets.QLineEdit('div_avg')

        # fit window
        self.remove_fits_button = QPushButton("Remove Fits")
        self.remove_fits_button.clicked.connect(self._remove_fits_clicked)

        # fit window
        self.view_fits_button = QPushButton("View Fits")
        self.view_fits_button.clicked.connect(self._view_fits_clicked)
        
        # spacer
        fixed_spacer = QtWidgets.QLabel('')
        fixed_spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )

        # spacer
        expanding_spacer = QtWidgets.QLabel('')
        expanding_spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        # layout
        settings_layout_config = {
            'type': QtWidgets.QVBoxLayout,
            'data_source': {
                'type': QtWidgets.QHBoxLayout,
                'label': QtWidgets.QLabel('Data Set: '),
                'edit': self.datasource_lineedit,
                'button': connect_button,
            },
            'config': {
                'type': QtWidgets.QHBoxLayout,
                'plot': {
                    'type': QtWidgets.QVBoxLayout,
                    'label': plot_settings_label,
                    'settings': {
                        'type': QtWidgets.QVBoxLayout,
                        'name': {
                            'type': QtWidgets.QHBoxLayout,
                            'label': QtWidgets.QLabel('Plot Name: '),
                            'edit': self.plot_name_lineedit,
                        },
                        'series': {
                            'type': QtWidgets.QHBoxLayout,
                            'label': QtWidgets.QLabel('Data Series: '),
                            'edit': self.plot_series_lineedit,
                        },
                        'index': {
                            'type': QtWidgets.QHBoxLayout,
                            'l1': QtWidgets.QLabel('Scan'),
                            'i': self.add_plot_scan_i_textbox,
                            'l2': QtWidgets.QLabel(' to '),
                            'j': self.add_plot_scan_j_textbox,
                        },
                        'processing': {
                            'type': QtWidgets.QHBoxLayout,
                            'label': plot_processing_label,
                            'dropdown': self.plot_processing_dropdown,
                        },
                        'boxcar': {
                            'type': QtWidgets.QHBoxLayout,
                            'label': plot_boxcar_label,
                            'edit': self.plot_boxcar_spinbox,
                        },
                        'spacer': expanding_spacer,
                    },
                },
                'settings_buttons': {
                    'type': QtWidgets.QVBoxLayout,
                    'spacer_t': fixed_spacer,
                    'update': update_plot_button,
                    'add': add_plot_button,
                    'remove': remove_button,
                    'spacer_b': expanding_spacer,
                },
                'plots': {
                    'type': QtWidgets.QVBoxLayout,
                    'label': plots_label,
                    'list': self.plots_list_widget,
                },
                'list_buttons': {
                    'type': QtWidgets.QVBoxLayout,
                    'spacer_t': fixed_spacer,
                    'show': show_button,
                    'hide': hide_button,
                    'spacer_b': expanding_spacer,
                },
            },
            'cursor_source': {
                'type': QtWidgets.QHBoxLayout,
                'button': self.cursor_button,
            },
            'curve_fit_source': {
                'type': QtWidgets.QHBoxLayout,
                'label': QtWidgets.QLabel('Curve Fit: '),
                'edit': self.curvefit_lineedit,
                'fit_buttons': {
                    'type': QtWidgets.QVBoxLayout,
                    'spacer_t': fixed_spacer,
                    'fit': self.fit_button,
                    'remove': self.remove_fits_button,
                    'view': self.view_fits_button,
                },
            }
        }
        self.layout_tree = tree_layout(settings_layout_config)
        # make the plots list (index=2) take up all extra space (stretch=1)
        self.layout_tree.config.layout.setStretch(2, 1)

        # splitter
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.line_plot)
        layout_container = QtWidgets.QWidget()
        layout_container.setLayout(self.layout_tree.layout)
        splitter.addWidget(layout_container)

        # main layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter)

        self.setLayout(layout)

        if color_flip:
            self.plot_color_manager = PlotColorManager(self)
            layout.addWidget(self.plot_color_manager.color_flip_button)

    def _plot_selection_changed(self):
        """Called when the selected plot changes."""
        # selected QListWidgetItem
        selected_item = self.plots_list_widget.currentItem()
        if selected_item is None:
            return
        # get the selected plot name
        name = selected_item.text()
        self.line_plot.plot_settings.run_safe(
            self.line_plot.plot_settings.get_settings,
            name,
            callback=self._plot_selection_changed_callback,
        )

    def _plot_selection_changed_callback(
        self, name: str, settings: _FlexLinePlotSeriesSettings
    ):
        """Called after the selection is changed to update the plot settings GUI
        elements."""
        self.plot_name_lineedit.setText(name)
        self.plot_series_lineedit.setText(settings.series)
        self.add_plot_scan_i_textbox.setText(settings.scan_i)
        self.add_plot_scan_j_textbox.setText(settings.scan_j)
        self.plot_processing_dropdown.setCurrentText(settings.processing)
        self.plot_boxcar_spinbox.setValue(int(settings.boxcar_width))

    def _get_plot_settings(self):
        """Retrieve the user-entered plot settings from the GUI and check them for
        errors."""
        scan_i = self.add_plot_scan_i_textbox.text()
        try:
            if scan_i != '':
                int(scan_i)
        except ValueError as err:
            raise ValueError(
                f'Scan start [{scan_i}] must be either an integer or empty.'
            ) from err
        scan_j = self.add_plot_scan_j_textbox.text()
        try:
            if scan_j != '':
                int(scan_j)
        except ValueError as err:
            raise ValueError(
                f'Scan end [{scan_j}] must be either an integer or empty.'
            ) from err
        name = self.plot_name_lineedit.text()
        series = self.plot_series_lineedit.text()
        processing = self.plot_processing_dropdown.currentText()
        boxcar_width = self.plot_boxcar_spinbox.value()

        return name, series, scan_i, scan_j, processing, boxcar_width

    def _update_plot_clicked(self):
        """Called when the user clicks the update button."""
        name, series, scan_i, scan_j, processing, boxcar_width = self._get_plot_settings()
        # set the plot settings
        self.line_plot.plot_settings.run_safe(
            self.line_plot.plot_settings.update_settings,
            name,
            series,
            scan_i,
            scan_j,
            processing,
            boxcar_width,
        )

    def _add_plot_clicked(self):
        """Called when the user clicks the add button."""
        name, series, scan_i, scan_j, processing, boxcar_width = self._get_plot_settings()
        self.add_plot(name, series, scan_i, scan_j, processing, boxcar_width)

    def add_plot(
        self, 
        name: str, 
        series: str, 
        scan_i: str, 
        scan_j: str, 
        processing: str,
        boxcar_width: int = 1,
    ):
        """Add a new subplot. Thread safe.

        Args:
            name: Name for the new plot.
            series: The data series name pushed by the \
                :py:class:`~nspyre.data.source.DataSource`, e.g. \
                :code:`channel_1` for the example given in \
                :py:class:`~nspyre.gui.widgets.flex_line_plot.FlexLinePlotWidget`
            scan_i: String value of the scan to start plotting from.
            scan_j: String value of the scan to stop plotting at. \
                Use Python list indexing notation, e.g.:

                - :code:`scan_i = '-1'`, :code:`scan_j = ''` for the last element
                - :code:`scan_i = '0'`, :code:`scan_j = '1'` for the first element
                - :code:`scan_i = '-3'`, :code:`scan_j = ''` for the last 3 elements.

            processing: 'Average' to average the x and y values of scans i
                through j, 'Append' to concatenate them.
            boxcar_width: Integer width of the boxcar window for smoothing. 
                Default is 1 (no smoothing).
        """
        self.line_plot.plot_settings.run_safe(
            self.line_plot.plot_settings.add_plot,
            name=name,
            series=series,
            scan_i=scan_i,
            scan_j=scan_j,
            processing=processing,
            hidden=False,
            boxcar_width=boxcar_width,
            callback=self._add_plot_callback,
        )

    def _add_plot_callback(self, name: str):
        """Called in main thread after a plot is added."""
        self.plots_list_widget.addItem(name)
        self.line_plot.add_plot(name)

        # NEW: assign shuffled color + apply current mode styling
        if hasattr(self, "plot_color_manager"):
            self.plot_color_manager.apply_current_style_all()

    def _find_plot_item(self, name):
        """Return the index of the list widget plot item with the given name."""
        list_widget_index = None
        for i in range(self.plots_list_widget.count()):
            if self.plots_list_widget.item(i).text() == name:
                list_widget_index = i
                break
        if list_widget_index is None:
            raise RuntimeError(
                f'Internal error: plot [{name}] not found in list widget.'
            )

        return list_widget_index

    def _remove_plot_clicked(self):
        """Called when the user clicks the remove button."""
        # array of selected QListWidgetItems
        selected_items = self.plots_list_widget.selectedItems()
        for i in selected_items:
            name = i.text()
            self.remove_plot(name)

    def remove_plot(self, name: str):
        """Remove a subplot. Thread safe.

        Args:
            name: Name of the subplot.
        """
        # remove the plot settings
        self.line_plot.plot_settings.run_safe(
            self.line_plot.plot_settings.remove_plot,
            name,
            callback=self._remove_plot_callback,
        )

    def _remove_plot_callback(self, name: str):
        """Called in main thread after a plot is removed."""
        # remove the plot name from the list of plots
        self.plots_list_widget.takeItem(self._find_plot_item(name))
        # remove the plot from the pyqtgraph plotwidget
        self.line_plot.remove_plot(name)

    def _hide_plot_clicked(self):
        """Called when the user clicks the hide button."""
        # array of selected QListWidgetItems
        selected_items = self.plots_list_widget.selectedItems()
        for i in selected_items:
            name = i.text()
            self.hide_plot(name)

    def hide_plot(self, name: str):
        """Hide a subplot. Thread safe.

        Args:
            name: Name of the subplot.
        """
        # update the settings
        self.line_plot.plot_settings.run_safe(
            self.line_plot.plot_settings.hide_plot,
            name,
            callback=self._hide_plot_callback,
        )

    def _hide_plot_callback(self, name: str):
        """Called in main thread after a plot is hidden."""
        # hide the plot in the pyqtgraph plotting widget
        self.line_plot.hide_plot(name)
        # change the list widget item color scheme
        idx = self._find_plot_item(name)

        if hasattr(self, "plot_color_manager"):
            self.plot_color_manager.apply_current_style_all()

        self.plots_list_widget.item(idx).setForeground(QtCore.Qt.GlobalColor.gray)
        self.plots_list_widget.item(idx).setBackground(
            self.palette().color(QtGui.QPalette.ColorRole.Mid)
        )

    def _show_plot_clicked(self):
        """Called when the user clicks the show button."""
        # array of selected QListWidgetItems
        selected_items = self.plots_list_widget.selectedItems()
        for i in selected_items:
            name = i.text()
            self.show_plot(name)

    def show_plot(self, name: str):
        """Show a previously hidden subplot. Thread safe.

        Args:
            name: Name of the subplot.
        """
        # update the settings
        self.line_plot.plot_settings.run_safe(
            self.line_plot.plot_settings.show_plot,
            name,
            callback=self._show_plot_callback,
        )

    def _show_plot_callback(self, name: str):
        """Called after a plot is shown."""
        # show the plot in the pyqtgraph plotting widget
        self.line_plot.show_plot(name)
        # return list widget item to normal color scheme
        idx = self._find_plot_item(name)

        # NEW: re-style when shown (important if it was hidden initially)
        if hasattr(self, "plot_color_manager"):
            self.plot_color_manager.apply_current_style_all()

        normal_text_color = self.palette().color(QtGui.QPalette.ColorRole.Text)
        normal_bg_color = self.palette().color(QtGui.QPalette.ColorRole.Base)
        self.plots_list_widget.item(idx).setForeground(normal_text_color)
        self.plots_list_widget.item(idx).setBackground(normal_bg_color)

    def _update_source_clicked(self):
        """Called when the user clicks the connect button."""
        self.current_exp_type = self.datasource_lineedit.text()
        self.line_plot.new_source(self.current_exp_type)

        if hasattr(self, "plot_color_manager"):
            self.plot_color_manager.reshuffle_for_new_connection()

        self.line_plot.plot_widget.getPlotItem().setDownsampling(ds=True, auto=True, mode='mean')
        # clear previously loaded plots
        for plot_name in list(self.line_plot.plot_settings.series_settings):
            self.remove_plot(plot_name)
        
        match self.current_exp_type:
            case 'sigvstime':
                self.add_plot('monitor 1',        series='signal',   scan_i='',     scan_j='',  processing='Append')
                self.add_plot('monitor 2',        series='background',   scan_i='',     scan_j='',  processing='Append')
                self.hide_plot('monitor 2')

            case 'odmr':
                # create default fit plot
                self.add_plot('fit',            series='fit',   scan_i='',      scan_j='',  processing='Average')
                self.hide_plot('fit')
                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_avg')
                self.hide_plot('diff_latest')

                # create some default div plots
                self.add_plot('div_avg',       series='div',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('div_latest',    series='div',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('div_latest')
                
                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')

            case 'odmr rf':
                # # create default fit plot
                # self.add_plot('fit',            series='fit',   scan_i='',      scan_j='',  processing='Average')
                # self.hide_plot('fit')
                # create some default dark, echo plots for DEER. Otherwise, just duplicate div plots
                self.add_plot('rf_avg',       series='div_rf',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('no_rf_avg',       series='div',  scan_i='',    scan_j='',  processing='Average')
                
                # create some default dark signal plots
                self.add_plot('rf_sig_avg',        series='rf_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('rf_sig_latest',     series='rf_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('rf_sig_avg')
                self.hide_plot('rf_sig_latest')

                # create some default dark background plots
                self.add_plot('rf_bg_avg',         series='rf_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('rf_bg_latest',      series='rf_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('rf_bg_avg')
                self.hide_plot('rf_bg_latest')                

                # create some default echo signal plots
                self.add_plot('no_rf_sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('no_rf_sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('no_rf_sig_avg')
                self.hide_plot('no_rf_sig_latest')

                # create some default echo background plots
                self.add_plot('no_rf_bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('no_rf_bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('no_rf_bg_avg')
                self.hide_plot('no_rf_bg_latest')

            case 'rabi':
                # create default fit plot
                self.add_plot('fit',            series='fit',   scan_i='',      scan_j='',  processing='Average')
                self.hide_plot('fit')
                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_avg')
                self.hide_plot('diff_latest')

                # create some default div plots
                self.add_plot('div_avg',       series='div',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('div_latest',    series='div',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('div_latest')
                
                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')

            case 't1':
                # create default fit plot
                self.add_plot('fit',            series='fit',   scan_i='',      scan_j='',  processing='Average')
                self.hide_plot('fit')
                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_latest')

                # create some default contrast plots
                self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_avg')
                self.hide_plot('contrast_latest')
                
                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')

            case 't2':
                # create default fit plot
                self.add_plot('fit',            series='fit',   scan_i='',      scan_j='',  processing='Average')
                self.hide_plot('fit')
                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_latest')

                # create some default contrast plots
                self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_avg')
                self.hide_plot('contrast_latest')
                
                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')

                # create some default fft plots
                self.add_plot('fft_avg',       series='fft',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('fft_latest',    series='fft',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('fft_avg')
                self.hide_plot('fft_latest')

            case 'dq':
                # create some default diff plots
                self.add_plot('S0,0 - S0,-1 avg',       series='diff dq1',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('S0,0 - S0,-1 latest',    series='diff dq1',  scan_i='-1',    scan_j='',  processing='Average')

                self.add_plot('S-1,-1 - S-1,+1 avg',       series='diff dq2',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('S-1,-1 - S-1,+1 latest',    series='diff dq2',  scan_i='-1',    scan_j='',  processing='Average')

                self.hide_plot('S0,0 - S0,-1 latest')
                self.hide_plot('S-1,-1 - S-1,+1 latest')

                # create some default contrast plots
                # self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                # self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                # self.hide_plot('contrast_latest')
                
                # create some default S0,0 plots
                self.add_plot('S0,0 avg',        series='S0,0',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('S0,0 latest',     series='S0,0',   scan_i='-1',   scan_j='',  processing='Average')
                # self.add_plot('S0,0 first',      series='S0,0',   scan_i='0',    scan_j='1', processing='Average')
                # self.add_plot('S0,0 latest 10',  series='S0,0',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('S0,0 avg')
                self.hide_plot('S0,0 latest')
                # self.hide_plot('S0,0 first')
                # self.hide_plot('S0,0 latest 10')

                # create some default S0,-1 plots
                self.add_plot('S0,-1 avg',        series='S0,-1',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('S0,-1 latest',     series='S0,-1',   scan_i='-1',   scan_j='',  processing='Average')
                # self.add_plot('S0,-1 first',      series='S0,-1',   scan_i='0',    scan_j='1', processing='Average')
                # self.add_plot('S0,-1 latest 10',  series='S0,-1',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('S0,-1 avg')
                self.hide_plot('S0,-1 latest')
                # self.hide_plot('S0,-1 first')
                # self.hide_plot('S0,-1 latest 10')

                # create some default S-1,-1 plots
                self.add_plot('S-1,-1 avg',        series='S-1,-1',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('S-1,-1 latest',     series='S-1,-1',   scan_i='-1',   scan_j='',  processing='Average')
                # self.add_plot('S-1,-1 first',      series='S-1,-1',   scan_i='0',    scan_j='1', processing='Average')
                # self.add_plot('S-1,-1 latest 10',  series='S-1,-1',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('S-1,-1 avg')
                self.hide_plot('S-1,-1 latest')
                # self.hide_plot('S-1,-1 first')
                # self.hide_plot('S-1,-1 latest 10')

                # create some default S-1,+1 plots
                self.add_plot('S-1,+1 avg',        series='S-1,+1',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('S-1,+1 latest',     series='S-1,+1',   scan_i='-1',   scan_j='',  processing='Average')
                # self.add_plot('S-1,+1 first',      series='S-1,+1',   scan_i='0',    scan_j='1', processing='Average')
                # self.add_plot('S-1,+1 latest 10',  series='S-1,+1',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('S-1,+1 avg')
                self.hide_plot('S-1,+1 latest')
                # self.hide_plot('S-1,+1 first')
                # self.hide_plot('S-1,+1 latest 10')

            case 'deer':
                # create default fit plot
                self.add_plot('fit',            series='fit',   scan_i='',      scan_j='',  processing='Average')
                self.hide_plot('fit')
                # create some default dark, echo plots for DEER. Otherwise, just duplicate div plots
                self.add_plot('dark_avg',       series='dark_contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('echo_avg',       series='echo_contrast',  scan_i='',    scan_j='',  processing='Average')
                self.hide_plot('dark_avg')
                self.hide_plot('echo_avg')

                # create some default contrast plots
                self.add_plot('contrast_avg',       series='deer_contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='deer_contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_latest')
                
                # create some default dark signal plots
                self.add_plot('dark_sig_avg',        series='dark_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_sig_latest',     series='dark_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_sig_avg')
                self.hide_plot('dark_sig_latest')

                # create some default dark background plots
                self.add_plot('dark_bg_avg',         series='dark_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_bg_latest',      series='dark_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_bg_avg')
                self.hide_plot('dark_bg_latest')                

                # create some default echo signal plots
                self.add_plot('echo_sig_avg',        series='echo_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_sig_latest',     series='echo_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_sig_avg')
                self.hide_plot('echo_sig_latest')

                # create some default echo background plots
                self.add_plot('echo_bg_avg',         series='echo_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_bg_latest',      series='echo_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_bg_avg')
                self.hide_plot('echo_bg_latest') 

            case 'deer rabi':
                # create default fit plot
                self.add_plot('fit',            series='fit',   scan_i='',      scan_j='',  processing='Average')
                self.hide_plot('fit')
                # create some default dark, echo plots for DEER. Otherwise, just duplicate div plots
                self.add_plot('dark_avg',       series='dark_contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('echo_avg',       series='echo_contrast',  scan_i='',    scan_j='',  processing='Average')
                self.hide_plot('dark_avg')
                self.hide_plot('echo_avg')

                # create some default contrast plots
                self.add_plot('contrast_avg',       series='deer_contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='deer_contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_latest')
                
                # create some default dark signal plots
                self.add_plot('dark_sig_avg',        series='dark_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_sig_latest',     series='dark_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_sig_avg')
                self.hide_plot('dark_sig_latest')

                # create some default dark background plots
                self.add_plot('dark_bg_avg',         series='dark_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_bg_latest',      series='dark_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_bg_avg')
                self.hide_plot('dark_bg_latest')                

                # create some default echo signal plots
                self.add_plot('echo_sig_avg',        series='echo_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_sig_latest',     series='echo_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_sig_avg')
                self.hide_plot('echo_sig_latest')

                # create some default echo background plots
                self.add_plot('echo_bg_avg',         series='echo_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_bg_latest',      series='echo_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_bg_avg')
                self.hide_plot('echo_bg_latest') 

            case 'fid':
                # create some default dark, echo plots for DEER. Otherwise, just duplicate div plots
                self.add_plot('dark_avg',       series='dark_contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('echo_avg',       series='echo_contrast',  scan_i='',    scan_j='',  processing='Average')
                self.hide_plot('dark_avg')
                self.hide_plot('echo_avg')

                # create some default contrast plots
                self.add_plot('contrast_avg',       series='deer_contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='deer_contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_latest')
                
                # TODO: create a double log plot - fix the unsolvable values in array
                # self.add_plot('doublelog_avg',       series='deer_log_contrast',  scan_i='',      scan_j='',  processing='Average')
                # self.add_plot('doublelog_latest',    series='deer_log_contrast',  scan_i='-1',    scan_j='',  processing='Average')
                # self.hide_plot('doublelog_avg')
                # self.hide_plot('doublelog_latest')

                # create some default dark signal plots
                self.add_plot('dark_sig_avg',        series='dark_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_sig_latest',     series='dark_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_sig_avg')
                self.hide_plot('dark_sig_latest')

                # create some default dark background plots
                self.add_plot('dark_bg_avg',         series='dark_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_bg_latest',      series='dark_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_bg_avg')
                self.hide_plot('dark_bg_latest')                

                # create some default echo signal plots
                self.add_plot('echo_sig_avg',        series='echo_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_sig_latest',     series='echo_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_sig_avg')
                self.hide_plot('echo_sig_latest')

                # create some default echo background plots
                self.add_plot('echo_bg_avg',         series='echo_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_bg_latest',      series='echo_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_bg_avg')
                self.hide_plot('echo_bg_latest') 

            case 'fid cd':
                # create some default dark, echo plots for DEER. Otherwise, just duplicate div plots
                self.add_plot('dark_avg',       series='dark_contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('echo_avg',       series='echo_contrast',  scan_i='',    scan_j='',  processing='Average')
                self.add_plot('cd_avg',       series='cd_contrast',  scan_i='',    scan_j='',  processing='Average')
                # self.hide_plot('dark_avg')
                # self.hide_plot('echo_avg')
                # self.hide_plot('cd_avg')

                # create some default dark signal plots
                self.add_plot('dark_sig_avg',        series='dark_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_sig_latest',     series='dark_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_sig_avg')
                self.hide_plot('dark_sig_latest')

                # create some default dark background plots
                self.add_plot('dark_bg_avg',         series='dark_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('dark_bg_latest',      series='dark_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('dark_bg_avg')
                self.hide_plot('dark_bg_latest')                

                # create some default echo signal plots
                self.add_plot('echo_sig_avg',        series='echo_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_sig_latest',     series='echo_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_sig_avg')
                self.hide_plot('echo_sig_latest')

                # create some default echo background plots
                self.add_plot('echo_bg_avg',         series='echo_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('echo_bg_latest',      series='echo_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('echo_bg_avg')
                self.hide_plot('echo_bg_latest')

                # create some default cd signal plots
                self.add_plot('cd_sig_avg',        series='cd_signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('cd_sig_latest',     series='cd_signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('cd_sig_avg')
                self.hide_plot('cd_sig_latest')

                # create some default cd background plots
                self.add_plot('cd_bg_avg',         series='cd_background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('cd_bg_latest',      series='cd_background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('cd_bg_avg')
                self.hide_plot('cd_bg_latest')

            case 'corr rabi':
                # create some default contrast plots
                self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_latest')

                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_avg')
                self.hide_plot('diff_latest')

                # create some default fft plots
                self.add_plot('fft_avg',       series='fft',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('fft_latest',    series='fft',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('fft_avg')
                self.hide_plot('fft_latest')

                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')
            
            case 'corr t1 simple':
                # create some default contrast plots
                self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_latest')

                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_avg')
                self.hide_plot('diff_latest')

                # create some default fft plots
                self.add_plot('fft_avg',       series='fft',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('fft_latest',    series='fft',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('fft_avg')
                self.hide_plot('fft_latest')

                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')

            case 'deer t1':
                # create some default dark, echo plots for DEER. Otherwise, just duplicate div plots
                self.add_plot('diff +y surface T1 decay',       series='diff_py',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff -y surface T1 decay',       series='diff_ny',  scan_i='',    scan_j='',  processing='Average')
                self.add_plot('diff overall surface T1 decay',       series='diff_overall',  scan_i='',      scan_j='',  processing='Average')
                
                self.add_plot('diff nuclear oscillation with pi pulse',       series='diff_osc_with_pulse',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff nuclear oscillation no pi pulse',       series='diff_osc_without_pulse',  scan_i='',    scan_j='',  processing='Average')
                self.add_plot('sum pure nuclear oscillation',       series='sum_nuclear',  scan_i='',      scan_j='',  processing='Average')
                
                self.hide_plot('diff -y surface T1 decay')
                self.hide_plot('diff overall surface T1 decay')
                self.hide_plot('diff nuclear oscillation with pi pulse')
                self.hide_plot('diff nuclear oscillation no pi pulse')
                self.hide_plot('sum pure nuclear oscillation')

                # create some default signal plots (norm method 1 - pi/no pi pulse)
                self.add_plot('with pi pulse +y avg',        series='with_py',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('with pi pulse +y latest',     series='with_py',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('with pi pulse +y avg')
                self.hide_plot('with pi pulse +y latest')

                # create some default signal plots (norm method 1 - pi/no pi pulse)
                self.add_plot('with pi pulse -y avg',        series='with_ny',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('with pi pulse -y latest',     series='with_ny',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('with pi pulse -y avg')
                self.hide_plot('with pi pulse -y latest')

                # create some default signal plots (norm method 1 - pi/no pi pulse)
                self.add_plot('no pi pulse +y avg',        series='without_py',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('no pi pulse +y latest',     series='without_py',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('no pi pulse +y avg')
                self.hide_plot('no pi pulse +y latest')

                # create some default signal plots (norm method 1 - pi/no pi pulse)
                self.add_plot('no pi pulse -y avg',        series='without_ny',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('no pi pulse -y latest',     series='without_ny',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('no pi pulse -y avg')
                self.hide_plot('no pi pulse -y latest')

            case 'deer t2':
                # create some default contrast plots
                self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_latest')

                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_avg')
                self.hide_plot('diff_latest')

                # create some default fft plots
                self.add_plot('fft_avg',       series='fft',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('fft_latest',    series='fft',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('fft_avg')
                self.hide_plot('fft_latest')

                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')

            case 'nmr':
                # create some default contrast plots
                self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_avg')
                self.hide_plot('contrast_latest')
                
                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_avg')
                self.hide_plot('diff_latest')

                # create some default fft plots
                self.add_plot('fft_avg',       series='fft',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('fft_latest',    series='fft',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('fft_latest')

                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')

            case 'casr':
                self.line_plot.plot_widget.getPlotItem().setDownsampling(ds=10, auto=False, mode='mean')
                # create some default contrast plots
                self.add_plot('contrast_avg',       series='contrast',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('contrast_latest',    series='contrast',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('contrast_avg')
                self.hide_plot('contrast_latest')

                # create some default diff plots
                self.add_plot('diff_avg',       series='diff',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('diff_latest',    series='diff',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('diff_avg')
                self.hide_plot('diff_latest')

                # create some default fft plots
                self.add_plot('fft_avg',       series='fft',  scan_i='',      scan_j='',  processing='Average')
                self.add_plot('fft_latest',    series='fft',  scan_i='-1',    scan_j='',  processing='Average')
                self.hide_plot('fft_latest')

                # create some default signal plots
                self.add_plot('sig_avg',        series='signal',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('sig_latest',     series='signal',   scan_i='-1',   scan_j='',  processing='Average')
                self.add_plot('sig_first',      series='signal',   scan_i='0',    scan_j='1', processing='Average')
                self.add_plot('sig_latest_10',  series='signal',   scan_i='-10',  scan_j='',  processing='Average')
                self.hide_plot('sig_avg')
                self.hide_plot('sig_latest')
                self.hide_plot('sig_first')
                self.hide_plot('sig_latest_10')

                # create some default background plots
                self.add_plot('bg_avg',         series='background',   scan_i='',     scan_j='',  processing='Average')
                self.add_plot('bg_latest',      series='background',   scan_i='-1',   scan_j='',  processing='Average')
                self.hide_plot('bg_avg')
                self.hide_plot('bg_latest')


    def _cursor_clicked(self):
        def mouse_moved(evt):
            pos = evt[0]
            if self.line_plot.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.vb.mapSceneToView(pos)
                index = int(mouse_point.x())
                # if index >= 0 and index < self.data_length:
                #     self.cursor_label2.setText(str(round(mouse_point.x(), 3)))
                #     self.cursor_label4.setText(str(round(mouse_point.y(), 3)))
                    
                self.v_line.setPos(mouse_point.x())
                self.h_line.setPos(mouse_point.y())

            def mouse_clicked():
                # x_pos = mouse_point.x()
                # y_pos = mouse_point.y()
                # self.cursor_label2.setText(str(round(x_pos, 3)))
                # self.cursor_label4.setText(str(round(y_pos, 3)))
                pass
            
            self.line_plot.plot_widget.scene().sigMouseClicked.connect(mouse_clicked)  

        if self.cursor_button.text() == "Cursor":
            self.cursor_button.setText("Delete Cursor")
            
            if self.datasource_lineedit.text() == 'odmr':
                self.cursor_label1 = QLabel("Frequency [GHz] = ")
            else:
                self.cursor_label1 = QLabel("Tau [\u03BCs] = ")
            
            self.cursor_label3 = QLabel("PL Voltage or Contrast = ")
 
            self.cursor_label2 = QLabel()
            self.cursor_label4 = QLabel()

            # self.layout.addWidget(self.cursor_label1,4,1)
            # self.layout.addWidget(self.cursor_label2,4,2)
            # self.layout.addWidget(self.cursor_label3,5,1)
            # self.layout.addWidget(self.cursor_label4,5,2)

            self.line_plot.plot_widget.addItem(self.v_line, ignoreBounds = True)
            self.line_plot.plot_widget.addItem(self.h_line, ignoreBounds = True)
        
            self.proxy = SignalProxy(self.line_plot.plot_widget.scene().sigMouseMoved, rateLimit = 60, slot = mouse_moved)

        else:
            self.cursor_button.setText("Cursor")
            self.line_plot.plot_widget.removeItem(self.v_line)
            self.line_plot.plot_widget.removeItem(self.h_line)

            self.cursor_label1.deleteLater()
            self.cursor_label2.deleteLater()
            self.cursor_label3.deleteLater()
            self.cursor_label4.deleteLater()

            # self.layout.removeWidget(self.cursor_label1)
            # self.layout.removeWidget(self.cursor_label2)
            # self.layout.removeWidget(self.cursor_label3)
            # self.layout.removeWidget(self.cursor_label4)

            del(self.proxy)

    @pyqtSlot(list)
    def _update_fit_parameters(self, params):
        self.fit_type = params[0]
        self.fit_params = params[1:]
        self.fit_params = [float(f) for f in self.fit_params]

        # print("Curve to fit: ", self.curve_to_fit['x'])
        if self.fit_type == 'Exponential':
            coeffs = ['A', 'T']
            def exp_fit(xs, A, T):
                # from nspyre import qt_set_trace; qt_set_trace()
                return A*np.exp(-xs/T)
            
            param, param_cov = curve_fit(exp_fit, self.xs, self.ys, self.fit_params)
            
            # add fit_ys curve to current plot
            fit_ys = exp_fit(self.xs, *param)

        elif self.fit_type == 'Exponential (stretched)':
            coeffs = ['A', 'T', 'N']
            def exp_str_fit(xs, A, T, N):
                return A*np.exp(-(xs/T)**N)
            
            param, param_cov = curve_fit(exp_str_fit, self.xs, self.ys, self.fit_params)

            # add fit_ys curve to current plot
            fit_ys = exp_str_fit(self.xs, *param)

        elif self.fit_type == 'Linear':
            coeffs = ['m', 'b']
            def lin_fit(xs, m, b):
                return m*xs + b
            
            param, param_cov = curve_fit(lin_fit, self.xs, self.ys, self.fit_params)

            # add fit_ys curve to current plot
            fit_ys = lin_fit(self.xs, *param)

        elif self.fit_type == 'Trig':
            coeffs = ['A', 'w', 'phi', 'y0']
            def trig_fit(xs, A, w, phi, y0):
                return A*np.cos(w*xs + phi) + y0
            
            param, param_cov = curve_fit(trig_fit, self.xs, self.ys, self.fit_params)

            # add fit_ys curve to current plot
            fit_ys = trig_fit(self.xs, *param)
        
        elif self.fit_type == 'Trig Decay':
            coeffs = ['A', 'lamb', 'w', 'phi', 'y0']
            def trig_decay_fit(xs, A, lamb, w, phi, y0):
                return A*np.exp(-lamb*xs)*np.cos(w*xs + phi) + y0
            
            param, param_cov = curve_fit(trig_decay_fit, self.xs, self.ys, self.fit_params)

            # add fit_ys curve to current plot
            fit_ys = trig_decay_fit(self.xs, *param)
        
        elif self.fit_type == 'Lorentzian':
            coeffs = ['A', 'x0', 'gamma', 'y0']
            def lorentz_fit(xs, A, x0, gamma, y0):
                return A/(1 + ((xs - x0)/gamma)**2) + y0
            
            param, param_cov = curve_fit(lorentz_fit, self.xs, self.ys, self.fit_params)

            # add fit_ys curve to current plot
            fit_ys = lorentz_fit(self.xs, *param)
        
        elif self.fit_type == 'Gaussian':
            coeffs = ['A', 'x0', 'sigma', 'y0']
            def gauss_fit(xs, A, x0, sigma, y0):
                return A*np.exp(-0.5*((xs-x0)/sigma)**2)+y0
            
            param, param_cov = curve_fit(gauss_fit, self.xs, self.ys, self.fit_params)

            # add fit_ys curve to current plot
            fit_ys = gauss_fit(self.xs, *param)

        p = dict(zip(coeffs, param)) # create dictionary with coefficients as keys and fit params as values to display in ViewFitDialog

        title = f"Fit #{self.fit_counter}: {self.fit_type}"

        self.fits[title] = {
            'dataset': self.datasource_lineedit.text(),
            'xs': self.xs,
            'ys': fit_ys,
            'curve': self.fits.get(title, dict()).get('curve'),
            'params': p,
            'param_cov': param_cov
        }

        self._update_fits()
    
        # fitted_curve = self.line_plot.plot_widget.plot(pen=pg.mkPen(color=(255,255,0), width=1), antialias=True)
        # fitted_curve.setData(x = self.xs, y = fit_ys)

        self.fit_counter += 1

    def _update_fits(self):
            for trace_name, data in self.fits.items():
                xs = data['xs']
                ys = data['ys']
                curve = data['curve']
                if curve is None:
                    curve = self.line_plot.plot_widget.plot(pen=pg.mkPen(color=(255,255,0), width=10), antialias=True)
                    data['curve'] = curve
                curve.setData(x=xs, y=ys)
            return
    
    def _fit_clicked(self):
        # self.new_plot('Curve Fit', QColor(255, 0, 0))
        # print("SERIES SETTINGS: ", self.line_plot.plot_settings.series_settings)
        try:
            self.xs, self.ys = self.line_plot.processed_data_dict[self.curvefit_lineedit.text()] # access processed x and y data sets for fitting
            # print("X data: ", self.xs)
            # print("Y data: ", self.ys)
        except Exception as e:
            print("ERROR: ", e)
        else:
            self.curve_fitter = FitCurveDialog()
            self.curve_fitter.fit_parameters.connect(self._update_fit_parameters)       
            self.curve_fitter.show()

    def _remove_fits_clicked(self):
        print("keys: ", self.fits.keys())
        for key in self.fits.keys():
            try:
                fit_data = self.fits[key]
                fit = fit_data['curve']
            except KeyError:
                continue
            self.line_plot.plot_widget.removeItem(fit)

            self.fit_counter -= 1
        
        self.fits.clear()

        return
    
    @pyqtSlot(int)
    def _update_fit_delete_parameters(self, param):
        pass

    @pyqtSlot(int)
    def _update_fit_save_parameters(self, param):
        pass

    def _view_fits_clicked(self):
        self.view_fitter = ViewFitDialog(self.fits)
        self.view_fitter.fit_to_delete.connect(self._update_fit_delete_parameters) 
        self.view_fitter.fit_to_save.connect(self._update_fit_save_parameters) 
        self.view_fitter.show()

class _FlexLinePlotWidget(LinePlotWidget):
    """See FlexLinePlotWidget."""

    def __init__(self, timeout: float):
        """
        Args:
            timeout: see :py:class:`FlexLinePlotWidget`.
            # OBSOLETE: data_processing_func: see :py:class:`FlexLinePlotWidget`.
        """
        self.timeout = timeout
        # self.data_processing_func = data_processing_func
        self.processed_data_dict = dict()
        self.plot_settings = _FlexLinePlotSettings()
        self.plot_settings.start()
        super().__init__()

    def _stop(self):
        """Stop the updating and plot data management threads."""
        self.plot_settings.stop()
        super()._stop()

    def new_source(self, data_set_name: str):
        """Connect to a new data set on the data server.

        Args:
            data_set_name: Name of the new data set.
        """
        # run on the plot_settings thread since we'll need to acquire mutexes
        self.plot_settings.run_safe(self._new_source, data_set_name)

    def _new_source(self, data_set_name: str):
        
        self.dataset_name = data_set_name
        # connect to a new data set
        with QtCore.QMutexLocker(self.plot_settings.sink_mutex):
            self.clear_plots()
            try:
                # connect to the new data source
                self.plot_settings.sink = DataSink(data_set_name)
                self.plot_settings.sink.start()
                
                # try to get the plot title and x/y labels
                self.plot_settings.sink.pop(timeout=self.timeout)

                # set title
                try:
                    title = self.plot_settings.sink.title
                except AttributeError:
                    _logger.info(
                        f'Data source [{data_set_name}] has no "title" '
                        'attribute. Not setting the plot title...'
                    )
                    title = None

                # set xlabel
                try:
                    xlabel = self.plot_settings.sink.xlabel
                except AttributeError:
                    _logger.info(
                        f'Data source [{data_set_name}] has no "xlabel" '
                        'attribute. Not setting the plot x-axis label...'
                    )
                    xlabel = None

                # set ylabel
                try:
                    ylabel = self.plot_settings.sink.ylabel
                except AttributeError:
                    _logger.info(
                        f'Data source [{data_set_name}] has no "ylabel" '
                        'attribute. Not setting the plot y-axis label...'
                    )
                    ylabel = None

                # try to access datasets
                try:
                    dsets = self.plot_settings.sink.datasets
                except AttributeError as err:
                    raise RuntimeError(
                        f'Data source [{data_set_name}] has no "datasets" attribute - '
                        'exiting...'
                    ) from err
                else:
                    if not isinstance(dsets, dict):
                        raise RuntimeError(
                            f'Data source [{data_set_name}] "datasets" attribute is '
                            'not a dictionary - exiting...'
                        )

                # set the new title/labels in the main thread
                self.plot_settings.run_main(
                    self._new_source_callback, title, xlabel, ylabel, blocking=True
                )

                # add the existing plots
                with QtCore.QMutexLocker(self.plot_settings.mutex):
                    for plot_name in self.plot_settings.series_settings:
                        self.add_plot(plot_name)
                        if self.plot_settings.series_settings[plot_name].hidden:
                            self.hide_plot(plot_name)

                # force plot the data since we used the first pop() to extract the
                # plot info
                self.plot_settings.force_update = True
            except (TimeoutError, RuntimeError) as err:
                self.teardown()
                raise RuntimeError(
                    f'Could not connect to new data source [{data_set_name}]'
                ) from err

    def _new_source_callback(self, title, xlabel, ylabel):
        """Callback for when a new data source connects."""
        if title is not None:
            self.set_title(title)
        if xlabel is not None:
            self.xaxis.setLabel(text=xlabel)
        if ylabel is not None:
            self.yaxis.setLabel(text=ylabel)

    def teardown(self):
        """Clean up."""
        # run on the plot_settings thread since we'll need to acquire mutexes
        self.plot_settings.run_safe(self._close_source)

    def _close_source(self):
        """Disconnect from the data source."""
        with QtCore.QMutexLocker(self.plot_settings.sink_mutex):
            if self.plot_settings.sink is not None:
                self.plot_settings.sink.stop()
                self.plot_settings.sink = None

    def update(self):
        """Update the plot if there is new data available."""
        with QtCore.QMutexLocker(self.plot_settings.sink_mutex):
            if self.plot_settings.sink is None:
                # rate limit how often update() runs if there is no sink connected
                time.sleep(0.1)
                return

            if self.plot_settings.force_update:
                self.plot_settings.force_update = False
            else:
                try:
                    # wait for new data to be available from the sink
                    self.plot_settings.sink.pop(timeout=self.timeout)
                except TimeoutError:
                    return

            with QtCore.QMutexLocker(self.plot_settings.mutex):
                datasets = self.plot_settings.sink.datasets
                avg_cache = {}

                for plot_name in self.plot_settings.series_settings:
                    settings = self.plot_settings.series_settings[plot_name]
                    if settings.hidden:
                        continue
                    series = settings.series
                    scan_i = settings.scan_i
                    scan_j = settings.scan_j
                    processing = settings.processing
                    
                    scan_key = (scan_i, scan_j, processing)
                    
                    # pick out the particular data series
                    try:
                        if series in ('diff', 'div', 'contrast', 'fft'):
                            sig_name = 'signal'
                            bg_name = 'background'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'div_rf':
                            sig_name = 'rf_signal'
                            bg_name = 'rf_background'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'diff dq1':
                            sig_name = 'S0,-1'
                            bg_name = 'S0,0'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'diff dq2':
                            sig_name = 'S-1,+1'
                            bg_name = 'S-1,-1'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'deer_contrast' or series == 'deer_log_contrast' or series == 'deer_diff':
                            data_dark_sig = datasets['dark_signal']
                            data_dark_bg = datasets['dark_background']
                            data_echo_sig = datasets['echo_signal']
                            data_echo_bg = datasets['echo_background']
                        elif series == 'dark_contrast':
                            sig_name = 'dark_signal'
                            bg_name = 'dark_background'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'echo_contrast':
                            sig_name = 'echo_signal'
                            bg_name = 'echo_background'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'cd_contrast':
                            sig_name = 'cd_signal'
                            bg_name = 'cd_background'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]    
                        elif series == 'diff_py':
                            sig_name = 'with_py'
                            bg_name = 'without_py'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'diff_ny':
                            sig_name = 'with_ny'
                            bg_name = 'without_ny'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'diff_osc_with_pulse':
                            sig_name = 'with_py'
                            bg_name = 'with_ny'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'diff_osc_without_pulse':
                            sig_name = 'without_py'
                            bg_name = 'without_ny'
                            data_sig = datasets[sig_name]
                            data_bg = datasets[bg_name]
                        elif series == 'diff_overall' or series == 'sum_nuclear':
                            data_wpy = datasets['with_py']
                            data_wny = datasets['with_ny']
                            data_nopy = datasets['without_py']
                            data_nony = datasets['without_ny']
                        elif series == 'fit':
                            data_x_fit = datasets['x_fit']
                            data_y_fit = datasets['y_fit']
                        else:
                            data = datasets[series]

                    except KeyError:
                        # _logger.error(f'Data series [{series}] does not exist.')
                        continue

                    else:
                        # check for numpy array
                        # if not isinstance(data[0], np.ndarray):
                        #     raise ValueError(
                        #         f'Data series [{series}] must be a list of numpy '
                        #         'arrays, but the first list element has type '
                        #         f'[{type(data[0])}].'
                        #     )
                        # # check numpy array shape
                        # if data[0].shape[0] != 2 or len(data[0].shape) != 2:
                        #     raise ValueError(
                        #         f'Data series [{series}] first list element has '
                        #         f'shape {data.shape}, but should be (2, n).'
                        #     )

                        try:
                            if series == 'diff' or series == 'diff dq1' or series == 'diff dq2' or series == 'div' or series == 'div_rf' or series == 'contrast' or series == 'dark_contrast' or series == 'echo_contrast' or series == 'cd_contrast' or series == 'fft' or series == 'diff_py' or series == 'diff_ny' or series == 'diff_osc_with_pulse' or series == 'diff_osc_without_pulse':
                                if scan_i == '' and scan_j == '':
                                    data_subset_sig = data_sig[:]
                                    data_subset_bg = data_bg[:]
                                elif scan_j == '':
                                    data_subset_sig = data_sig[int(scan_i) :]
                                    data_subset_bg = data_bg[int(scan_i) :]
                                elif scan_i == '':
                                    data_subset_sig = data_sig[: int(scan_j)]
                                    data_subset_bg = data_bg[: int(scan_j)]
                                else:
                                    data_subset_sig = data_sig[int(scan_i) : int(scan_j)]
                                    data_subset_bg = data_bg[int(scan_i) : int(scan_j)]
                            elif series == 'deer_contrast' or series == 'deer_log_contrast' or series == 'deer_diff':
                                if scan_i == '' and scan_j == '':
                                    data_subset_dark_sig = data_dark_sig[:]
                                    data_subset_dark_bg = data_dark_bg[:]
                                    data_subset_echo_sig = data_echo_sig[:]
                                    data_subset_echo_bg = data_echo_bg[:]
                                elif scan_j == '':
                                    data_subset_dark_sig = data_dark_sig[int(scan_i) :]
                                    data_subset_dark_bg = data_dark_bg[int(scan_i) :]
                                    data_subset_echo_sig = data_echo_sig[int(scan_i) :]
                                    data_subset_echo_bg = data_echo_bg[int(scan_i) :]
                                elif scan_i == '':
                                    data_subset_dark_sig = data_dark_sig[: int(scan_j)]
                                    data_subset_dark_bg = data_dark_bg[: int(scan_j)]
                                    data_subset_echo_sig = data_echo_sig[: int(scan_j)]
                                    data_subset_echo_bg = data_echo_bg[: int(scan_j)]
                                else:
                                    data_subset_dark_sig = data_dark_sig[int(scan_i) : int(scan_j)]
                                    data_subset_dark_bg = data_dark_bg[int(scan_i) : int(scan_j)]
                                    data_subset_echo_sig = data_echo_sig[int(scan_i) : int(scan_j)]
                                    data_subset_echo_bg = data_echo_bg[int(scan_i) : int(scan_j)]
                            elif series == 'diff_overall' or series == 'sum_nuclear':
                                if scan_i == '' and scan_j == '':
                                    data_subset_wpy = data_wpy[:]
                                    data_subset_wny = data_wny[:]
                                    data_subset_nopy = data_nopy[:]
                                    data_subset_nony = data_nony[:]
                                elif scan_j == '':
                                    data_subset_wpy = data_wpy[int(scan_i) :]
                                    data_subset_wny = data_wny[int(scan_i) :]
                                    data_subset_nopy = data_nopy[int(scan_i) :]
                                    data_subset_nony = data_nony[int(scan_i) :]
                                elif scan_i == '':
                                    data_subset_wpy = data_wpy[: int(scan_j)]
                                    data_subset_wny = data_wny[: int(scan_j)]
                                    data_subset_nopy = data_nopy[: int(scan_j)]
                                    data_subset_nony = data_nony[: int(scan_j)]
                                else:
                                    data_subset_wpy = data_wpy[int(scan_i) : int(scan_j)]
                                    data_subset_wny = data_wny[int(scan_i) : int(scan_j)]
                                    data_subset_nopy = data_nopy[int(scan_i) : int(scan_j)]
                                    data_subset_nony = data_nony[int(scan_i) : int(scan_j)]
                            elif series == 'fit':
                                pass
                            else:
                                if scan_i == '' and scan_j == '':
                                    data_subset = data[:]
                                elif scan_j == '':
                                    data_subset = data[int(scan_i) :]
                                elif scan_i == '':
                                    data_subset = data[: int(scan_j)]
                                else:
                                    data_subset = data[int(scan_i) : int(scan_j)]

                        except IndexError:
                            _logger.warning(
                                f'Data series [{series}] invalid scan indices '
                                f'[{scan_i}, {scan_j}].'
                            )
                            continue

                        if processing == 'Append':
                            # concatenate the numpy arrays
                            processed_data = np.concatenate(data_subset, axis=1)

                        elif processing == 'Average':
                            if series in (
                                'diff', 'diff dq1', 'diff dq2', 'div', 'div_rf', 'contrast', 
                                'dark_contrast', 'echo_contrast', 'cd_contrast', 'fft', 
                                'diff_py', 'diff_ny', 'diff_osc_with_pulse', 'diff_osc_without_pulse'
                            ):
                                cache_key = (sig_name, bg_name, scan_key)

                                if cache_key in avg_cache:
                                    processed_data_sig, processed_data_bg = avg_cache[cache_key]
                                else:
                                    # create a single numpy array
                                    stacked_data_sig = np.stack(data_subset_sig)
                                    stacked_data_bg = np.stack(data_subset_bg)

                                    # average the numpy arrays
                                    processed_data_sig = np.nanmean(stacked_data_sig, axis=0)
                                    processed_data_bg = np.nanmean(stacked_data_bg, axis=0)

                                    avg_cache[cache_key] = (processed_data_sig, processed_data_bg)

                            elif series == 'deer_contrast' or series == 'deer_log_contrast' or series == 'deer_diff':
                                # create a single numpy array
                                stacked_data_dark_sig = np.stack(data_subset_dark_sig)
                                stacked_data_dark_bg = np.stack(data_subset_dark_bg)
                                stacked_data_echo_sig = np.stack(data_subset_echo_sig)
                                stacked_data_echo_bg = np.stack(data_subset_echo_bg)
                                
                                # average the numpy arrays
                                processed_data_dark_sig = np.nanmean(stacked_data_dark_sig, axis=0)
                                processed_data_dark_bg = np.nanmean(stacked_data_dark_bg, axis=0)
                                processed_data_echo_sig = np.nanmean(stacked_data_echo_sig, axis=0)
                                processed_data_echo_bg = np.nanmean(stacked_data_echo_bg, axis=0)
                            elif series == 'diff_overall' or series == 'sum_nuclear':
                                # create a single numpy array
                                stacked_data_wpy = np.stack(data_subset_wpy)
                                stacked_data_wny = np.stack(data_subset_wny)
                                stacked_data_nopy = np.stack(data_subset_nopy)
                                stacked_data_nony = np.stack(data_subset_nony)
                                
                                # average the numpy arrays
                                processed_data_wpy = np.nanmean(stacked_data_wpy, axis=0)
                                processed_data_wny = np.nanmean(stacked_data_wny, axis=0)
                                processed_data_nopy = np.nanmean(stacked_data_nopy, axis=0)
                                processed_data_nony = np.nanmean(stacked_data_nony, axis=0)
                            elif series == 'fit':
                                x = np.asarray(data_x_fit)
                                y = np.asarray(data_y_fit)

                                mask = np.isfinite(x) & np.isfinite(y)

                                processed_fit_data = [x[mask], y[mask]]
                            else:
                                # create a single numpy array
                                stacked_data = np.stack(data_subset)
                                
                                # average the numpy arrays
                                processed_data = np.nanmean(stacked_data, axis=0)
                        
                        else:
                            raise ValueError(
                                f'Processing has unsupported value [{processing}].'
                            )

                    # update the plot
                    try:
                        if series == 'diff' or series == 'diff dq1' or series == 'diff dq2' or series == 'diff_py' or series == 'diff_ny' or series == 'diff_osc_with_pulse' or series == 'diff_osc_without_pulse':
                            processed_data = [processed_data_sig[0], processed_data_bg[1] - processed_data_sig[1]]                        
                        elif series == 'div' or series == 'div_rf':
                            processed_data = [processed_data_sig[0], processed_data_sig[1] / processed_data_bg[1]]
                        elif series == 'contrast' or series == 'dark_contrast' or series == 'echo_contrast' or series == 'cd_contrast':
                            contrast = (processed_data_bg[1] - processed_data_sig[1]) / (processed_data_bg[1] + processed_data_sig[1])
                            processed_data = [processed_data_sig[0], contrast]
                        elif series == 'deer_contrast':
                            deer = (processed_data_dark_bg[1] - processed_data_dark_sig[1]) / (processed_data_dark_bg[1] + processed_data_dark_sig[1])
                            echo = (processed_data_echo_bg[1] - processed_data_echo_sig[1]) / (processed_data_echo_bg[1] + processed_data_echo_sig[1])
                            processed_data = [processed_data_dark_sig[0], deer / echo]
                        elif series == 'deer_log_contrast':
                            deer = (processed_data_dark_bg[1] - processed_data_dark_sig[1]) / (processed_data_dark_bg[1] + processed_data_dark_sig[1])
                            echo = (processed_data_echo_bg[1] - processed_data_echo_sig[1]) / (processed_data_echo_bg[1] + processed_data_echo_sig[1])
                            processed_data = [processed_data_dark_sig[0], np.log(-np.log(deer / echo))]
                        elif series == 'deer_diff':
                            deer = (processed_data_dark_bg[1] - processed_data_dark_sig[1]) / (processed_data_dark_bg[1] + processed_data_dark_sig[1])
                            echo = (processed_data_echo_bg[1] - processed_data_echo_sig[1]) / (processed_data_echo_bg[1] + processed_data_echo_sig[1])
                            processed_data = [processed_data_dark_sig[0], deer - echo]
                        elif series == 'diff_overall':
                            processed_data = [processed_data_wpy[0], (processed_data_nopy[1] - processed_data_wpy[1]) - (processed_data_nony[1] - processed_data_wny[1])]
                        elif series == 'sum_nuclear':
                            processed_data = [processed_data_wpy[0], (processed_data_wny[1] - processed_data_wpy[1]) + (processed_data_nony[1] - processed_data_nopy[1])]
                        elif series == 'fft':
                            time_axis = np.asarray(processed_data_sig[0], dtype=float)
                            sig = np.asarray(processed_data_sig[1], dtype=float)
                            bg = np.asarray(processed_data_bg[1], dtype=float)

                            time_trace = bg - sig # averaged time-domain signal
                            time_trace -= np.mean(time_trace) # zero-center the time trace before FFT

                            if len(time_axis) < 2:
                                continue

                            dt = time_axis[1] - time_axis[0] # sampling interval

                            ### --- FFT of averaged time trace --- ###
                            ft = rfft(time_trace)
                            freqs = rfftfreq(len(time_trace), d=dt)
                            
                            ### --- Boxcar smoothing in frequency space --- ###
                            boxcar_width = settings.boxcar_width
                            ft_smoothed = boxcar_complex_smooth(ft, boxcar_width)

                            processed_data = [freqs[1:], np.abs(ft_smoothed[1:])**2] # power spectrum - skip DC bin
                        elif series == 'fit':
                            processed_data = [processed_fit_data[0], processed_fit_data[1]]
                        else:
                            pass
                    
                    except KeyError:
                        # _logger.error(f'Data series [{series}] does not exist.')
                        continue

                    else:
                        self.set_data(plot_name, processed_data[0], processed_data[1])
                        
                        # print("updating data in dictionary...")
                        self.processed_data_dict[plot_name] = processed_data


class PlotColorManager:
    """Manage plot color schemes and shuffle colors on each new connection."""

    def __init__(self, plot_widget):
        """
        Args:
            plot_widget: Your FlexLinePlotWidget instance (self from FlexLinePlotWidget)
        """
        self.plot_widget = plot_widget

        # original palette
        self.base_colors = list(cyclic_colors)

        # shuffled per connection
        self.shuffled_colors = list(self.base_colors)
        self.plot_colors = {}   # plot_name -> QColor (or tuple)
        self.next_color_i = 0

        # current display mode
        self.mode = "dark"  # "light" or "dark"

        self.setup_color_controls()

    def setup_color_controls(self) -> None:
        self.color_flip_button = QtWidgets.QStackedWidget()
        self.light_plot_button = QtWidgets.QPushButton("Light Plot Mode")
        self.dark_plot_button = QtWidgets.QPushButton("Dark Plot Mode")

        self.light_plot_button.clicked.connect(self.light_plot)
        self.dark_plot_button.clicked.connect(self.dark_plot)

        # index 0: Light button shown, index 1: Dark button shown
        self.color_flip_button.addWidget(self.light_plot_button)
        self.color_flip_button.addWidget(self.dark_plot_button)

    # ---------- core helpers ----------

    def reshuffle_for_new_connection(self):
        """Call this once each time you connect to a dataset."""
        self.shuffled_colors = list(self.base_colors)
        random.shuffle(self.shuffled_colors)
        self.plot_colors.clear()
        self.next_color_i = 0

    def ensure_color(self, plot_name: str):
        """Assign a shuffled color to a plot name if it doesn't have one yet."""
        if plot_name not in self.plot_colors:
            self.plot_colors[plot_name] = self.shuffled_colors[
                self.next_color_i % len(self.shuffled_colors)
            ]
            self.next_color_i += 1

    def _named_curve_items(self) -> dict:
        """
        Return a mapping {plot_name: PlotDataItem} if possible.
        We try several common attribute names, and finally fall back to pyqtgraph's listDataItems().
        """
        lw = self.plot_widget.line_plot  # your _FlexLinePlotWidget / LinePlotWidget subclass

        # Try common dict attributes
        for attr in ("plots", "plot_items", "curves", "plotdataitems", "plotDataItems"):
            d = getattr(lw, attr, None)
            if isinstance(d, dict) and d:
                # ensure values are PlotDataItem-ish
                out = {}
                for k, v in d.items():
                    if isinstance(v, pg.PlotDataItem):
                        out[str(k)] = v
                if out:
                    return out

        # Fallback: ask the PlotItem for its data items
        try:
            items = self.plot_widget.line_plot.plot_widget.getPlotItem().listDataItems()
        except Exception:
            items = []

        out = {}
        for it in items:
            if not isinstance(it, pg.PlotDataItem):
                continue
            name = it.name()
            if not name:
                name = it.opts.get("name", None)
            if name:
                out[str(name)] = it
        return out

    def _data_items(self):
        """All curve items currently in the plot."""
        return self.plot_widget.line_plot.plot_widget.getPlotItem().listDataItems()

    def _style_plotdataitem(self, item: pg.PlotDataItem, color=None):
        """
        Apply thickness + marker visibility.
        If color is provided, also set color; otherwise keep existing color.
        """

        plot_name = item.name() or item.opts.get("name", "")
        is_fit = (plot_name == "fit")

        # keep existing pen color unless caller provides a color
        pen = pg.mkPen(item.opts.get("pen", None))

        # Special styling for fit only
        if is_fit:
            fit_color = "k" if self.mode == "light" else "w" # black fit for light mode, white fit for dark mode
            fit_pen = pg.mkPen(
                color=fit_color,
                width=5,
                style=Qt.PenStyle.SolidLine,
            )
            
            item.setPen(fit_pen)

            # fit should never have markers
            item.setSymbol(None)
            item.setSymbolBrush(None)
            item.setSymbolPen(None)
            if getattr(item, "scatter", None) is not None:
                item.scatter.setVisible(False)
            return

        if color is not None:
            pen.setColor(color)

        if self.mode == "light":
            pen.setWidth(8)
            item.setPen(pen)

            # hard kill symbols
            item.setSymbol(None)
            item.setSymbolBrush(None)
            item.setSymbolPen(None)

            # IMPORTANT: also hide the internal scatter item if it exists
            if getattr(item, "scatter", None) is not None:
                item.scatter.setVisible(False)

        else:
            pen.setWidth(6)
            item.setPen(pen)

            item.setSymbol("o")
            item.setSymbolSize(12)
            item.setSymbolBrush(pg.mkBrush(240, 240, 240, 100))
            item.setSymbolPen(pg.mkPen(240, 240, 240, 120))

            if getattr(item, "scatter", None) is not None:
                item.scatter.setVisible(True)

    def apply_current_style_all(self):
        """Force current style onto all curves *currently shown*."""
        for item in self._data_items():
            if isinstance(item, pg.PlotDataItem):
                self._style_plotdataitem(item)

    def light_plot(self):
        self.mode = "light"
        self.plot_widget.line_plot.plot_widget.setBackground("white")
        self.apply_current_style_all()
        self.color_flip_button.setCurrentIndex(1)

    def dark_plot(self):
        self.mode = "dark"
        self.plot_widget.line_plot.plot_widget.setBackground("k")
        self.apply_current_style_all()
        self.color_flip_button.setCurrentIndex(0)
