from nspyre import DataSink
from pyqtgraph import SpinBox, ComboBox
from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QRadioButton, QSlider, QDoubleSpinBox
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QWidget
from PyQt6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect, QApplication
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap
from PyQt6.QtCore import Qt, QTimer



import numpy as np
from nspyre import ParamsWidget
pi=np.pi
import sys
from pyqtgraph.Qt import QtWidgets

BOLTZMANN = 1.380649e-23  # J/K
AVOGADRO = 6.02214076e23  # mol^-1


class CalcWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.white_border = "border: 1px solid white"
        self.setStyleSheet("background-color: grey; QLineEdit, QLineEdit * { color: white }; QLabel, QLabel * { background: 2b2b2b; color: white }")
        self.font = "Arial"
        self.font_color= "white"

        self.diffusion_widget_init()
        self.b_widget_init()
        self.widgetlayout()
        self.resize(1300,900)

    def get_spin_value(self, SpinBox):
        return SpinBox.value()

    def get_combobox_val(self, combobox):
        return str(combobox.value())

    def diffusion_widget_init(self):
        self.diffuse_label = QLabel("Predicted diffusion, corr time, fwhm")
        self.diffuse_label.setFixedHeight(22)
        self.diffuse_label.setFont(QFont(self.font, 15))
        self.diffuse_label.setStyleSheet("color:white; background-color: #22512d")
        self.diffuse_params_widget = ParamsWidget(
            {
                'temperature': {
                    'display_text': 'Temp, K: ',
                    'widget': SpinBox(
                        value=298,
                        suffix='K',
                        siPrefix=False,
                        bounds=(0, 400),
                        dec=True,
                    ),
                },

                'density': {
                    'display_text': 'Sample density: ',
                    'widget': SpinBox(
                        suffix='g/ml',
                        siPrefix=True,
                        dec=True
                    )
                },

                'kinematic_viscosity': {
                    'display_text': 'Sample KV: ',
                    'widget': SpinBox(
                        suffix='St',
                        siPrefix=True,
                        dec=True
                    )
                },

                'viscosity': {
                    'display_text': 'Viscosity: ',
                    'widget': SpinBox(
                        suffix='P',
                        siPrefix=True,
                        dec=True
                    )
                },

                'hydro_radius': {
                    'display_text': 'Hydrodynamic r: ',
                    'widget': SpinBox(
                        value=1,
                        suffix='nm',
                        siPrefix=True,
                        dec=True
                    )
                },

                'NV_depth': {
                    'display_text': 'NV depth: ',
                    'widget': SpinBox(
                        value=7,
                        suffix='nm',
                        siPrefix=True,
                        dec=True
                    )
                },

            },
            get_param_value_funs={SpinBox: self.get_spin_value},

        )
        self.diffuse_params_widget.setStyleSheet("background-color: #22512d; color:white")

        self.diffusion_coef_label = QLabel("Diffusion coef, m^2/s: ")
        self.diffusion_coef_label.setFixedHeight(65)
        self.diffusion_coef_label.setFixedWidth(180)
        self.diffusion_coef_label.setStyleSheet("background-color: #4f7458; border-radius: 5px; color:white")
        self.diffusion_coef_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diffusion_coef_label.setFont(QFont(self.font, 8))

        self.tcorr_label = QLabel("Corr. time, \n microseconds: ")
        self.tcorr_label.setFixedHeight(65)
        self.tcorr_label.setFixedWidth(180)
        self.tcorr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tcorr_label.setStyleSheet("background-color: #4f7458; border-radius: 5px; color:white")
        self.tcorr_label.setFont(QFont(self.font, 8))

        self.fwhm_label = QLabel("FWHM, kHz: ")
        self.fwhm_label.setFixedHeight(65)
        self.fwhm_label.setFixedWidth(180)
        self.fwhm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fwhm_label.setStyleSheet("background-color: #4f7458; border-radius: 5px; color:white")
        self.fwhm_label.setFont(QFont(self.font, 8))

        self.calc_diffusion_button = QPushButton("Calculate")
        self.calc_diffusion_button.setStyleSheet("background-color:#9ab9a2")
        self.calc_diffusion_button.clicked.connect(self.calc_diffusion)

    def calc_diffusion(self):

        fun_kwargs = dict(**self.diffuse_params_widget.all_params())

        try:
            nv_d= fun_kwargs['NV_depth']
            temp_k= fun_kwargs['temperature']
            r_hydrodynamic= fun_kwargs['hydro_radius']
            visc = fun_kwargs['viscosity']

            diffusion = (BOLTZMANN * float(temp_k)) / (6 * pi * visc * r_hydrodynamic*10**-9)
            self.diffusion_coef_label.setText(f"Diffusion coef:\n {diffusion:.2e} m^2/s")

            corrtime = (2 * nv_d * nv_d) / diffusion /(10**6)**2
            self.tcorr_label.setText(f"Correlation time:\n {corrtime:.2f} microseconds")

            fwhm_theoretical = 2 / corrtime*(10**6)
            self.fwhm_label.setText(f"FWHM:\n {fwhm_theoretical/1000:.5f} kHz")

        except (ValueError, TypeError, ZeroDivisionError):
            dens = fun_kwargs['density']
            k_visc = fun_kwargs['kinematic_viscosity']
            visc = float(dens) * float(k_visc)

            diffusion = (BOLTZMANN * float(temp_k)) / (6 * pi * visc * r_hydrodynamic*10**-9)
            self.diffusion_coef_label.setText(f"Diffusion coef:\n {diffusion:.2e} m^2/s")

            corrtime = (2 * nv_d * nv_d) / diffusion /(10**6)**2
            self.tcorr_label.setText(f"Corr. time:\n {corrtime:.2f} microseconds")

            fwhm_theoretical = 2 / corrtime*(10**6)
            self.fwhm_label.setText(f"FWHM:\n {fwhm_theoretical/1000:.5f} kHz")

    def b_widget_init(self):
        self.b_label = QLabel("Field and resonance")
        self.b_label.setFixedHeight(22)
        self.b_label.setStyleSheet("color:white")
        self.b_label.setFont(QFont(self.font, 15))

        self.b_unit_select= QComboBox()
        self.b_unit_select.addItems(["T", "mT", "G"])
        self.b_unit_select.setStyleSheet("color:white; border: 1px solid white;")
        self.b_unit_select.setFixedWidth(60)
        self.b_input = QLineEdit("")
        self.b_input.setStyleSheet("color:white")
        self.b_input.setFixedWidth(75)

        self.nv_m_label= QLabel("NV minus: ")
        self.nv_m_label.setStyleSheet("background-color: #2f7c75; border-top-left-radius: 5px; border-top-right-radius: 5px; color:white;padding:5px")
        self.nv_m_label.setFixedHeight(50)
        self.nv_p_label= QLabel("NV plus: ")
        self.nv_p_label.setStyleSheet("background-color: #2f7c75; border-bottom-left-radius: 5px; border-bottom-right-radius: 5px; color:white;padding:5px")
        self.nv_p_label.setFixedHeight(50)

        self.b_calc_button = QPushButton("Calculate")
        self.b_calc_button.setFixedWidth(100)
        self.b_calc_button.setStyleSheet("background-color:#88bdb7")
        self.b_output_label = QLabel("Field in G:")
        self.b_output_label.setStyleSheet("color:white")
        self.b_output = QLabel("")
        self.b_output.setStyleSheet("color:white")
        self.b_calc_button.clicked.connect(self.convert_to_gauss)

        self.spin_select = QComboBox()
        self.spin_select.setFixedHeight(20)
        self.spin_select.setPlaceholderText("SPIN")
        self.spin_select.addItems(["H1", "N15", "N14", "C13", "F19", "Al27", "P31"])
        self.spin_select.currentIndexChanged.connect(lambda: self.calc_nuclear_resonance())
        self.spin_select.setStyleSheet("color:white; border:1px solid white")
        self.larmor_output = QLabel("Larmor. : ")
        self.larmor_output.setStyleSheet("background-color: #2f7c75; border-top-left-radius: 5px; border-top-right-radius: 5px; color:white;padding:5px;")
        self.larmor_output.setFixedHeight(40)
        self.period_output= QLabel("Period: ")
        self.period_output.setStyleSheet("background-color: #2f7c75; color:white;padding:5px")
        self.period_output.setFixedHeight(40)
        self.halfperiod_output= QLabel("Half period: ")
        self.halfperiod_output.setStyleSheet("background-color: #2f7c75; border-bottom-left-radius: 5px; border-bottom-right-radius: 5px; color:white;padding:5px")
        self.halfperiod_output.setFixedHeight(40)



    def convert_to_gauss(self):
        units = self.b_unit_select.currentText()
        B_field = float(self.b_input.text())
        match units:  # convert all B field units to Gauss for calculation
            case 'T':
                B_field = B_field * 1e4
            case 'mT':
                B_field = B_field * 10
            case _:
                pass  # default in G

        self.b_output.setText(f"{B_field}")

        ### computes NV resonances in [GHz] from a B field
        nv_zfs = 2.87  # GHz
        gyro_e = 2.8025  # MHz/G

        nv_minus1 = abs(nv_zfs - (gyro_e * B_field) / 1000)
        nv_plus1 = nv_zfs + (gyro_e * B_field) / 1000

        self.nv_m_label.setText(f"NV minus: {nv_minus1:6f}")
        self.nv_p_label.setText(f"NV minus: {nv_plus1:6f}")

    def calc_nuclear_resonance(self):
        ### computes nuclear spin resonance [MHz] for a B field
        bfield= float(self.b_output.text())
        spin = self.spin_select.currentText()
        match spin:  # gyromagnetic ratio in [MHz/T]
            case 'H1':
                gyro_n = 42.5775
            case 'N15':
                gyro_n = abs(-4.316)
            case 'N14':
                gyro_n = 3.077
            case 'C13':
                gyro_n = 10.7084
            case 'F19':
                gyro_n = 40.078
            case 'Al27':
                gyro_n = 11.103
            case 'P31':
                gyro_n = 17.235

        gyro_n = gyro_n * 1e-4  # convert to [MHz/G]

        larmor = gyro_n * bfield  # nuclear Larmor frequency

        period = 1 / larmor  # period in [us]
        half_period = period / 2

        self.larmor_output.setText(f"Larmor: {larmor:6f} MHz")
        self.period_output.setText(f"Period: {period:6f} us")
        self.halfperiod_output.setText(f"Half period: {half_period:6f} us")


    def widgetlayout(self):
        self.widgetlayout = QGridLayout()

        self.diffusion_frame = QFrame(self)
        self.diffusion_frame.setObjectName("dFrame")
        self.diffusion_frame.setStyleSheet(
            "QFrame#dFrame {background-color: #22512d;}")
        self.diffusion_layout = QGridLayout(self.diffusion_frame)
        self.diffusion_frame.setFixedHeight(325)
        self.diffusion_frame.setFixedWidth(450)
        self.diffusion_layout.setSpacing(0)
        self.diffusion_layout.addWidget(self.diffuse_label, 0, 0,1,0, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.diffuse_params_widget, 1, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.diffusion_coef_label, 2, 0, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.tcorr_label, 2, 1, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.fwhm_label, 2, 2, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.calc_diffusion_button, 1, 2, Qt.AlignmentFlag.AlignCenter)

        self.b_frame = QFrame(self)
        self.b_frame_layout = QGridLayout(self.b_frame)
        self.b_frame_layout.addWidget(self.b_label, 0, 0, Qt.AlignmentFlag.AlignCenter)
        self.b_frame.setStyleSheet("background-color: #07665d;")
        self.b_frame.setFixedWidth(600)
        self.b_frame.setFixedHeight(260)

        self.b_subframe = QFrame(self)
        self.b_subframe_layout = QGridLayout(self.b_subframe)
        self.b_subframe.setFixedWidth(600)
        self.b_subframe.setFixedHeight(200)
        self.b_subframe1 = QFrame(self)
        self.b_subframe1_layout = QHBoxLayout(self.b_subframe1)
        self.b_subframe1.setFixedWidth(400)
        self.b_subframe1.setFixedHeight(50)

        self.b_subframe1_layout.addWidget(self.b_input)
        self.b_subframe1_layout.addWidget(self.b_unit_select)
        self.b_subframe1_layout.addWidget(self.b_output_label)
        self.b_subframe1_layout.addWidget(self.b_output)
        self.b_subframe_layout.addWidget(self.b_subframe1,0,0)
        self.b_subframe_layout.addWidget(self.nv_m_label, 1,0)
        self.b_subframe_layout.addWidget(self.nv_p_label, 2,0)
        self.b_subframe_layout.addWidget(self.b_calc_button,3, 0, Qt.AlignmentFlag.AlignCenter)

        self.b_subframe2 = QFrame(self)
        self.b_subframe2_layout = QVBoxLayout(self.b_subframe2)

        self.b_subframe2_layout.addWidget(self.spin_select)
        self.b_subframe2.setFixedWidth(250)
        self.b_subframe2_layout.addWidget(self.larmor_output)
        self.b_subframe2_layout.addWidget(self.period_output)
        self.b_subframe2_layout.addWidget(self.halfperiod_output)

        self.b_subframe_layout.addWidget(self.b_subframe1,0,0)
        self.b_subframe_layout.addWidget(self.b_subframe2, 0, 1,3,1)

        self.b_frame_layout.addWidget(self.b_subframe, 1,0)

        self.widgetlayout.addWidget(self.diffusion_frame, 0, 0)
        self.widgetlayout.addWidget(self.b_frame, 0, 1)
        self.setLayout(self.widgetlayout)

