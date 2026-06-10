from nspyre import DataSink
from pyqtgraph import SpinBox, ComboBox
from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QRadioButton, QSlider, QDoubleSpinBox
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QWidget
from PyQt6.QtWidgets import QStackedWidget, QWidget, QGraphicsOpacityEffect, QApplication, QSpacerItem
from PyQt6.QtWidgets import QSizePolicy
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

        # Professional color scheme
        self.bg_dark = "#0f0f0f"
        self.bg_light = "#1a1a1a"
        self.bg_elevated = "#252525"
        self.accent_primary = "#0d7377"
        self.accent_secondary = "#14919b"
        self.accent_warm = "#b8956a"
        self.bg_surface = "#3a3a3a"
        self.text_primary = "#ffffff"
        self.text_secondary = "#e0e0e0"
        self.text_tertiary = "#a0a0a0"
        self.button_hover = "#14919b"
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.bg_dark};
                color: {self.text_primary};
            }}
        """)
        self.font = "Segoe UI"
        self.font_color = self.text_primary

        self.diffusion_widget_init()
        self.b_widget_init()
        self.power_widget_init()
        self.tir_widget_init()
        self.widgetlayout()
        self.resize(1400, 1250)
        # Set minimum size to prevent text from being hidden when window is resized
        # Height accounts for: Row 0 (430) + spacing (20) + Row 1 (230) + spacing (20) + Row 2 (200) + margins (40) = 940
        self.setMinimumHeight(950)
        self.setMinimumWidth(1050)

    def get_spin_value(self, SpinBox):
        return SpinBox.value()

    def get_combobox_val(self, combobox):
        return str(combobox.value())

    def diffusion_widget_init(self):
        self.diffuse_label = QLabel("Diffusion Parameters")
        self.diffuse_label.setFixedHeight(40)
        self.diffuse_label.setFont(QFont(self.font, 13, QFont.Weight.Bold))
        self.diffuse_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diffuse_label.setStyleSheet(f"color: {self.text_primary}; background-color: {self.accent_primary}; padding: 6px; border-radius: 4px;")
        
        self.diffuse_params_widget = ParamsWidget(
            {
                'temperature': {
                    'display_text': 'Temperature: ',
                    'widget': SpinBox(
                        value=298,
                        suffix=' K',
                        siPrefix=False,
                        bounds=(0, 400),
                        dec=True,
                    ),
                },

                'density': {
                    'display_text': 'Density: ',
                    'widget': SpinBox(
                        suffix=' g/ml',
                        siPrefix=True,
                        dec=True
                    )
                },

                'kinematic_viscosity': {
                    'display_text': 'Kinematic Viscosity: ',
                    'widget': SpinBox(
                        suffix=' St',
                        siPrefix=True,
                        dec=True
                    )
                },

                'viscosity': {
                    'display_text': 'Viscosity: ',
                    'widget': SpinBox(
                        suffix=' P',
                        siPrefix=True,
                        dec=True
                    )
                },

                'hydro_radius': {
                    'display_text': 'Hydrodynamic Radius: ',
                    'widget': SpinBox(
                        value=1,
                        suffix=' nm',
                        siPrefix=True,
                        dec=True
                    )
                },

                'NV_depth': {
                    'display_text': 'NV Depth: ',
                    'widget': SpinBox(
                        value=7,
                        suffix=' nm',
                        siPrefix=True,
                        dec=True
                    )
                },

            },
            get_param_value_funs={SpinBox: self.get_spin_value},

        )
        self.diffuse_params_widget.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}")

        # Output labels with consistent styling
        self.output_label_style = f"""
            background-color: {self.bg_elevated};
            color: {self.text_primary};
            padding: 8px;
            font-size: 14px;
            border-radius: 4px;
        """

        self.diffusion_coef_label = QLabel("Diffusion Coef:\n—")
        self.diffusion_coef_label.setFixedHeight(65)
        self.diffusion_coef_label.setFixedWidth(170)
        self.diffusion_coef_label.setStyleSheet(self.output_label_style)
        self.diffusion_coef_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diffusion_coef_label.setFont(QFont(self.font, 13))

        self.tcorr_label = QLabel("Correlation Time:\n—")
        self.tcorr_label.setFixedHeight(65)
        self.tcorr_label.setFixedWidth(170)
        self.tcorr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tcorr_label.setStyleSheet(self.output_label_style)
        self.tcorr_label.setFont(QFont(self.font, 13))

        self.fwhm_label = QLabel("FWHM:\n—")
        self.fwhm_label.setFixedHeight(65)
        self.fwhm_label.setFixedWidth(170)
        self.fwhm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fwhm_label.setStyleSheet(self.output_label_style)
        self.fwhm_label.setFont(QFont(self.font, 13))

        self.calc_diffusion_button = QPushButton("Calculate")
        self.calc_diffusion_button.setFixedHeight(40)
        self.calc_diffusion_button.setFont(QFont(self.font, 12, QFont.Weight.Bold))
        self.calc_diffusion_button.setStyleSheet(f"QPushButton {{background-color: {self.accent_primary}; color: {self.text_primary}; border: none; border-radius: 6px; padding: 6px; font-weight: bold;}} QPushButton:hover {{background-color: {self.accent_secondary};}} QPushButton:pressed {{background-color: {self.accent_primary};}}")
        self.calc_diffusion_button.clicked.connect(self.calc_diffusion)

    def calc_diffusion(self):

        fun_kwargs = dict(**self.diffuse_params_widget.all_params())

        try:
            nv_d= fun_kwargs['NV_depth']
            temp_k= fun_kwargs['temperature']
            r_hydrodynamic= fun_kwargs['hydro_radius']
            visc = fun_kwargs['viscosity']

            diffusion = (BOLTZMANN * float(temp_k)) / (6 * pi * visc * r_hydrodynamic*10**-9)
            self.diffusion_coef_label.setText(f"Diffusion Coef:\n{diffusion:.2e} m²/s")

            corrtime = (2 * nv_d * nv_d) / diffusion /(10**6)**2
            self.tcorr_label.setText(f"Correlation Time:\n{corrtime:.2f} µs")

            fwhm_theoretical = 2 / corrtime*(10**6)
            self.fwhm_label.setText(f"FWHM:\n{fwhm_theoretical/1000:.5f} kHz")

        except (ValueError, TypeError, ZeroDivisionError):
            dens = fun_kwargs['density']
            k_visc = fun_kwargs['kinematic_viscosity']
            visc = float(dens) * float(k_visc)

            diffusion = (BOLTZMANN * float(temp_k)) / (6 * pi * visc * r_hydrodynamic*10**-9)
            self.diffusion_coef_label.setText(f"Diffusion Coef:\n{diffusion:.2e} m²/s")

            corrtime = (2 * nv_d * nv_d) / diffusion /(10**6)**2
            self.tcorr_label.setText(f"Correlation Time:\n{corrtime:.2f} µs")

            fwhm_theoretical = 2 / corrtime*(10**6)
            self.fwhm_label.setText(f"FWHM:\n{fwhm_theoretical/1000:.5f} kHz")

    def b_widget_init(self):
        self.b_label = QLabel("Magnetic Field & Nuclear Resonance")
        self.b_label.setFixedHeight(40)
        self.b_label.setFont(QFont(self.font, 13, QFont.Weight.Bold))
        self.b_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.b_label.setStyleSheet(f"color: {self.text_primary}; background-color: {self.accent_primary}; padding: 6px; border-radius: 4px;")

        self.b_unit_select = QComboBox()
        self.b_unit_select.addItems(["G", "T", "mT"])
        self.b_unit_select.setFixedWidth(70)
        self.b_unit_select.setFixedHeight(30)
        
        self.b_input = QLineEdit("")
        self.b_input.setFixedWidth(100)
        self.b_input.setFixedHeight(30)
        self.b_input.setPlaceholderText("Enter value")

        # NV resonance labels
        self.nv_m_label = QLabel("NV − Resonance:\n—")
        self.nv_m_label.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}; padding: 8px; border-radius: 4px;")
        self.nv_m_label.setFixedHeight(65)
        self.nv_m_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nv_m_label.setWordWrap(True)
        self.nv_m_label.setFont(QFont(self.font, 12))
        
        self.nv_p_label = QLabel("NV + Resonance:\n—")
        self.nv_p_label.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}; padding: 8px; border-radius: 4px;")
        self.nv_p_label.setFixedHeight(65)
        self.nv_p_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nv_p_label.setWordWrap(True)
        self.nv_p_label.setFont(QFont(self.font, 12))

        self.b_calc_button = QPushButton("Calculate")
        self.b_calc_button.setFixedWidth(120)
        self.b_calc_button.setFixedHeight(35)
        self.b_calc_button.setFont(QFont(self.font, 11, QFont.Weight.Bold))
        self.b_calc_button.setStyleSheet(f"QPushButton {{background-color: {self.accent_secondary}; color: {self.text_primary}; border: none; border-radius: 6px; padding: 4px; font-weight: bold;}} QPushButton:hover {{background-color: {self.accent_primary};}} QPushButton:pressed {{background-color: {self.accent_secondary};}}")
        self.b_calc_button.clicked.connect(self.convert_to_gauss)

        self.b_output_label = QLabel("Field (Gauss):")
        self.b_output_label.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}; padding: 8px; border-radius: 4px;")
        self.b_output_label.setFixedHeight(65)
        self.b_output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.b_output_label.setWordWrap(True)
        self.b_output_label.setFont(QFont(self.font, 12))
        self.b_output = QLabel("—")
        self.b_output.setStyleSheet(f"color: {self.accent_secondary}; font-weight: bold; font-size: 13px;")

        self.spin_label = QLabel("Nuclear Spin:")
        self.spin_label.setStyleSheet(f"color: {self.text_secondary}; font-size: 14px; font-weight: bold;")
        
        self.spin_select = QComboBox()
        self.spin_select.setFixedWidth(130)
        self.spin_select.setFixedHeight(30)
        self.spin_select.setPlaceholderText("Select Spin")
        self.spin_select.addItems(["H1", "N15", "N14", "C13", "F19", "Al27", "P31"])

        # Nuclear resonance output labels
        self.larmor_output = QLabel("Larmor Frequency:\n—")
        self.larmor_output.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}; padding: 8px; border-radius: 4px;")
        self.larmor_output.setFixedHeight(65)
        self.larmor_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.larmor_output.setWordWrap(True)
        self.larmor_output.setFont(QFont(self.font, 12))
        
        self.period_output = QLabel("Period:\n—")
        self.period_output.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}; padding: 8px; border-radius: 4px;")
        self.period_output.setFixedHeight(65)
        self.period_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.period_output.setWordWrap(True)
        self.period_output.setFont(QFont(self.font, 12))
        
        self.halfperiod_output = QLabel("Half Period:\n—")
        self.halfperiod_output.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}; padding: 8px; border-radius: 4px;")
        self.halfperiod_output.setFixedHeight(65)
        self.halfperiod_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.halfperiod_output.setWordWrap(True)
        self.halfperiod_output.setFont(QFont(self.font, 12))

    def power_widget_init(self):
        self.power_label = QLabel("Power Unit Conversion")
        self.power_label.setFixedHeight(40)
        self.power_label.setFont(QFont(self.font, 13, QFont.Weight.Bold))
        self.power_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.power_label.setStyleSheet(f"color: {self.text_primary}; background-color: {self.accent_primary}; padding: 6px; border-radius: 4px;")

        self.power_unit_select = QComboBox()
        self.power_unit_select.addItems(["W", "mW", "dBm"])
        self.power_unit_select.setFixedWidth(70)
        self.power_unit_select.setFixedHeight(30)
        
        self.power_input = QLineEdit("")
        self.power_input.setFixedWidth(100)
        self.power_input.setFixedHeight(30)
        self.power_input.setPlaceholderText("Enter value")

        self.power_calc_button = QPushButton("Calculate")
        self.power_calc_button.setFixedWidth(120)
        self.power_calc_button.setFixedHeight(35)
        self.power_calc_button.setFont(QFont(self.font, 11, QFont.Weight.Bold))
        self.power_calc_button.setStyleSheet(f"QPushButton {{background-color: {self.accent_primary}; color: {self.text_primary}; border: none; border-radius: 6px; padding: 4px; font-weight: bold;}} QPushButton:hover {{background-color: {self.accent_secondary};}} QPushButton:pressed {{background-color: {self.accent_primary};}}")
        self.power_calc_button.clicked.connect(self.convert_power)

        # Power output labels
        self.watts_output_label = QLabel("Watts (W):")
        self.watts_output_label.setStyleSheet(f"color: {self.text_secondary}; font-size: 14px; font-weight: bold;")
        self.watts_output = QLabel("—")
        self.watts_output.setStyleSheet(f"color: {self.text_primary}; font-weight: bold; font-size: 11px;")

        self.milliwatts_output_label = QLabel("Milliwatts (mW):")
        self.milliwatts_output_label.setStyleSheet(f"color: {self.text_secondary}; font-size: 14px; font-weight: bold;")
        self.milliwatts_output = QLabel("—")
        self.milliwatts_output.setStyleSheet(f"color: {self.text_primary}; font-weight: bold; font-size: 11px;")

        self.dbm_output_label = QLabel("Decibel-milliwatts (dBm):")
        self.dbm_output_label.setStyleSheet(f"color: {self.text_secondary}; font-size: 14px; font-weight: bold;")
        self.dbm_output = QLabel("—")
        self.dbm_output.setStyleSheet(f"color: {self.text_primary}; font-weight: bold; font-size: 11px;")

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

        self.b_output.setText(f"{B_field:.2f} G")

        ### computes NV resonances in [GHz] from a B field
        nv_zfs = 2.87  # GHz
        gyro_e = 2.8025  # MHz/G

        nv_minus1 = abs(nv_zfs - (gyro_e * B_field) / 1000)
        nv_plus1 = nv_zfs + (gyro_e * B_field) / 1000

        # Use HTML formatting to make NV values stand out
        self.nv_m_label.setText(f"NV − Resonance:<br/><span style='color: {self.accent_secondary}; font-weight: bold;'>{nv_minus1:.4f} GHz</span>")
        self.nv_p_label.setText(f"NV + Resonance:<br/><span style='color: {self.accent_secondary}; font-weight: bold;'>{nv_plus1:.4f} GHz</span>")
        
        # Calculate nuclear resonance now that we have a B field
        self.calc_nuclear_resonance()

    def calc_nuclear_resonance(self):
        ### computes nuclear spin resonance [MHz] for a B field
        bfield= float(self.b_output.text().split()[0]) if self.b_output.text() != "—" else 0
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
            case _:
                # Keep dashes if no valid spin selected
                self.larmor_output.setText(f"Larmor Frequency:\n—")
                self.period_output.setText(f"Period:\n—")
                self.halfperiod_output.setText(f"Half Period:\n—")
                return

        gyro_n = gyro_n * 1e-4  # convert to [MHz/G]

        larmor = gyro_n * bfield  # nuclear Larmor frequency

        period = 1 / larmor if larmor != 0 else 0  # period in [us]
        half_period = period / 2

        # Use HTML formatting to make values brighter/more prominent
        self.larmor_output.setText(f"Larmor Frequency:<br/><span style='color: {self.accent_secondary}; font-weight: bold;'>{larmor:.6f} MHz</span>")
        self.period_output.setText(f"Period:<br/><span style='color: {self.accent_secondary}; font-weight: bold;'>{period:.6f} µs</span>")
        self.halfperiod_output.setText(f"Half Period:<br/><span style='color: {self.accent_secondary}; font-weight: bold;'>{half_period:.6f} µs</span>")

    def convert_power(self):
        """Convert between power units: Watts, mW, and dBm"""
        try:
            unit = self.power_unit_select.currentText()
            value = float(self.power_input.text())
            
            # Convert input to Watts first
            if unit == "W":
                watts = value
            elif unit == "mW":
                watts = value / 1000.0
            elif unit == "dBm":
                watts = 10.0 ** (value / 10.0) / 1000.0
            
            # Convert to all units
            milliwatts = watts * 1000.0
            if watts > 0:
                dbm = 10.0 * np.log10(milliwatts)
            else:
                dbm = float('-inf')
            
            # Update output labels
            self.watts_output.setText(f"{watts:.6e} W")
            self.milliwatts_output.setText(f"{milliwatts:.6e} mW")
            self.dbm_output.setText(f"{dbm:.4f} dBm")
            
        except (ValueError, TypeError):
            self.watts_output.setText("—")
            self.milliwatts_output.setText("—")
            self.dbm_output.setText("—")

    def tir_widget_init(self):
        self.tir_label = QLabel("Total Internal Reflection")
        self.tir_label.setFixedHeight(40)
        self.tir_label.setFont(QFont(self.font, 13, QFont.Weight.Bold))
        self.tir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tir_label.setStyleSheet(f"color: {self.text_primary}; background-color: {self.accent_primary}; padding: 6px; border-radius: 4px;")

        self.medium1_label = QLabel("Medium 1:")
        self.medium1_label.setStyleSheet(f"color: {self.text_secondary}; font-size: 14px; font-weight: bold;")

        self.medium1_select = QComboBox()
        self.medium1_select.addItems(["Air", "Water", "Glass", "Im. Oil", "Diamond"])
        self.medium1_select.setCurrentText("Diamond")
        self.medium1_select.setFixedWidth(120)
        self.medium1_select.setFixedHeight(30)
        self.medium1_select.currentIndexChanged.connect(self.calc_tir_snell)

        self.medium2_label = QLabel("Medium 2:")
        self.medium2_label.setStyleSheet(f"color: {self.text_secondary}; font-size: 14px; font-weight: bold;")

        self.medium2_select = QComboBox()
        self.medium2_select.addItems(["Air", "Water", "Glass", "Im. Oil", "Diamond"])
        self.medium2_select.setCurrentText("Water")
        self.medium2_select.setFixedWidth(120)
        self.medium2_select.setFixedHeight(30)
        self.medium2_select.currentIndexChanged.connect(self.calc_tir_snell)

        self.tir_angle_label = QLabel("Critical Angle:\n—")
        self.tir_angle_label.setStyleSheet(f"background-color: {self.bg_elevated}; color: {self.text_primary}; padding: 8px; border-radius: 4px;")
        self.tir_angle_label.setFixedHeight(65)
        
        # Calculate initial critical angle
        self.calc_tir_snell()

    def calc_tir_snell(self):
        """Calculate critical angle for total internal reflection using Snell's law. Assume 532 nm wavelength for dispersion effects."""
        medium1 = self.medium1_select.currentText()
        medium2 = self.medium2_select.currentText()

        # Refractive indices
        refractive_indices = {
            "Air": 1.0003,
            "Water": 1.333,
            "Glass": 1.5,
            "Im. Oil": 1.518,
            "Diamond": 2.42
        }

        n1 = refractive_indices[medium1]
        n2 = refractive_indices[medium2]

        if n1 > n2:
            critical_angle_rad = np.arcsin(n2 / n1)
            critical_angle_deg = np.degrees(critical_angle_rad)
            self.tir_angle_label.setText(f"Critical Angle:\n{critical_angle_deg:.2f}°")
        else:
            self.tir_angle_label.setText("Critical Angle:\n— (No TIR)")

    def widgetlayout(self):
        self.main_layout = QGridLayout()
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Diffusion Frame
        self.diffusion_frame = QFrame(self)
        self.diffusion_frame.setObjectName("dFrame")
        self.diffusion_frame.setStyleSheet(f"QFrame#dFrame {{background-color: {self.bg_light}; border-radius: 6px; border: 2px solid {self.accent_primary}; box-shadow: 0 0 12px rgba(13, 115, 119, 0.25);}}")
        self.diffusion_layout = QGridLayout(self.diffusion_frame)
        self.diffusion_layout.setSpacing(12)
        self.diffusion_layout.setContentsMargins(15, 15, 15, 15)
        self.diffusion_frame.setMinimumHeight(410)
        self.diffusion_frame.setMinimumWidth(480)
        
        self.diffusion_layout.addWidget(self.diffuse_label, 0, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.diffuse_params_widget, 1, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.diffusion_coef_label, 2, 0, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.tcorr_label, 2, 1, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.fwhm_label, 2, 2, Qt.AlignmentFlag.AlignCenter)
        self.diffusion_layout.addWidget(self.calc_diffusion_button, 3, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)

        # B Field Frame
        self.b_frame = QFrame(self)
        self.b_frame_layout = QGridLayout(self.b_frame)
        self.b_frame_layout.setSpacing(12)
        self.b_frame_layout.setContentsMargins(15, 15, 15, 15)
        self.b_frame_layout.addWidget(self.b_label, 0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        self.b_frame.setStyleSheet(f"background-color: {self.bg_light}; border-radius: 6px; border: 2px solid {self.accent_secondary}; box-shadow: 0 0 12px rgba(20, 145, 155, 0.25);")
        self.b_frame.setMinimumWidth(520)
        self.b_frame.setMinimumHeight(430)

        # B Field Input Subframe
        self.b_input_frame = QFrame(self)
        self.b_input_frame.setStyleSheet(f"background-color: {self.bg_elevated}; border: none; border-radius: 4px;")
        self.b_input_layout = QHBoxLayout(self.b_input_frame)
        self.b_input_layout.setSpacing(10)
        self.b_input_layout.setContentsMargins(0, 0, 0, 0)
        self.b_input_layout.addWidget(QLabel("B Field:"), 0, Qt.AlignmentFlag.AlignRight)
        self.b_input_layout.addWidget(self.b_input)
        self.b_input_layout.addWidget(self.b_unit_select)
        self.b_input_layout.addWidget(self.b_calc_button)
        
        # NV Resonances
        self.nv_frame = QFrame(self)
        self.nv_frame.setStyleSheet(f"background-color: transparent; border: none;")
        self.nv_layout = QVBoxLayout(self.nv_frame)
        self.nv_layout.setSpacing(6)
        self.nv_layout.setContentsMargins(0, 0, 0, 0)
        
        # B Field output (now in NV frame)
        self.b_output_subframe = QFrame(self)
        self.b_output_subframe.setStyleSheet(f"background-color: transparent; border: none;")
        self.b_output_sublayout = QHBoxLayout(self.b_output_subframe)
        self.b_output_sublayout.setSpacing(6)
        self.b_output_sublayout.setContentsMargins(0, 0, 0, 0)
        self.b_output_sublayout.addWidget(self.b_output_label, 0, Qt.AlignmentFlag.AlignRight)
        self.b_output_sublayout.addWidget(self.b_output, 0, Qt.AlignmentFlag.AlignLeft)
        self.b_output_sublayout.addStretch()
        self.nv_layout.addWidget(self.b_output_subframe)
        
        self.nv_layout.addWidget(self.nv_m_label)
        self.nv_layout.addWidget(self.nv_p_label)

        # Nuclear Resonances
        self.nuclear_frame = QFrame(self)
        self.nuclear_frame.setStyleSheet(f"background-color: transparent; border: none;")
        self.nuclear_layout = QVBoxLayout(self.nuclear_frame)
        self.nuclear_layout.setSpacing(0)
        self.nuclear_layout.setContentsMargins(0, 0, 0, 0)
        self.nuclear_layout.addWidget(self.larmor_output)
        self.nuclear_layout.addWidget(self.period_output)
        self.nuclear_layout.addWidget(self.halfperiod_output)

        # Spin Selection
        self.spin_frame = QFrame(self)
        self.spin_frame.setStyleSheet(f"background-color: transparent; border: none;")
        self.spin_layout = QVBoxLayout(self.spin_frame)
        self.spin_layout.setSpacing(8)
        self.spin_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_layout.addWidget(self.spin_label)
        self.spin_layout.addWidget(self.spin_select)

        # Right side content (B Field resonances and nuclear resonances)
        self.b_content_frame = QFrame(self)
        self.b_content_frame.setStyleSheet(f"background-color: {self.bg_elevated}; border: none; border-radius: 4px;")
        self.b_content_layout = QHBoxLayout(self.b_content_frame)
        self.b_content_layout.setSpacing(12)
        self.b_content_layout.setContentsMargins(0, 0, 0, 0)
        self.b_content_layout.addWidget(self.nv_frame, 1)
        self.b_content_layout.addWidget(self.spin_frame)
        self.b_content_layout.addWidget(self.nuclear_frame, 1)

        # Add all to main B frame layout
        self.b_frame_layout.addWidget(self.b_input_frame, 1, 0, 1, 2)
        self.b_frame_layout.addWidget(self.b_content_frame, 2, 0, 1, 2)

        # Power Frame
        self.power_frame = QFrame(self)
        self.power_frame_layout = QGridLayout(self.power_frame)
        self.power_frame_layout.setSpacing(12)
        self.power_frame_layout.setContentsMargins(15, 15, 15, 15)
        self.power_frame_layout.addWidget(self.power_label, 0, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        self.power_frame.setStyleSheet(f"background-color: {self.bg_light}; border-radius: 6px; border: 2px solid {self.accent_primary}; box-shadow: 0 0 12px rgba(13, 115, 119, 0.25);")
        self.power_frame.setMinimumHeight(230)
        self.power_frame.setMinimumWidth(500)

        # Power Input Subframe
        self.power_input_frame = QFrame(self)
        self.power_input_frame.setStyleSheet(f"background-color: {self.bg_elevated}; border: none; border-radius: 4px;")
        self.power_input_layout = QHBoxLayout(self.power_input_frame)
        self.power_input_layout.setSpacing(10)
        self.power_input_layout.setContentsMargins(0, 0, 0, 0)
        self.power_input_layout.addWidget(QLabel("Power:"), 0, Qt.AlignmentFlag.AlignRight)
        self.power_input_layout.addWidget(self.power_input)
        self.power_input_layout.addWidget(self.power_unit_select)
        self.power_input_layout.addWidget(self.power_calc_button)
        self.power_input_layout.addStretch()
        
        # Power Outputs Frame
        self.power_output_frame = QFrame(self)
        self.power_output_frame.setStyleSheet(f"background-color: {self.bg_elevated}; border: none; border-radius: 4px;")
        self.power_output_layout = QVBoxLayout(self.power_output_frame)
        self.power_output_layout.setSpacing(8)
        self.power_output_layout.setContentsMargins(0, 0, 0, 0)

        # Create output pairs
        self.watts_pair_layout = QHBoxLayout()
        self.watts_pair_layout.setSpacing(10)
        self.watts_pair_layout.addWidget(self.watts_output_label, 0, Qt.AlignmentFlag.AlignRight)
        self.watts_pair_layout.addWidget(self.watts_output)
        self.watts_pair_layout.addStretch()
        
        self.milliwatts_pair_layout = QHBoxLayout()
        self.milliwatts_pair_layout.setSpacing(10)
        self.milliwatts_pair_layout.addWidget(self.milliwatts_output_label, 0, Qt.AlignmentFlag.AlignRight)
        self.milliwatts_pair_layout.addWidget(self.milliwatts_output)
        self.milliwatts_pair_layout.addStretch()
        
        self.dbm_pair_layout = QHBoxLayout()
        self.dbm_pair_layout.setSpacing(10)
        self.dbm_pair_layout.addWidget(self.dbm_output_label, 0, Qt.AlignmentFlag.AlignRight)
        self.dbm_pair_layout.addWidget(self.dbm_output)
        self.dbm_pair_layout.addStretch()

        self.power_output_layout.addLayout(self.watts_pair_layout)
        self.power_output_layout.addLayout(self.milliwatts_pair_layout)
        self.power_output_layout.addLayout(self.dbm_pair_layout)

        self.power_frame_layout.addWidget(self.power_input_frame, 1, 0, 1, 2)
        self.power_frame_layout.addWidget(self.power_output_frame, 2, 0, 1, 2)

        # TIR Frame
        self.tir_frame = QFrame(self)
        self.tir_frame_layout = QGridLayout(self.tir_frame)
        self.tir_frame_layout.setSpacing(12)
        self.tir_frame_layout.setContentsMargins(15, 15, 15, 15)
        self.tir_frame_layout.addWidget(self.tir_label, 0, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)
        self.tir_frame.setStyleSheet(f"background-color: {self.bg_light}; border-radius: 6px; border: 2px solid {self.accent_secondary}; box-shadow: 0 0 12px rgba(20, 145, 155, 0.25);")
        self.tir_frame.setMinimumHeight(200)
        self.tir_frame.setMinimumWidth(500)

        # TIR Input Subframe
        self.tir_input_frame = QFrame(self)
        self.tir_input_frame.setStyleSheet(f"background-color: {self.bg_elevated}; border: none; border-radius: 4px;")
        self.tir_input_layout = QGridLayout(self.tir_input_frame)
        self.tir_input_layout.setSpacing(10)
        self.tir_input_layout.setContentsMargins(0, 0, 0, 0)
        
        # Medium 1 selection
        self.tir_input_layout.addWidget(self.medium1_label, 0, 0, Qt.AlignmentFlag.AlignRight)
        self.tir_input_layout.addWidget(self.medium1_select, 0, 1, Qt.AlignmentFlag.AlignLeft)
        
        # Medium 2 selection
        self.tir_input_layout.addWidget(self.medium2_label, 0, 2, Qt.AlignmentFlag.AlignRight)
        self.tir_input_layout.addWidget(self.medium2_select, 0, 3, Qt.AlignmentFlag.AlignLeft)
        self.tir_input_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum), 0, 4)

        # TIR Output
        self.tir_output_frame = QFrame(self)
        self.tir_output_frame.setStyleSheet(f"background-color: {self.bg_elevated}; border: none; border-radius: 4px;")
        self.tir_output_layout = QHBoxLayout(self.tir_output_frame)
        self.tir_output_layout.setSpacing(10)
        self.tir_output_layout.setContentsMargins(0, 0, 0, 0)
        self.tir_output_layout.addStretch()
        self.tir_output_layout.addWidget(self.tir_angle_label, 0, Qt.AlignmentFlag.AlignCenter)
        self.tir_output_layout.addStretch()

        self.tir_frame_layout.addWidget(self.tir_input_frame, 1, 0, 1, 3)
        self.tir_frame_layout.addWidget(self.tir_output_frame, 2, 0, 1, 3)

        # Main layout - 3 rows
        self.main_layout.addWidget(self.diffusion_frame, 0, 0)
        self.main_layout.addWidget(self.b_frame, 0, 1)
        self.main_layout.addWidget(self.power_frame, 1, 0, 1, 2)
        self.main_layout.addWidget(self.tir_frame, 2, 0, 1, 2)
        self.setLayout(self.main_layout)

