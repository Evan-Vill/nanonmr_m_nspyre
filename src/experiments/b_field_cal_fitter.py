# %% Imports and file loading
from cProfile import label

from collections import defaultdict
from scipy.optimize import curve_fit, fsolve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

mag_cal_file = "b_field_calibration_2x2x2Mag_May2026_neg35deg_cal_interpolated.xlsx"
df = pd.read_excel(mag_cal_file, sheet_name="Calibration Matrix", header=None)

numeric_df = df.apply(pd.to_numeric, errors='coerce')  # convert all values to numeric, non-convertible values become NaN

print(f"Numeric data type: {numeric_df.dtypes}")  # check data types of each column


# %% Testing data value retrieval

num_z = numeric_df.shape[0] - 1  # number of z positions
num_azi = numeric_df.shape[1] - 1  # number of azimuthal angles

z_positions = numeric_df.iloc[1:num_z + 1, 0].tolist()  # z positions are in the first column, starting from the second row (Row 1)
azi_angles = numeric_df.iloc[0, 1:num_azi + 1].tolist()  # azimuthal angles are in the first row, starting from the second column (Column 1)

print(f'Number of z positions: {num_z}')
print(f'Number of azimuthal angles: {num_azi}')
# print(f"Calibration data at (0,1): {numeric_df.iloc[0, 1]}")
# print(f"Calibration data at (0,{num_azi}): {numeric_df.iloc[0, num_azi]}")

print(f"Z positions list: {z_positions}")  # should print the z positions
print(f"Azimuthal angles list: {azi_angles}")  # should print the azimuthal angles
print(f"Azi angles type: {type(azi_angles)}")  # check the type of azi_angles


# %% Determine z range for each azimuthal angle
b_field_calibration_data = defaultdict(dict)

for i in range(0, num_azi):
    b_values = numeric_df.iloc[1:num_z + 1, i + 1].dropna()  # get z values for the current azimuthal angle, excluding NaN
    z_values = z_positions[0: len(b_values)]  # get corresponding z positions for the non-NaN b values
    if not b_values.empty:
        # print(f"Len z values for azimuthal angle {azi_angles[i]} degrees: {len(z_values)}")
        b_field_calibration_data[azi_angles[i]]['b_measured_values'] = b_values
        b_field_calibration_data[azi_angles[i]]['b_max'] = b_values.max()  # max b corresponds to the first entry in b_values
        b_field_calibration_data[azi_angles[i]]['b_min'] = b_values.min()  # min b corresponds to the last entry in b_values
        b_field_calibration_data[azi_angles[i]]['z_stage_values'] = z_values
        b_field_calibration_data[azi_angles[i]]['z_max'] = z_values[0]  # max z corresponds to the first entry in z_values
        b_field_calibration_data[azi_angles[i]]['z_min'] = z_values[-1]  # min z corresponds to the last entry in z_values
        b_field_calibration_data[azi_angles[i]]['num_z_values'] = len(b_values)

print("Z ranges for each azimuthal angle:")
for azi, data in b_field_calibration_data.items():
    print(f"Azimuthal angle {azi} degrees: b range = ({data['b_min']}, {data['b_max']}) mm")
    print(f"Corresponding z range = ({data['z_min']}, {data['z_max']}) mm")
    print(f"Num. z values for azimuthal angle {azi} degrees: {data['num_z_values']}")


# %% Test data fit and display
# def fit_exp(z, a, b, c, d):
#     return a * np.exp(-(b / z) ** c) + d

# def fit_poly5th(z, a, b, c, d, e, f):
#     return a * z**5 + b * z**4 + c * z**3 + d * z**2 + e * z + f

# def fit_poly3rd(z, a, b, c, d):
#     return a * z**3 + b * z**2 + c * z + d

def fit_inv_cube(z, a, b, c):
    return a / (z + b)**3 + c

azi_to_fit = 100  # degrees
b, z, len_z = b_field_calibration_data[azi_to_fit]['b_measured_values'], b_field_calibration_data[azi_to_fit]['z_stage_values'], b_field_calibration_data[azi_to_fit]['num_z_values']
b = np.array(b)
z = np.array(z)

# popt_exp, pcov_exp = curve_fit(fit_exp, z, b, p0=(2000, 100, 1, 1000), maxfev=10000)
# popt_poly, pcov_poly = curve_fit(fit_poly5th, z, b, p0=(1e-5, 1e-3, 1, 1, 1, 1e3), maxfev=10000)
# popt_poly3rd, pcov_poly3rd = curve_fit(fit_poly3rd, z, b, p0=(1e-5, 1e-3, 1, 1e3), maxfev=10000)
popt_inv_cube, pcov_inv_cube = curve_fit(fit_inv_cube, z, b, p0=(1e9, 50, 1), maxfev=10000)

x_fit = np.linspace(z[0], z[-1], 1000)
# y_fit_exp = fit_exp(x_fit, *popt_exp)
# y_fit_poly = fit_poly5th(x_fit, *popt_poly)
# y_fit_poly3rd = fit_poly3rd(x_fit, *popt_poly3rd)
y_fit_inv_cube = fit_inv_cube(x_fit, *popt_inv_cube)

plt.figure(figsize=(8, 6))
plt.scatter(z, b, marker='o', s=60, label='Data', color='orange')
# plt.plot(x_fit, y_fit_exp, linewidth=4, label='Exp Fit', color='blue')
plt.plot(x_fit, y_fit_inv_cube, linewidth=4, label='Inv Cube Fit', color='black')
# plt.plot(x_fit, y_fit_poly3rd, linewidth=4, label='Poly3rd Fit', color='green')
plt.title(f'B vs. Z for Azimuthal Angle {azi_to_fit} degrees (Inv Cube Fit)', fontsize=16)
plt.xlabel('Z (mm)')
plt.ylabel('B (G)')
plt.legend()
plt.grid()
plt.show()

print(f"type z: {type(z)}, type b: {type(b)}")
print(f"param values for inv cube fit: {popt_inv_cube}")

# %% Goodness of fit

# y_fit_exp = fit_exp(z, *popt_exp)
# y_fit_poly = fit_poly5th(z, *popt_poly)
# y_fit_poly3rd = fit_poly3rd(z, *popt_poly3rd)
y_fit_inv_cube = fit_inv_cube(z, *popt_inv_cube)

# r2_exp = r2_score(b, y_fit_exp)
# r2_poly = r2_score(b, y_fit_poly)
# r2_poly3rd = r2_score(b, y_fit_poly3rd)
r2_inv_cube = r2_score(b, y_fit_inv_cube)

# print(f"R^2 for exponential fit: {r2_exp:.4f}")
# print(f"R^2 for polynomial fit: {r2_poly:.4f}")
# print(f"R^2 for polynomial 3rd fit: {r2_poly3rd:.4f}")
print(f"R^2 for inverse cube fit: {r2_inv_cube:.4f}")

# %% Fit all azimuthal angle data and save fitted parameters to file
for azi, data in b_field_calibration_data.items():
    b = np.array(data['b_measured_values'])
    z = np.array(data['z_stage_values'])

    popt_inv_cube, pcov_inv_cube = curve_fit(fit_inv_cube, z, b, p0=(1e9, 50, 1), maxfev=10000)
    b_field_calibration_data[azi]['fit_parameters'] = popt_inv_cube

# Save fitted parameters to Excel file
b_field_cal_df = pd.DataFrame({
    'Azimuthal Angle (degrees)': list(b_field_calibration_data.keys()),
    'B Max (G)': [data['b_max'] for data in b_field_calibration_data.values()],
    'B Min (G)': [data['b_min'] for data in b_field_calibration_data.values()],
    'Z Max (mm)': [data['z_max'] for data in b_field_calibration_data.values()],
    'Z Min (mm)': [data['z_min'] for data in b_field_calibration_data.values()],
    'Fit Param a': [data['fit_parameters'][0] for data in b_field_calibration_data.values()],
    'Fit Param b': [data['fit_parameters'][1] for data in b_field_calibration_data.values()],
    'Fit Param c': [data['fit_parameters'][2] for data in b_field_calibration_data.values()]
})

# Save the DataFrame to an Excel file
with pd.ExcelWriter("b_field_calibration_fitted_params.xlsx", mode='a', engine='openpyxl') as writer:
    b_field_cal_df.to_excel(writer, sheet_name="Polar = -35 deg", index=False)

# b_field_cal_df.to_excel("b_field_calibration_fitted_params.xlsx", sheet_name="Polar = 35 deg", index=False)


# %%
# Print r2 and plot fitted parameter for given azimuthal angle
azi_to_show = 53  # degrees
b_azi_to_show = np.array(b_field_calibration_data[azi_to_show]['b_measured_values'])
z_azi_to_show = np.array(b_field_calibration_data[azi_to_show]['z_stage_values'])
x_fit = np.linspace(z_azi_to_show[0], z_azi_to_show[-1], 1000)

if azi_to_show in azi_angles:
    popt_inv_cube = b_field_calibration_data[azi_to_show]['fit_parameters']
    y_fit_inv_cube = fit_inv_cube(z_azi_to_show, *popt_inv_cube)
    r2_inv_cube = r2_score(b_azi_to_show, y_fit_inv_cube)
    print(f"R^2 for inverse cube fit at azimuthal angle {azi_to_show} degrees: {r2_inv_cube:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(z_azi_to_show, b_azi_to_show, marker='o', s=60, label='Data', color='orange')
    plt.plot(x_fit, fit_inv_cube(x_fit, *popt_inv_cube), linewidth=4, label='Inv Cube Fit', color='black')
    plt.title(f'B vs. Z for Azimuthal Angle {azi_to_show} degrees (Inv Cube Fit)', fontsize=16)
    plt.xlabel('Z (mm)')
    plt.ylabel('B (G)')
    plt.legend()
    plt.grid()
    plt.show()


# %% Determine z position for given B value using fit
def find_z_for_b(b_target, popt, fit_func, z_min, z_max):
    print(f"{(z_min + z_max) / 2} is the initial guess for z_solver")
    def z_solver(z):
        print(f"z: {z}")
        return fit_func(z, *popt) - b_target

    z_solution = fsolve(z_solver, x0=(z_min + z_max) / 2)
    return z_solution[0]

azi_choice = 53  # degrees
popt_inv_cube = b_field_calibration_data[azi_choice]['fit_parameters']

b_target = 1456  # G
z_max, z_min = b_field_calibration_data[azi_choice]['z_max'], b_field_calibration_data[azi_choice]['z_min'] 
print(f"Zmin: {z_min}, zmax: {z_max}")

z_for_b_inv_cube = find_z_for_b(b_target, popt_inv_cube, fit_inv_cube, z_min, z_max)

if z_for_b_inv_cube < z_min or z_for_b_inv_cube > z_max:
    print(f"Estimated z position for B = {b_target} G using inverse cube fit is out of range: {z_for_b_inv_cube:.2f} mm")
else:
    print(f"Estimated z position for B = {b_target} G using inverse cube fit: {z_for_b_inv_cube:.2f} mm")


# %% Determine z position for B value using b_field_calibration_fitted_params file
def fit_off_inv_cube(z, a, b, c):
    return a / (z + b)**3 + c

def find_z_for_b_from_fits(b_target, azi, polar):
    b_field_cal_file = "b_field_calibration_fitted_params.xlsx"
    b_cal_df = pd.read_excel(b_field_cal_file, sheet_name=f"Polar = {polar} deg")
    
    # Read fit parameters from separate columns
    row = b_cal_df.loc[b_cal_df['Azimuthal Angle (degrees)'] == azi]
    a = row['Fit Param a'].iloc[0]
    b = row['Fit Param b'].iloc[0]
    c = row['Fit Param c'].iloc[0]
    popt = (a, b, c)
    
    z_min = row['Z Min (mm)'].iloc[0]
    z_max = row['Z Max (mm)'].iloc[0]
    print(f"Retrieved fit parameters for azi {azi} degrees, polar {polar} deg: {popt}")
    print(f"Retrieved z range for azi {azi} degrees, polar {polar} deg: z_min = {z_min} mm, z_max = {z_max} mm")
    def z_solver(z):
        print(f"Initial guess: {z}")
        return fit_off_inv_cube(z, *popt) - b_target

    z_solution = fsolve(z_solver, x0=(z_min + z_max) / 2)
    return z_solution[0]

azi_choice = 53  # degrees
polar_choice = -35  # degrees
b_target = 1456  # G
z_for_b_from_fits = find_z_for_b_from_fits(b_target, azi_choice, polar_choice)
print(f"Estimated z position for B = {b_target} G using fit parameters from file for azi {azi_choice} degrees, polar {polar_choice} deg: {z_for_b_from_fits:.2f} mm")
# %%
