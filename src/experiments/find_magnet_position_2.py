from scipy.optimize import curve_fit, fsolve
import numpy as np

def find_mag_pos(phi, b):
    # file_path = 'C:\\Users\\ejvil\\nspyre\\src\\experiments\\all_b_field_fitted_params.csv'
    file_path = 'C:/Users/nanon/nanon-nspyre/nanonmr_m_nspyre/src/experiments/all_b_field_fitted_params.csv'
    data = np.loadtxt(file_path, delimiter=',')

    mag_pos = fsolve(pos_solver, x0=10, args=(data, round(phi), float(b))) 
    mag_pos = round(mag_pos[0], 2)

    return mag_pos

def pos_solver(x, data, azi, b):
    return data[azi][0] * np.exp(-(data[azi][1]/x)**data[azi][2]) + data[azi][3] - float(b)



def find_z_for_b(b_target, popt, fit_func, z_min, z_max):
    def z_solver(z):
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