from scipy.optimize import curve_fit, fsolve
import numpy as np

def find_mag_pos(phi, b):
    file_path = 'C:\\Users\\ejvil\\nspyre\\src\\experiments\\all_b_field_fitted_params.csv'
    data = np.loadtxt(file_path, delimiter=',')

    mag_pos = fsolve(pos_solver, x0=10, args=(data, round(phi), float(b))) 
    mag_pos = round(mag_pos[0], 2)

    return mag_pos

def pos_solver(x, data, azi, b):
    return data[azi][0] * np.exp(-(data[azi][1]/x)**data[azi][2]) + data[azi][3] - float(b)
