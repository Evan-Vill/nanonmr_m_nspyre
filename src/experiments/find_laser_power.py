from scipy.optimize import curve_fit, fsolve
import numpy as np
import pandas as pd 

def find_las_power(percentage):
    # file_path = 'C:\\Users\\ejvil\\nspyre\\src\\experiments\\all_b_field_fitted_params.csv'
    file_path = 'C:/Users/nanon/nanon-nspyre/nanonmr_m_nspyre/src/experiments/laser_power_data.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')


    fit_coeff_row = df.iloc[23, 0:6]  # row 24 (index 23), columns A to G (0 to 6)

    coefficients = fit_coeff_row.to_list()
    a1, a2, a3, a4, a5, b0 = coefficients

    power = a5*percentage**5 + a4*percentage**4 + a3*percentage**3 + a2*percentage**2 + a1*percentage + b0
    las_pow = round(power, 2)

    return las_pow

