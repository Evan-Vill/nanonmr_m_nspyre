import numpy as np
from scipy.optimize import curve_fit

def negative_lorentzian(x, A, x0, gamma, c):
    return -A / (1 + ((x - x0) / gamma)**2) + c

def multi_negative_lorentzian(x, *params):
    """Sum of N negative Lorentzian peaks with a shared baseline.

    Parameter order is:
    [A1, x01, gamma1, A2, x02, gamma2, ..., AN, x0N, gammaN, c]

    where N is flexible and c is a single shared offset.
    """
    if len(params) < 4 or (len(params) - 1) % 3 != 0:
        raise ValueError(
            "Expected parameters [A1, x01, gamma1, ..., AN, x0N, gammaN, c]."
        )

    x = np.asarray(x, dtype=float)
    y = np.full_like(x, float(params[-1]), dtype=float)

    for i in range(0, len(params) - 1, 3):
        A, x0, gamma = params[i : i + 3]
        y += -A / (1 + ((x - x0) / gamma) ** 2)

    return y

def fit_multi_negative_lorentzian(
    x_values,
    y_values,
    n_peaks,
    initial_guess,
    bounds=(-np.inf, np.inf),
    maxfev=20000,
):
    """Fit an arbitrary number of negative Lorentzian peaks.

    Args:
        x_values: x data used for fitting.
        y_values: y data used for fitting.
        n_peaks: Number of negative Lorentzian peaks to fit.
        initial_guess: Initial parameter guess in the order:
            [A1, x01, gamma1, ..., AN, x0N, gammaN, c].
        bounds: Bounds passed to scipy.optimize.curve_fit.
        maxfev: Maximum number of function evaluations for curve_fit.

    Returns:
        params: Best-fit parameters.
        param_errors: One-sigma fit errors from covariance diagonal.
        x_fit: Dense x-axis for plotting the fitted curve.
        y_fit: Fitted curve values on x_fit.
    """
    if not isinstance(n_peaks, int) or n_peaks <= 0:
        raise ValueError("n_peaks must be a positive integer.")

    expected_len = 3 * n_peaks + 1
    if len(initial_guess) != expected_len:
        raise ValueError(
            f"initial_guess length must be {expected_len} for {n_peaks} peaks."
        )

    params, covariance = curve_fit(
        multi_negative_lorentzian,
        x_values,
        y_values,
        p0=initial_guess,
        bounds=bounds,
        maxfev=maxfev,
    )
    param_errors = np.sqrt(np.diag(covariance))

    x_fit = np.linspace(np.min(x_values), np.max(x_values), 1000)
    y_fit = multi_negative_lorentzian(x_fit, *params)

    return params, param_errors, x_fit, y_fit

def positive_lorentzian(x, A, x0, gamma, c):
    return A / (1 + ((x - x0) / gamma)**2) + c

def gaussian(x, A, x0, sigma, c):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + c

def decaying_cosine(x, A, t_decay, T, phi, c):
    return A * np.exp(-x / t_decay) * np.cos(2 * np.pi * x / T + phi) + c

def stretched_exponential(x, A, T, n, c):
    return A*np.exp(-(x/T)**n) + c

def mod_stretched_exponential(x, A, T, n, a1, f1, phi1, a2, f2, phi2):
    return A * np.exp(-(x / T)**n) * (1 - a1 * np.sin(2 * np.pi * f1 * x / 4 + phi1)**2) * (1 - a2 * np.sin(2 * np.pi * f2 * x / 4 + phi2)**2)

# FIXME: make T1_nv and n_nv fixed parameters, not fit parameters
def deer_t1_stretched_exponential(x, A, T1_nv, n_nv, T1_e, n_e, c):
    return A * np.exp(-(x / T1_nv)**n_nv - (x / T1_e)**n_e) + c

def fit_data(self, fit_type, exp, sig_data, back_data, *args):
    # Combine all signal sweeps into a single 3D array and average
    all_signal_data = np.stack(sig_data, axis=-1)  # Shape: (2, 10, 5)
    averaged_sig = np.mean(all_signal_data[1, :, :], axis=1)  # Shape: (10,)

    # Combine all background sweeps into a single 3D array and average
    all_background_data = np.stack(back_data, axis=-1)  # Shape: (2, 10, 5)
    averaged_bg = np.mean(all_background_data[1, :, :], axis=1)  # Shape: (10,)

    # Compute the microwave_times (assumed constant across sweeps)
    x_values = all_signal_data[0, :, 0]  # Shape: (10,)
    x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting

    # Compute the ratio/difference of averaged signal and background for fitting
    if exp == 'odmr' or exp == 'rabi':
        y_values = averaged_sig / averaged_bg  # Shape: (10,)
    elif exp == 't1' or exp == 't2':
        y_values = averaged_bg - averaged_sig

    # Initial guesses for parameters: A, gamma, f, phi, C
    # initial_guess = [0.02, 0.001, 200, 0, 1]
    initial_guess = list(args)
    
    # Perform curve fitting 
    """
    "Neg. Lorentz.",
    "Pos. Lorentz.",
    "Two Neg. Lorentz."
    "Decaying Cos.",
    "Stretched Exp.",
    "Modulated Str. Exp.",
    "DEER T1 Str. Exp."
    """
    match fit_type:
        case 'Neg. Lorentz.':
            if exp == 'odmr':
                initial_guess[1] /= 1e9 # convert [Hz] to [GHz]
                initial_guess[2] /= 1e9 # convert [Hz] to [GHz]
            
            ### --- Perform fit --- ###
            params, covariance = curve_fit(negative_lorentzian, x_values, y_values, p0=initial_guess)
            y_fit = negative_lorentzian(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 4) for i in params]
            fitted_errors = [round(i, 4) for i in param_errors]
        
        case 'Pos. Lorentz.':
            if exp == 'nmr':
                initial_guess[1] /= 1e6 # convert [Hz] to [MHz]
                initial_guess[2] /= 1e3 # convert [Hz] to [kHz]
            elif exp == 'casr':
                initial_guess[1] /= 1e3 # convert [Hz] to [kHz]
            
            ### --- Perform fit --- ###
            params, covariance = curve_fit(positive_lorentzian, x_values, y_values, p0=initial_guess)
            y_fit = positive_lorentzian(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 4) for i in params]
            fitted_errors = [round(i, 4) for i in param_errors]

        case 'Two Neg. Lorentz.':
            if exp == 'odmr rf':
                initial_guess[1] /= 1e9 # convert [Hz] to [GHz]
                initial_guess[2] /= 1e9 # convert [Hz] to [GHz]
                initial_guess[5] /= 1e9 # convert [Hz] to [GHz]
                initial_guess[6] /= 1e9 # convert [Hz] to [GHz]
            
            ### --- Perform fit --- ###
            params, covariance = curve_fit(negative_lorentzian, x_values, y_values, p0=initial_guess)
            y_fit = negative_lorentzian(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 4) for i in params]
            fitted_errors = [round(i, 4) for i in param_errors]

        case 'Decaying Cos.':
            if exp in ('rabi', 't2'):
                initial_guess[1] *= 1e9 # convert [s] to [us]
                initial_guess[2] *= 1e9 # convert [s] to [ns]

            ### --- Perform fit --- ###
            params, covariance = curve_fit(decaying_cosine, x_values, y_values, p0=initial_guess)
            y_fit = decaying_cosine(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 2) for i in params]
            fitted_errors = [round(i, 2) for i in param_errors]
            fitted_values[2] = round(x_fit[np.argmin(y_fit)],2)
            fitted_errors[2] = 0
        
        case 'Stretched Exp.':
            if exp == 't1':
                initial_guess[1] *= 1e3 # convert [s] to [ms]
            elif exp == 't2':
                initial_guess[1] *= 1e6 # convert [s] to [us]
            
            ### --- Perform fit --- ###
            params, covariance = curve_fit(stretched_exponential, x_values, y_values, p0=initial_guess)
            y_fit = stretched_exponential(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 3) for i in params]
            fitted_errors = [round(i, 3) for i in param_errors]
        
        case 'Modulated Str. Exp.':
            if exp == 't2':
                initial_guess[1] *= 1e6 # convert [s] to [ms]
                initial_guess[4] /= 1e6 # convert [Hz] to [MHz]
                initial_guess[7] /= 1e6 # convert [Hz] to [MHz]
            
            ### --- Perform fit --- ###
            params, covariance = curve_fit(mod_stretched_exponential, x_values, y_values, p0=initial_guess)
            y_fit = mod_stretched_exponential(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            # compute fitted value of interest (resonance for ODMR, pi pulse for Rabi, etc.)
            fitted_values = [round(i, 3) for i in params]
            fitted_errors = [round(i, 3) for i in param_errors]
        
        case _:
            print("No fit type selected.")
            return 0, 0, 0, 0
            
    return fitted_values, fitted_errors, x_fit, y_fit

def fit_deer_data(self, fit_type, exp, dark_sig_data, dark_back_data, echo_sig_data, echo_back_data, *args):
    # Combine all dark signal sweeps into a single 3D array and average
    all_dark_signal_data = np.stack(dark_sig_data, axis=-1)  # Shape: (2, 10, 5)
    averaged_dark_sig = np.mean(all_dark_signal_data[1, :, :], axis=1)  # Shape: (10,)

    # Combine all dark background sweeps into a single 3D array and average
    all_dark_background_data = np.stack(dark_back_data, axis=-1)  # Shape: (2, 10, 5)
    averaged_dark_bg = np.mean(all_dark_background_data[1, :, :], axis=1)  # Shape: (10,)

    # Combine all dark signal sweeps into a single 3D array and average
    all_echo_signal_data = np.stack(echo_sig_data, axis=-1)  # Shape: (2, 10, 5)
    averaged_echo_sig = np.mean(all_echo_signal_data[1, :, :], axis=1)  # Shape: (10,)

    # Combine all dark background sweeps into a single 3D array and average
    all_echo_background_data = np.stack(echo_back_data, axis=-1)  # Shape: (2, 10, 5)
    averaged_echo_bg = np.mean(all_echo_background_data[1, :, :], axis=1)  # Shape: (10,)

    # Compute the microwave_times (assumed constant across sweeps)
    x_values = all_dark_signal_data[0, :, 0]  # Shape: (10,)
    x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting

    # Compute the ratio/difference of averaged signal and background for fitting
    deer = (averaged_dark_bg - averaged_dark_sig) / (averaged_dark_bg + averaged_dark_sig)
    echo = (averaged_echo_bg - averaged_echo_sig) / (averaged_echo_bg + averaged_echo_sig)

    y_values = deer / echo

    initial_guess = list(args)

    # Perform curve fitting 
    match fit_type:
        case 'Neg. Lorentz.':
            if exp == 'deer':
                initial_guess[1] /= 1e6 # convert [Hz] to [MHz]
                initial_guess[2] /= 1e6 # convert [Hz] to [MHz]
            
            ### --- Perform fit --- ###
            params, covariance = curve_fit(negative_lorentzian, x_values, y_values, p0=initial_guess)
            y_fit = negative_lorentzian(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 3) for i in params]
            fitted_errors = [round(i, 3) for i in param_errors]
            # fitted_values[0] = 100*round(params[0], 3) # to display DEER contrast as percent
            # fitted_errors[0] = 100*round(param_errors[0], 3)

        case 'Decaying Cos.':
            if exp == 'deer rabi':
                initial_guess[1] /= 1e9 # convert [Hz] to [GHz]
                initial_guess[2] *= 1e9 # convert [s] to [ns]
            
            ### --- Perform fit --- ###
            params, covariance = curve_fit(decaying_cosine, x_values, y_values, p0=initial_guess)
            y_fit = decaying_cosine(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 2) for i in params]
            fitted_errors = [round(i, 2) for i in param_errors]
            fitted_values[2] = round(x_fit[np.argmin(y_fit)],2)
            fitted_errors[2] = 0

        case _:
            print("No fit type selected.")
            return 0, 0, 0, 0

    return fitted_values, fitted_errors, x_fit, y_fit

# TODO: update this function to fit DEER T1
def fit_deer_t1_data(self, fit_type, with_pulse_py_sweeps, without_pulse_py_sweeps, with_pulse_ny_sweeps, without_pulse_ny_sweeps, *args):
    # Combine all dark signal sweeps into a single 3D array and average
    all_with_pulse_py_data = np.stack(with_pulse_py_sweeps, axis=-1)  # Shape: (2, 10, 5)
    averaged_with_pulse_py = np.mean(all_with_pulse_py_data[1, :, :], axis=1)  # Shape: (10,)

    # Combine all dark background sweeps into a single 3D array and average
    all_without_pulse_py_data = np.stack(without_pulse_py_sweeps, axis=-1)  # Shape: (2, 10, 5)
    averaged_without_pulse_py = np.mean(all_without_pulse_py_data[1, :, :], axis=1)  # Shape: (10,)

    # Combine all dark signal sweeps into a single 3D array and average
    all_with_pulse_ny_data = np.stack(with_pulse_ny_sweeps, axis=-1)  # Shape: (2, 10, 5)
    averaged_with_pulse_ny = np.mean(all_with_pulse_ny_data[1, :, :], axis=1)  # Shape: (10,)

    # Combine all dark background sweeps into a single 3D array and average
    all_without_pulse_ny_data = np.stack(without_pulse_ny_sweeps, axis=-1)  # Shape: (2, 10, 5)
    averaged_without_pulse_ny = np.mean(all_without_pulse_ny_data[1, :, :], axis=1)  # Shape: (10,)

    # Compute the microwave_times (assumed constant across sweeps)
    x_values = all_with_pulse_py_data[0, :, 0]  # Shape: (10,)
    x_fit = np.linspace(min(x_values), max(x_values), 1000) # finer resolution for fitting

    # Compute the ratio/difference of averaged signal and background for fitting
    # deer = (averaged_dark_bg - averaged_dark_sig) / (averaged_dark_bg + averaged_dark_sig)
    # echo = (averaged_echo_bg - averaged_echo_sig) / (averaged_echo_bg + averaged_echo_sig)

    # y_values = deer / echo
    diff_overall = (averaged_without_pulse_py - averaged_with_pulse_py) - (averaged_without_pulse_ny - averaged_with_pulse_ny)
    y_values = diff_overall

    initial_guess = list(args)

    # Perform curve fitting 
    match fit_type:
        case 'DEER T1 Str. Exp.':
            ### --- Perform fit --- ### 
            params, covariance = curve_fit(deer_t1_stretched_exponential, x_values, y_values, p0=initial_guess)
            y_fit = deer_t1_stretched_exponential(x_fit, *params)
            param_errors = np.sqrt(np.diag(covariance))
            fitted_values = [round(i, 3) for i in params]
            fitted_errors = [round(i, 3) for i in param_errors]

        case _:
            print("No fit type selected.")
            return 0, 0, 0, 0

    return fitted_values, fitted_errors, x_fit, y_fit


