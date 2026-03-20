def build_experiment_defaults():
    # parameter defaults for different experiments
    sideband_opts = ["Lower", "Upper"]
    sideband_cw_opts = ["Lower", "Upper"]

    dig_ro_chan_opts = ["0", "1"]
    dig_coupling_opts = ["DC", "AC"]
    dig_termination_opts = ["1M", "50"]

    rabi_axis_opts = ["y", "x"]
    opt_t1_array_opts = ["geomspace", "linspace"]
    mw_t1_array_opts = ["geomspace", "linspace"]
    t2_array_opts = ["geomspace", "linspace"]
    t2_rf_array_opts = ["geomspace", "linspace"]
    dq_array_opts = ["geomspace", "linspace"]
    fid_array_opts = ["geomspace", "linspace"]
    fid_cd_array_opts = ["geomspace", "linspace"]
    t2_seq_opts = ["Ramsey", "Echo", "XY4", "YY4", "XY8", "YY8", "CPMG", "PulsePol"]
    deer_drive_opts = ["Pulsed", "Continuous"]
    fid_drive_opts = ["Pulsed", "Continuous"]
    corr_t1_array_opts = ["geomspace", "linspace"]
    
    corr_spec_sig_opts = ["Sample", "Coil"]
    casr_sig_opts = ["Sample", "Coil"]
    dnp_opts = ["Off", "Overhauser"]

    sigvstime_params_defaults = {
        "exp_sampling_rate": 1e3, 
    }

    sigvstime_mw_params_defaults = {
        "exp_sampling_rate_0": 1e3,
    }

    laser_params_defaults = {
        "laser_power": 0,
        "laser_init": 15e-6,
        "laser_readout": 2.5e-6,
        "sideband_freq": 30e6,
        "sideband_power": 0.45,
        "sideband": sideband_opts,
        "i_offset": -0.002,
        "q_offset": -0.004,
    }

    laser_params_sigvstime_defaults = {
        "laser_power": 0,
    }

    digitizer_defaults = {
        "segment_size": 1024,
        "dig_sampling_freq": 500e6,
        "dig_amplitude": 1,
        "read_channel": dig_ro_chan_opts,
        "both_channels": False,
        "dig_coupling": dig_coupling_opts,
        "dig_termination": dig_termination_opts,
        "pretrig_size": 32,
        "dig_timeout": 5,
    }

    odmr_params_defaults = {
        "runs": 120,
        "iters": 10,
        "num_pts": 50,
        "enable_pl_trace": False,
        "pl_pt": 25,
    }

    odmr_mw_params_defaults = {
        "center_freq": 2.87e9,
        "half_span_sideband_freq": 100e6,
        "rf_power": 1e-9,
        "probe": 25e-6,
    }

    odmr_smart_params_defaults = {
        "runs": 120,
        "num_pts": 50,
        "start_angle": 45,
        "stop_angle": 60,
        "iters": 20,
    }

    odmr_smart_mw_params_defaults = {
        "center_freq": 2.87e9,
        "half_span_sideband_freq": 100e6,
        "rf_power": 1e-9,
        "probe": 25e-6,
    }

    rabi_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 0,
        "stop": 200e-9,
        "num_pts": 50,
        "enable_pl_trace": False,
        "pl_pt": 25
    } 

    rabi_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pulse_axis": rabi_axis_opts
    }

    pulsed_odmr_params_defaults = {
        "runs": 120,
        "iters": 10,
        "num_pts": 50,
        "enable_pl_trace": False,
        "pl_pt": 25
    }
    pulsed_odmr_mw_params_defaults = {
        "center_freq": 2.87e9,
        "half_span_sideband_freq": 100e6,
        "rf_power": 1e-9,
        "pi": 100e-9
    }
    
    pulsed_odmr_rf_params_defaults = {
        "runs": 120,
        "iters": 10,
        "num_pts": 50
    }

    pulsed_odmr_rf_mw_params_defaults = {
        "center_freq": 2.87e9,
        "half_span_sideband_freq": 100e6,
        "rf_power": 1e-9,
        "pi": 100e-9,
        "rf_pulse_freq": 500e3,
        "rf_pulse_power": 0.3,
        "rf_pulse_phase": 0
    }

    opt_t1_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 100e-6,
        "num_pts": 50,
        "array_type": opt_t1_array_opts
    }
    opt_t1_mw_params_defaults = {
        "runs": 12,
        "iters": 10,
        "start": 50e-9,
        "stop": 100e-6,
        "num_pts": 50,
        "array_type": opt_t1_array_opts
    } # not needed - hidden in GUI

    mw_t1_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 100e-6,
        "num_pts": 50,
        "array_type": mw_t1_array_opts,
        "enable_pl_trace": False,
        "pl_pt": 25
    }
    mw_t1_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y"
    }

    t2_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 20e-6,
        "num_pts": 50,
        "array_type": t2_array_opts,
        "enable_pl_trace": False,
        "pl_pt": 25
    }
    t2_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "t2_seq": t2_seq_opts,
        "n": 1
    }

    t2_rf_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 20e-6,
        "num_pts": 50,
        "array_type": t2_rf_array_opts
    }
    t2_rf_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "rf_pulse_freq": 1e6,
        "rf_pulse_power": 0.1,
        "rf_pulse_phase": 0,
        "n": 1
    }

    dq_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 100e-6,
        "num_pts": 50,
        "array_type": dq_array_opts
    }
    dq_mw_params_defaults = {
        "freq_minus": 2.87e9,
        "rf_power_minus": 1e-9,
        "pi_minus": 20e-9,
        "freq_plus": 2.87e9,
        "rf_power_plus": 1e-9,
        "pi_plus": 20e-9,
        "pulse_axis": "y"
    }

    deer_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 350e6,
        "stop": 750e6,
        "num_pts": 100,
        "tau": 800e-9
    }      
    deer_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "awg_power": 0.2,
        "dark_pi": 40e-9,
        "drive_type": deer_drive_opts
    } 

    deer_rabi_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 3e-9,
        "stop": 100e-9,
        "num_pts": 100,
        "tau": 800e-9
    }     
    deer_rabi_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "dark_freq": 560e6,
        "awg_power": 0.2
    }     

    deer_fid_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 20e-6,
        "num_pts": 100,
        "array_type": fid_array_opts
    }    
    deer_fid_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "dark_freq": 560e6,
        "awg_power": 0.2,
        "dark_pi": 40e-9,
        "n": 1
    }    

    deer_fid_cd_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 20e-6,
        "num_pts": 100,
        "array_type": fid_cd_array_opts
    }        
    deer_fid_cd_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "dark_freq": 560e6,
        "awg_power": 0.2,
        "dark_pi": 40e-9,
        "awg_cd_power": 0.1,
        "n": 1
    }

    deer_corr_rabi_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 3e-9,
        "stop": 100e-9,
        "num_pts": 100,
        "tau": 800e-9,
        "t_corr": 1e-6
    }
    deer_corr_rabi_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "dark_freq": 560e6,
        "dark_pi": 40e-9,
        "awg_power": 0.2
    }

    deer_corr_t1_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 1e-6,
        "num_pts": 100,
        "array_type": corr_t1_array_opts,
        "tau": 800e-9
    }
    deer_corr_t1_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "dark_freq": 560e6,
        "dark_pi": 40e-9,
        "awg_power": 0.2
    }

    deer_t2_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 1e-6,
        "num_pts": 100,
        "array_type": corr_t1_array_opts,
        "tau": 800e-9,
        "deer_t2_buffer": 500e-9
    }
    deer_t2_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "dark_freq": 560e6,
        "dark_pi": 40e-9,
        "awg_power": 0.2
    } 

    corr_spec_params_defaults = {
        "runs": 120,
        "iters": 10,
        "start": 50e-9,
        "stop": 100e-6,
        "num_pts": 100,
        "tau": 1e-6,
        "sig_opt": corr_spec_sig_opts,
        "enable_pl_trace": False,
        "pl_pt": 50
    }
    corr_spec_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "pulse_axis": "y",
        "n": 1
    }

    casr_params_defaults = {
        "runs": 120,
        "iters": 10,
        "num_pts": 10,
        "tau": 200e-9,
        "sig_opt": casr_sig_opts,
        "dnp": dnp_opts
    }
    casr_mw_params_defaults = {
        "freq": 2.87e9,
        "rf_power": 1e-9,
        "pi": 20e-9,
        "rf_pulse_freq": 1e6,
        "rf_pulse_power": 0.1,
        "rf_pi_half": 1e-6,
        "rf_pulse_phase": 0,
        "n": 1
    }

    fit_none_default = {
        "A": 0
    }
    fit_neg_lorentz_defaults = {
        "A": 0.01,
        "x0": 1e9,
        "gamma": 6e6,
        "c": 1
    } # defaults = 1% contrast, 1 GHz central freq, 6 MHz linewidth, 1 vertical offset
    fit_decaying_cosine_defaults = {
        "A": 0.02,
        "t_decay": 1e-6,
        "T": 200e-9,
        "phi": 0,
        "c": 1
    } # defaults = 2% contrast, 1 us decay time, 200 ns period, 0 phase, 1 vertical offset
    fit_two_neg_lorentz_defaults = {
        "A": 0.01,
        "x0": 1e9,
        "gamma": 6e6,
        "c": 1,
        "A_rf": 0.01,
        "x0_rf": 1e9,
        "gamma_rf": 6e6,
        "c_rf": 1
    } # defaults = 1% contrast, 1 GHz central freq, 6 MHz linewidth, 1 vertical offset, 1% contrast, 1 GHz central freq, 6 MHz linewidth, 1 vertical offset for the two overlapping Lorentzians
    fit_str_exp_defaults = {
        "A": 0.01,
        "T1": 1e-3,
        "n": 1,
        "c": 0
    } # defaults = 0.01 amplitude, 1 ms T1, 1 stretching factor, 0 vertical offset
    fit_mod_str_exp_defaults = {
        "A": 0.1,
        "T2": 2e-6,
        "n": 1,
        "a1": 1,
        "f1": 0.2e6,
        "phi1": 0,
        "a2": 1,
        "f2": 0.2e6,
        "phi2": 0
    } # defaults = 0.1 amplitude, 2 us T2, 1 stretching factor, 1 amp first sine wave, 0.2 MHz first sine wave, 0 phase first sine wave, 1 amp second sine wave, 0.2 MHz second sine wave, 0 phase second sine wave
    fit_deer_t1_str_exp_defaults = {
        "A": 0.1,
        "T1_nv": 1e-3,
        "n_nv": 1,
        "T1_e": 1e-3,
        "n_e": 1,
        "c": 0
    } # defaults = 0.1 amplitude, 1 ms T1_nv, 1 n_nv, 1 ms T1_e, 1 n_e, 0 vertical offset
    fit_pos_lorentz_defaults = {
        "A": 0.01,
        "x0": 2e6,
        "gamma": 100e3,
        "c": 1
    } # defaults = 1% contrast, 2 MHz central freq, 100 kHz linewidth, 1 vertical offset

    return {
        'sigvstime_params_defaults': sigvstime_params_defaults,
        'sigvstime_mw_params_defaults': sigvstime_mw_params_defaults,
        'laser_params_defaults': laser_params_defaults,
        'laser_params_sigvstime_defaults': laser_params_sigvstime_defaults,
        'digitizer_defaults': digitizer_defaults,
        'odmr_params_defaults': odmr_params_defaults,
        'odmr_mw_params_defaults': odmr_mw_params_defaults,
        'odmr_smart_params_defaults': odmr_smart_params_defaults,
        'odmr_smart_mw_params_defaults': odmr_smart_mw_params_defaults,
        'pulsed_odmr_params_defaults': pulsed_odmr_params_defaults,
        'pulsed_odmr_mw_params_defaults': pulsed_odmr_mw_params_defaults,
        'pulsed_odmr_rf_params_defaults': pulsed_odmr_rf_params_defaults,
        'pulsed_odmr_rf_mw_params_defaults': pulsed_odmr_rf_mw_params_defaults,
        'rabi_params_defaults': rabi_params_defaults,
        'rabi_mw_params_defaults': rabi_mw_params_defaults,
        'opt_t1_params_defaults': opt_t1_params_defaults,
        'opt_t1_mw_params_defaults': opt_t1_mw_params_defaults,
        'mw_t1_params_defaults': mw_t1_params_defaults,
        'mw_t1_mw_params_defaults': mw_t1_mw_params_defaults,
        't2_params_defaults': t2_params_defaults,
        't2_mw_params_defaults': t2_mw_params_defaults,
        't2_rf_params_defaults': t2_rf_params_defaults,
        't2_rf_mw_params_defaults': t2_rf_mw_params_defaults,
        'dq_params_defaults': dq_params_defaults,
        'dq_mw_params_defaults': dq_mw_params_defaults,
        'deer_params_defaults': deer_params_defaults,
        'deer_mw_params_defaults': deer_mw_params_defaults,
        'deer_rabi_params_defaults': deer_rabi_params_defaults,
        'deer_rabi_mw_params_defaults': deer_rabi_mw_params_defaults,
        'deer_fid_params_defaults': deer_fid_params_defaults,
        'deer_fid_mw_params_defaults': deer_fid_mw_params_defaults,
        'deer_fid_cd_params_defaults': deer_fid_cd_params_defaults,
        'deer_fid_cd_mw_params_defaults': deer_fid_cd_mw_params_defaults,
        'deer_corr_rabi_params_defaults': deer_corr_rabi_params_defaults,
        'deer_corr_rabi_mw_params_defaults': deer_corr_rabi_mw_params_defaults,
        'deer_corr_t1_params_defaults': deer_corr_t1_params_defaults,
        'deer_t1_params_defaults': deer_corr_t1_params_defaults,
        'deer_corr_t1_mw_params_defaults': deer_corr_t1_mw_params_defaults,
        'deer_t2_params_defaults': deer_t2_params_defaults,
        'deer_t2_mw_params_defaults': deer_t2_mw_params_defaults,
        'corr_spec_params_defaults': corr_spec_params_defaults,
        'corr_spec_mw_params_defaults': corr_spec_mw_params_defaults,
        'casr_params_defaults': casr_params_defaults,
        'casr_mw_params_defaults': casr_mw_params_defaults,
        'fit_none_default': fit_none_default,
        'fit_neg_lorentz_defaults': fit_neg_lorentz_defaults,
        'fit_decaying_cosine_defaults': fit_decaying_cosine_defaults,
        'fit_two_neg_lorentz_defaults': fit_two_neg_lorentz_defaults,
        'fit_str_exp_defaults': fit_str_exp_defaults,
        'fit_mod_str_exp_defaults': fit_mod_str_exp_defaults,
        'fit_deer_t1_str_exp_defaults': fit_deer_t1_str_exp_defaults,
        'fit_pos_lorentz_defaults': fit_pos_lorentz_defaults,
    }