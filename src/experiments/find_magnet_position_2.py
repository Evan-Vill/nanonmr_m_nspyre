# %%
from scipy.optimize import curve_fit, fsolve
import numpy as np
import pandas as pd
import os

import logging
logger = logging.getLogger(__name__)

def fit_off_inv_cube(z, a, b, c):
    """Model function for fitting B-field vs z-position data to an off-center inverse cube law."""
    return a / (z + b)**3 + c

def open_calibration_spreadsheet(polar):
    """Open the calibration spreadsheet and return the DataFrame for the specified polar angle."""
    b_field_cal_file = "b_field_calibration_fitted_params.xlsx"
    
    try:
        b_cal_df = pd.read_excel(b_field_cal_file, sheet_name=f"Polar = {polar} deg")
        
        # Verify required columns exist
        required_cols = ['Azimuthal Angle (degrees)', 'Fit Param a', 'Fit Param b', 'Fit Param c', 'Z Min (mm)', 'Z Max (mm)']
        missing_cols = [col for col in required_cols if col not in b_cal_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in calibration file: {missing_cols}")
            logger.error(f"Available columns: {list(b_cal_df.columns)}")
            raise ValueError(f"Calibration file missing columns: {missing_cols}")
        
        if b_cal_df.empty:
            logger.error(f"Calibration sheet 'Polar = {polar} deg' is empty")
            raise ValueError(f"Calibration sheet 'Polar = {polar} deg' is empty")
        
        return b_cal_df
        
    except FileNotFoundError as e:
        logger.error(f"Calibration file '{b_field_cal_file}' not found at current working directory")
        raise FileNotFoundError(f"Calibration file '{b_field_cal_file}' not found. Current directory: {os.getcwd()}") from e
    except ValueError as e:
        if "no sheet named" in str(e).lower():
            logger.error(f"Sheet 'Polar = {polar} deg' not found in calibration file")
            raise ValueError(f"Sheet 'Polar = {polar} deg' not found in '{b_field_cal_file}'") from e
        raise

def try_open_calibration_spreadsheet(polar: int):
    """
    Try to open the exact polar-angle calibration sheet.

    Returns
    -------
    df : pandas.DataFrame | None
        Calibration dataframe if found, otherwise None.
    found : bool
        True if exact polar sheet exists, False otherwise.
    """
    try:
        df = open_calibration_spreadsheet(polar=polar)
        return df, True

    except Exception as e:
        logger.warning(
            f"No exact calibration spreadsheet found for polar={polar}. "
            f"Exception: {e}"
        )
        return None, False

def open_other_calibration_spreadsheet():
    """Open the 'other' calibration spreadsheet and return the DataFrame for the specified polar angle."""
    b_field_cal_file = "b_field_calibration_fitted_params.xlsx"
    
    try:
        b_cal_df = pd.read_excel(b_field_cal_file, sheet_name=f"Polar = other")
        return b_cal_df
    except FileNotFoundError as e:
        logger.error(f"Calibration file '{b_field_cal_file}' not found at current working directory")
        raise FileNotFoundError(f"Calibration file '{b_field_cal_file}' not found. Current directory: {os.getcwd()}") from e
    except ValueError as e:
        if "no sheet named" in str(e).lower():
            logger.error(f"Sheet 'Polar = other' not found in calibration file")
            raise ValueError(f"Sheet 'Polar = other' not found in '{b_field_cal_file}'") from e
        raise

def find_b_for_z_from_fits(z_target, azi, polar):
    """Find the B-field value corresponding to a target z-position using fit parameters from 
    a calibration file for given azimuthal and polar angles."""

    b_cal_df = open_calibration_spreadsheet(round(polar))
    
    # Find nearest azimuthal angle in calibration data (handles floating-point precision)
    try:
        idx = (b_cal_df['Azimuthal Angle (degrees)'] - azi).abs().idxmin()
    except KeyError as e:
        logger.error(f"Column 'Azimuthal Angle (degrees)' not found in calibration data")
        raise KeyError(f"Calibration file missing 'Azimuthal Angle (degrees)' column") from e
    
    # Validate that the index is valid (not NaN or out of bounds)
    if pd.isna(idx):
        logger.error(f"Invalid index returned from idxmin() for polar angle {round(polar)} deg")
        raise ValueError(f"Failed to find valid azimuthal angle index in calibration data for polar {round(polar)} deg")
    
    try:
        row = b_cal_df.loc[[idx]]
    except Exception as e:
        logger.error(f"Failed to access row with index {idx} in calibration data")
        raise ValueError(f"Cannot access calibration data at index {idx}") from e
    
    if row.empty:
        logger.error(f"No data row found for azimuthal angle near {azi} degrees at polar {round(polar)} deg")
        raise ValueError(f"No calibration data found for azi={azi}, polar={round(polar)}")
    
    # Read fit parameters from separate columns
    try:
        a = row['Fit Param a'].iloc[0]
        b = row['Fit Param b'].iloc[0]
        c = row['Fit Param c'].iloc[0]
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to read fit parameters from calibration row")
        raise ValueError(f"Cannot read fit parameters from calibration data") from e
    
    popt = (a, b, c)
    
    azi_used = row['Azimuthal Angle (degrees)'].iloc[0]
    print(f"Retrieved fit parameters for azi {azi} degrees (nearest: {azi_used} degrees), polar {polar} deg: {popt}")
    
    b_estimated = fit_off_inv_cube(z_target, *popt)
    print(f"Estimated B-field for z = {z_target} mm using fit parameters from polar {round(polar)} deg file for azi {azi} degrees (nearest: {azi_used}), polar {polar} deg: {b_estimated:.2f} G")
    
    return b_estimated

def find_z_for_b_from_fits(b_target, azi, polar):
    """Find the z-position corresponding to a target B-field value using fit parameters from 
    a calibration file for given azimuthal and polar angles."""

    b_cal_df = open_calibration_spreadsheet(round(polar))
    
    # Find nearest azimuthal angle in calibration data (handles floating-point precision)
    try:
        idx = (b_cal_df['Azimuthal Angle (degrees)'] - azi).abs().idxmin()
    except KeyError as e:
        logger.error(f"Column 'Azimuthal Angle (degrees)' not found in calibration data")
        raise KeyError(f"Calibration file missing 'Azimuthal Angle (degrees)' column") from e
    
    # Validate that the index is valid (not NaN or out of bounds)
    if pd.isna(idx):
        logger.error(f"Invalid index returned from idxmin() for polar angle {round(polar)} deg")
        raise ValueError(f"Failed to find valid azimuthal angle index in calibration data for polar {round(polar)} deg")
    
    try:
        row = b_cal_df.loc[[idx]]
    except Exception as e:
        logger.error(f"Failed to access row with index {idx} in calibration data")
        raise ValueError(f"Cannot access calibration data at index {idx}") from e
    
    if row.empty:
        logger.error(f"No data row found for azimuthal angle near {azi} degrees at polar {round(polar)} deg")
        raise ValueError(f"No calibration data found for azi={azi}, polar={round(polar)}")
    
    # Read fit parameters from separate columns
    try:
        a = row['Fit Param a'].iloc[0]
        b = row['Fit Param b'].iloc[0]
        c = row['Fit Param c'].iloc[0]
        z_min = row['Z Min (mm)'].iloc[0]
        z_max = row['Z Max (mm)'].iloc[0]
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to read parameters from calibration row")
        raise ValueError(f"Cannot read parameters from calibration data") from e
    
    popt = (a, b, c)
    azi_used = row['Azimuthal Angle (degrees)'].iloc[0]
    print(f"Retrieved fit parameters for azi {azi} degrees (nearest: {azi_used} degrees), polar {polar} deg: {popt}")
    print(f"Retrieved z range for azi {azi} degrees (nearest: {azi_used} degrees), polar {polar} deg: z_min = {z_min} mm, z_max = {z_max} mm")
    
    def z_solver(z):
        return fit_off_inv_cube(z, *popt) - b_target

    z_solution = fsolve(z_solver, x0=(z_min + z_max) / 2)
    if z_solution[0] < z_min or z_solution[0] > z_max:
        print(f"Warning: Estimated z position {z_solution[0]:.2f} mm for B = {b_target} G is out of range ({z_min} mm to {z_max} mm) for azi {azi} degrees, polar {polar} deg.")
        out_of_bounds = True
    else:
        out_of_bounds = False

    return z_solution[0], out_of_bounds, z_min, z_max


def exclusion_zone_check(stage_name, current_positions, target_position):
    """
    Return True if the move breaches or may breach the exclusion zone.
    Return False if the move appears safe.
    """

    if stage_name == 'thor_azi':
        try:
            b_cal_df = open_calibration_spreadsheet(
                polar=round(current_positions['thor_polar'])
            )
        except Exception as e:
            logger.error(f"Error opening calibration spreadsheet for \u03b8 = {current_positions['thor_polar']} degrees.")
            logger.error(f"Exception: {e}")
            return True  # fail safe

        phi_start = current_positions['thor_azi']
        phi_target = target_position
        zaber_pos = current_positions['zaber']

        # Find nearest calibration angles for start and target positions (handles floating-point precision)
        idx_start = (b_cal_df['Azimuthal Angle (degrees)'] - phi_start).abs().idxmin()
        idx_target = (b_cal_df['Azimuthal Angle (degrees)'] - phi_target).abs().idxmin()
        
        phi_start_cal = b_cal_df.loc[idx_start, 'Azimuthal Angle (degrees)']
        phi_target_cal = b_cal_df.loc[idx_target, 'Azimuthal Angle (degrees)']

        phi_min = min(phi_start_cal, phi_target_cal)
        phi_max = max(phi_start_cal, phi_target_cal)

        rows_to_sweep = b_cal_df.loc[
            (b_cal_df['Azimuthal Angle (degrees)'] >= phi_min)
            & (b_cal_df['Azimuthal Angle (degrees)'] <= phi_max)
        ]

        if rows_to_sweep.empty:
            logger.warning(
                f"No exclusion-zone calibration data found between "
                f"{phi_min} and {phi_max} degrees. Blocking move."
            )
            return True

        breach_mask = zaber_pos < rows_to_sweep['Z Min (mm)']

        if breach_mask.any():
            breached_rows = rows_to_sweep.loc[breach_mask]
            worst_row = breached_rows.loc[breached_rows['Z Min (mm)'].idxmax()]

            logger.warning(
                f"Requested {stage_name} move from {phi_start} to {phi_target} degrees "
                f"may breach exclusion zone. Current zaber position is {zaber_pos} mm, "
                f"but required z_min reaches {worst_row['Z Min (mm)']} mm "
                f"at azimuthal angle {worst_row['Azimuthal Angle (degrees)']} degrees."
            )

            return True

        return False
    
    elif stage_name == 'thor_polar':
        current_polar = round(current_positions['thor_polar'])
        target_polar = round(target_position)
        azi = current_positions['thor_azi']
        zaber_pos = current_positions['zaber']

        b_cal_df_start, start_sheet_exists = open_calibration_spreadsheet(
            polar=current_polar
        )

        b_cal_df_target, target_sheet_exists = open_calibration_spreadsheet(
            polar=target_polar
        )

        b_cal_df_other = None

        if not start_sheet_exists or not target_sheet_exists:
            b_cal_df_other = open_other_calibration_spreadsheet()

        df_start = b_cal_df_start if start_sheet_exists else b_cal_df_other
        df_target = b_cal_df_target if target_sheet_exists else b_cal_df_other
        df_other = b_cal_df_other

        if start_sheet_exists and target_sheet_exists:
            try:
                idx_start = (
                    df_start['Azimuthal Angle (degrees)'] - azi
                ).abs().idxmin()

                idx_target = (
                    df_target['Azimuthal Angle (degrees)'] - azi
                ).abs().idxmin()

                row_start = df_start.loc[idx_start]
                row_target = df_target.loc[idx_target]

            except Exception as e:
                logger.error(f"Error finding nearest azimuthal calibration row: {e}")
                return True

            # azi_start_used = row_start['Azimuthal Angle (degrees)']
            # azi_target_used = row_target['Azimuthal Angle (degrees)']

            z_min_start = row_start['Z Min (mm)']
            z_min_target = row_target['Z Min (mm)']

            breach_start = zaber_pos < z_min_start
            breach_target = zaber_pos < z_min_target

            if breach_start or breach_target:
                if breach_start and breach_target:
                    failed_location = "both the current/start polar angle and the target polar angle"
                elif breach_start:
                    failed_location = "the current/start polar angle"
                else:
                    failed_location = "the target polar angle"

                logger.warning(
                    f"Requested {stage_name} move from {current_positions['thor_polar']} "
                    f"to {target_position} degrees may breach exclusion zone. "
                    f"The exclusion-zone test failed at {failed_location}. "
                    f"Current zaber position is {zaber_pos} mm. "
                    f"Required z_min is {z_min_start} mm at the current polar angle and "
                    f"{z_min_target} mm at the target polar angle. "
                    f"Maximum required z_min is {max(z_min_start, z_min_target)} mm."
                )
                return True

            return False
        
        ### --- If target polar angle is not in calibration file ---- ###
        elif start_sheet_exists and not target_sheet_exists:
            logger.warning(
                f"No exclusion-zone calibration data found for target polar angle {target_polar} degrees. Interpolating from 'other' spreadsheet."
            )

            # --- 1. Get z_min at the current azimuth from the exact start polar sheet --- ###
            azi_idx_polar_start = (
                df_start['Azimuthal Angle (degrees)'] - azi
            ).abs().idxmin()

            azi_row_start = df_start.loc[azi_idx_polar_start]
            z_min_start = azi_row_start['Z Min (mm)']
            
            z_min_array = [z_min_start]

            # --- 2. Build the integer polar sweep excluding the start, since it's already handled --- ###
            step = 1 if target_polar > current_polar else -1

            # Polar angle sweep range should include target polar angle but not current, 
            # since current is already handled in z_min_array
            polar_sweep = range(
                int(current_polar) + step, 
                int(target_polar) + step, 
                step
            )
            
            # --- 3. Extract the rows from the "other" sheet for those polar angles --- ###
            polar_rows = df_other[
                df_other['Polar Angle (degrees)'].isin(polar_sweep)
            ].copy()

            # Preserve the sweep order
            polar_rows['__sweep_order'] = polar_rows['Polar Angle (degrees)'].apply(
                lambda p: list(polar_sweep).index(p)
            )
            polar_rows = polar_rows.sort_values('__sweep_order')

            for _, row in polar_rows.iterrows():
                # For each polar angle in sweep, check if current azimuthal angle falls within the angle range for the measured minimum z value. 
                # If not, interpolate a conservative z_min for that polar angle from the azi = 0 and 100 degree measurements. 
                polar = row['Polar Angle (degrees)']
                
                max_azi_for_min_z = row['Azimuthal Max (degrees)']
                min_azi_for_min_z = row['Azimuthal Min (degrees)']
                z_min = row['Z Min (mm)']

                if pd.isna(max_azi_for_min_z) or pd.isna(min_azi_for_min_z) or pd.isna(z_min):
                    logger.warning(
                        f"Missing azimuthal angle range for minimum z value at polar angle {polar} degrees in 'other' calibration sheet. Skipping this polar angle in interpolation."
                    )
                    continue

                if min_azi_for_min_z <= azi <= max_azi_for_min_z:
                    z_min_array.append(z_min)
                else:
                    pass

        elif not start_sheet_exists and target_sheet_exists:
            logger.warning(
                f"No exclusion-zone calibration data found for current polar angle {current_polar} degrees. Interpolating from 'other' spreadsheet."
            )
            
            # --- 1. Get z_min at the current azimuth from the exact target polar sheet --- ###
            azi_idx_polar_target = (
                df_target['Azimuthal Angle (degrees)'] - azi
            ).abs().idxmin()

            azi_row_target = df_target.loc[azi_idx_polar_target]
            z_min_target = azi_row_target['Z Min (mm)']

            z_min_array = [z_min_target]

            # --- 2. Build the integer polar sweep excluding the target, since it's already handled --- ###
            step = 1 if target_polar > current_polar else -1

            # Polar angle sweep range should include current polar angle but not target, 
            # since target is already handled in z_min_array
            polar_sweep = range(
                int(current_polar) + step, 
                int(target_polar) + step, 
                step
            )

            # --- 3. Extract the rows from the "other" sheet for those polar angles --- ###
            polar_rows = df_other[
                df_other['Polar Angle (degrees)'].isin(polar_sweep)
            ].copy()

            # Preserve the sweep order
            polar_rows['__sweep_order'] = polar_rows['Polar Angle (degrees)'].apply(
                lambda p: list(polar_sweep).index(p)
            )
            polar_rows = polar_rows.sort_values('__sweep_order')

            for _, row in polar_rows.iterrows():
                # For each polar angle in sweep, check if current azimuthal angle falls within the angle range for the measured minimum z value. 
                # If not, interpolate a conservative z_min for that polar angle from the azi = 0 and 100 degree measurements. 
                polar = row['Polar Angle (degrees)']
                
                max_azi_for_min_z = row['Azimuthal Max (degrees)']
                min_azi_for_min_z = row['Azimuthal Min (degrees)']
                z_min = row['Z Min (mm)']

                if pd.isna(max_azi_for_min_z) or pd.isna(min_azi_for_min_z) or pd.isna(z_min):
                    logger.warning(
                        f"Missing azimuthal angle range for minimum z value at polar angle {polar} degrees in 'other' calibration sheet. Skipping this polar angle in interpolation."
                    )
                    continue

                if min_azi_for_min_z <= azi <= max_azi_for_min_z:
                    z_min_array.append(z_min)
                else:
                    pass


        elif not start_sheet_exists and not target_sheet_exists:
            logger.warning(
                f"No exclusion-zone calibration data found for either current polar angle {current_polar} degrees or target polar angle {target_polar} degrees. Interpolating from 'other' spreadsheet."
            )

            

        else:
            logger.error("Logic error in exclusion zone check for polar angle move. This should never happen.")
            return True # fail safe

        # Determine if move breaches exclusion zone based on maximum required z_min in sweep from current to target position
        breach_mask = zaber_pos < max(z_min_array)

    elif stage_name == 'zaber':
        try:
            b_cal_df = open_calibration_spreadsheet(
                polar=round(current_positions['thor_polar'])
            )
        except Exception as e:
            logger.error(f"Error opening calibration spreadsheet for exclusion zone check: {e}")
            return True  # fail safe
        
        phi_current = current_positions['thor_azi']
        
        # Find nearest azimuthal angle in calibration data (handles floating-point precision)
        idx = (b_cal_df['Azimuthal Angle (degrees)'] - phi_current).abs().idxmin()
        row = b_cal_df.loc[[idx]]
        
        if row.empty:
            logger.warning(
                f"No exclusion-zone calibration data found for current azimuthal angle {phi_current} degrees. Blocking move."
            )
            return True
        
        z_min = row['Z Min (mm)'].iloc[0]
        if target_position < z_min:
            logger.warning(
                f"Requested zaber move to {target_position} mm may breach exclusion zone. "
                f"Current zaber position is {current_positions['zaber']} mm, but required z_min is {z_min} mm at current azimuthal angle {phi_current} degrees."
            )
            return True
        
        return False
    
    else:
        logger.error(f"Unknown stage name '{stage_name}' for exclusion zone check. Blocking move.")
        return True
    
# def exclusion_zone():
#     pass

# def exclusion_zone_check(current_positions, target_position, stage_name, exclusion_zones):
#     """Check if the target position for a move is within any defined exclusion zones."""
#     for zone in exclusion_zones:
#         if stage_name == zone['stage']:
#             if abs(target_position - current_positions[stage_name]) < zone['radius']:
#                 logger.warning(f"Requested move to position {target_position} for stage {stage_name} is within exclusion zone around current position {current_positions[stage_name]} with radius {zone['radius']}. Move may not be safe.")
#                 return False
#     return True


# if __name__ == "__main__":
#     # azi_choice = 56  # degrees
#     # polar_choice = 35  # degrees
#     # b_target = 2137  # G
#     # z_for_b_from_fits, safe, z_min, z_max = find_z_for_b_from_fits(b_target, azi_choice, polar_choice)

#     # print(f"Z range: {z_min} mm to {z_max} mm")
#     # if not safe:
#     #     print(f"Warning: Estimated z position for B = {b_target} G using fit parameters from file for azi {azi_choice} degrees, polar {polar_choice} deg is out of range.")
#     # else:
#     #     print(f"Estimated z position for B = {b_target} G using fit parameters from file for azi {azi_choice} degrees, polar {polar_choice} deg: {z_for_b_from_fits:.2f} mm")

#     stage_name = 'thor_azi'
#     current_positions = {'thor_azi': 50, 'thor_polar': int(35.0), 'zaber': 20}  # example current positions
#     target_position = 35 # example target position for thor_azi

#     exc = exclusion_zone_check(stage_name, current_positions, target_position)
#     print(f"Exclusion zone check result for {stage_name}: {exc}")

#     b_estimated = find_b_for_z_from_fits(z_target=88.5, azi=50, polar=35)
#     print(f"Estimated B-field for z = 20 mm: {b_estimated:.2f} G")

# %%