#!/usr/bin/python3
# -*- coding: utf-8 -*-

from ctypes import WinDLL
import ctypes as C
from pathlib import Path
from openpyxl import Workbook, load_workbook
from datetime import datetime

### --- Hall sensor calibration parameters from zero-field calibration --- ###
B_OFFSET = 65.68069249  # G, determined by averaging 20000 readings with no applied magnetic field
Bx_OFFSET = -26.23473900  # G, zero-field x-component offset (updated via zero-field calibration)
By_OFFSET = 48.92314850  # G, zero-field y-component offset (updated via zero-field calibration)
Bz_OFFSET = 34.89738200  # G, zero-field z-component offset (updated via zero-field calibration)
SE_Bx_OFFSET = 0.01898306  # G, standard error in Bx_OFFSET (from 20000-measurement zero-field calibration)
SE_By_OFFSET = 0.01906927  # G, standard error in By_OFFSET (from 20000-measurement zero-field calibration)
SE_Bz_OFFSET = 0.01857257  # G, standard error in Bz_OFFSET (from 20000-measurement zero-field calibration)

### --- Calibration settings --- ###
THETA_FIXED = 0  # degrees, fixed angle above horizontal (111 direction in diamond)
NUM_SAMPLES = 500  # samples per measurement
CALIBRATION_FILE = Path("b_field_calibration_2x2x2Mag_May2026_neg35deg_cal_newpos_full.xlsx")
ZERO_FIELD_CALIB_FILE = Path("b_field_zero_calibration_May2026.xlsx")
SAVE_UNIT_VECTORS = False  # Will be set by user in main()

# A3mtslib = C.CDLL("A3mtslib64")
A3mtslib = C.CDLL("C:\\Users\\nanon\\Senis Hall Sensor\\Installation Software\\matlab example\\64bit\\a3mtslib64.dll")


def initialize_device():
    """Initialize Hall sensor device and return device_number."""
    i = C.c_ushort()
    result = A3mtslib.count_devices(C.byref(i))
    print(f"Number of devices: {i.value}")

    p = C.create_string_buffer(40)
    device_number = C.c_int()
    result = A3mtslib.get_device_name_ch(C.byref(device_number), C.byref(p))
    print(f"Device name: {p.value}")

    device_number.value = 0
    A3mtslib.open_device(C.byref(device_number))

    range_value = C.c_ushort()
    A3mtslib.get_range(C.byref(device_number), C.byref(range_value))
    print(f"Range setting: {range_value.value}")

    return device_number


def collect_zero_field_measurements(device_number, num_measurements=20000):
    """Collect zero-field measurements (no magnetic field applied). Returns list of (bx, by, bz, magnitude) tuples."""
    measurements = []
    
    print(f"\nCollecting {num_measurements} zero-field measurements...")
    print("Make sure NO magnetic field is applied to the sensor!")
    
    for i in range(num_measurements):
        timestamp = C.c_ulong()
        sensorx = C.c_float()
        sensory = C.c_float()
        sensorz = C.c_float()
        A3mtslib.get_sensor_values_fl(
            C.byref(device_number),
            C.byref(timestamp),
            C.byref(sensorx),
            C.byref(sensory),
            C.byref(sensorz),
        )
        bx_gauss = sensorx.value / 100
        by_gauss = sensory.value / 100
        bz_gauss = sensorz.value / 100
        b_magnitude = (bx_gauss**2 + by_gauss**2 + bz_gauss**2) ** 0.5
        
        measurements.append({
            'bx': bx_gauss,
            'by': by_gauss,
            'bz': bz_gauss,
            'magnitude': b_magnitude
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Measurement {i + 1}/{num_measurements}")
    
    return measurements


def save_zero_field_measurements(measurements):
    """Save zero-field measurements to Excel and calculate/display offsets."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Zero Field Measurements"
    
    # Write headers
    ws["A1"] = "Measurement #"
    ws["B1"] = "Bx (G)"
    ws["C1"] = "By (G)"
    ws["D1"] = "Bz (G)"
    ws["E1"] = "Magnitude (G)"
    
    # Write measurements
    for idx, meas in enumerate(measurements, start=1):
        ws[f"A{idx + 1}"] = idx
        ws[f"B{idx + 1}"] = meas['bx']
        ws[f"C{idx + 1}"] = meas['by']
        ws[f"D{idx + 1}"] = meas['bz']
        ws[f"E{idx + 1}"] = meas['magnitude']
    
    # Calculate averages and add them as a summary section
    avg_bx = sum(m['bx'] for m in measurements) / len(measurements)
    avg_by = sum(m['by'] for m in measurements) / len(measurements)
    avg_bz = sum(m['bz'] for m in measurements) / len(measurements)
    avg_mag = sum(m['magnitude'] for m in measurements) / len(measurements)
    
    # Calculate standard deviations
    var_bx = sum((m['bx'] - avg_bx)**2 for m in measurements) / len(measurements)
    var_by = sum((m['by'] - avg_by)**2 for m in measurements) / len(measurements)
    var_bz = sum((m['bz'] - avg_bz)**2 for m in measurements) / len(measurements)
    var_mag = sum((m['magnitude'] - avg_mag)**2 for m in measurements) / len(measurements)
    
    std_bx = var_bx ** 0.5
    std_by = var_by ** 0.5
    std_bz = var_bz ** 0.5
    std_mag = var_mag ** 0.5
    
    # Calculate standard errors (uncertainty in the offset estimates)
    n = len(measurements)
    se_bx = std_bx / (n ** 0.5)
    se_by = std_by / (n ** 0.5)
    se_bz = std_bz / (n ** 0.5)
    se_mag = std_mag / (n ** 0.5)
    
    # Add summary section
    summary_row = len(measurements) + 3
    ws[f"A{summary_row}"] = "ZERO-FIELD OFFSETS (Average Values)"
    ws[f"A{summary_row + 1}"] = "Component"
    ws[f"B{summary_row + 1}"] = "Average (G)"
    ws[f"C{summary_row + 1}"] = "Std Error (G)"
    
    ws[f"A{summary_row + 2}"] = "Bx_OFFSET"
    ws[f"B{summary_row + 2}"] = avg_bx
    ws[f"C{summary_row + 2}"] = se_bx
    
    ws[f"A{summary_row + 3}"] = "By_OFFSET"
    ws[f"B{summary_row + 3}"] = avg_by
    ws[f"C{summary_row + 3}"] = se_by
    
    ws[f"A{summary_row + 4}"] = "Bz_OFFSET"
    ws[f"B{summary_row + 4}"] = avg_bz
    ws[f"C{summary_row + 4}"] = se_bz
    
    ws[f"A{summary_row + 5}"] = "B_OFFSET (Magnitude)"
    ws[f"B{summary_row + 5}"] = avg_mag
    ws[f"C{summary_row + 5}"] = se_mag
    
    wb.save(ZERO_FIELD_CALIB_FILE)
    
    print(f"\n✓ Zero-field measurements saved to: {ZERO_FIELD_CALIB_FILE}")
    print("\n" + "=" * 60)
    print("ZERO-FIELD OFFSETS (Use these in your calibration code)")
    print("=" * 60)
    print(f"Bx_OFFSET = {avg_bx:.8f}  # ± {se_bx:.8f} G")
    print(f"By_OFFSET = {avg_by:.8f}  # ± {se_by:.8f} G")
    print(f"Bz_OFFSET = {avg_bz:.8f}  # ± {se_bz:.8f} G")
    print(f"B_OFFSET  = {avg_mag:.8f}  # ± {se_mag:.8f} G (magnitude-based, legacy)")
    print("=" * 60)
    
    return avg_bx, avg_by, avg_bz, avg_mag


def run_zero_field_calibration(device_number):
    """Run complete zero-field calibration: collect 20000 measurements and save offsets."""
    print("\n" + "=" * 60)
    print("ZERO-FIELD CALIBRATION")
    print("=" * 60)
    print("This will collect 20000 measurements with NO magnetic field applied.")
    print("These measurements will be averaged to determine component-wise offsets.")
    confirm = input("Continue? (y/n) [n]: ").strip().lower()
    if confirm != "y":
        print("Zero-field calibration cancelled.")
        return
    
    measurements = collect_zero_field_measurements(device_number, num_measurements=20000)
    save_zero_field_measurements(measurements)


def take_measurement(device_number, num_samples=NUM_SAMPLES):
    """Acquire num_samples from Hall sensor and return average B-field (offset-corrected) and xyz components.
    
    Error propagation:
    - Measurement error: std_dev / sqrt(num_samples) for each component
    - Offset error: SE_Bx_OFFSET, SE_By_OFFSET, SE_Bz_OFFSET (from zero-field calibration)
    - Total component error: quadrature sum of measurement and offset errors
    - B_mag error: propagated from component errors using partial derivatives
    """
    bx_values = []
    by_values = []
    bz_values = []

    print(f"\nAcquiring {num_samples} samples...")
    for i in range(num_samples):
        timestamp = C.c_ulong()
        sensorx = C.c_float()
        sensory = C.c_float()
        sensorz = C.c_float()
        A3mtslib.get_sensor_values_fl(
            C.byref(device_number),
            C.byref(timestamp),
            C.byref(sensorx),
            C.byref(sensory),
            C.byref(sensorz),
        )
        bx_gauss = sensorx.value / 100
        by_gauss = sensory.value / 100
        bz_gauss = sensorz.value / 100
        bx_values.append(bx_gauss)
        by_values.append(by_gauss)
        bz_values.append(bz_gauss)

        if (i + 1) % 10 == 0:
            print(f"  Sample {i + 1}/{num_samples}: Bx={bx_gauss:.2f} G, By={by_gauss:.2f} G, Bz={bz_gauss:.2f} G (raw)")

    # Compute average xyz components
    bx_avg = sum(bx_values) / len(bx_values)
    by_avg = sum(by_values) / len(by_values)
    bz_avg = sum(bz_values) / len(bz_values)

    # Calculate standard deviations for each component from the NUM_SAMPLES measurements
    var_bx = sum((x - bx_avg)**2 for x in bx_values) / len(bx_values)
    var_by = sum((y - by_avg)**2 for y in by_values) / len(by_values)
    var_bz = sum((z - bz_avg)**2 for z in bz_values) / len(bz_values)
    
    std_bx = var_bx ** 0.5
    std_by = var_by ** 0.5
    std_bz = var_bz ** 0.5
    
    # Calculate measurement standard errors (from NUM_SAMPLES)
    n = len(bx_values)
    se_bx_meas = std_bx / (n ** 0.5)
    se_by_meas = std_by / (n ** 0.5)
    se_bz_meas = std_bz / (n ** 0.5)
    
    # Combine measurement and offset errors in quadrature
    se_bx_total = (se_bx_meas**2 + SE_Bx_OFFSET**2) ** 0.5
    se_by_total = (se_by_meas**2 + SE_By_OFFSET**2) ** 0.5
    se_bz_total = (se_bz_meas**2 + SE_Bz_OFFSET**2) ** 0.5
    
    # Apply offsets to get corrected components
    bx_avg_corrected = bx_avg - Bx_OFFSET
    by_avg_corrected = by_avg - By_OFFSET
    bz_avg_corrected = bz_avg - Bz_OFFSET

    # Compute total magnitude from corrected components
    b_mag = (bx_avg_corrected**2 + by_avg_corrected**2 + bz_avg_corrected**2) ** 0.5
    
    # Propagate component errors to b_mag using partial derivatives
    # For b_mag = sqrt(bx^2 + by^2 + bz^2):
    # db_mag/dbx = bx/b_mag, db_mag/dby = by/b_mag, db_mag/dbz = bz/b_mag
    if b_mag != 0:
        se_b_mag = ((bx_avg_corrected/b_mag * se_bx_total)**2 + 
                    (by_avg_corrected/b_mag * se_by_total)**2 + 
                    (bz_avg_corrected/b_mag * se_bz_total)**2) ** 0.5
    else:
        se_b_mag = 0
    
    # Calculate unit vector (normalized by magnitude of corrected components)
    if b_mag != 0:
        bx_unit = bx_avg_corrected / b_mag
        by_unit = by_avg_corrected / b_mag
        bz_unit = bz_avg_corrected / b_mag
    else:
        bx_unit, by_unit, bz_unit = 0, 0, 0
    
    print(f"Average Bx (offset-corrected): {bx_avg_corrected:.4f} ± {se_bx_total:.4f} G")
    print(f"Average By (offset-corrected): {by_avg_corrected:.4f} ± {se_by_total:.4f} G")
    print(f"Average Bz (offset-corrected): {bz_avg_corrected:.4f} ± {se_bz_total:.4f} G")
    print(f"Average B-field (magnitude): {b_mag:.4f} ± {se_b_mag:.4f} G")
    if SAVE_UNIT_VECTORS:
        print(f"Unit vector: ({bx_unit:.4f}, {by_unit:.4f}, {bz_unit:.4f})")

    return b_mag, se_b_mag, bx_unit, by_unit, bz_unit


def initialize_spreadsheet():
    """Create or load Excel calibration file with two sheets: Raw Data and Calibration Matrix."""
    if CALIBRATION_FILE.exists():
        wb = load_workbook(CALIBRATION_FILE)
        print(f"Loaded existing calibration file: {CALIBRATION_FILE}")
    else:
        wb = Workbook()
        
        # Sheet 1: Raw Data (append-only measurement log)
        ws_raw = wb.active
        ws_raw.title = "Raw Data"
        ws_raw["A1"] = "Z (stage position)"
        ws_raw["B1"] = "Theta (degrees)"
        ws_raw["C1"] = "Phi (degrees)"
        ws_raw["D1"] = "B_total (G, offset-corrected)"
        ws_raw["E1"] = "Standard Error (G)"
        
        # Add unit vector columns if enabled
        if SAVE_UNIT_VECTORS:
            ws_raw["F1"] = "Bx_unit"
            ws_raw["G1"] = "By_unit"
            ws_raw["H1"] = "Bz_unit"
            ws_raw["I1"] = "Timestamp"
        else:
            ws_raw["F1"] = "Timestamp"
        
        # Sheet 2: Calibration Matrix (analysis-ready pivot table)
        ws_calib = wb.create_sheet("Calibration Matrix")
        ws_calib["A1"] = "Z \ Phi"
        
        wb.save(CALIBRATION_FILE)
        print(f"Created new calibration file: {CALIBRATION_FILE}")
    
    return wb


def find_measurement_row(ws_raw, z, phi):
    """Find the row index of an existing (z, phi) measurement. Returns None if not found."""
    for row_idx, row in enumerate(ws_raw.iter_rows(min_row=2, values_only=True), start=2):
        # Extract first 3 columns regardless of how many columns exist
        if len(row) < 3:
            continue
        z_val, theta_val, phi_val = row[0], row[1], row[2]
        try:
            if float(z_val) == z and float(phi_val) == phi:
                return row_idx
        except (ValueError, TypeError):
            continue
    return None


def save_to_spreadsheet(wb, z, theta, phi, b_corrected, std_error, bx_unit=None, by_unit=None, bz_unit=None, row_to_update=None):
    """Append a measurement row to the Raw Data sheet or update existing row."""
    ws_raw = wb["Raw Data"]
    
    if row_to_update is not None:
        # Update existing row
        ws_raw[f"D{row_to_update}"] = b_corrected
        ws_raw[f"E{row_to_update}"] = std_error
        
        if SAVE_UNIT_VECTORS:
            ws_raw[f"F{row_to_update}"] = bx_unit
            ws_raw[f"G{row_to_update}"] = by_unit
            ws_raw[f"H{row_to_update}"] = bz_unit
            ws_raw[f"I{row_to_update}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            ws_raw[f"F{row_to_update}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        wb.save(CALIBRATION_FILE)
        print(f"✓ Data updated in Raw Data sheet, row {row_to_update}")
    else:
        # Append new row
        row = ws_raw.max_row + 1
        ws_raw[f"A{row}"] = z
        ws_raw[f"B{row}"] = theta
        ws_raw[f"C{row}"] = phi
        ws_raw[f"D{row}"] = b_corrected
        ws_raw[f"E{row}"] = std_error
        
        if SAVE_UNIT_VECTORS:
            ws_raw[f"F{row}"] = bx_unit
            ws_raw[f"G{row}"] = by_unit
            ws_raw[f"H{row}"] = bz_unit
            ws_raw[f"I{row}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            ws_raw[f"F{row}"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        wb.save(CALIBRATION_FILE)
        print(f"✓ Data saved to Raw Data sheet, row {row}")
    
    # Update Calibration Matrix
    update_calibration_matrix(wb)


def update_calibration_matrix(wb):
    """Pivot raw data into calibration matrix format (z as rows, phi as columns)."""
    ws_raw = wb["Raw Data"]
    ws_calib = wb["Calibration Matrix"]
    
    # Read all raw data and build matrix
    data_matrix = {}  # {z: {phi: b_total}}
    
    # Iterate through all rows in Raw Data sheet
    for row in ws_raw.iter_rows(min_row=2, values_only=True):
        # Extract first 5 columns (z, theta, phi, b_total, std_error)
        # Ignore unit vectors and timestamp columns
        if len(row) < 5:
            continue
        z, theta, phi, b_total, std_error = row[0], row[1], row[2], row[3], row[4]
        
        # Skip incomplete rows
        if z is None or phi is None or b_total is None:
            continue
        
        try:
            z = float(z)
            phi = float(phi)
            b_total = float(b_total)
        except (ValueError, TypeError):
            continue
        
        if z not in data_matrix:
            data_matrix[z] = {}
        data_matrix[z][phi] = b_total
    
    # Clear all rows from calibration sheet
    for row in ws_calib.iter_rows():
        for cell in row:
            cell.value = None
    
    # Get sorted unique z and phi values
    z_values = sorted(data_matrix.keys(), reverse=True)  # High to low (100 to 0)
    all_phi = set()
    for phi_dict in data_matrix.values():
        all_phi.update(phi_dict.keys())
    phi_values = sorted(all_phi)
    
    # Write header row with phi values
    ws_calib.cell(row=1, column=1, value="Z \ Phi")
    for col_idx, phi in enumerate(phi_values, start=2):
        ws_calib.cell(row=1, column=col_idx, value=phi)
    
    # Write data rows with z values and b_total values
    for row_idx, z in enumerate(z_values, start=2):
        ws_calib.cell(row=row_idx, column=1, value=z)
        for col_idx, phi in enumerate(phi_values, start=2):
            b_value = data_matrix.get(z, {}).get(phi, None)
            ws_calib.cell(row=row_idx, column=col_idx, value=b_value)
    
    wb.save(CALIBRATION_FILE)
    print(f"✓ Calibration Matrix updated ({len(z_values)} z-levels, {len(phi_values)} phi angles)")


def get_coordinates():
    """Prompt user for z and phi; theta is fixed."""
    while True:
        try:
            z = float(input(f"\nEnter Z position (0-100): "))
            if not 0 <= z <= 100:
                print("Z must be between 0 and 100")
                continue

            phi = float(input(f"Enter Phi/Azimuthal angle (0-100 degrees): "))
            if not 0 <= phi <= 100:
                print("Phi must be between 0 and 100 degrees")
                continue

            return z, phi

        except ValueError:
            print("Invalid input. Please enter numeric values.")


def main():
    global SAVE_UNIT_VECTORS
    
    print("=" * 60)
    print("B-Field Calibration System (111 Diamond Direction)")
    print("=" * 60)
    print(f"Fixed Theta: {THETA_FIXED}°")
    print(f"B-Offset (magnitude): {B_OFFSET} G")
    print(f"Bx_OFFSET: {Bx_OFFSET} G")
    print(f"By_OFFSET: {By_OFFSET} G")
    print(f"Bz_OFFSET: {Bz_OFFSET} G")
    print(f"Samples per measurement: {NUM_SAMPLES}")
    print("=" * 60)
    
    # Initialize device once
    device_number = initialize_device()
    
    try:
        # Offer zero-field calibration option
        print("\n" + "=" * 60)
        mode = input("Select mode:\n(1) Run zero-field calibration (collect offsets)\n(2) Run field calibration (measure field at positions)\n[2]: ").strip()
        if mode == "1":
            run_zero_field_calibration(device_number)
            print("\nZero-field calibration complete.")
            print("You can now use the Bx_OFFSET, By_OFFSET, Bz_OFFSET values in your calibration.")
            return
        
        # Regular field calibration workflow
        print("=" * 60)
        
        # Ask user if they want to save unit vectors
        save_vectors = input("\nSave unit vectors for each measurement? (y/n) [n]: ").strip().lower()
        SAVE_UNIT_VECTORS = save_vectors == "y"
        if SAVE_UNIT_VECTORS:
            print("✓ Unit vectors will be saved (Bx_unit, By_unit, Bz_unit)")
        else:
            print("✓ Unit vectors will NOT be saved")

        # Initialize or load spreadsheet
        wb = initialize_spreadsheet()

        while True:
            print("\n" + "=" * 60)
            print("Move magnet stage to desired position")
            print("=" * 60)

            z, phi = get_coordinates()

            # Check if measurement already exists at this (z, phi)
            ws_raw = wb["Raw Data"]
            existing_row = find_measurement_row(ws_raw, z, phi)
            
            if existing_row is not None:
                print(f"\n⚠ Measurement already exists at Z={z}, Phi={phi} (row {existing_row})")
                override = input("Override existing data? (y/n) [n]: ").strip().lower()
                if override != "y":
                    print("Measurement skipped.")
                    again = input("\nRun another measurement? (y/n) [y]: ").strip().lower()
                    if again == "n":
                        break
                    continue
            else:
                existing_row = None

            # Take measurement (now returns unit vector components)
            b_corrected, std_error, bx_unit, by_unit, bz_unit = take_measurement(device_number)

            # Save to spreadsheet and auto-update calibration matrix
            save_to_spreadsheet(wb, z, THETA_FIXED, phi, b_corrected, std_error, 
                              bx_unit, by_unit, bz_unit, row_to_update=existing_row)

            # Prompt for next measurement
            again = input(
                "\nRun another measurement? (y/n) [y]: "
            ).strip().lower()
            if again == "n":
                break

    except KeyboardInterrupt:
        print("\n\nExiting...")

    finally:
        print("Closing device...")
        A3mtslib.close_device(C.byref(device_number))
        print("Done.")


if __name__ == "__main__":
    main()