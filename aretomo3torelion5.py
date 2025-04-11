#!/usr/bin/env python3
import os
import json
import csv
import argparse
import math
import numpy as np
import sys
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Convert AreTomo3 output to RELION5 star files')
    parser.add_argument('aretomo_dir', type=str, help='Directory containing AreTomo3 output')
    parser.add_argument('--output_dir', type=str, default='relion_star_files', help='Output directory for RELION5 star files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    # The dose (in e-/Å²) per reference tilt step is now required.
    parser.add_argument('--dose', type=float, required=True, 
                        help='Electron dose per tilt (in e-/Å²).')
    return parser.parse_args()

def read_session_json(aretomo_dir):
    """Read AreTomo3 Session.json file."""
    json_file = os.path.join(aretomo_dir, 'AreTomo3_Session.json')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Session JSON file not found: {json_file}")
    with open(json_file, 'r') as f:
        session_data = json.load(f)
    return session_data

def get_tomo_prefix(aretomo_dir):
    """Extract tomogram prefix from directory contents using a .mrc file (excluding Vol, EVN, ODD, CTF)."""
    mrc_files = [f for f in os.listdir(aretomo_dir)
                 if f.endswith('.mrc') and not any(x in f for x in ['Vol', 'EVN', 'ODD', 'CTF'])]
    if not mrc_files:
        raise FileNotFoundError(f"No tomogram MRC files found in {aretomo_dir}")
    return os.path.splitext(mrc_files[0])[0]

def read_order_list(aretomo_dir, tomo_prefix):
    """
    Read tilt order and tilt angles from the IMOD order list CSV file.
    Expected file path: {aretomo_dir}/{tomo_prefix}_Imod/{tomo_prefix}_order_list.csv
    Assumes the CSV has a header with a column that contains 'tilt' (case insensitive).
    Returns a list of tilt angles (floats) in acquisition order.
    """
    order_list_file = os.path.join(aretomo_dir, f"{tomo_prefix}_Imod", f"{tomo_prefix}_order_list.csv")
    if not os.path.exists(order_list_file):
        raise FileNotFoundError(f"Order list CSV file not found: {order_list_file}")
    tilt_angles = []
    with open(order_list_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # Try to find a field whose name includes "tilt" (case-insensitive)
        key = None
        for field in reader.fieldnames:
            if "tilt" in field.lower():
                key = field
                break
        if key is None:
            # If no matching header, assume the first column is the tilt angle.
            for row in reader:
                angle = float(list(row.values())[0])
                tilt_angles.append(angle)
        else:
            for row in reader:
                angle = float(row[key])
                tilt_angles.append(angle)
    return tilt_angles

def read_tlt_file(aretomo_dir, tomo_prefix):
    """Read tilt angles from the .tlt file."""
    tlt_file = os.path.join(aretomo_dir, f"{tomo_prefix}_Imod", f"{tomo_prefix}_st.tlt")
    if not os.path.exists(tlt_file):
        raise FileNotFoundError(f"Tilt file not found: {tlt_file}")
    with open(tlt_file, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def read_xf_file(aretomo_dir, tomo_prefix):
    """Read transformation matrices from the .xf file.
    Returns a list of lists, each with 6 floats: [A11, A12, A21, A22, DX, DY].
    """
    xf_file = os.path.join(aretomo_dir, f"{tomo_prefix}_Imod", f"{tomo_prefix}_st.xf")
    if not os.path.exists(xf_file):
        raise FileNotFoundError(f"XF file not found: {xf_file}")
    xf_data = []
    with open(xf_file, 'r') as f:
        for line in f:
            if line.strip():
                xf_data.append([float(x) for x in line.strip().split()])
    return xf_data

def read_ctf_file(aretomo_dir, tomo_prefix):
    """Read CTF defocus information from CTF_Imod.txt."""
    ctf_file = os.path.join(aretomo_dir, f"{tomo_prefix}_CTF.txt")
    if not os.path.exists(ctf_file):
        raise FileNotFoundError(f"CTF file not found: {ctf_file}")
    ctf_data = []
    with open(ctf_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 8:
                    ctf_data.append({
                        'frame': int(parts[0]),
                        'tilt_angle': None,
                        'defocus_u': float(parts[1]),
                        'defocus_v': float(parts[2]),
                        'astigmatism_angle': float(parts[3])
                    })
    return ctf_data

def read_metrics_csv(aretomo_dir):
    """Read tilt series metrics from TiltSeries_Metrics.csv (optional)."""
    metrics_file = os.path.join(aretomo_dir, 'TiltSeries_Metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found: {metrics_file}")
        return None
    metrics_data = {}
    with open(metrics_file, 'r') as f:
        header = f.readline().strip().split(',')
        data_line = f.readline().strip().split(',')
        if len(header) != len(data_line):
            print("Warning: Metrics file has mismatched header and data columns")
            return None
        for i, key in enumerate(header):
            metrics_data[key.strip()] = data_line[i].strip()
    return metrics_data

def read_aln_file(aretomo_dir, tomo_prefix):
    """Read alignment info from the .aln file."""
    aln_file = os.path.join(aretomo_dir, f"{tomo_prefix}.aln")
    if not os.path.exists(aln_file):
        raise FileNotFoundError(f"ALN file not found: {aln_file}")
    
    aln_data = []
    with open(aln_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("# SEC"):
                for j in range(i+1, len(lines)):
                    data_line = lines[j].strip()
                    if data_line and data_line.lstrip()[0].isdigit():
                        parts = data_line.split()
                        if len(parts) >= 3:
                            aln_data.append(parts)
                break
    if not aln_data:
        print("ALN file content preview:")
        with open(aln_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(f"Line {i+1}: {repr(line)}")
        raise ValueError("No alignment data found in ALN file.")
    return aln_data

def read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix):
    """
    Read volume dimensions from the header of the .aln file.
    Returns a tuple (vol_x, vol_y, None).
    """
    aln_file = os.path.join(aretomo_dir, f"{tomo_prefix}.aln")
    if not os.path.exists(aln_file):
        raise FileNotFoundError(f"ALN file not found: {aln_file}")
    with open(aln_file, 'r') as f:
        for line in f:
            if "RawSize" in line:
                parts = line.strip().split("=")
                if len(parts) == 2:
                    dims = parts[1].strip().split()
                    if len(dims) >= 2:
                        try:
                            vol_x = int(dims[0])
                            vol_y = int(dims[1])
                            return vol_x, vol_y, None
                        except ValueError:
                            raise ValueError(f"Failed to parse dimensions from: {line.strip()}")
    raise ValueError("Dimensions not found in ALN file.")

def read_volZ_from_json(session_data):
    """Read the Z dimension from the session JSON."""
    vol_z = session_data['parameters'].get('VolZ')
    if vol_z is None:
        raise ValueError("VolZ not found in session metadata.")
    return vol_z

def read_ctf_txt(aretomo_dir, tomo_prefix):
    """
    Read additional CTF information (including handedness, dfHand) from the _CTF.txt file.
    """
    ctf_txt_file = os.path.join(aretomo_dir, f"{tomo_prefix}_CTF.txt")
    if not os.path.exists(ctf_txt_file):
        raise FileNotFoundError(f"CTF txt file not found: {ctf_txt_file}")
    ctf_txt_data = []
    with open(ctf_txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                entry = {
                    'micrograph_num': int(parts[0]),
                    'defocus1': float(parts[1]),
                    'defocus2': float(parts[2]),
                    'astig_azimuth': float(parts[3]),
                    'phase_shift': float(parts[4]),
                    'cross_corr': float(parts[5]),
                    'ctf_fit_range': float(parts[6]),
                    'dfHand': float(parts[7])
                }
                ctf_txt_data.append(entry)
    if not ctf_txt_data:
        raise ValueError("No CTF information (including handedness) found in _CTF.txt.")
    return ctf_txt_data

def compute_tilt_alignment(xf_row, pixel_size):
    """
    Compute RELION5 tilt parameters from the IMOD .xf transformation matrix.
    Returns (x_tilt, y_tilt, z_rot, x_shift_angst, y_shift_angst).
    """
    A11, A12, A21, A22, DX, DY = xf_row
    tr_matrix = np.array([[A11, A12], [A21, A22]])
    z_rot = math.degrees(math.atan2(A21, A11))
    i_tr_matrix = np.linalg.inv(tr_matrix)
    x_shift = i_tr_matrix[0, 0] * DX + i_tr_matrix[0, 1] * DY
    y_shift = i_tr_matrix[1, 0] * DX + i_tr_matrix[1, 1] * DY
    x_shift_angst = x_shift * pixel_size
    y_shift_angst = y_shift * pixel_size
    x_tilt = 0.0
    y_tilt = 0.0  # will be replaced with actual tilt angle later
    return x_tilt, y_tilt, z_rot, x_shift_angst, y_shift_angst

def calculate_cumulative_exposure(tilt_angles, dose_per_tilt):
    """
    Dynamically calculate cumulative exposure based on the acquired tilt angles.
    The idea is to compute the median difference between successive tilts, then 
    for each tilt, accumulate (|difference| / median_diff) * dose. The tilt closest 
    to 0° is assigned a cumulative dose of 0.
    This means that if a tilt is missing (i.e. a larger gap occurs), the dose increment 
    will be proportionally larger.
    """
    n = len(tilt_angles)
    if dose_per_tilt <= 0:
        return [0.0] * n

    # Calculate differences (in absolute value) between consecutive tilt angles.
    diffs = [abs(tilt_angles[i] - tilt_angles[i-1]) for i in range(1, n)]
    sorted_diffs = sorted(diffs)
    m = len(sorted_diffs)
    if m % 2 == 1:
        ref_inc = sorted_diffs[m//2]
    else:
        ref_inc = 0.5 * (sorted_diffs[m//2 - 1] + sorted_diffs[m//2])
    if ref_inc == 0:
        ref_inc = 1.0  # safeguard

    # Find the index of the tilt closest to 0°.
    zero_idx = min(range(n), key=lambda i: abs(tilt_angles[i]))
    exposures = [0.0] * n
    exposures[zero_idx] = 0.0

    current_exp = 0.0
    # Forward accumulation (from zero to the end)
    for i in range(zero_idx + 1, n):
        diff = tilt_angles[i] - tilt_angles[i - 1]  # should be positive if tilts sorted in increasing order
        current_exp += (abs(diff) / ref_inc) * dose_per_tilt
        exposures[i] = current_exp

    current_exp = 0.0
    # Backward accumulation (from zero to the beginning)
    for i in range(zero_idx - 1, -1, -1):
        diff = tilt_angles[i + 1] - tilt_angles[i]
        current_exp += (abs(diff) / ref_inc) * dose_per_tilt
        exposures[i] = current_exp

    return exposures

def create_softlinks(aretomo_dir, output_dir, tomo_prefix):
    """Create softlinks with .mrcs extension for .mrc files."""
    os.makedirs(output_dir, exist_ok=True)
    abs_aretomo_dir = os.path.abspath(aretomo_dir)
    abs_output_dir = os.path.abspath(output_dir)
    
    mrc_file = os.path.join(abs_aretomo_dir, f"{tomo_prefix}.mrc")
    evn_file = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_EVN.mrc")
    odd_file = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_ODD.mrc")
    ctf_file = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_CTF.mrc")
    vol_file = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_Vol.mrc")
    
    mrc_link = os.path.join(abs_output_dir, f"{tomo_prefix}.mrcs")
    evn_link = os.path.join(abs_output_dir, f"{tomo_prefix}_EVN.mrcs")
    odd_link = os.path.join(abs_output_dir, f"{tomo_prefix}_ODD.mrcs")
    ctf_link = os.path.join(abs_output_dir, f"{tomo_prefix}_CTF.mrcs")
    
    links_created = []
    for src, dst in [(mrc_file, mrc_link), (evn_file, evn_link),
                     (odd_file, odd_link), (ctf_file, ctf_link)]:
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            links_created.append((src, dst))
            print(f"Created softlink: {dst} -> {src}")
        else:
            print(f"Warning: Source file not found: {src}")
    return links_created, vol_file

def create_tomogram_star(session_data, output_dir, tomo_prefix, aretomo_dir, vol_file):
    """Create the tomogram.star file for RELION5 using metadata from AreTomo3."""
    os.makedirs(output_dir, exist_ok=True)
    voltage      = session_data['parameters']['kV']
    cs           = session_data['parameters']['Cs']
    amp_contrast = session_data['parameters']['AmpContrast']
    pixel_size   = session_data['parameters']['PixSize']
    optics_group = "optics1"
    bin_factor   = session_data['parameters'].get('AtBin', [1])[0]
    
    vol_size_x, vol_size_y, _ = read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix)
    vol_size_z = read_volZ_from_json(session_data)
    ctf_txt_data = read_ctf_txt(aretomo_dir, tomo_prefix)
    hand = ctf_txt_data[0]['dfHand']
    
    abs_output_dir = os.path.abspath(output_dir)
    tilt_series_star = os.path.join(abs_output_dir, f"{tomo_prefix}.star")
    etomo_directive  = os.path.join(abs_output_dir, f"{tomo_prefix}.edf")
    reconstructed_tomo = vol_file
    
    tomogram_star_path = os.path.join(output_dir, 'tomograms.star')
    with open(tomogram_star_path, 'w') as f:
        f.write("# version 50001\n\n")
        f.write("data_global\n")
        f.write("loop_\n")
        f.write("_rlnTomoName #1\n")
        f.write("_rlnVoltage #2\n")
        f.write("_rlnSphericalAberration #3\n")
        f.write("_rlnAmplitudeContrast #4\n")
        f.write("_rlnMicrographOriginalPixelSize #5\n")
        f.write("_rlnTomoHand #6\n")
        f.write("_rlnOpticsGroupName #7\n")
        f.write("_rlnTomoTiltSeriesPixelSize #8\n")
        f.write("_rlnTomoTiltSeriesStarFile #9\n")
        f.write("_rlnEtomoDirectiveFile #10\n")
        f.write("_rlnTomoTomogramBinning #11\n")
        f.write("_rlnTomoSizeX #12\n")
        f.write("_rlnTomoSizeY #13\n")
        f.write("_rlnTomoSizeZ #14\n")
        f.write("_rlnTomoReconstructedTomogram #15\n")
        f.write(f"{tomo_prefix}   {voltage:.6f}   {cs:.6f}   {amp_contrast:.6f}   {pixel_size:.6f}   {hand:.6f}   {optics_group}   {pixel_size:.6f}   {tilt_series_star}   {etomo_directive}   {bin_factor:.6f}   {vol_size_x}   {vol_size_y}   {vol_size_z}   {reconstructed_tomo}\n")
    
    print(f"Created tomogram star file: {tomogram_star_path}")
    return tomogram_star_path

def create_tilt_series_star(session_data, output_dir, tomo_prefix, aretomo_dir, tilt_angles, xf_data, ctf_data, aln_data, dose_per_tilt):
    """Create the tilt-series star file using AreTomo3 metadata and .mrcs softlinks."""
    os.makedirs(output_dir, exist_ok=True)
    pixel_size = session_data['parameters']['PixSize']
    if "TiltAxis" not in session_data['parameters']:
        raise ValueError("TiltAxis parameter not found in session metadata.")
    tilt_axis = session_data['parameters']["TiltAxis"][0]
    
    raw_dims = read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix)
    image_dims = (raw_dims[0], raw_dims[1])
    
    abs_output_dir = os.path.abspath(output_dir)
    even_mrcs_file    = os.path.join(abs_output_dir, f"{tomo_prefix}_EVN.mrcs")
    odd_mrcs_file     = os.path.join(abs_output_dir, f"{tomo_prefix}_ODD.mrcs")
    aligned_mrcs_file = os.path.join(abs_output_dir, f"{tomo_prefix}.mrcs")
    ctf_mrcs_file     = os.path.join(abs_output_dir, f"{tomo_prefix}_CTF.mrcs")
    
    if ctf_data and ctf_data[0]['tilt_angle'] is None:
        for entry in ctf_data:
            frame_idx = entry['frame'] - 1
            if 0 <= frame_idx < len(tilt_angles):
                entry['tilt_angle'] = tilt_angles[frame_idx]
    
    # Dynamically calculate cumulative exposures based on tilt angles and dose.
    exposures = calculate_cumulative_exposure(tilt_angles, dose_per_tilt)
    
    tilt_series_star_path = os.path.join(output_dir, f"{tomo_prefix}.star")
    with open(tilt_series_star_path, 'w') as f:
        f.write("# Generated by AreTomo3 to RELION5 converter\n")
        f.write("# Relion star file version 50001\n\n")
        f.write(f"data_{tomo_prefix}\n\n")
        f.write("loop_\n")
        f.write("_rlnMicrographMovieName\n")
        f.write("_rlnTomoTiltMovieFrameCount\n")
        f.write("_rlnTomoNominalStageTiltAngle\n")
        f.write("_rlnTomoNominalTiltAxisAngle\n")
        f.write("_rlnMicrographPreExposure\n")
        f.write("_rlnTomoNominalDefocus\n")
        f.write("_rlnCtfPowerSpectrum\n")
        f.write("_rlnMicrographNameEven\n")
        f.write("_rlnMicrographNameOdd\n")
        f.write("_rlnMicrographName\n")
        f.write("_rlnMicrographMetadata\n")
        f.write("_rlnAccumMotionTotal\n")
        f.write("_rlnAccumMotionEarly\n")
        f.write("_rlnAccumMotionLate\n")
        f.write("_rlnCtfImage\n")
        f.write("_rlnDefocusU\n")
        f.write("_rlnDefocusV\n")
        f.write("_rlnCtfAstigmatism\n")
        f.write("_rlnDefocusAngle\n")
        f.write("_rlnCtfFigureOfMerit\n")
        f.write("_rlnCtfMaxResolution\n")
        f.write("_rlnCtfIceRingDensity\n")
        f.write("_rlnTomoXTilt\n")
        f.write("_rlnTomoYTilt\n")
        f.write("_rlnTomoZRot\n")
        f.write("_rlnTomoXShiftAngst\n")
        f.write("_rlnTomoYShiftAngst\n")
        f.write("_rlnCtfScalefactor\n")
        
        for i, tilt_angle in enumerate(tilt_angles):
            defocus_u = defocus_v = astigmatism_angle = 0.0
            for ctf_entry in ctf_data:
                if 'frame' in ctf_entry and ctf_entry['frame'] == i + 1:
                    defocus_u = ctf_entry['defocus_u']
                    defocus_v = ctf_entry['defocus_v']
                    astigmatism_angle = ctf_entry['astigmatism_angle']
                    break
                elif abs(ctf_entry['tilt_angle'] - tilt_angle) < 0.1:
                    defocus_u = ctf_entry['defocus_u']
                    defocus_v = ctf_entry['defocus_v']
                    astigmatism_angle = ctf_entry['astigmatism_angle']
                    break
            astigmatism = abs(defocus_u - defocus_v)
            defocus_angle = astigmatism_angle
            
            x_tilt, _, z_rot, x_shift_angst, y_shift_angst = compute_tilt_alignment(xf_data[i], pixel_size)
            y_tilt = tilt_angle
            
            even_entry    = f"{i+1:06d}@{even_mrcs_file}"
            odd_entry     = f"{i+1:06d}@{odd_mrcs_file}"
            aligned_entry = f"{i+1}@{aligned_mrcs_file}"
            ctf_entry_str = f"{i+1}@{ctf_mrcs_file}"
            ctf_scalefactor = math.cos(math.radians(tilt_angle))
            exposure = exposures[i]
            
            f.write(
                f"FileNotFound   1   {tilt_angle:.6f}   {tilt_axis:.6f}   {exposure:.6f}   0.000000   FileNotFound   "
                f"{even_entry}   {odd_entry}   {aligned_entry}   FileNotFound   0   0   0   {ctf_entry_str}   "
                f"{defocus_u:.6f}   {defocus_v:.6f}   {astigmatism:.6f}   {defocus_angle:.6f}   0   "
                f"10.000000   0.010000   {x_tilt:.6f}   {y_tilt:.6f}   {z_rot:.6f}   {x_shift_angst:.6f}   {y_shift_angst:.6f}   {ctf_scalefactor:.6f}\n"
            )
    
    print(f"Created tilt series star file: {tilt_series_star_path}")
    return tilt_series_star_path

def print_banner():
    """Print a banner with version information."""
    print(f"""
╔══════════════════════════════════════════════════╗
║                                                  ║
║             AreTomo3 to RELION v5.0.0            ║
║                                                  ║
║     Convert AreTomo3 output to RELION5 format    ║
║                                                  ║
╚══════════════════════════════════════════════════╝
""")

def main():
    print_banner()
    args = parse_args()
    
    if not os.path.exists(args.aretomo_dir):
        print(f"Error: AreTomo3 directory not found: {args.aretomo_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        session_data = read_session_json(args.aretomo_dir)
        tomo_prefix = get_tomo_prefix(args.aretomo_dir)
        print(f"Processing tomogram: {tomo_prefix}")
        
        # Prefer tilt angles from the order list CSV if available.
        order_list_path = os.path.join(args.aretomo_dir, f"{tomo_prefix}_Imod", f"{tomo_prefix}_order_list.csv")
        if os.path.exists(order_list_path):
            tilt_angles = read_order_list(args.aretomo_dir, tomo_prefix)
            print(f"Using tilt angles from order list ({len(tilt_angles)} entries).")
        else:
            tilt_angles = read_tlt_file(args.aretomo_dir, tomo_prefix)
            print(f"Found {len(tilt_angles)} tilt angles from .tlt file")
        
        xf_data = read_xf_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(xf_data)} transformation matrices")
        
        ctf_data = read_ctf_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(ctf_data)} CTF entries")
        
        metrics_data = read_metrics_csv(args.aretomo_dir)
        aln_data = read_aln_file(args.aretomo_dir, tomo_prefix)
        if aln_data:
            print(f"Found {len(aln_data)} ALN entries")
        
        os.makedirs(args.output_dir, exist_ok=True)
        print("Creating softlinks with .mrcs extension...")
        links_created, vol_file = create_softlinks(args.aretomo_dir, args.output_dir, tomo_prefix)
        
        start_time = datetime.now()
        create_tomogram_star(session_data, args.output_dir, tomo_prefix, args.aretomo_dir, vol_file)
        create_tilt_series_star(session_data, args.output_dir, tomo_prefix, args.aretomo_dir,
                                tilt_angles, xf_data, ctf_data, aln_data, dose_per_tilt=args.dose)
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        print(f"Successfully created RELION5 star files in {args.output_dir}")
        print(f"Processing completed in {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
