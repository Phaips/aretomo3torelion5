#!/usr/bin/env python3
import os
import json
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
    parser.add_argument('--dose', type=float, required=True,
                        help='Electron dose per tilt in e-/Å². This value is used to calculate cumulative exposure.')
    return parser.parse_args()

def read_session_json(aretomo_dir):
    """Read AreTomo3_Session.json file."""
    json_file = os.path.join(aretomo_dir, 'AreTomo3_Session.json')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Session JSON file not found: {json_file}")
    with open(json_file, 'r') as f:
        session_data = json.load(f)
    return session_data

def get_tomo_prefix(aretomo_dir):
    """
    Extract tomogram prefix from directory contents by looking for a .mrc file
    that doesn't contain 'Vol', 'EVN', 'ODD', or 'CTF' in the filename.
    """
    mrc_files = [
        f for f in os.listdir(aretomo_dir)
        if f.endswith('.mrc') and not any(x in f for x in ['Vol', 'EVN', 'ODD', 'CTF'])
    ]
    if not mrc_files:
        raise FileNotFoundError(f"No suitable tomogram MRC files found in {aretomo_dir}")
    return os.path.splitext(mrc_files[0])[0]

def read_tlt_file(aretomo_dir, tomo_prefix):
    """Read tilt angles from the .tlt file."""
    tlt_file = os.path.join(aretomo_dir, f"{tomo_prefix}_Imod", f"{tomo_prefix}_st.tlt")
    if not os.path.exists(tlt_file):
        raise FileNotFoundError(f"Tilt file not found: {tlt_file}")
    with open(tlt_file, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def read_xf_file(aretomo_dir, tomo_prefix):
    """Read transformation matrices from the .xf file (IMOD format).

    Returns:
      A list of lists, each with 6 floats: [A11, A12, A21, A22, DX, DY].
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
    """
    Read CTF defocus information from the .txt produced by AreTomo3 (e.g. Position_1_CTF.txt).
    Adjust parsing logic as needed if your file differs in format.
    """
    ctf_file = os.path.join(aretomo_dir, f"{tomo_prefix}_CTF.txt")
    if not os.path.exists(ctf_file):
        raise FileNotFoundError(f"CTF file not found: {ctf_file}")
    ctf_data = []
    with open(ctf_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                # Expecting something like: frame defU defV astigAngle ...
                if len(parts) >= 8:
                    ctf_data.append({
                        'frame': int(parts[0]),
                        'tilt_angle': None,  # We'll match with tilt angles if needed
                        'defocus_u': float(parts[1]),
                        'defocus_v': float(parts[2]),
                        'astigmatism_angle': float(parts[3])
                    })
    return ctf_data

def read_aln_file(aretomo_dir, tomo_prefix):
    """Read alignment info from the .aln file if needed."""
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
        with open(aln_file, 'r') as f2:
            for i, line in enumerate(f2):
                if i < 10:
                    print(f"Line {i+1}: {repr(line)}")
        raise ValueError("No alignment data found in ALN file.")
    return aln_data

def read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix):
    """
    Read volume dimensions from a line like "# RawSize = 4096 4096" in the .aln file.
    Returns (vol_x, vol_y, None).
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
    """Read the Z dimension (VolZ) from AreTomo3 session JSON."""
    vol_z = session_data['parameters'].get('VolZ')
    if vol_z is None:
        raise ValueError("VolZ not found in session metadata.")
    return vol_z

def read_ctf_txt(aretomo_dir, tomo_prefix):
    """
    (Optional) read additional CTF info (including 'dfHand') from the same _CTF.txt file,
    if you need to store e.g. handedness. Adjust logic as needed.
    """
    ctf_txt_file = os.path.join(aretomo_dir, f"{tomo_prefix}_CTF.txt")
    if not os.path.exists(ctf_txt_file):
        return []
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
                    'dfHand': float(parts[7])  # handedness
                }
                ctf_txt_data.append(entry)
    return ctf_txt_data

def compute_tilt_alignment(xf_row, pixel_size):
    """
    Compute RELION tilt parameters from an IMOD .xf transformation matrix.
    following:
    https://github.com/scipion-em/scipion-em-reliontomo/blob/8d538ca04f8d02d7a9978e594876bbf7617dcf5f/reliontomo/convert/convert50_tomo.py#L363
    Shout-out to Scipion developers

    Returns (x_tilt, y_tilt, z_rot, x_shift_angst, y_shift_angst).
    Typically, x_tilt=0, y_tilt=actual stage tilt, z_rot=rotation.
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
    y_tilt = 0.0
    return x_tilt, y_tilt, z_rot, x_shift_angst, y_shift_angst

def read_acquisition_order_csv(aretomo_dir, tomo_prefix):
    """
    Read the tilt acquisition order from e.g. "Position_1_order_list.csv",
    which typically has 2 columns: ImageNumber, TiltAngle.
    """
    csv_file = os.path.join(aretomo_dir, f"{tomo_prefix}_Imod", f"{tomo_prefix}_order_list.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Acquisition order CSV file not found: {csv_file}")

    acquisition_data = []
    with open(csv_file, 'r') as f:
        # Try skipping a header line:
        header = f.readline().strip()
        if not ("ImageNumber" in header and "TiltAngle" in header):
            # If no real header, rewind
            f.seek(0)

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    image_num = int(parts[0].strip())
                    tilt_angle = float(parts[1].strip())
                    acquisition_data.append((image_num, tilt_angle))
                except ValueError:
                    continue

    if not acquisition_data:
        raise ValueError(f"No valid data found in acquisition order CSV: {csv_file}")
    return acquisition_data

def calculate_cumulative_exposure(tilt_angles, acquisition_order, dose_per_tilt):
    """
    Calculate cumulative exposure for each tilt based on the acquisition order
    (i.e. the actual order in which images were acquired).

    Args:
      tilt_angles: Sorted tilt angles from the .tlt file (lowest to highest, typically).
      acquisition_order: List of (image_num, tilt_angle) from the CSV in acquisition order.
      dose_per_tilt: The user-provided per-tilt dose (e-/Å²).

    Returns:
      A list of exposures, matching length/order of tilt_angles.
      exposures[i] = pre-exposure for tilt_angles[i].
    """
    if dose_per_tilt <= 0:
        return [0.0] * len(tilt_angles)

    # Map tilt angles in the order of acquisition to cumulative exposure
    angle_to_exposure = {}
    current_exposure = 0.0

    # For each image in the real acquisition order, assign the current exposure, then increment
    for _, tilt_angle in acquisition_order:
        # Round tilt_angle to 2 decimals to match .tlt rounding
        rangle = round(tilt_angle, 2)
        angle_to_exposure[rangle] = current_exposure
        current_exposure += dose_per_tilt

    # Now we build a final array, in the sorted order of tilt_angles (as read from .tlt)
    exposures = []
    for angle in tilt_angles:
        rangle = round(angle, 2)
        if rangle in angle_to_exposure:
            exposures.append(angle_to_exposure[rangle])
        else:
            # If there's not an exact match, pick the closest tilt in angle_to_exposure
            if len(angle_to_exposure) == 0:
                # no data at all
                exposures.append(0.0)
                continue
            closest = min(angle_to_exposure.keys(), key=lambda x: abs(x - rangle))
            # If difference is too large, warn
            if abs(closest - rangle) > 0.1:
                print(f"Warning: No matching acquisition angle found for {angle:.2f}, assigning 0.0 exposure")
                exposures.append(0.0)
            else:
                exposures.append(angle_to_exposure[closest])

    return exposures

def create_softlinks(aretomo_dir, output_dir, tomo_prefix):
    """
    Create softlinks with .mrcs extension for .mrc files. This helps RELION treat them as stacks.
    """
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
    for src, dst in [
        (mrc_file, mrc_link),
        (evn_file, evn_link),
        (odd_file, odd_link),
        (ctf_file, ctf_link)
    ]:
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
    """
    Create the tomogram.star file for RELION5 using metadata from AreTomo3.
    This file references the tilt series .star and other parameters needed by RELION.
    """
    os.makedirs(output_dir, exist_ok=True)

    voltage      = session_data['parameters']['kV']
    cs           = session_data['parameters']['Cs']
    amp_contrast = session_data['parameters']['AmpContrast']
    pixel_size   = session_data['parameters']['PixSize']
    optics_group = "optics1"
    bin_factor   = session_data['parameters'].get('AtBin', [1])[0]

    vol_size_x, vol_size_y, _ = read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix)
    vol_size_z = read_volZ_from_json(session_data)

    # Read CTF text for handedness if needed
    ctf_txt_data = read_ctf_txt(aretomo_dir, tomo_prefix)
    hand = ctf_txt_data[0]['dfHand'] if ctf_txt_data else 0.0

    abs_output_dir = os.path.abspath(output_dir)
    tilt_series_star = os.path.join(abs_output_dir, f"{tomo_prefix}.star")
    etomo_directive  = os.path.join(abs_output_dir, f"{tomo_prefix}.edf")
    reconstructed_tomo = vol_file
    tomogram_star_path = os.path.join(output_dir, 'tomograms.star')

    with open(tomogram_star_path, 'w') as f:
        f.write("# version 50001\n\n")
        f.write("data_global\n\n")
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
        f.write("_rlnTomoReconstructedTomogram #15\n\n")

        f.write(
            f"{tomo_prefix} "
            f"{voltage:.6f} {cs:.6f} {amp_contrast:.6f} {pixel_size:.6f} {hand:.6f} {optics_group} "
            f"{pixel_size:.6f} {tilt_series_star} {etomo_directive} {bin_factor:.6f} "
            f"{vol_size_x} {vol_size_y} {vol_size_z} {reconstructed_tomo}\n"
        )

    print(f"Created tomogram star file: {tomogram_star_path}")
    return tomogram_star_path

def create_tilt_series_star(
    session_data,
    output_dir,
    tomo_prefix,
    aretomo_dir,
    tilt_angles,
    xf_data,
    ctf_data,
    aln_data,
    dose_per_tilt
):
    """
    Create the tilt-series star file using AreTomo3 metadata, referencing the .mrcs softlinks.
    Includes the per-tilt cumulative exposure in _rlnMicrographPreExposure.
    """
    os.makedirs(output_dir, exist_ok=True)
    pixel_size = session_data['parameters']['PixSize']

    if "TiltAxis" not in session_data['parameters']:
        raise ValueError("TiltAxis parameter not found in session metadata.")
    tilt_axis = session_data['parameters']["TiltAxis"][0]

    abs_output_dir = os.path.abspath(output_dir)
    even_mrcs_file    = os.path.join(abs_output_dir, f"{tomo_prefix}_EVN.mrcs")
    odd_mrcs_file     = os.path.join(abs_output_dir, f"{tomo_prefix}_ODD.mrcs")
    aligned_mrcs_file = os.path.join(abs_output_dir, f"{tomo_prefix}.mrcs")
    ctf_mrcs_file     = os.path.join(abs_output_dir, f"{tomo_prefix}_CTF.mrcs")

    # If the CTF entries do not have tilt_angle set, match them by frame index
    if ctf_data and ctf_data[0].get('tilt_angle') is None:
        for entry in ctf_data:
            frame_idx = entry['frame'] - 1
            if 0 <= frame_idx < len(tilt_angles):
                entry['tilt_angle'] = tilt_angles[frame_idx]

    # Attempt to read real acquisition order from CSV
    try:
        acquisition_order = read_acquisition_order_csv(aretomo_dir, tomo_prefix)
        print(f"Found acquisition order data with {len(acquisition_order)} entries.")
        exposures = calculate_cumulative_exposure(tilt_angles, acquisition_order, dose_per_tilt)
    except FileNotFoundError:
        print("Warning: Acquisition order CSV file not found. Using default incremental exposure.")
        # Fallback: just do an incremental from 0, 1*dose, 2*dose, ...
        exposures = [i * dose_per_tilt for i in range(len(tilt_angles))]

    tilt_series_star_path = os.path.join(output_dir, f"{tomo_prefix}.star")
    with open(tilt_series_star_path, 'w') as f:
        f.write("# Generated by AreTomo3 to RELION5 converter\n")
        f.write("# Relion star file version 50001\n\n")
        f.write(f"data_{tomo_prefix}\n\n")

        # Define the columns in the standard RELION tilt-series star format
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
        f.write("_rlnCtfScalefactor\n\n")

        # Loop through the tilt angles, in sorted order (the .tlt order)
        for i, tilt_angle in enumerate(tilt_angles):
            # pre-exposure from the computed exposures array
            pre_exposure = exposures[i]

            # Attempt to match a defocus from ctf_data by tilt angle
            defocus_u = 0.0
            defocus_v = 0.0
            astigmatism_angle = 0.0
            for ctf_entry in ctf_data:
                if ctf_entry.get('tilt_angle') is not None:
                    if abs(ctf_entry['tilt_angle'] - tilt_angle) < 0.1:
                        defocus_u = ctf_entry['defocus_u']
                        defocus_v = ctf_entry['defocus_v']
                        astigmatism_angle = ctf_entry['astigmatism_angle']
                        break
                else:
                    # if no tilt_angle field, attempt frame-based
                    if ctf_entry['frame'] == (i + 1):
                        defocus_u = ctf_entry['defocus_u']
                        defocus_v = ctf_entry['defocus_v']
                        astigmatism_angle = ctf_entry['astigmatism_angle']
                        break

            astigmatism = abs(defocus_u - defocus_v)
            defocus_angle = astigmatism_angle

            x_tilt, _, z_rot, x_shift_angst, y_shift_angst = compute_tilt_alignment(xf_data[i], pixel_size)
            y_tilt = tilt_angle

            # Format the MicrographName fields
            even_entry    = f"{(i+1):06d}@{even_mrcs_file}"
            odd_entry     = f"{(i+1):06d}@{odd_mrcs_file}"
            aligned_entry = f"{i+1}@{aligned_mrcs_file}"
            ctf_entry_str = f"{i+1}@{ctf_mrcs_file}"

            # For typical single-tilt geometry, you might scale some factors with cos(tilt)
            ctf_scalefactor = math.cos(math.radians(tilt_angle))

            f.write(
                f"FileNotFound 1 {tilt_angle:.6f} {tilt_axis:.6f} {pre_exposure:.6f} 0.000000 FileNotFound "
                f"{even_entry} {odd_entry} {aligned_entry} FileNotFound 0 0 0 {ctf_entry_str} "
                f"{defocus_u:.6f} {defocus_v:.6f} {astigmatism:.6f} {defocus_angle:.6f} 0 "
                f"10.000000 0.010000 {x_tilt:.6f} {y_tilt:.6f} {z_rot:.6f} {x_shift_angst:.6f} {y_shift_angst:.6f} {ctf_scalefactor:.6f}\n"
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

    # Check for required directory
    if not os.path.exists(args.aretomo_dir):
        print(f"Error: AreTomo3 directory not found: {args.aretomo_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        session_data = read_session_json(args.aretomo_dir)
        tomo_prefix = get_tomo_prefix(args.aretomo_dir)
        print(f"Processing tomogram: {tomo_prefix}")

        tilt_angles = read_tlt_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(tilt_angles)} tilt angles")

        xf_data = read_xf_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(xf_data)} transformation matrices")

        ctf_data = read_ctf_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(ctf_data)} CTF entries")

        aln_data = read_aln_file(args.aretomo_dir, tomo_prefix)
        if aln_data:
            print(f"Found {len(aln_data)} ALN entries")

        os.makedirs(args.output_dir, exist_ok=True)
        print("Creating softlinks with .mrcs extension...")
        links_created, vol_file = create_softlinks(args.aretomo_dir, args.output_dir, tomo_prefix)

        start_time = datetime.now()

        # Create the "tomograms.star"
        create_tomogram_star(session_data, args.output_dir, tomo_prefix, args.aretomo_dir, vol_file)

        # Create the tilt-series star file
        create_tilt_series_star(
            session_data,
            args.output_dir,
            tomo_prefix,
            args.aretomo_dir,
            tilt_angles,
            xf_data,
            ctf_data,
            aln_data,
            args.dose
        )

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
