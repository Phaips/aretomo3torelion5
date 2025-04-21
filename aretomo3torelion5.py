#!/usr/bin/env python3
import os
import json
import argparse
import math
import numpy as np
import sys
import glob
import re
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Convert AreTomo3 output to RELION5 star files')
    parser.add_argument('aretomo_dir', type=str, help='Directory containing AreTomo3 output')
    parser.add_argument('--output_dir', type=str, default='relion_star_files', help='Output directory for RELION5 star files')
    parser.add_argument('--dose', type=float, required=True,
                        help='Electron dose per tilt in e-/Å². This value is used to calculate cumulative exposure.')
    parser.add_argument('--include', type=str, nargs='+', default=None, 
                        help='Include only these tomogram prefixes (e.g., Position_1 Position_2)')
    parser.add_argument('--exclude', type=str, nargs='+', default=None,
                        help='Exclude these tomogram prefixes (e.g., Position_3 Position_4)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def find_all_tomo_prefixes(aretomo_dir):
    """
    Find all tomogram prefixes in the directory by looking for .mrc files
    that don't contain 'Vol', 'EVN', 'ODD', or 'CTF' in the filename.
    """
    mrc_files = [
        f for f in os.listdir(aretomo_dir)
        if f.endswith('.mrc') and not any(x in f for x in ['Vol', 'EVN', 'ODD', 'CTF'])
    ]
    
    if not mrc_files:
        raise FileNotFoundError(f"No suitable tomogram MRC files found in {aretomo_dir}")
    
    prefixes = [os.path.splitext(f)[0] for f in mrc_files]
    return sorted(prefixes)

def filter_prefixes(all_prefixes, include=None, exclude=None):
    """
    Filter tomogram prefixes based on include/exclude lists.
    
    Args:
        all_prefixes: List of all detected tomogram prefixes
        include: List of prefixes to include (if None, include all)
        exclude: List of prefixes to exclude (if None, exclude none)
    
    Returns:
        Filtered list of prefixes
    """
    result = list(all_prefixes)
    
    if include is not None:
        result = [p for p in result if any(
            re.match(f"^{pattern.replace('*', '.*')}$", p) 
            for pattern in include
        )]
    
    if exclude is not None:
        result = [p for p in result if not any(
            re.match(f"^{pattern.replace('*', '.*')}$", p) 
            for pattern in exclude
        )]
        
    return result

def read_session_json(aretomo_dir, tomo_prefix):
    """Read AreTomo3_Session.json file."""
    json_file = os.path.join(aretomo_dir, 'AreTomo3_Session.json')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Session JSON file not found: {json_file}")
    with open(json_file, 'r') as f:
        session_data = json.load(f)
    return session_data

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
    Read defocus1/defocus2 (in Å) from the AreTomo3 _CTF.txt
    and convert them to microns, along with the astig angle.
    """
    ctf_file = os.path.join(aretomo_dir, f"{tomo_prefix}_CTF.txt")
    if not os.path.exists(ctf_file):
        raise FileNotFoundError(f"CTF file not found: {ctf_file}")
    ctf_data = []
    with open(ctf_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            # parts[1]=defocus1 [Å], parts[2]=defocus2 [Å], parts[3]=astig-angle
            defocusU = float(parts[1])  
            defocusV = float(parts[2])
            ctf_data.append({
                'frame':   int(parts[0]),
                'defocus_u': defocusU,
                'defocus_v': defocusV,
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
     Following:
     https://github.com/scipion-em/scipion-em-reliontomo/blob/8d538ca04f8d02d7a9978e594876bbf7617dcf5f/reliontomo/convert/convert50_tomo.py
     and
     https://github.com/teamtomo/yet-another-imod-wrapper/blob/main/src/yet_another_imod_wrapper/utils/xf.py#L52

    """
    A11, A12, A21, A22, DX, DY = xf_row
    # Build the full 3x3 transformation matrix
    T = np.array([[A11, A12, DX],
                  [A21, A22, DY],
                  [0.0, 0.0, 1.0]])
    # Note: np.arctan2(A12, A11) gives the proper sign.
    z_rot = np.degrees(np.arctan2(A12, A11))
    
    # Invert the full transformation matrix to get the corrected translation
    T_inv = np.linalg.inv(T)
    # The translation (shift) is given by the third column of the inverted matrix
    x_shift_angst = T_inv[0, 2] * pixel_size
    y_shift_angst = T_inv[1, 2] * pixel_size

    x_tilt = 0.0  # by default
    y_tilt = 0.0  # we populate this later
    
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

def create_dummy_edf_file(output_dir, tomo_prefix):
    """
    Create a dummy ETOMO directive (.edf) file for the tomogram.
    """
    edf_file_path = os.path.join(output_dir, f"{tomo_prefix}.edf")
    
    # Create an empty file or with minimal content
    with open(edf_file_path, 'w') as f:
        f.write("# Dummy ETOMO directive file for RELION5\n")
        f.write(f"# Generated for tomogram: {tomo_prefix}\n")
        f.write("# This is a placeholder file\n")
    
    print(f"Created dummy ETOMO directive file: {edf_file_path}")
    return edf_file_path

def collect_tomogram_data(aretomo_dir, tomo_prefix, dose_per_tilt):
    """Process a single tomogram and return its data"""
    try:
        session_data = read_session_json(aretomo_dir, tomo_prefix)
        
        tilt_angles = read_tlt_file(aretomo_dir, tomo_prefix)
        print(f"Found {len(tilt_angles)} tilt angles")

        xf_data = read_xf_file(aretomo_dir, tomo_prefix)
        print(f"Found {len(xf_data)} transformation matrices")

        ctf_data = read_ctf_file(aretomo_dir, tomo_prefix)
        print(f"Found {len(ctf_data)} CTF entries")

        aln_data = read_aln_file(aretomo_dir, tomo_prefix)
        if aln_data:
            print(f"Found {len(aln_data)} ALN entries")
            
        # Read CTF text for handedness
        ctf_txt_data = read_ctf_txt(aretomo_dir, tomo_prefix)
        hand = ctf_txt_data[0]['dfHand'] if ctf_txt_data else 0.0
            
        # Read dimensions
        vol_size_x, vol_size_y, _ = read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix)
        vol_size_z = read_volZ_from_json(session_data)
        
        # Get voltage, pixel size, etc.
        voltage = session_data['parameters']['kV']
        cs = session_data['parameters']['Cs']
        amp_contrast = session_data['parameters']['AmpContrast']
        pixel_size = session_data['parameters']['PixSize']
        bin_factor = session_data['parameters'].get('AtBin', [1])[0]
        
        # Get tilt axis from ALN file
        try:
            tilt_axis = float(aln_data[0][1])
        except Exception:
            tilt_axis = 0.0  # Default if not found
        
        # Assume vol file path
        vol_file = os.path.join(aretomo_dir, f"{tomo_prefix}_Vol.mrc")
        
        # Handle tilt series data
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
            
        # Create a tilt series data array
        tilt_series_data = []
        
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

            # For typical single-tilt geometry, you might scale some factors with cos(tilt)
            ctf_scalefactor = math.cos(math.radians(tilt_angle))
            
            tilt_series_data.append({
                'index': i,
                'tilt_angle': tilt_angle,
                'pre_exposure': pre_exposure,
                'defocus_u': defocus_u,
                'defocus_v': defocus_v,
                'astigmatism': astigmatism,
                'defocus_angle': defocus_angle,
                'x_tilt': x_tilt,
                'y_tilt': y_tilt,
                'z_rot': z_rot,
                'x_shift_angst': x_shift_angst,
                'y_shift_angst': y_shift_angst,
                'ctf_scalefactor': ctf_scalefactor
            })
        
        return {
            'prefix': tomo_prefix,
            'voltage': voltage,
            'cs': cs,
            'amp_contrast': amp_contrast,
            'pixel_size': pixel_size,
            'hand': hand,
            'bin_factor': bin_factor,
            'vol_size_x': vol_size_x,
            'vol_size_y': vol_size_y,
            'vol_size_z': vol_size_z,
            'tilt_axis': tilt_axis,
            'vol_file': vol_file,
            'tilt_series_data': tilt_series_data
        }
    
    except Exception as e:
        print(f"Error processing tomogram {tomo_prefix}: {str(e)}")
        return None

def create_combined_tomogram_star(tomogram_data_list, output_dir):
    """
    Create a combined tomograms.star file for all tomograms.
    """
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
        f.write("_rlnTomoReconstructedTomogram #15\n")

        for data in tomogram_data_list:
            if data is None:
                continue
                
            tomo_prefix = data['prefix']
            optics_group = "optics1"  # Default optics group
            
            # Relative paths for star and etomo directive files
            tilt_series_star_rel = f"{tomo_prefix}.star"
            etomo_directive_rel = f"{tomo_prefix}.edf"

            f.write(
                f"{tomo_prefix}   {data['voltage']:.6f}   {data['cs']:.6f}   {data['amp_contrast']:.6f}   "
                f"{data['pixel_size']:.6f}   {data['hand']:.6f}   {optics_group}   {data['pixel_size']:.6f}   "
                f"{tilt_series_star_rel}   {etomo_directive_rel}   {data['bin_factor']:.6f}   "
                f"{data['vol_size_x']}   {data['vol_size_y']}   {data['vol_size_z']}   {data['vol_file']}\n"
            )
            
    print(f"Created combined tomogram star file: {tomogram_star_path}")
    return tomogram_star_path

def create_individual_tilt_series_star(tomogram_data, output_dir):
    """
    Create an individual tilt-series star file for a single tomogram.
    """
    tomo_prefix = tomogram_data['prefix']
    pixel_size = tomogram_data['pixel_size']
    tilt_axis = tomogram_data['tilt_axis']
    
    abs_output_dir = os.path.abspath(output_dir)
    even_mrcs_file = os.path.join(abs_output_dir, f"{tomo_prefix}_EVN.mrcs")
    odd_mrcs_file = os.path.join(abs_output_dir, f"{tomo_prefix}_ODD.mrcs")
    aligned_mrcs_file = os.path.join(abs_output_dir, f"{tomo_prefix}.mrcs")
    ctf_mrcs_file = os.path.join(abs_output_dir, f"{tomo_prefix}_CTF.mrcs")
    
    tilt_series_star_path = os.path.join(output_dir, f"{tomo_prefix}.star")
    
    with open(tilt_series_star_path, 'w') as f:
        f.write("# Generated by AreTomo3 to RELION5-multi converter\n")
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
        
        # For each tilt image in this tomogram:
        for entry in tomogram_data['tilt_series_data']:
            # Extract parameters from the tilt series data structure
            i = entry['index']
            tilt_angle = entry['tilt_angle']
            pre_exposure = entry['pre_exposure']
            defocus_u = entry['defocus_u']
            defocus_v = entry['defocus_v']
            astigmatism = entry['astigmatism']
            defocus_angle = entry['defocus_angle']
            x_tilt = entry['x_tilt']
            y_tilt = entry['y_tilt']
            z_rot = entry['z_rot']
            x_shift_angst = entry['x_shift_angst']
            y_shift_angst = entry['y_shift_angst']
            ctf_scalefactor = entry['ctf_scalefactor']
            
            even_entry = f"{(i+1):06d}@{even_mrcs_file}"
            odd_entry = f"{(i+1):06d}@{odd_mrcs_file}"
            aligned_entry = f"{i+1}@{aligned_mrcs_file}"
            ctf_entry_str = f"{i+1}@{ctf_mrcs_file}"
            
            # Write the row
            f.write(
                f"FileNotFound 1 {tilt_angle:.6f} {tilt_axis:.6f} {pre_exposure:.6f} 0.000000 FileNotFound "
                f"{even_entry} {odd_entry} {aligned_entry} FileNotFound 0 0 0 {ctf_entry_str} "
                f"{defocus_u:.6f} {defocus_v:.6f} {astigmatism:.6f} {defocus_angle:.6f} 0 "
                f"10.000000 0.010000 {x_tilt:.6f} {y_tilt:.6f} {z_rot:.6f} {x_shift_angst:.6f} {y_shift_angst:.6f} {ctf_scalefactor:.6f}\n"
            )
    
    print(f"Created individual tilt series star file: {tilt_series_star_path}")
    return tilt_series_star_path

def main():
    args = parse_args()
    
    if not os.path.exists(args.aretomo_dir):
        print(f"Error: AreTomo3 directory not found: {args.aretomo_dir}", file=sys.stderr)
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find and filter tomogram prefixes
    all_prefixes = find_all_tomo_prefixes(args.aretomo_dir)
    tomo_prefixes = filter_prefixes(all_prefixes, args.include, args.exclude)
    
    if not tomo_prefixes:
        print("No tomogram prefixes found matching the criteria.", file=sys.stderr)
        sys.exit(1)
    
    print("Processing tomograms:")
    for prefix in tomo_prefixes:
        print(f"  {prefix}")
    
    # Process each tomogram folder and collect data into a list
    tomogram_data_list = []
    for prefix in tomo_prefixes:
        # Create softlinks for the current tomogram
        try:
            links, _ = create_softlinks(args.aretomo_dir, args.output_dir, prefix)
        except Exception as e:
            print(f"Warning: Could not create softlinks for {prefix}: {e}")
            continue
        
        data = collect_tomogram_data(args.aretomo_dir, prefix, args.dose)
        if data is None:
            print(f"Skipping tomogram {prefix} due to errors.")
            continue
        tomogram_data_list.append(data)
        
        # Create an individual tilt-series star file for this tomogram
        create_individual_tilt_series_star(data, args.output_dir)

        create_dummy_edf_file(args.output_dir, prefix)
    
    # Create a combined tomogram.star file (including all tomograms)
    if tomogram_data_list:
        create_combined_tomogram_star(tomogram_data_list, args.output_dir)
    else:
        print("No valid tomogram data was collected.", file=sys.stderr)
    
    print("Processing completed.")

if __name__ == "__main__":
    main()
