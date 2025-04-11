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
    return parser.parse_args()

def read_session_json(aretomo_dir):
    """Read AreTomo3 Session.json file"""
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

def read_tlt_file(aretomo_dir, tomo_prefix):
    """Read tilt angles from the .tlt file."""
    tlt_file = os.path.join(aretomo_dir, f"{tomo_prefix}_Imod", f"{tomo_prefix}_st.tlt")
    if not os.path.exists(tlt_file):
        raise FileNotFoundError(f"Tilt file not found: {tlt_file}")
    with open(tlt_file, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def read_xf_file(aretomo_dir, tomo_prefix):
    """Read transformation matrices from the .xf file.
    
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
                        'tilt_angle': None,  # We'll match with tilt angles later
                        'defocus_u': float(parts[1]),  # defocus1 in _CTF.txt
                        'defocus_v': float(parts[2]),  # defocus2 in _CTF.txt
                        'astigmatism_angle': float(parts[3])  # astig_azimuth in _CTF.txt
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
    header_line = None
    
    with open(aln_file, 'r') as f:
        lines = f.readlines()
        
        # First, find the header line
        for i, line in enumerate(lines):
            if line.strip().startswith("# SEC"):
                header_line = line.strip()
                # Start processing from the next line
                for j in range(i+1, len(lines)):
                    data_line = lines[j].strip()
                    # Check if this is a data line (starts with a number with possible whitespace)
                    if data_line and data_line.lstrip()[0].isdigit():
                        parts = data_line.split()
                        if len(parts) >= 3:  # Ensure at least a few columns are present
                            aln_data.append(parts)
                break
    
    if not aln_data:
        # Add debug information
        print(f"ALN file content preview:")
        with open(aln_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 10:  # Print first 10 lines for debugging
                    print(f"Line {i+1}: {repr(line)}")
        raise ValueError("No alignment data found in ALN file.")
    
    return aln_data

def read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix):
    """
    Read volume dimensions from the header of the .aln file.
    
    Expects a line like "# RawSize = 4096 4096".
    Raises an error if dimensions are not found.
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
    """Read the Z dimension from the session JSON; raises an error if missing."""
    vol_z = session_data['parameters'].get('VolZ')
    if vol_z is None:
        raise ValueError("VolZ not found in session metadata.")
    return vol_z

def read_ctf_txt(aretomo_dir, tomo_prefix):
    """
    Read additional CTF information (including handedness, dfHand) from the _CTF.txt file.
    
    Expects at least 8 columns.
    Raises an error if no valid data is found.
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
    Compute RELION5 tilt parameters from IMOD .xf transformation matrix.
    
    Parameters:
      xf_row (list): [A11, A12, A21, A22, DX, DY] from .xf file
      pixel_size (float): Pixel size in Angstrom
      
    Returns:
      tuple: (x_tilt, y_tilt, z_rot, x_shift_angst, y_shift_angst)
    """
    A11, A12, A21, A22, DX, DY = xf_row
    
    # Create transformation matrix
    tr_matrix = np.array([[A11, A12], [A21, A22]])
    
    # Calculate rotation angle (in degrees)
    z_rot = math.degrees(math.atan2(A21, A11))
    
    # Calculate shifts using inverse transform
    i_tr_matrix = np.linalg.inv(tr_matrix)
    x_shift = i_tr_matrix[0, 0] * DX + i_tr_matrix[0, 1] * DY
    y_shift = i_tr_matrix[1, 0] * DX + i_tr_matrix[1, 1] * DY
    
    # Convert shifts to Angstroms
    x_shift_angst = x_shift * pixel_size
    y_shift_angst = y_shift * pixel_size
    
    # For now, set x_tilt to 0 and y_tilt to tilt_angle (will be filled later)
    x_tilt = 0.0
    y_tilt = 0.0  # This will be replaced with the actual tilt angle
    
    return x_tilt, y_tilt, z_rot, x_shift_angst, y_shift_angst


def create_tomogram_star(session_data, output_dir, tomo_prefix, aretomo_dir):
    """Create the tomogram.star file for RELION5 using metadata from AreTomo3."""
    os.makedirs(output_dir, exist_ok=True)

    voltage      = session_data['parameters']['kV']
    cs           = session_data['parameters']['Cs']
    amp_contrast = session_data['parameters']['AmpContrast']
    pixel_size   = session_data['parameters']['PixSize']
    optics_group = "optics1"
    bin_factor   = session_data['parameters'].get('AtBin', [1])[0]

    # Read X and Y dimensions strictly from the ALN file.
    vol_size_x, vol_size_y, _ = read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix)
    # Read Z dimension from session JSON.
    vol_size_z = read_volZ_from_json(session_data)

    # Read handedness from the _CTF.txt file.
    ctf_txt_data = read_ctf_txt(aretomo_dir, tomo_prefix)
    hand = ctf_txt_data[0]['dfHand']

    # Create absolute paths for all file references
    abs_output_dir = os.path.abspath(output_dir)
    abs_aretomo_dir = os.path.abspath(aretomo_dir)
    
    tilt_series_star = os.path.join(abs_output_dir, f"{tomo_prefix}.star")
    etomo_directive  = os.path.join(abs_output_dir, f"{tomo_prefix}.edf")  # Dummy file.
    reconstructed_tomo = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_Vol.mrc")
    
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

def create_tilt_series_star(session_data, output_dir, tomo_prefix, aretomo_dir, tilt_angles, xf_data, ctf_data, aln_data):
    """Create the tilt-series star file using AreTomo3 metadata."""
    os.makedirs(output_dir, exist_ok=True)
    pixel_size = session_data['parameters']['PixSize']

    # Require that TiltAxis is present in the session JSON.
    if "TiltAxis" not in session_data['parameters']:
        raise ValueError("TiltAxis parameter not found in session metadata.")
    tilt_axis = session_data['parameters']["TiltAxis"][0]

    # Get raw image dimensions (X, Y) from ALN file.
    raw_dims = read_dimensions_from_aln_strict(aretomo_dir, tomo_prefix)
    image_dims = (raw_dims[0], raw_dims[1])
    
    # Create absolute paths for all file references
    abs_aretomo_dir = os.path.abspath(aretomo_dir)
    
    even_mrc_file    = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_EVN.mrc")
    odd_mrc_file     = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_ODD.mrc")
    aligned_mrc_file = os.path.join(abs_aretomo_dir, f"{tomo_prefix}.mrc")
    ctf_mrc_file     = os.path.join(abs_aretomo_dir, f"{tomo_prefix}_CTF.mrc")

    # Map micrograph numbers to tilt angles if needed
    if ctf_data and ctf_data[0]['tilt_angle'] is None:
        # We're using _CTF.txt data which doesn't have tilt angles directly
        # Map the micrograph numbers (starting from 1) to tilt angles array (0-indexed)
        for entry in ctf_data:
            frame_idx = entry['frame'] - 1
            if 0 <= frame_idx < len(tilt_angles):
                entry['tilt_angle'] = tilt_angles[frame_idx]

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
        
        # Loop over tilt images.
        for i, tilt_angle in enumerate(tilt_angles):
            # Obtain defocus parameters from CTF data (matched by tilt angle).
            defocus_u = defocus_v = astigmatism_angle = 0.0
            for ctf_entry in ctf_data:
                # Match by frame number if available, otherwise by tilt angle
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
            
            # Calculate astigmatism as the absolute difference between defocus_u and defocus_v
            astigmatism = abs(defocus_u - defocus_v)
            
            # Defocus angle in RELION is the angle between X and defocus U direction, in degrees
            # The astigmatism_angle from CTF.txt is already in the correct format
            defocus_angle = astigmatism_angle

            # Compute effective alignment parameters.
            x_tilt, _, z_rot, x_shift_angst, y_shift_angst = compute_tilt_alignment(xf_data[i], pixel_size)

            y_tilt = tilt_angle

            even_entry    = f"{i+1:06d}@{even_mrc_file}"
            odd_entry     = f"{i+1:06d}@{odd_mrc_file}"
            aligned_entry = f"{i+1}@{aligned_mrc_file}"
            ctf_entry_str = f"{i+1}@{ctf_mrc_file}"
            
            # _rlnCtfScalefactor is computed as cosine of the tilt angle.
            ctf_scalefactor = math.cos(math.radians(tilt_angle))
            
            # Write the STAR file row.
            # Field order (1 to 28):
            # 1: MicrographMovieName -> "FileNotFound"
            # 2: TiltMovieFrameCount -> 1
            # 3: NominalStageTiltAngle -> tilt_angle
            # 4: NominalTiltAxisAngle -> tilt_axis
            # 5: MicrographPreExposure -> 0.000000
            # 6: NominalDefocus -> 0.000000 (dummy)
            # 7: CtfPowerSpectrum -> "FileNotFound"
            # 8: MicrographNameEven -> even_entry
            # 9: MicrographNameOdd -> odd_entry
            # 10: MicrographName -> aligned_entry
            # 11: MicrographMetadata -> "FileNotFound"
            # 12-14: AccumMotionTotal/Early/Late -> 0
            # 15: CtfImage -> ctf_entry_str
            # 16: DefocusU -> defocus_u
            # 17: DefocusV -> defocus_v
            # 18: CtfAstigmatism -> astigmatism
            # 19: DefocusAngle -> defocus_angle
            # 20: CtfFigureOfMerit -> 0
            # 21: CtfMaxResolution -> 10.000000
            # 22: CtfIceRingDensity -> 0.010000
            # 23: rlnTomoXTilt -> x_tilt
            # 24: rlnTomoYTilt -> y_tilt
            # 25: rlnTomoZRot -> z_rot
            # 26: rlnTomoXShiftAngst -> x_tilt
            # 27: rlnTomoYShiftAngst -> y_tilt
            # 28: rlnCtfScalefactor -> cosine of tilt_angle
            f.write(
                f"FileNotFound   1   {tilt_angle:.6f}   {tilt_axis:.6f}   0.000000   0.000000   FileNotFound   "
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
║             AreTomo to RELION v5.0.0             ║
║                                                  ║
║     Convert AreTomo3 output to RELION5 format    ║
║                                                  ║
╚══════════════════════════════════════════════════╝
""")

def main():
    print_banner()
    args = parse_args()

    # Check for required files
    if not os.path.exists(args.aretomo_dir):
        print(f"Error: AreTomo3 directory not found: {args.aretomo_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Process the data
        session_data = read_session_json(args.aretomo_dir)
        tomo_prefix = get_tomo_prefix(args.aretomo_dir)
        print(f"Processing tomogram: {tomo_prefix}")
        
        tilt_angles = read_tlt_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(tilt_angles)} tilt angles")
        
        xf_data = read_xf_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(xf_data)} transformation matrices")
        
        ctf_data = read_ctf_file(args.aretomo_dir, tomo_prefix)
        print(f"Found {len(ctf_data)} CTF entries")
        
        metrics_data = read_metrics_csv(args.aretomo_dir)  # Optional.
        aln_data = read_aln_file(args.aretomo_dir, tomo_prefix)
        if aln_data:
            print(f"Found {len(aln_data)} ALN entries")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate STAR files
        start_time = datetime.now()
        
        create_tomogram_star(session_data, args.output_dir, tomo_prefix, args.aretomo_dir)
        create_tilt_series_star(session_data, args.output_dir, tomo_prefix, args.aretomo_dir,
                                tilt_angles, xf_data, ctf_data, aln_data)
        
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