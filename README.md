# AreTomo to RELION

A Python utility for converting AreTomo3 alignment and reconstruction output to RELION5 star files.

## Description

This script converts metadata from AreTomo3 tomographic reconstruction software into the STAR file format used by RELION5 for subtomogram averaging.

## Features

- Extracts alignment parameters from AreTomo3 output
- Generates `tomograms.star` and tilt-series STAR files for RELION5
- Handles CTF estimation data
- Preserves transformation matrices from the reconstruction

## Usage

```bash
python aretomo2relion.py /path/to/aretomo/output/ --output_dir relion_star_files
```

### Arguments

- `aretomo_dir`: Directory containing AreTomo3 output
- `--output_dir`: (Optional) Output directory for RELION5 star files (default: 'relion_star_files')

## Requirements

- Python 3.6+
- NumPy

## License

[MIT](LICENSE)