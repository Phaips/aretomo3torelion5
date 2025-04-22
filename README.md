# AreTomo3 to RELION5

A Python utility for converting AreTomo3 alignment and reconstruction output to RELION5 star files.

## Features

- Extracts alignment parameters from AreTomo3 output
- Handles CTF estimation data
- Preserves transformation matrices from the reconstruction
- Automatically detects and processes multiple tomogram outputs (e.g., `Position_1`, `Position_2`, etc.) with exclude and include options
- Generates `tomograms.star` and `tilt-series.star` files for RELION5

## Usage

```bash
 aretomo3torelion5.py /path/to/aretomo_output/ --dose 2 # --output_dir relion_star_files --include Position_1 Position_2 or e.g. --exclude Position_3 Position_4
```

### Arguments

- `aretomo_dir`: Directory containing AreTomo3 outputs
- `dose`: Electron dose per tilt in e-/Å². This value is used to calculate cumulative exposure
- `--output_dir`: (Optional) Output directory name for RELION5 star files (default: 'relion_star_files')
- `--include`: (Optional) One or more tomogram prefixes to include (e.g., `Position_1` `Position_2`). If not provided, all detected prefixes are processed.
- `--exclude`: (Optional) One or more tomogram prefixes to exclude (e.g., `Position_3` `Position_4`).

## License

[MIT](LICENSE)
