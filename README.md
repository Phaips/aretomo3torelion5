# AreTomo3 to RELION5

A Python utility for converting AreTomo3 alignment and reconstruction output to RELION5 star files.


## Features

- Extracts alignment parameters from AreTomo3 output
- Generates `tomograms.star` and `tilt-series.star` files for RELION5
- Handles CTF estimation data
- Preserves transformation matrices from the reconstruction

## Usage

```bash
python aretomo2relion.py /path/to/aretomo_output/ --dose 2
```

### Arguments

- `aretomo_dir`: Directory containing AreTomo3 outputs
- `dose`: Electron dose per tilt in e-/Å². This value is used to calculate cumulative exposure
- `--output_dir`: (Optional) Output directory name for RELION5 star files (default: 'relion_star_files')

## Requirements

- Python 3.6+
- NumPy

## License

[MIT](LICENSE)
