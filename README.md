# AreTomo3 to RELION5

A Python utility for converting [AreTomo3](https://github.com/czimaginginstitute/AreTomo3) alignment and reconstruction output to [RELION5](https://github.com/3dem/relion/tree/ver5.0) star files.

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
By default the `--output_dir` is called relion_star_files. The RELION5 project needs to be initialized with respect to this folder (one above if `--output_dir` is relative).


### Arguments

- `aretomo_dir`: Directory containing AreTomo3 outputs
- `dose`: Electron dose per tilt in e-/Å². This value is used to calculate cumulative exposure
- `--output_dir`: (Optional) Output directory name for RELION5 star files (default: 'relion_star_files')
- `--include`: (Optional) One or more tomogram prefixes to include (e.g., `Position_1` `Position_2`). If not provided, all detected prefixes are processed.
- `--exclude`: (Optional) One or more tomogram prefixes to exclude (e.g., `Position_3` `Position_4`).

Assuming your AreTomo3 output folder should look like this:

```
aretomo3/
├─ AreTomo3_Session.json
├─ TiltSeries_Metrics.csv
├─ TiltSeries_TimeStamp.csv
├─ Position_1.aln
├─ Position_1.mrc
├─ Position_1_ODD.mrc
├─ Position_1_EVN.mrc
├─ Position_1_Vol.mrc
├─ Position_1_CTF.txt
├─ Position_1_TLT.txt
├─ Position_1_CTF_Imod.txt
├─ Position_1_Imod/
│   ├─ Position_1_st.tlt
│   ├─ Position_1_st.xf
│   └─ Position_1_order_list.csv
├─ Position_2.aln
├─ Position_2.mrc
...
```

## License

[MIT](LICENSE)
