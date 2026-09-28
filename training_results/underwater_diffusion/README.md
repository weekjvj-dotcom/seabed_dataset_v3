# Underwater diffusion training results

This directory groups the selected model checkpoint, comparison outputs, and diagnostic artifacts for the underwater diffusion experiment.

## Contents

- `gen`: PyTorch checkpoint archive. Its internal archive prefix is `I5000_E42_gen`.
- `insert_gen.py`: prints the checkpoint state-dictionary keys and tensor shapes. This repository copy resolves `gen` beside the script; the source file outside this repository was left unchanged.
- `comparisons_10/`: ten comparison PNGs and `metrics.csv` for scenes 0451–0454.
- `comparisons_best_random_v2/`: ten comparison PNGs and `summary.json` for scenes 0081–0090.
- `diagnostic_report/`: extracted diagnostic images, logs, and report metadata.
- `diagnostic_report.tar.gz`: compressed archive supplied alongside the extracted diagnostic directory; the archive contains the same 96 report files.

The diagnostic folder's macOS `.DS_Store` metadata file is excluded by the repository's existing ignore rule.
