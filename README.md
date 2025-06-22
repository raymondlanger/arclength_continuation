# Pseudo-Arclength Continuation Example

This repository contains a Python script (`arclength_continuation.py`) that demonstrates how to use Newton’s method in conjunction with a pseudo-arclength continuation technique to find points on the unit circle. This approach also illustrates how pseudo-arclength continuation can handle singular points where naive continuation might fail.

## Usage

Example:
```shell
python arclength_continuation.py -d -0.2 -n 30 -t 1e-9 --no-predictor
```

## Dependencies

- Python 3.7 or later
- NumPy
- Matplotlib

Most other imports (e.g., `dataclasses`, `typing`, `argparse`) come with the 
Python standard library in versions ≥3.7, so no extra installation is needed 
for them.
