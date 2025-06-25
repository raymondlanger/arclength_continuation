[![DOI](https://zenodo.org/badge/557482325.svg)](https://doi.org/10.5281/zenodo.15733515)
[![pip](https://img.shields.io/pypi/v/arclength_continuation)](https://pypi.org/project/arclength_continuation/)

# Pseudo-Arclength Continuation Example

This repository contains a Python script (`arclength_continuation.py`) that demonstrates how to use Newton's method in conjunction with a pseudo-arclength continuation technique to find points on the unit circle. This approach can handle singular points where naive continuation might fail.

## Installation


`arclength-continuation` can be installed via `pip` or an equivalent via:

```console
pip install arclength-continuation
```

## Usage

```console
arclength_continuation -d -0.2 -n 30 -t 1e-9 --no-predictor
```

