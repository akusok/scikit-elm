# Scikit-ELM

![Codecov](https://codecov.io/gh/akusok/scikit-elm/branch/master/graph/badge.svg)
![ReadTheDocs](https://readthedocs.org/projects/scikit-elm/badge/?version=latest)

[scikit-learn](https://scikit-learn.org) | [dask](https://dask.org) | [plaidml](https://github.com/plaidml/plaidml/blob/master/docs/install.md#macos)

**Scikit-elm** is a scikit-learn compatible Extreme Learning Machine (ELM) regressor/classifier.

It features a very high degree of model flexibility: dynamically added classes,
`partial_fit` without performance penalties, wide data format compatibility,
optimization and parameter selection without full re-training.

Big Data and out-of-core learning support through a dask-powered backend.
GPU acceleration support with NVidia hardware, and on macOS through plaidml.


## Testing

`uv run pytest skelm`
