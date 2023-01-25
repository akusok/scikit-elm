.. -*- mode: rst -*-

|ReadTheDocs|_ |Codecov|_

.. |Codecov| image:: https://codecov.io/gh/akusok/scikit-elm/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/akusok/scikit-elm

.. |ReadTheDocs| image:: https://readthedocs.org/projects/scikit-elm/badge/?version=latest
.. _ReadTheDocs: https://scikit-elm.readthedocs.io/en/latest/?badge=latest

Scikit-ELM
============================================================

.. _scikit-learn: https://scikit-learn.org
.. _dask: https://dask.org
.. _plaidml: https://github.com/plaidml/plaidml/blob/master/docs/install.md#macos

**scikit-elm** is a scikit-learn_ compatible Extreme Learning Machine (ELM) regressor/classifier.

It features very high degree of model flexibility: dynamically added classes,
``partial_fit`` without performance penalties, wide data format compatibility,
optimization and parameter selection without full re-training.

Big Data and out-of-core learning support through dask_-powered backend.
GPU acceleration support with NVidia hardware, and on macOS through plaidml_.

