.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

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

*Toolbox is in active development, initial release soon.*

