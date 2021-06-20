#############
API Reference
#############

.. currentmodule:: skelm

Extreme Learning Machine is a native regression method, that also supports classification with extra features
designed to support classes in targets.

Large ELM refers to a parallel implementation with Dask that takes data
in a bunch of files on disk, and supports extremely large numbers of neurons - at a cost of slightly longer
processing times.

Hidden layer, solvers and utilities are references to internal implementations for those interested in extending
or re-using parts of Scikit-ELM.


Regressor
=========
.. autosummary::
   :toctree: generated/
   :template: class.rst

   ELMRegressor
   LargeELMRegressor

Classifier
==========
.. autosummary::
   :toctree: generated/
   :template: class.rst

   ELMClassifier

Hidden Layer
============
.. autosummary::
   :toctree: generated/
   :template: class.rst

    HiddenLayer

Solvers
=======
.. autosummary::
   :toctree: generated/

    BatchCholeskySolver
    
    
Utilities
=========
.. autosummary::
   :toctree: generated/

    utils