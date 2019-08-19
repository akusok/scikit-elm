.. title:: Advanced Techniques : contents

.. _techniques:

=================================
Advanced data analysis techniques
=================================
 

Some functionality relies on programmer to understand the concepts, and cannot be implemented as a one-click button. Here are the most important methods.

Big Data Analysis with Dask
----------------------------------

Practical problems often use huge datasets that would not fit in memory of a single computer, or build huge models that would take impractically long on a single CPU. This is known as *Big Data*, a general term referring to analytical methods running at such a large scale that standard applications and hardware become insufficient.

Scikit-ELM employs :mod:`Dask` library to tackle Big Data. It can store and use datasets in parallel from multiple files. Internal processing is done in data batches small enough to fit into memory, and intermediate results are dumped onto hard drive. That way Scikit-ELM can build models larger than the computer memory. Batch processing can also be run in parallel on many machines, but one has to be causious as slow data exchange through the network may undo any performance improvements. GPU acceleration is an alternative way of training large models faster on one machine -- however it requires rather large models (>1000 neurons) and a powerful GPU for significant time savings.

blah-blah-blah



Fine-Tuning the Solution
------------------------

basic regularisation and mss


Save&Load with Checkpoints
--------------------------

update old model with new data


Export as Deep Learning Model
-----------------------------

... then run on mobile phone -- cool!


