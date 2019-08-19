Understanding ELM
=================

Yep, this is the **Methodology** section


How It Works?
-------------

General stuff about linear model and its non-linear extension

Liner model learning capacity vs. number of inputs

Obtaining *the best* solution -- easily


Two-Step Solution
-----------------

that particular version that we gonna use

mention that many solutions exist, and they are better for specific purposes -- like iterative update for PageRank web search algorithm working with millions of inputs

motivation and benefits of that solution:
    * Typical use case with more samples than hidden neurons
    * Other cases make no sense, as with same or large amount of hidden neurons we learn perfectly the training set with its noise that goes against the learning theory of ignoring the noise in data
    * Uses very efficient and highly-optimised computate operations (matrix multiplication and Cholesky solver) that give literally 10x-100x speedup vs manual code, optimally use both CPU and memory, and are available everywhere including mobile hardware and mobile GPUs
    * First step covers the most computations but needs to be done only once. Results of first step are simply added for all data chunks making it super easy to add more data to already existin model. We can even forget some data with it! Super simple to accelerate with GPU or run in chunks for Big Data applications.
    * Second step computes an actual solution and is very fast. Only the second step needs to be repeated for model tuning. 
    * Because the second step is run from scratch every time more data is added to model, there is no error accumulation as would otherwise be if we would update the solution itself. Also, it's very easy to fine-tune model again on new data.
    * Easy to include L2 regularization, and change L2 regularization coefficient on a live model. Very easy to test smaller numbers of neurons on already trained model. L1 regularization kinda runs but it still requires the whole dataset for the best results...
    

Batch Processing
----------------

All about Dask and out-of-core processing when part of data remains on SSD (yes, I purposedly ignore hard drives as an outdated technology in active working storage)


Forgetting Mechanism
--------------------

Its the same as adding data but with other sign. Re-calculate solution after that.


Stability & Optimization
------------------------

Basic tricks of linear models - increase neuron numbers until overfit, kill overfitting with L2 regularization. 


Combining With Deep Learning
----------------------------

Like, add deep learning for feature extraction. It workz! It fast! Get all the features as super-fast learning, adding more data, or forgetting something.

You can even export trained DL+ELM model as one large model for inference, and even fine-tune it if you wish so -- alghouth gains on the *test* set are questionable.


(upcoming) Deep ELM
-------------------

There are multi-layer extensions of ELM that actually make sense and work well.



