import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, check_array

import dask.array as da
import dask.dataframe as dd

from .elm import _BaseELM
from dask.distributed import Client, LocalCluster, wait
from .utils import _is_list_of_strings, _dense, HiddenLayerType, dummy


def _read_numeric_file(fname):
    try:
        return dd.read_parquet(fname)
    except:
        pass

    try:
        return dd.read_csv(fname)
    except:
        pass

    try:
        return np.load(fname)
    except:
        pass

class LargeELMRegressor(_BaseELM, RegressorMixin):
    """ELM Regressor for larger-than-memory problems.

    Uses `Dask <https://dask.org>`_ for batch analysis of data in Parquet files.

    .. attention:: Why do I need Parquet files?

        Parquet files provide necessary information about the data without loading whole file content from
        disk. It makes a tremendous runtime difference compared to simpler `.csv` or `.json` file formats.
        Reading from files saves memory by loading data in small chunks, supporting arbitrary large input files.
        It also solves current memory leaks with Numpy matrix inputs in Dask.

        Any data format can be easily converted to Parquet, see `Analytical methods <techniques.html>`_ section.

        HDF5 is almost as good as Parquet, but performs worse with Dask due to internal data layout.

    .. todo: Write converters.

    .. todo: Memo about number of workers: one is good, several cover disk read latency but need more memory.
        On one machine matrix operators always run in parallel, do not benefit from Dask.

    .. todo: Memory consumption with large number of neurons - 100,000 neurons require 200GB or swap space, with
        read+write reaching 1GB/s. Suggested a fast SSD, or HDD + extra workers to hide swap latency.
        Mention that Dask is not the perfect solution, kept here for future updates. And it actually solves
        stuff larger than memory, albeit at a very high time+swap cost.

    .. todo: Avoid large batch sizes as workers can fail, safe bet is 2000-5000 range.

    .. todo: Fast HtH and in-place Cholesky solver.

    .. todo: Pro tip in documentation: run ELM with dummy 1000 data samples and 1e+9 regularization,
        This will test possible memory issues for workers without wasting your time on computing full HH.

    .. todo: Option to keep full HH permanently somewhere at disk. Saves before the final step,
        avoids failures from memory issues during Cholesky solver.

    .. todo: GPU + batch Cholesky solver, for both ELM and LargeELM.

    Requirements
    ------------
        * Pandas
        * pyarrow
        * python-snappy

    Parameters
    ----------

    batch_size : int
        Batch size used for both data samples and hidden neurons. With batch Cholesky solver, allows for very large
        numbers of hidden neurons of over 100,000; limited only by the computation time and disk swap space.

        .. hint:: Include bias and original features for best performance.

        ELM will include a bias term (1 extra feature), and the original features with `include_original_features=True`.
        For optimal performance, choose `batch_size` to be equal or evenly divide the
        `n_neurons + 1 (bias) + n_inputs (if include_original_features=True)`.

        .. todo:: Exact batch_size vs. GPU performance
    """


    def __del__(self):
        if hasattr(self, 'client_'):
            self.client_.close()
            self.cluster_.close()

    def _setup_dask_client(self):
        self.cluster_ = LocalCluster(
            n_workers=4, threads_per_worker=1,
            local_dir="/Users/akusok/wrkdir/dask-temp",
            memory_limit="8GB"
        )
        self.client_ = Client(self.cluster_)

        W_list = [hl.projection_.components_ for hl in self.hidden_layers_]
        W_dask = [da.from_array(_dense(W), chunks=self.bsize_) for W in W_list]
        self.W_ = self.client_.persist(W_dask)

        def foo():
            import os
            os.environ['OMP_NUM_THREADS'] = '1'
        self.client_.run(foo)

        print("Running on:", self.client_)

        try:
            dashboard = self.client_.scheduler_info()['address'].split(":")
            dashboard[0] = "http"
            dashboard[-1] = str(self.client_.scheduler_info()['services']['dashboard'])
            print("Dashboard at", ":".join(dashboard))
        except:
            pass

    def _project(self, X_dask):
        """Compute hidden layer output with Dask functionality.
        """
        H_list = []
        for hl, W in zip(self.hidden_layers_, self.W_):
            if hl.hidden_layer_ == HiddenLayerType.PAIRWISE:
                H0 = X_dask.map_blocks(
                    pairwise_distances,
                    W,
                    dtype=X_dask.dtype,
                    chunks=(X_dask.chunks[0], (W.shape[0],)),
                    metric=hl.pairwise_metric
                )
            else:
                XW_dask = da.dot(X_dask, W.transpose())
                if hl.ufunc_ is dummy:
                    H0 = XW_dask
                elif hl.ufunc_ is np.tanh:
                    H0 = da.tanh(XW_dask)
                else:
                    H0 = XW_dask.map_blocks(hl.ufunc_)
            H_list.append(H0)

        if self.include_original_features:
            H_list.append(X_dask)
        H_list.append(da.ones((X_dask.shape[0], 1)))

        H_dask = da.concatenate(H_list, axis=1).rechunk(self.bsize_)
        return H_dask

    def _compute(self, X, y, sync_every, HH=None, HY=None):
        """Computing matrices HH and HY, the actually long part.

        .. todo: actually distributed computations that scatter batches of data file names,
            and reduce-sum the HH,HY matrices.
        """

        # processing files
        for i, X_file, y_file in zip(range(len(X)), X, y):
            X_dask = dd.read_parquet(X_file).to_dask_array(lengths=True)
            Y_dask = dd.read_parquet(y_file).to_dask_array(lengths=True)
            H_dask = self._project(X_dask)

            if HH is None:  # first iteration
                HH = da.dot(H_dask.transpose(), H_dask)
                HY = da.dot(H_dask.transpose(), Y_dask)
            else:
                HH += da.dot(H_dask.transpose(), H_dask)
                HY += da.dot(H_dask.transpose(), Y_dask)
                if sync_every is not None and i % sync_every == 0:
                    wait([HH, HY])

            # synchronization
            if sync_every is not None and i % sync_every == 0:
                HH, HY = self.client_.persist([HH, HY])

        # finishing solution
        if sync_every is not None:
            wait([HH, HY])
        return HH, HY

    def _solve(self, HH, HY):
        """Compute output weights from HH and HY using Dask functionality.
        """
        # make HH/HY divisible by chunk size
        n_features, _ = HH.shape
        padding = 0
        if n_features > self.bsize_ and n_features % self.bsize_ > 0:
            print("Adjusting batch size {} to n_features {}".format(self.bsize_, n_features))
            padding = self.bsize_ - (n_features % self.bsize_)
            P01 = da.zeros((n_features, padding))
            P10 = da.zeros((padding, n_features))
            P11 = da.zeros((padding, padding))
            HH = da.block([[HH,  P01],
                           [P10, P11]])

            P1 = da.zeros((padding, HY.shape[1]))
            HY = da.block([[HY],
                           [P1]])

        # rechunk, add bias, and solve
        HH = HH.rechunk(self.bsize_) + self.alpha * da.eye(HH.shape[1], chunks=self.bsize_)
        HY = HY.rechunk(self.bsize_)

        B = da.linalg.solve(HH, HY, sym_pos=True)
        if padding > 0:
            B = B[:n_features]

        return B

    def fit(self, X, y=None, sync_every=10):
        """Fits an ELM with data in a bunch of files.

        Model will use the set of features from the first file.
        Same features must have same names across the whole dataset.

        .. todo: Check what happens if features are in different order or missing.

        Does **not** support sparse data.

        .. todo: Check if some sparse data would work.

        .. todo: Check that sync_every does not affect results

        .. todo: Add single precision

        .. todo: Parquet file format examples in documentation

        Original features and bias are added to the end of data, for easier rechunk-merge. This way full chunks
        of hidden neuron outputs stay intact.


        Parameters
        ----------

        X : [str]
            List of input data files in Parquet format.

        y : [str]
            List of target data files in Parquet format.

        sync_every : int or None
            Synchronize computations after this many files are processed. None for running without synchronization.
            Less synchronization improves run speed with smaller data files, but may result in large swap space usage
            for large data problems. Use smaller number for more frequent synchronization if swap space
            becomes a problem.
        """

        if not _is_list_of_strings(X) or not _is_list_of_strings(y):
            raise ValueError("Expected X and y as lists of file names.")

        if len(X) != len(y):
            raise ValueError("Expected X and y as lists of files with the same length. "
                             "Got len(X)={} and len(y)={}".format(len(X), len(y)))

        # read first file and get parameters
        X_dask = dd.read_parquet(X[0]).to_dask_array(lengths=True)
        Y_dask = dd.read_parquet(y[0]).to_dask_array(lengths=True)

        n_samples, n_features = X_dask.shape
        if hasattr(self, 'n_features_') and self.n_features_ != n_features:
            raise ValueError('Shape of input is different from what was seen in `fit`')

        _, n_outputs = Y_dask.shape
        if hasattr(self, 'n_outputs_') and self.n_outputs_ != n_outputs:
            raise ValueError('Shape of outputs is different from what was seen in `fit`')

        # set batch size, default is bsize=2000 or all-at-once with less than 10_000 samples
        self.bsize_ = self.batch_size
        if self.bsize_ is None:
            self.bsize_ = n_samples if n_samples < 10 * 1000 else 2000

        # init model if not fit yet
        if not hasattr(self, 'hidden_layers_'):
            self.n_features_ = n_features
            self.n_outputs_ = n_outputs

            X_sample = X_dask[:10].compute()
            self._init_hidden_layers(X_sample)
            self._setup_dask_client()

        HH, HY = self._compute(X, y, sync_every=sync_every)
        self.B = self._solve(HH, HY)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Prediction works with both lists of Parquet files and numeric arrays.

        Parameters
        ----------

        X : array-like, [str]
            Input data as list of Parquet files, or as a numeric array.

        Returns
        -------
        Yh : array, shape (n_samples, n_outputs)
            Predicted values for all input samples.

            .. attention:: Returns all outputs as a single in-memory array!

                Danger of running out out memory for high-dimensional outputs, if a large set of input
                files is provided. Feed data in smaller batches in such case.
        """
        check_is_fitted(self, 'is_fitted_')

        if _is_list_of_strings(X):
            Yh_list = []

            # processing files
            for X_file in X:
                X_dask = dd.read_parquet(X_file).to_dask_array(lengths=True)
                H_dask = self._project(X_dask)
                Yh_list.append(da.dot(H_dask, self.B))

            Yh_dask = da.concatenate(Yh_list, axis=0)
            return Yh_dask.compute()

        else:
            X = check_array(X, accept_sparse=True)
            H = [np.ones((X.shape[0], 1))]
            if self.include_original_features:
                H.append(_dense(X))
            H.extend([hl.transform(X) for hl in self.hidden_layers_])

            return np.hstack(H) @ self.B.compute()
