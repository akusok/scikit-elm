import pytest
from pytest import approx
import numpy as np
from tempfile import TemporaryDirectory
from sklearn.datasets import load_iris, make_multilabel_classification, load_diabetes
from skelm import LargeELMRegressor

pd = pytest.importorskip("pandas")  # tests if Pandas is installed
pytest.importorskip("pyarrow")
dask = pytest.importorskip("dask")


@pytest.fixture
def data_class():
    return load_iris(return_X_y=True)

@pytest.fixture
def data_ml():
    return make_multilabel_classification()

@pytest.fixture
def data_reg():
    return load_diabetes(return_X_y=True)


def test_Input_DifferentLengths_Raises():
    elm = LargeELMRegressor()
    with pytest.raises(ValueError):
        elm.fit(['a', 'b', 'c'], ['d'])


def test_Input_MultipleFiles(data_reg):
    X, y = data_reg
    with TemporaryDirectory() as data_dir:
        X_files = [data_dir + "/X_{}.parquet".format(i) for i in range(3)]
        y_files = [data_dir + "/y_{}.parquet".format(i) for i in range(3)]
        for i in range(3):
            pd.DataFrame(X[i::3], columns=[str(c) for c in range(X.shape[1])]).to_parquet(fname=X_files[i])
            pd.DataFrame(y[i::3], columns=['Class']).to_parquet(fname=y_files[i])

        elm = LargeELMRegressor(batch_size=10)
        elm.fit(X_files, y_files)
        y_hat = elm.predict(X_files)

        assert np.mean(y_hat != y) < 0.33
