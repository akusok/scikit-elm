import scipy as sp


def flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        # don't break strings into characters
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def _dense(X):
    if sp.sparse.issparse(X):
        return X.todense()
    else:
        return X