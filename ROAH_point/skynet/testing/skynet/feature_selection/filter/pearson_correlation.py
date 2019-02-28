import numpy as np
import pandas as pd


def pearson_correlation(X, y, depth=None):
    drk = []
    for key in X:
        if len(np.unique(X[key])) == 1:
            drk.append(key)
    X = X.drop(drk, axis=1)

    fets = np.array(X.keys())
    Xy = pd.concat([X, y], axis=1)
    corr = np.absolute(np.corrcoef(Xy.T)[-1, :-1])

    idx = np.argsort(corr)[::-1]
    if depth is None:
        fets = list(fets[idx]) + drk
    else:
        fets = fets[idx][:depth]

    return np.array(fets)
