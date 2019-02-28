import numpy as np
import pandas as pd

from sklearn import clone


def cross_validation(model, X, y, cv=3, scoring="f1"):
    from skynet.datasets import convert
    from skynet.evaluation import SCORES
    idx = {int(l): np.where(y.values[:, 0] == l)[0] for l in np.unique(y.values[:, 0])}
    spidx = {}
    for i in range(cv):
        spidx[i] = {int(l): idx[l][i * int(len(idx[l]) / cv):(i + 1) * int(len(idx[l]) / cv)] for l in idx}
        cnc = []
        for k in spidx[i]:
            cnc += list(spidx[i][k])
        spidx[i] = cnc

    spX = {i: X.loc[spidx[i]] for i in range(cv)}

    if type(y) != pd.DataFrame:
        y = pd.DataFrame(y)
    spy = {i: y.loc[spidx[i]] for i in range(cv)}

    cv_scores = []
    for i in range(cv):
        X_train = pd.concat([spX[n] for n in spX if n != i])
        y_train = pd.concat([spy[n] for n in spy if n != i])
        X_train, y_train = convert.balanced(X_train, y_train)
        X_train = X_train.values
        y_train = y_train.values[:, 0]

        X_test = spX[i]
        y_test = spy[i]
        X_test = X_test.values
        y_test = y_test.values[:, 0]

        model = clone(model)
        model.fit(X_train, y_train)
        p = model.predict(X_test)

        y_test = np.where(y_test > 1, 0, 1)
        p = np.where(p > 1, 0, 1)

        if len(np.unique(y_test)) == 2 and len(np.unique(p)) == 2:
            f1 = SCORES.functions[scoring](y_true=y_test, y_pred=p)
            cv_scores.append(f1)

    return np.array(cv_scores)
