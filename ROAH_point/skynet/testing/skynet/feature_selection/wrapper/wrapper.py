import copy
import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.metrics import f1_score

from skynet import DATA_PATH
from skynet.data_handling import read_learning_data
from skynet.data_handling import balanced
from skynet.svm import SkySVM


class WrapperSelector(object):
    def __init__(self, classifier=SkySVM(), nof=10, param_grid=None, **kwargs):
        self.classifier = classifier
        self.nof = nof

        if param_grid is None:
            self.param_grid = {
                'C': [1, 10, 100, 1000, 10000],
                'kernel': ['rbf'],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
            }
        else:
            self.param_grid = param_grid

        if "cv" in kwargs.keys():
            self.cv = kwargs["cv"]
        else:
            self.cv = 3

        self.filter = None
        self.depth = None
        self.score = pd.DataFrame()

    def fit(self, X, y):
        shuffled = np.random.permutation(len(X))
        X = X.loc[shuffled]
        y = y.loc[shuffled]

        self.__forward_search(X, y)

    def __forward_search(self, X, y):
        X_train = X[:int(0.8 * len(X))].reset_index(drop=True)
        y_train = y[:int(0.8 * len(X))].reset_index(drop=True)

        X_test = X[int(0.8 * len(X)):].reset_index(drop=True)
        y_test = y[int(0.8 * len(X)):].reset_index(drop=True)

        fets = np.array(X.keys())
        exfs = self.__filter(X, y)

        bX, by = balanced(X_train, y_train)

        selected = [""]
        while True:
            bf = exfs[0]
            bs = {"score": 0, "params": {"C": 1, "kernel": "rbf", "gamma": 0.0001}, "features": []}
            for f in exfs:
                selected[-1] = f

                s, params = self.__grid_search_cv(X_train[selected], y_train)

                if s > bs["score"]:
                    bs["score"] = s
                    bs["params"] = copy.deepcopy(params)
                    bs["features"] = copy.deepcopy(selected)
                    bf = f

            selected[-1] = bf

            print(selected)
            print(bs["score"])
            print(bs["params"])
            print()

            if bs["score"] != 0:
                bclf = self.classifier.__class__(**bs["params"])
                bclf.fit(bX[selected].values, by.values[:, 0])

                p = bclf.predict(X_test[selected].values)

                uufs = [f for f in fets if not (f in selected)]

                tmp = y_test.values[:, 0]

                fX = X_test.loc[p != tmp, uufs]
                fy = y_test.loc[p != tmp]

                if len(fy) == 0:
                    fX = X[uufs]
                    fy = copy.deepcopy(y)

                exfs = self.__filter(fX, fy)
            else:
                uufs = [f for f in fets if not (f in selected)]
                exfs = self.__filter(X[uufs], y)

            dfap = pd.DataFrame()
            dfap["score"] = [bs["score"]]
            for k in bs["params"]:
                dfap[k] = bs["params"][k]

            for k, f in zip(["f%d" % (i + 1) for i in range(len(selected))], selected):
                dfap[k] = f

            self.score = self.score.append(dfap)
            keys = [k for k in ["score"] + list(bs["params"].keys()) + ["f%d" % (i + 1) for i in range(len(selected))]]
            self.score = self.score[keys]

            if len(selected) == self.nof:
                break

            selected.append("")

    def __filter(self, X, y):
        fets = np.array(X.keys())
        if self.filter is None:
            exfs = fets
        else:
            exfs = self.filter(X, y, depth=self.depth)
        return exfs

    def __grid_search_cv(self, X, y):
        grids = self.__transform_param_grid(0, {}, [])
        best_score = 0
        best_params = grids[0]
        for grid in grids:
            model = self.classifier.__class__(**grid)
            f1s = self.__cross_validation(model, X, y)

            if len(f1s):
                if f1s.mean() > best_score:
                    best_score = f1s.mean()
                    best_params = grid

        return best_score, best_params

    def __cross_validation(self, model, X, y):
        idx = {int(l): np.where(y.values[:, 0] == l)[0] for l in np.unique(y.values[:, 0])}
        spidx = {}
        for i in range(self.cv):
            spidx[i] = {int(l): idx[l][i * int(len(idx[l]) / self.cv):(i + 1) * int(len(idx[l]) / self.cv)]
                        for l in idx}
            cnc = []
            for k in spidx[i]:
                cnc += list(spidx[i][k])
            spidx[i] = cnc

        spX = {i: X.loc[spidx[i]] for i in range(self.cv)}

        if type(y) != pd.DataFrame:
            y = pd.DataFrame(y)
        spy = {i: y.loc[spidx[i]] for i in range(self.cv)}

        cv_scores = []
        for i in range(self.cv):
            X_train = pd.concat([spX[n] for n in spX if n != i])
            y_train = pd.concat([spy[n] for n in spy if n != i])
            X_train, y_train = balanced(X_train, y_train)
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
                f1 = f1_score(y_true=y_test, y_pred=p)
                cv_scores.append(f1)

        return np.array(cv_scores)

    def __transform_param_grid(self, cnt, cp, cps):
        keys = list(self.param_grid.keys())

        if cnt == len(keys):
            return
        else:
            val = self.param_grid[keys[cnt]]

            for v in val:
                cp[keys[cnt]] = v
                if cnt == len(keys) - 1:
                    cps.append(copy.deepcopy(cp))
                self.__transform_param_grid(cnt + 1, cp, cps)

        return cps


def main():
    icao = "RJFK"

    train = read_learning_data(DATA_PATH + "/skynet/train_%s.pkl" % icao)

    clf = WrapperSelector(classifier=SkySVM(), nof=2,
                          param_grid={
                              'C': [1],
                              'kernel': ['rbf'],
                              'gamma': [0.0001]
                          })
    clf.fit(train.iloc[:, :-1], train.iloc[:, -1:])


if __name__ == "__main__":
    main()
