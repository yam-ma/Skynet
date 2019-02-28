import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from skynet import OUTPUT_PATH
from skynet.data_handling import read_learning_data
from skynet.data_handling import balanced
from skynet.data_handling import split_time_series
from skynet.data_handling import get_init_response
from skynet.preprocessing import PreProcessor
from skynet.ensemble import SkyRandomForest
from skynet.ensemble import SkyGradientBoosting
from skynet.svm import SkySVM
from skynet.neural_network import SkyMLP
from skynet.feature_selection.filter import pearson_correlation

from mlxtend.classifier import StackingClassifier


def __grid_search_cv(model, X, y, param_grid, cv=3):
    grids = __transform_param_grid(param_grid, 0, {}, [])
    best_score = 0
    best_params = grids[0]
    for grid in grids:
        clf = model(**grid)

        f1s = __cross_validation(clf, X, y, cv)

        if len(f1s):
            if f1s.mean() > best_score:
                best_score = f1s.mean()
                best_params = grid
                print(f1s.mean())
                print(grid)
                print()

    return best_score, best_params


def __cross_validation(clf, X, y, cv=3):
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
        X_train, y_train = balanced(X_train, y_train)
        X_train = X_train.values
        y_train = y_train.values[:, 0]

        X_test = spX[i]
        y_test = spy[i]
        X_test = X_test.values
        y_test = y_test.values[:, 0]

        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        y_test = np.where(y_test > 1, 0, 1)
        p = np.where(p > 1, 0, 1)

        if len(np.unique(y_test)) == 2 and len(np.unique(p)) == 2:
            f1 = f1_score(y_true=y_test, y_pred=p)
            cv_scores.append(f1)

    return np.array(cv_scores)


def __transform_param_grid(param, cnt, cp, cps):
    keys = list(param.keys())

    if cnt == len(keys):
        return
    else:
        val = param[keys[cnt]]

        for v in val:
            cp[keys[cnt]] = v
            if cnt == len(keys) - 1:
                cps.append(copy.deepcopy(cp))
            __transform_param_grid(param, cnt + 1, cp, cps)

    return cps


def result_report(y_true, y_pred, score="f1"):
    if score == "f1":
        func = f1_score
    else:
        func = None

    y_true = np.where(y_true > 1, 0, 1)

    for key in y_pred:
        p = np.where(y_pred[key] > 1, 0, 1)
        f1 = func(y_true=y_true, y_pred=p)
        print(key, f1)


def main():
    preprocess = PreProcessor(norm=False, binary=False)
    target = get_init_response()

    icao = "RJFK"

    train = read_learning_data(OUTPUT_PATH + "/datasets/apvis/train_%s.pkl" % icao)
    test = read_learning_data(OUTPUT_PATH + "/datasets/apvis/test_%s.pkl" % icao)

    # feature増やしてからデータ構造を（入力、正解）に戻す
    preprocess.fit(train.iloc[:, :-1], train.iloc[:, -1], test.iloc[:, :-1], test.iloc[:, -1])
    train = pd.concat([preprocess.X_train, preprocess.y_train], axis=1)
    test = pd.concat([preprocess.X_test, preprocess.y_test], axis=1)

    # 時系列でデータを分割
    sptrain = split_time_series(train, level="month", period=2)
    sptest = split_time_series(test, level="month", period=2)

    cv_svm = False
    grid_svm = [{'C': 1, 'kernel': 'rbf', 'gamma': 0.01},
                {'C': 100, 'kernel': 'rbf', 'gamma': 0.01},
                {'C': 100, 'kernel': 'rbf', 'gamma': 0.1},
                {'C': 10000, 'kernel': 'rbf', 'gamma': 0.0001}]

    ss = StandardScaler()
    for key in sptrain:
        data = pd.concat([sptrain[key], sptest[key]])
        fets = pearson_correlation(data.iloc[:, :-1], data.iloc[:, -1], depth=30)

        X_train = sptrain[key][fets]
        X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.keys())
        y_train = sptrain[key][target]
        X_train, y_train = balanced(X_train, y_train)

        X_test = sptest[key][fets]
        X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.keys())
        y_test = sptest[key][target]

        if cv_svm:
            best_score, best_params = __grid_search_cv(SkySVM,
                                                       X_train,
                                                       y_train,
                                                       param_grid={"C": [1, 10, 100, 1000, 10000],
                                                                   "kernel": ["rbf"],
                                                                   "gamma": [0.0001, 0.001, 0.01, 0.1, 1]})

            print("best_score and best params")
            print(best_score)
            print(best_params)

        X_train, y_train = X_train.values, y_train.values[:, 0]
        X_test, y_test = X_test.values, y_test.values[:, 0]

        clf = SkySVM(**{'C': 10, 'kernel': 'rbf', 'gamma': 0.1})
        clf.fit(X_train, y_train)
        p = clf.predict(X_test)

        y_true = np.where(y_test > 1, 0, 1)
        y_pred = np.where(p > 1, 0, 1)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        print("f1", f1)

        plt.plot(y_test)
        plt.plot(p)
        plt.show()

        if True:
            break


if __name__ == "__main__":
    main()
