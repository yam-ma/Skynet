import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from skynet.base import SkyMLBase
from skynet.data_handling import balanced
from skynet.ensemble import SkyGradientBoosting
from skynet.ensemble import SkyRandomForest
from skynet.neural_network import SkyMLP
from skynet.svm import SkySVM


class SkyStacking(SkyMLBase):
    def __init__(self, classifiers, meta_classifier, n_folds=3):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.n_folds = n_folds

    def fit(self, X=None, y=None):
        shuffled = np.random.permutation(len(X))
        X = X.loc[shuffled]
        y = y.loc[shuffled]

        self.__blend(X, y)

    def predict(self, X=None):
        blend_test = np.zeros((len(X), len(self.classifiers)))

        for n_clf in range(len(self.classifiers)):
            blend_test_ele = np.zeros((len(X), self.n_folds))
            for i in range(self.n_folds):
                blend_test_ele[:, i] = self.classifiers[n_clf].predict(X)

            blend_test[:, n_clf] = blend_test.mean(axis=1)

        return self.meta_classifier.predict(blend_test)

    def __split(self, X, y, cv=None):
        if cv is None:
            cv = self.n_folds

        idx = {int(l): np.where(y.values[:, 0] == l)[0] for l in np.unique(y.values[:, 0])}
        spidx = {}
        for i in range(cv):
            spidx[i] = {
                int(l): idx[l][i * int(len(idx[l]) / cv):(i + 1) * int(len(idx[l]) / cv)]
                for l in idx
            }
            cnc = []
            for k in spidx[i]:
                cnc += list(spidx[i][k])
            spidx[i] = cnc

        spX = {i: X.loc[spidx[i]] for i in range(cv)}

        if type(y) != pd.DataFrame:
            y = pd.DataFrame(y)
        spy = {i: y.loc[spidx[i]] for i in range(cv)}

        return spX, spy

    def __blend(self, X, y):
        spX, spy = self.__split(X, y, self.n_folds)

        blend_train = np.zeros((len(X), len(self.classifiers)))

        for n_clf in range(len(self.classifiers)):
            idx = 0
            for i in range(self.n_folds):
                X_train = pd.concat([spX[n] for n in spX if n != i])
                y_train = pd.concat([spy[n] for n in spy if n != i])
                X_train, y_train = balanced(X_train, y_train)
                X_train = X_train.values
                y_train = y_train.values[:, 0]

                X_valid = spX[i]
                X_valid = X_valid.values

                self.classifiers[n_clf].fit(X_train, y_train)
                blend_train[idx:idx + len(X_valid), n_clf] = self.classifiers[n_clf].predict(X_valid)

                idx += len(X_valid)

        self.meta_classifier.fit(blend_train, y.values[:, 0])


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from skynet import OUTPUT_PATH
    from skynet.data_handling import read_learning_data
    from skynet.preprocessing import PreProcessor
    from skynet.data_handling import get_init_response
    from skynet.data_handling import split_time_series
    from mlxtend.classifier import StackingClassifier

    icao = "RJFK"

    train = read_learning_data(OUTPUT_PATH + "/datasets/apvis/train_%s.pkl" % icao)
    test = read_learning_data(OUTPUT_PATH + "/datasets/apvis/test_%s.pkl" % icao)

    data = pd.concat([train, test]).reset_index(drop=True)

    preprocess = PreProcessor(norm=False, binary=False)
    preprocess.fit(data.iloc[:, :-1], data.iloc[:, -1])

    data = pd.concat([preprocess.X_train, preprocess.y_train], axis=1)

    spdata = split_time_series(data, level="month", period=2)

    selected = list(spdata.keys())[3:]
    spdata = {key: spdata[key] for key in selected}

    for key in spdata:
        ext = spdata[key]

        target = get_init_response()
        feats = [f for f in ext.keys() if not (f in target + ["date"])]

        X = ext[feats]
        ss = StandardScaler()
        X = pd.DataFrame(ss.fit_transform(X), columns=X.keys())
        y = ext[target]
        X, y = balanced(X, y)

        spX, spy = preprocess.split(X, y, n_folds=5)
        X = pd.concat([spX[n] for n in spX if n != 0]).reset_index(drop=True)
        y = pd.concat([spy[n] for n in spy if n != 0]).reset_index(drop=True)

        X_test = spX[0].reset_index(drop=True)
        y_test = spy[0].reset_index(drop=True)

        from sklearn.ensemble import RandomForestClassifier
        clf1 = RandomForestClassifier(max_features=2)
        clf2 = SkySVM()
        meta = LogisticRegression()

        # 学習
        # (注)balancedしてない
        sta = SkyStacking((clf1, clf2), meta)
        sta.fit(X, y)
        p = sta.predict(X_test)

        clf1.fit(X.values, y.values[:, 0])
        print(np.array(X.keys())[np.argsort(clf1.feature_importances_)[::-1]])
        p_rf = clf1.predict(X_test.values)

        # mlxtendのstacking
        sc = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=meta)
        sc.fit(X.values, y.values[:, 0])
        p_sc = sc.predict(X_test.values)

        y_test = np.where(y_test.values[:, 0] > 1, 0, 1)
        p = np.where(p > 1, 0, 1)
        p_rf = np.where(p_rf > 1, 0, 1)
        p_sc = np.where(p_sc > 1, 0, 1)

        f1 = f1_score(y_true=y_test, y_pred=p)
        print("stacking", f1)

        f1_rf = f1_score(y_true=y_test, y_pred=p_rf)
        print("random forest", f1_rf)

        f1_sc = f1_score(y_true=y_test, y_pred=p_sc)
        print("stacked classifier", f1_sc)

        if True:
            break


if __name__ == "__main__":
    main()
