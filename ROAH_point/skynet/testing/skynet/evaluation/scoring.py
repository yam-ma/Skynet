import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


class SCORES(object):
    functions = {
        "f1": f1_score,
        "recall": recall_score
    }


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
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from skynet import OUTPUT_PATH
    from skynet.data_handling import read_learning_data
    from skynet.data_handling import balanced
    from skynet.data_handling import split_time_series
    from skynet.data_handling import get_init_response
    from skynet.model_selection import grid_search_cv
    from skynet.preprocessing import PreProcessor
    from skynet.ensemble import SkyRandomForest
    from skynet.ensemble import SkyGradientBoosting
    from skynet.svm import SkySVM
    from skynet.neural_network import SkyMLP
    from sklearn.linear_model import LogisticRegression
    from mlxtend.classifier import StackingClassifier

    preprocess = PreProcessor(norm=False, binary=False)

    mode = 1
    if mode == 1:
        fets = [
            ['edge_Temp_6', 'edge_Temp_8', 'smooth_ks5_Rela_0', 'wspd500', 'Geop900', 'smooth_ks5_Rela_2', 'Rela925',
             'Medium cloud cover', 'smooth_ks7_Rela_0', 'v-co400', 'smooth_ks3_Rela_3', 'smooth_ks3_Rela_2',
             'smooth_ks9_Rela_0', 'smooth_ks7_Vert_3', 'Rela1000', 'smooth_ks3_Rela_4'],
            ['smooth_ks7_Rela_5', 'smooth_ks7_Temp_6', 'u-co925', 'smooth_ks9_Vert_3', 'Rela900', 'WX_telop_200',
             'smooth_ks7_Rela_0', 'u-co950', 'Temp400', 'u-co900', 'Temp700'],
            ['Relative humidity', 'u-co600', 'smooth_ks7_Vert_4', 'smooth_ks3_Temp_9', 'edge_Vert_3',
             'smooth_ks7_Vert_5',
             'smooth_ks7_Rela_3', 'u-co925', 'smooth_ks7_Rela_6', 'smooth_ks5_Rela_1', 'smooth_ks3_Rela_9', 'Rela500',
             'edge_Rela_4', 'u-co800', 'smooth_ks7_Rela_1', 'smooth_ks7_Vert_2', 'u-co700', 'Total precipitation',
             'smooth_ks3_Vert_8', 'smooth_ks9_Vert_0', 'Geop1000', 'edge_Temp_0', 'Vert500', 'Geop975',
             'smooth_ks7_Temp_4',
             'v-co600', 'v-co500', 'Pressure', 'Rela850', 'Low cloud cover', 'smooth_ks7_Vert_1', 'edge_Temp_3'],
            ['edge_Temp_1', 'smooth_ks9_Rela_2', 'smooth_ks7_Vert_4', 'wspd975', 'Vert600', 'smooth_ks5_Vert_7',
             'smooth_ks3_Vert_8', 'hour'],
            ['smooth_ks3_Rela_1', 'smooth_ks7_Vert_5', 'Rela1000', 'smooth_ks7_Rela_1', 'smooth_ks3_Vert_4',
             'smooth_ks7_Vert_2', 'smooth_ks9_Vert_3', 'Vert925', 'edge_Temp_1'],
            ['smooth_ks5_Rela_8', 'edge_Temp_3', 'smooth_ks3_Vert_5', 'Relative humidity']
        ]
        params = [
            {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 10},
            {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 1},
            {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1},
            {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 10},
            {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 1},
            {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 1}
        ]

    threat = [
        0.8,
        0.15,
        0.25,
        0.15,
        0.25,
        0.6
    ]

    n_clfs = [
        0,
        0,
        0,
        0,
        0,
        100
    ]

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

    cv_forest = False
    cv_boosting = False
    cv_svm = False
    cv_nn = False

    grid_rf = [{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1},
               {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1},
               {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 1},
               {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 10}]
    grid_gb = [{'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1,
                'max_depth': 5, 'subsample': 0.8},
               {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 10,
                'max_depth': 4, 'subsample': 0.8},
               {'n_estimators': 10, 'min_samples_split': 100, 'min_samples_leaf': 1,
                'max_depth': 5, 'subsample': 0.8},
               {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 6,
                'subsample': 0.8}]
    grid_svm = [{'C': 1, 'kernel': 'rbf', 'gamma': 0.01},
                {'C': 100, 'kernel': 'rbf', 'gamma': 0.01},
                {'C': 100, 'kernel': 'rbf', 'gamma': 0.1},
                {'C': 10000, 'kernel': 'rbf', 'gamma': 0.0001}]
    grid_nn = [{'alpha': 0.0001},
               {'alpha': 0.01}]
    ss = StandardScaler()
    for i_term, key in enumerate(sptrain):
        print(fets[i_term])
        print(params[i_term])
        print(threat[i_term])
        print()
        X_train = sptrain[key][fets[i_term]]
        X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.keys())
        y_train = sptrain[key][target]
        X_train, y_train = balanced(X_train, y_train)

        X_test = sptest[key][fets[i_term]]
        X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.keys())
        y_test = sptest[key][target]

        if cv_forest:
            best_score, best_params = grid_search_cv(SkyRandomForest,
                                                     X_train,
                                                     y_train,
                                                     param_grid={"n_estimators": [10, 100],
                                                                 "min_samples_split": [2, 10, 100],
                                                                 "min_samples_leaf": [1, 10, 50]
                                                                 })

            print("best_score and best params")
            print(best_score)
            print(best_params)
            print()

        if cv_boosting:
            best_score, best_params = grid_search_cv(SkyGradientBoosting,
                                                     X_train,
                                                     y_train,
                                                     param_grid={"n_estimators": [10, 100],
                                                                 "min_samples_split": [2, 10, 100],
                                                                 "min_samples_leaf": [1, 10, 50],
                                                                 "max_depth": [3, 4, 5, 6],
                                                                 "subsample": [0.8, 1.0]})

            print("best_score and best params")
            print(best_score)
            print(best_params)

        if cv_svm:
            best_score, best_params = grid_search_cv(SkySVM,
                                                     X_train,
                                                     y_train,
                                                     param_grid={"C": [1, 10, 100, 1000, 10000],
                                                                 "kernel": ["rbf"],
                                                                 "gamma": [0.0001, 0.001, 0.01, 0.1, 1]})

            print("best_score and best params")
            print(best_score)
            print(best_params)

        if cv_nn:
            best_score, best_params = grid_search_cv(SkyMLP,
                                                     X_train,
                                                     y_train,
                                                     param_grid={"alpha": [0.0001, 0.001, 0.01, 0.1]})

            print("best_score and best params")
            print(best_score)
            print(best_params)

        X_train, y_train = X_train.values, y_train.values[:, 0]
        X_test, y_test = X_test.values, y_test.values[:, 0]

        """
        for g in grid_boosting:
            clfs.append(SkyGradientBoosting(**g))

        for g in grid_svm:
            clfs.append(SkySVM(**g))

        for g in grid_nn:
            clfs.append(SkyMLP(**g))
        """

        y_true = np.where(y_test > 1, 0, 1)
        p_rf = np.zeros((len(X_test), n_clfs[i_term]))
        f1_rf = np.zeros(n_clfs[i_term])
        i = 0

        search = False
        if search:
            while True:
                clf = SkyRandomForest(**params[i_term])
                clf.fit(X_train, y_train)
                p = clf.predict(X_test)

                y_pred = np.where(p > 1, 0, 1)
                f1 = f1_score(y_true=y_true, y_pred=y_pred)

                if f1 >= threat[i_term]:
                    print(f1)
                    p_rf[:, i] = p
                    f1_rf[i] = f1
                    pickle.dump(clf, open(OUTPUT_PATH + "/learning_models/forest/rf%03d_%s.pkl" % (i, key), "wb"))
                    i += 1

                if i == n_clfs[i_term]:
                    break

        learning_model = True
        if learning_model:
            for i in range(n_clfs[i_term]):
                clf = pickle.load(open(OUTPUT_PATH + "/learning_models/forest/rf%03d_%s.pkl" % (i, key), "rb"))
                p_rf[:, i] = clf.predict(X_test)

                y_pred = np.where(p_rf[:, i] > 1, 0, 1)
                f1_rf[i] = f1_score(y_true=y_true, y_pred=y_pred)

                print(f1_rf[i])

        p = p_rf.mean(axis=1)
        f1_rf = f1_rf.mean()
        print("f1 mean", f1_rf)

        plt.plot(y_test)
        plt.plot(p)
        plt.show()

        """
        bos1 = SkyGradientBoosting(n_estimators=10, min_samples_split=2, min_samples_leaf=1,
                                   max_depth=6, subsample=0.8)
        bos1.fit(X_train, y_train)
        p_bos1 = bos1.predict(X_test)

        svm1 = SkySVM(kernel="rbf", gamma=0.01, C=1000)
        svm1.fit(X_train, y_train)
        p_svm1 = svm1.predict(X_test)

        nn1 = SkyMLP(alpha=0.1)
        nn1.fit(X_train, y_train)
        p_nn1 = nn1.predict(X_test)
        """


if __name__ == "__main__":
    main()
