import copy

from skynet.model_selection.validate import cross_validation


def convert_visibility_rank(vis):
    import numpy as np
    from skynet.datasets import learning_data

    label = np.zeros(len(vis))
    v = list(learning_data.get_init_vis_level().values()) + [100000]
    delta = np.diff(v)
    for i, d in enumerate(delta):
        indices = np.where((vis > v[i]) & (vis <= v[i] + d))[0]
        label[indices] = i
    return label


def grid_search_cv(model, X, y, param_grid, cv=3, scoring="f1"):
    grids = __transform_param_grid(param_grid, 0, {}, [])
    best_score = 0
    best_params = grids[0]
    for grid in grids:
        clf = model.__class__(**grid)
        f1s = cross_validation(clf, X, y, cv, scoring=scoring)

        if len(f1s):
            if f1s.mean() > best_score:
                best_score = f1s.mean()
                best_params = grid
                print(f1s.mean())
                print(grid)
                print()

    return best_score, best_params


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


def main():
    import os
    import re
    import datetime
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, recall_score
    from sklearn.ensemble import RandomForestClassifier
    from skynet import USER_DIR, DATA_DIR
    from skynet.nwp2d import NWPFrame
    from skynet.datasets import learning_data
    from skynet.datasets import convert

    params = [
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto'},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto'},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto'},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto'},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto'},
        {'n_estimators': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto'}
    ]

    threat = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]

    n_clfs = [
        100,
        100,
        100,
        100,
        100,
        100
    ]

    target = learning_data.get_init_response()

    icao = 'RJAA'
    # 'RJSS',
    # 'RJTT',
    # 'ROAH',
    # 'RJOC',
    # 'RJOO',
    # 'RJCH',
    # 'RJFF',
    # 'RJFK',
    # 'RJGG',
    # 'RJNK',
    # 'RJOA',

    '''
    # metar読み込み
    metar_path = '%s/text/metar' % DATA_DIR
    with open('%s/head.txt' % metar_path, 'r') as f:
        header = f.read()
    header = header.split(sep=',')

    data15 = pd.read_csv('%s/2015/%s.txt' % (metar_path, icao), sep=',')
    data16 = pd.read_csv('%s/2016/%s.txt' % (metar_path, icao), sep=',')
    data17 = pd.read_csv('%s/2017/%s.txt' % (metar_path, icao), sep=',', names=header)

    metar_data = pd.concat([data15, data16, data17])
    metar_data = NWPFrame(metar_data)

    metar_data.strtime_to_datetime('date', '%Y%m%d%H%M%S', inplace=True)
    metar_data.datetime_to_strtime('date', '%Y-%m-%d %H:%M', inplace=True)
    metar_data.drop_duplicates('date', inplace=True)
    metar_data.index = metar_data['date'].values

    metar_keys = ['date', 'visibility', 'str_cloud']
    metar_data = metar_data[metar_keys]
    metar_data['visibility_rank'] = convert_visibility_rank(metar_data['visibility'])

    # MSM読み込み
    msm_data = pd.read_csv('%s/msm_airport/%s.csv' % (DATA_DIR, icao))

    msm_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    msm_data.index = msm_data['date'].values
    msm_data.sort_index(inplace=True)

    fets = learning_data.get_init_features()
    res = learning_data.get_init_response()
    X = NWPFrame(pd.concat([msm_data[fets], metar_data[res]], axis=1))
    X.dropna(inplace=True)
    X.strtime_to_datetime('date', '%Y-%m-%d %H:%M', inplace=True)
    X.datetime_to_strtime('date', '%Y%m%d%H%M', inplace=True)
    X = X[fets + res]

    date = [d for d in X.index if not re.match('2017', d)]
    train = X.loc[date]
    '''
    train = learning_data.read_learning_data('%s/skynet/train_%s.pkl' % (DATA_DIR, icao))
    test = learning_data.read_learning_data('%s/skynet/test_%s.pkl' % (DATA_DIR, icao))

    # feature増やしてからデータ構造を（入力、正解）に戻す

    fets = [f for f in train.keys() if not (f in target)]
    print(fets)

    # 時系列でデータを分割
    sptrain = convert.split_time_series(train, train['date'], level="month", period=2)
    sptest = convert.split_time_series(test, test['date'], level="month", period=2)

    ss = StandardScaler()
    model_dir = '%s/PycharmProjects/SkyCC/trained_models' % USER_DIR

    period_keys = [
        'month:1-2',
        'month:3-4',
        'month:5-6',
        'month:7-8',
        'month:9-10',
        'month:11-12'
    ]
    for i_term, key in enumerate(period_keys):
        os.makedirs(
            '%s/%s/forest/%s'
            % (model_dir, icao, key), exist_ok=True
        )

        X_train = sptrain[key][fets]
        X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.keys())
        y_train = sptrain[key][target]
        X_train, y_train = convert.balanced(X_train, y_train)

        X_test = sptest[key][fets]
        X_test = pd.DataFrame(ss.fit_transform(X_test), columns=X_test.keys())
        y_test = sptest[key][target]

        cv = False
        if cv:
            best_score, best_params = grid_search_cv(
                RandomForestClassifier(),
                X_train,
                y_train,
                param_grid={"n_estimators": [10, 100],
                            "min_samples_split": [2, 10],
                            "min_samples_leaf": [1, 10],
                            "max_features": ["auto", 2, 10, 30, 70]},
                scoring="recall"
            )

            print("best score", best_score)
            print("best params", best_params)
            print()

            model = RandomForestClassifier(**best_params)
            model.fit(X_train.values, y_train.values[:, 0])
            p = model.predict(X_test.values)

            y_true = np.where(y_test > 1, 0, 1)
            y_pred = np.where(p > 1, 0, 1)

            print(recall_score(y_true=y_true, y_pred=y_pred))

        search = True
        y_true = np.where(y_test > 1, 0, 1)
        p_rf = np.zeros((len(X_test), n_clfs[i_term]))
        score_rf = np.zeros(n_clfs[i_term])
        i = 0
        if search:
            while True:
                clf = RandomForestClassifier(**params[i_term])
                clf.fit(X_train.values, y_train.values[:, 0])
                p = clf.predict(X_test.values)

                y_pred = np.where(p > 1, 0, 1)

                scr = recall_score(y_true=y_true, y_pred=y_pred)

                if scr >= threat[i_term]:
                    print(scr)
                    p_rf[:, i] = p
                    score_rf[i] = scr
                    pickle.dump(clf, open("%s/%s/forest/%s/rf%03d.pkl"
                                          % (model_dir, icao, key, i), "wb"))
                    i += 1

                if i == n_clfs[i_term]:
                    break

        learning_model = False
        if learning_model:
            for i in range(n_clfs[i_term]):
                clf = pickle.load(open("%s/%s/forest/%s/rf%03d.pkl"
                                       % (model_dir, icao, key, i), "rb"))
                p_rf[:, i] = clf.predict(X_test.values)

                y_pred = np.where(p_rf[:, i] > 1, 0, 1)
                score_rf[i] = recall_score(y_true=y_true, y_pred=y_pred)

                print(score_rf[i])

        p = p_rf.mean(axis=1)
        score_rf = score_rf.mean()
        print("f1 mean", score_rf)

        plt.figure()
        plt.plot(y_test)
        plt.plot(p)
    plt.show()


if __name__ == "__main__":
    main()
