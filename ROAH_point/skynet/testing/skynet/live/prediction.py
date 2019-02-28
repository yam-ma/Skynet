N_CLF = 100

W = {
    "RJAA": [
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 1, 1, 1, 1, 1, 1, 1, 5]
    ]
}


def predict(X, clfs, w, smooth=False, confidence=False):
    import numpy as np
    n_clf = len(clfs)
    p_rf = np.zeros((len(X), n_clf))

    for i, clf in enumerate(clfs):
        p_rf[:, i] = clf.predict(X.values)

    # p = p_rf.mean(axis=1)
    # p = majority_vote(p_rf, n_class=9, threshold=None)

    p = deflected_mean(p_rf, w, threshold=6)
    if smooth:
        p = smoothing(p, ksize=6, threshold=5)

    if confidence:
        c = confidence_factor(p_rf, n_class=9)
        return p, c

    else:
        return p


def confidence_factor(x, n_class):
    import numpy as np
    import skynet.nwp2d as npd
    mv = np.zeros((len(x), n_class))
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mv[idx, x[:, i].astype(int)] += 1

    confac = npd.NWPFrame(mv)

    return confac


def deflected_mean(x, w, threshold=None):
    import numpy as np
    xx = np.zeros_like(x)
    mp = np.zeros_like(x)
    ww = np.array([w for _ in range(len(x))])
    idx = np.arange(len(x))
    for i in range(x.shape[1]):
        mp[:, i] = ww[idx, x[:, i].astype(int)]
        xx[:, i] = x[:, i] * mp[:, i]

    p = xx.sum(axis=1) / mp.sum(axis=1)

    if threshold is not None:
        p[p > threshold] = p.max()

    return p


def smoothing(x, ksize, threshold=None):
    import numpy as np
    kernel = np.ones(ksize)
    x_sm = 1 / kernel.sum() * np.convolve(x, kernel, mode="same")

    if threshold is not None:
        extend = np.zeros_like(x)
        for i in range(len(x)):
            idx1 = i - int(ksize / 2)
            idx2 = i + int(ksize / 2)
            if x_sm[i] < threshold:
                if idx1 >= 0 and idx2 <= len(x):
                    extend[i] = x_sm[i] * x[idx1:idx2].min() / (x_sm[idx1:idx2].min() + 1e-2)
                elif idx1 < 0:
                    extend[i] = x_sm[i] * x[:idx2].min() / (x_sm[:idx2].min() + 1e-2)
                elif idx2 > 0:
                    extend[i] = x_sm[i] * x[idx1:].min() / (x_sm[idx1:].min() + 1e-2)

            else:
                extend[i] = x_sm[i]

        extend[extend > 6] = 8
        return extend

    return x_sm


def set_month_key(date, period=2):
    keys = [
        (0, "month:1-2"),
        (1, "month:3-4"),
        (2, "month:5-6"),
        (3, "month:7-8"),
        (4, "month:9-10"),
        (5, "month:11-12")
    ]
    i = 4
    e = 6
    d = int(date[i:e])

    if d % period == 0:
        idx = int(d / period) - 1
    else:
        idx = int(d / period)

    return keys[idx]


def adapt_visibility(v):
    import copy
    import numpy as np
    import skynet.datasets as skyds
    v = copy.deepcopy(v)
    vis_level = skyds.learning_data.get_init_vis_level()
    diff = np.diff(list(vis_level.values()) + [9999])
    for key in vis_level:
        v[(v > key) & (v <= (key + 1))] = \
            diff[key] * (v[(v > key) & (v <= (key + 1))] - key) + vis_level[key]
    v[v >= list(vis_level.values())[-1]] = 9999
    return v


def main():
    import argparse
    import pickle
    import pandas as pd
    import skynet.nwp2d as npd
    import skynet.datasets as skyds
    from skynet import MY_DIR, DATA_DIR
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser()
    parser.add_argument("--icao")
    parser.add_argument("--date")
    parser.add_argument("--time")

    args = parser.parse_args()

    icao = args.icao
    date = args.date
    time = args.time

    if args.icao is None:
        icao = "RJAA"

    if args.date is None:
        date = "20180620"

    if args.time is None:
        time = "060000"

    X = pd.read_csv('/Users/makino/PycharmProjects/SkyCC/data/compass_ark/GLOBAL_METAR-%s.csv' % icao)
    X = npd.NWPFrame(X)

    df_date = X[['HEAD:YEAR', 'MON', 'DAY', 'HOUR']]

    date_keys = ['HEAD:YEAR', 'MON', 'DAY', 'HOUR', 'MIN']
    X['MIN'] = [0] * len(X)
    for key in date_keys:
        if not key == 'HEAD:YEAR':
            X[key] = ['%02d' % int(d) for d in X[key]]

    X.merge_strcol(date_keys, 'date', inplace=True)
    X.drop(date_keys, axis=1, inplace=True)

    # print(X)
    wni_code = skyds.learning_data.get_init_features('wni')
    X = X[wni_code]

    long_code = skyds.learning_data.get_init_features('long')
    X.columns = long_code

    vt = len(X)

    pool = skyds.learning_data.read_learning_data('%s/skynet/train_%s.pkl' % (DATA_DIR, icao))[long_code]
    sppool = skyds.convert.split_time_series(pool, date=pool["date"].values, level="month", period=2, index_date=True)

    month_key_info = set_month_key(X['date'][0], period=2)
    X = pd.concat([X, sppool[month_key_info[1]]])

    ss = StandardScaler()
    X = npd.NWPFrame(ss.fit_transform(X), columns=X.keys())
    X = X.iloc[:vt]

    model_dir = '/Users/makino/PycharmProjects/SkyCC/trained_models'
    clfs = [
        pickle.load(
            open("%s/%s/forest/%s/rf%03d.pkl" % (model_dir, icao, month_key_info[1], i), "rb"))
        for i in range(N_CLF)
    ]

    p, c = predict(X, clfs, W[icao][month_key_info[0]], smooth=False, confidence=True)

    # vis = pd.read_csv("%s/live/input/%s.csv" % (MY_DIR, icao))[["date"]]
    vis_pred = adapt_visibility(p)
    vis = npd.NWPFrame(df_date)
    vis = pd.concat([vis, c], axis=1)
    vis['SkyNet'] = vis_pred
    vis.to_csv("%s/live/prediction/SkyNet_%s.vis.csv" % (MY_DIR, icao), index=False)

    print(vis)


if __name__ == "__main__":
    main()
