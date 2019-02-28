import datetime
import pandas as pd
import skynet.data_handling as dh

from sklearn.preprocessing import StandardScaler
from skynet.data_handling import msm
from skynet.data_handling import area_forecast as af
from skynet.data_handling.preprocessing import PreProcessor


def make_datetable(date):
    y = int(date[:4])
    mo = int(date[4:6])
    d = int(date[6:8])
    h = 0
    mi = 0

    s = datetime.datetime(y, mo, d, h, mi)
    datetable = [(s + datetime.timedelta(hours=i)).strftime("%Y%m%d%H%M")
                 for i in range(48)]

    df = pd.DataFrame(datetable, columns=["date"])

    return df


def convert_to_datestring(X, date_header):
    X[date_header] = X[date_header].astype(str)
    date = [y + mo.zfill(2) + d.zfill(2) + h.zfill(2) + mi.zfill(2)
            for y, mo, d, h, mi in X[date_header].values]
    X = X.drop(date_header, axis=1)
    X.insert(loc=0, column="date", value=date)
    return X


def make_input_data(icao, date, time):
    msm_sf = msm.read_airports(layer="surface", icaos=[icao], date=date)
    msm_ul = msm.read_airports(layer="upper", icaos=[icao], date=date)
    X_msm = msm.concat_surface_upper(msm_sf, msm_ul)
    X_msm = X_msm[icao]
    X_msm = X_msm.interpolate(axis=0)

    X_af = af.read(icaos=[icao], date=date, time=time)
    X_af = X_af[icao]
    X_af = convert_to_datestring(X_af, date_header=["year", "month", "day", "hour", "min"])

    X = make_datetable(date)
    for d in (X_msm, X_af):
        X = dh.sync_values(X, d, key="date")

    fets = dh.get_init_features()
    X = X[fets]

    preprocess = PreProcessor(norm=False, binary=False)
    preprocess.fit(X_test=X, y_test=None)
    X = preprocess.X_test

    return X


def normalization(X, icao):
    data = dh.read_learning_data(OUTPUT_PATH + "/datasets/apvis/test_%s.pkl" % icao)
    fets = dh.get_init_features()
    data = data[fets]
    preprocess = PreProcessor(norm=False, binary=False)
    preprocess.fit(X_test=data, y_test=None)
    data = preprocess.X_test
    X_size = len(X)
    X = pd.concat([X, data])
    ss = StandardScaler()
    X = pd.DataFrame(ss.fit_transform(X), columns=X.keys())
    X = X[:X_size]

    return X


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--icao")
    parser.add_argument("--date")
    parser.add_argument("--time")

    args = parser.parse_args()

    icao = args.icao
    date = args.date
    time = args.time

    if args.icao is None:
        icao = "RJFT"

    if args.date is None:
        date = "20180620"

    if args.time is None:
        time = "060000"

    X = make_input_data(icao=icao, date=date, time=time)
    X.to_csv("%s/live/input/%s.csv" % (OUTPUT_PATH, icao), index=False)

    X_norm = normalization(X, icao)
    X_norm.to_csv("%s/live/input/norm_%s.csv" % (OUTPUT_PATH, icao), index=False)


if __name__ == "__main__":
    main()
