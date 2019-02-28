import datetime
import numpy as np
import pandas as pd

from skynet.data_handling import area_forecast as af
from skynet.data_handling import human_edit as he


def daily_area_forecast(icao, date, time):
    data_af = af.read(icaos=[icao], date=date, time=time)
    data_af = data_af[icao]
    data_af = convert_to_datestring(data_af, date_header=["year", "month", "day", "hour", "min"])

    return data_af


def daily_human_edit(icao, date, time):
    data_he = he.read(icaos=[icao], date=date, time=time)
    data_he = data_he[icao]
    data_he = convert_to_datestring(data_he, date_header=["year", "month", "day", "hour", "min"])

    return data_he


def convert_to_datestring(X, date_header):
    X[date_header] = X[date_header].astype(str)
    date = [y + mo.zfill(2) + d.zfill(2) + h.zfill(2) + mi.zfill(2)
            for y, mo, d, h, mi in X[date_header].values]
    X = X.drop(date_header, axis=1)
    X.insert(loc=0, column="date", value=date)
    return X


def make_vis_table(init_date: str, end_date: str, METAR=None, AF=None, HE=None, ML=None):
    def __check_index(df):
        if "date" in df.keys():
            idx = df["date"].astype(int).astype(str).values
            df.index = idx
            df = df.drop("date", axis=1)
        return df

    def __append(df, values, kind):
        values = __check_index(values)
        for key in values:
            df["%s" % kind] = values[key]

        return df

    ids = [init_date[0:4], init_date[4:6], init_date[6:8], init_date[8:10], init_date[10:12]]
    ids = [int(d) for d in ids]
    eds = [end_date[0:4], end_date[4:6], end_date[6:8], end_date[8:10], end_date[10:12]]
    eds = [int(d) for d in eds]

    s = datetime.datetime(ids[0], ids[1], ids[2], ids[3], ids[4])
    e = datetime.datetime(eds[0], eds[1], eds[2], eds[3], eds[4])

    days = (e - s).days

    date = [(s + datetime.timedelta(hours=i)).strftime("%Y%m%d%H%M") for i in range(0, 24 * days)]

    vis = pd.DataFrame(index=date)
    if METAR is not None:
        vis = __append(vis, METAR, kind="metar")
    if AF is not None:
        vis = __append(vis, AF, kind="AF")
    if HE is not None:
        vis = __append(vis, HE, kind="human")
    if ML is not None:
        vis = __append(vis, ML, kind="SkyNet")

    return vis


def edit_visibility(vis, v_min=0, key="SkyNet"):
    p = vis[key].values
    p = smoothing(p, ksize=6, threshold=5000)
    p += v_min
    p[p > 8000] = 9999.
    p[p <= 8000] = np.round(p[p <= 8000], -2)
    vis[key] = p.astype(int)
    return vis


def smoothing(x, ksize, threshold=None, cutoff=7000):
    kernel = np.ones(ksize)
    x_sm = 1 / kernel.sum() * np.convolve(x, kernel, mode="same")

    if threshold is not None:
        extend = np.zeros_like(x)
        for i in range(len(x)):
            idx1 = i - int(ksize / 2)
            idx2 = i + int(ksize / 2)
            if x_sm[i] < threshold:
                if idx1 >= 0 and idx2 <= len(x):
                    extend[i] = x_sm[i] * (x[idx1:idx2].min()) / (x_sm[idx1:idx2].min() + 1e-2)
                elif idx1 < 0:
                    extend[i] = x_sm[i] * x[:idx2].min() / (x_sm[:idx2].min() + 1e-2)
                elif idx2 > 0:
                    extend[i] = x_sm[i] * x[idx1:].min() / (x_sm[idx1:].min() + 1e-2)

            else:
                extend[i] = x_sm[i]

        return extend

    return x_sm


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
        icao = "RJFK"

    if args.date is None:
        date = "20180620"

    if args.time is None:
        time = "060000"

    vis_pred = pd.read_csv("%s/live/prediction/%s.csv" % (SKYNET_PATH, icao))
    vis_pred = edit_visibility(vis_pred, v_min=300, key="SkyNet")
    vis_human = daily_human_edit(icao, date, time)[["date", "VIS"]]
    vis_af = daily_area_forecast(icao, date, time)[["date", "VIS"]]

    init_date = date + time
    end_date = datetime.datetime.strftime(
        datetime.datetime.strptime(init_date, "%Y%m%d%H%M%S") + datetime.timedelta(hours=24),
        "%Y%m%d%H%M%S"
    )

    vis = make_vis_table(init_date=init_date, end_date=end_date, AF=vis_af, HE=vis_human, ML=vis_pred)
    vis = vis.fillna("")
    date = vis.index
    date = [datetime.datetime.strptime(d, "%Y%m%d%H%M%S") for d in date]
    vis.insert(loc=0, column="date", value=date)
    vis.columns = [
        [icao, "VIS", "VIS", "VIS"],
        vis.columns
    ]
    print(vis)
    vis.to_html("%s/%s.html" % (LIVE_PATH, icao), index=False)


if __name__ == "__main__":
    main()
