def read(icaos, date, time):
    import glob
    import json
    import pandas as pd
    from RU import RU

    files = glob.glob(DATA_PATH + "/402100440/%s_%s*" % (date, time))

    df_metars = {icao: None for icao in icaos}
    for icao in icaos:
        for f in files:
            ru = RU(f)
            metar = json.loads(ru.to_json())
            metar = convert_dict_construction(metar, {}, "/", depth=1)
            metar.pop("station_count")
            if icao in metar["data/station_code"]:
                rvr = __metar_count_data(metar, "rvr")
                wx = __metar_count_data(metar, "wx")
                cloud = __metar_count_data(metar, "cloud")

                metar = pd.DataFrame(metar, columns=list(metar.keys()))
                metar = pd.concat([metar, rvr, wx, cloud], axis=1)
                df_metars[icao] = metar[metar["data/station_code"] == icao]
    return df_metars


def live(icaos, date, time):
    import glob
    import datetime
    import pandas as pd

    files = glob.glob(LIVE_PATH + "/*.html")

    metar = read(icaos, date, time)

    for icao in icaos:
        if metar[icao] is not None:
            for file in files:
                if file[-9:-5] == icao:
                    f = file
            df = pd.read_html(f)[0]
            header = list(df.columns.levels[1][df.columns.labels[1]])
            df = pd.DataFrame(df.values, columns=header)
            df.index = df["date"].values.astype(str)

            header = ["announced_date/year",
                      "announced_date/mon",
                      "announced_date/day",
                      "announced_date/hour",
                      "announced_date/min"]
            date = df["date"]
            vis = pd.DataFrame(index=date, columns=["METAR"])
            y, mo, d, h, mi = tuple(metar[icao][header].values[0])
            metar_date = str(datetime.datetime(y, mo, d, h, mi))
            if metar[icao]["data/visibility"].values[0] == -1:
                metar_vis = "CAVOK"
            else:
                metar_vis = metar[icao]["data/visibility"].values[0]

            if "METAR" in df.keys():
                vis["METAR"] = df["METAR"]
                vis.loc[metar_date, "METAR"] = metar_vis
                df["METAR"] = vis["METAR"]
            else:
                vis.loc[metar_date, "METAR"] = metar_vis
                df["METAR"] = vis["METAR"]

            header = [[icao, "VIS", "VIS", "VIS", "VIS"],
                      ["date", "Area Forecast", "Human Edit", "SkyNet", "METAR"]]
            df = pd.DataFrame(df.values, columns=header)
            df.fillna("", inplace=True)
            df.reset_index(drop=True, inplace=True)

            df.to_html(f, index=False)


def __metar_count_data(metar, ext):
    import pandas as pd
    count = metar.pop("data/%s_count" % ext)
    sl = len(ext)
    exts = [key[sl + 1:] for key in metar.keys() if key[:sl] == ext]
    df_ext = pd.DataFrame(columns=["%s_%d" % (k, c) for c in range(max(count) + 1) for k in exts])
    for key in exts:
        i = 0
        x = metar.pop("%s/%s" % (ext, key))
        for idx, c in enumerate(count):
            h = ["%s_%d" % (key, c) for c in range(c)]
            df_ext.loc[idx, h] = x[i:i + c]
            i += c
    df_ext.fillna(0, inplace=True)
    return df_ext


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

    live([icao], date, time)
    print(icao)


if __name__ == "__main__":
    main()
