import numpy as np
import pandas as pd

from RU import *


def read(icaos, date, time):
    files = os.listdir(DATA_PATH + "/411023118")
    fidx = np.array([int(t[9:15]) - int(time) for t in files]).argmin()

    f = DATA_PATH + "/411023118/%s" % files[fidx]

    ru = RU(f)
    af = json.loads(ru.to_json())
    area_count = __get_icao_count(af, icaos)

    df_afs = {icao: None for icao in icaos}

    for icao in icaos:
        area = af["area"][area_count[icao]]
        area = convert_dict_construction(area, {}, "/", depth=0)

        area.pop("AREA")
        area.pop("ICAO")
        area.pop("FCST_count")

        wx_count = area.pop("WX_count")
        wx_telop = area.pop("WX_telop")
        wx_prob = area.pop("WX_prob")

        telop = [100, 200, 300, 340, 400, 430, 500, 600, 610]
        wx = pd.DataFrame(columns=["WX_telop_%d" % t for t in telop])

        i = 0
        for idx, c in enumerate(wx_count):
            telop = ["WX_telop_%d" % t for t in wx_telop[i:i + c]]
            wx.loc[idx, telop] = wx_prob[i:i + c]
            i += c
        wx.fillna(0, inplace=True)

        area = pd.DataFrame(area, columns=list(area.keys()))
        df_afs[icao] = pd.concat([area, wx], axis=1)

    return df_afs


def __get_icao_count(af, icaos):
    return {area["ICAO"]: i for i, area in enumerate(af["area"]) if area["ICAO"] in icaos}


def main():
    date = "20180620"
    time = "060000"

    # icaos = [area["ICAO"] for area in af["area"] if area["ICAO"][:2] == "RJ"]
    # icaos = ["RJFT", "RJFK", "RJOT", "RJCC"]
    icaos = ["RJFT"]
    icaos.sort()

    df_afs = read(icaos, date, time)
    print(df_afs["RJFT"])


if __name__ == "__main__":
    main()
