import glob
import shutil
import numpy as np
import pandas as pd

from RU import *


def read(icaos, date, time):
    files = glob.glob(DATA_PATH + "/411023800/%s_%s*" % (date, time))
    files.sort()

    files = [f for f in files if json.loads(RU(f).to_json())["ICAO"] in icaos]
    files.sort()
    latest_file = files[-1]

    df_hes = {icao: None for icao in icaos}
    for icao in icaos:
        ru = RU(latest_file)
        he = json.loads(ru.to_json())
        he = convert_dict_construction(he, {}, "/", depth=0)

        fcst_count = he.pop("FCST_count")[0]
        runway_count = he.pop("runway_count")

        for key in he:
            if len(he[key]) == 1:
                he[key] = [he[key][0] for _ in range(fcst_count)]

        runway_no = he.pop("runway_no")
        pop_keys = [key for key in he if len(he[key]) != fcst_count]
        header = ["%s_runway_%s" % (key, rno) for rno in np.unique(runway_no) for key in pop_keys]
        runway = pd.DataFrame(columns=header)
        for key in pop_keys:
            i = 0
            x_runway = he.pop(key)
            for idx, c in enumerate(runway_count):
                h = ["%s_runway_%s" % (key, rno) for rno in runway_no[:c]]
                runway.loc[idx, h] = x_runway[i:i + c]
                i += c
        runway.fillna(0, inplace=True)

        he = pd.DataFrame(he, columns=list(he.keys()))
        df_hes[icao] = pd.concat([he, runway], axis=1)

    return df_hes


def __human_edit_file_arrangement(file_path):
    list_dir = os.listdir(file_path)
    list_dir.sort()
    for d in list_dir:
        if os.path.isdir(file_path + "/" + d):
            __human_edit_file_arrangement(file_path + "/" + d)
        else:
            if os.path.isfile(DATA_PATH + "/411023800/" + d):
                print(d, "already exist.")
            else:
                print(d)
                shutil.copyfile(file_path + "/" + d, DATA_PATH + "/411023800/" + d)
    return


def main():
    # file_arrangement(DATA_PATH + "/411023800",DATA_PATH + "/411023800","411023800")
    date = "20180620"
    time = "060000"

    # files = glob.glob(DATA_PATH + "/411023800/%s_%s*" % (date, time))
    # icaos = [json.loads(RU(f).to_json())["ICAO"] for f in files]
    # icaos = ["RJFT", "RJFK", "RJOT", "RJCC"]
    icaos = ["RJFT"]

    df_hes = read(icaos, date, time)
    print(df_hes["RJFT"])


if __name__ == "__main__":
    main()
