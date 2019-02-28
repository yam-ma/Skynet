import os
import glob
import json
import pygrib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.basemap import Basemap

from skynet import DATA_PATH, OUTPUT_PATH

GRIB = {
    "surface": {
        "tag_id": {
            "FT_0-15": "400220001",
            "FT_16-33": "400220115"
        },
        'Pressure reduced to MSL': 'Pressure reduced to MSL',
        'Pressure': 'Pressure',
        'Wind speed': 'Wind speed',
        'Wind direction': 'Wind direction',
        'u-component of wind': 'u-component of wind',
        'v-component of wind': 'v-component of wind',
        'Temperature': 'Temperature',
        'Relative humidity': 'Relative humidity',
        'Low cloud cover': 'Low cloud cover',
        'Medium cloud cover': 'Medium cloud cover',
        'High cloud cover': 'High cloud cover',
        'Total cloud cover': 'Total cloud cover',
        'Total precipitation': 'Total precipitation',
        'Downward short-wave radiation flux': 'Downward short-wave radiation flux'
    },
    "upper": {
        "tag_id": {
            "FT_0-15": "400220009",
            "FT_16-33": "400220123"
        },
        'Geopotential height': 'Geopotential height',
        'Wind speed': 'Wind speed',
        'Wind direction': 'Wind direction',
        'u-component of wind': 'u-component of wind',
        'v-component of wind': 'v-component of wind',
        'Temperature': 'Temperature',
        'Vertical velocity': 'Vertical velocity [pressure]',
        'Relative humidity': 'Relative humidity'

    }
}


def read_airports(layer, icaos, date):
    msm_dir = {"surface": ["400220001", "400220115"], "upper": ["400220009", "400220123"]}

    if layer == "surface":
        file_pair = __get_file_pair(dir_pair=msm_dir[layer], date=date)

    elif layer == "upper":
        file_pair = __get_file_pair(dir_pair=msm_dir[layer], date=date)

    else:
        raise Exception("layer must be 'surface' or 'upper'.")

    df_grbs = {icao: pd.DataFrame() for icao in icaos}

    latlon = __get_airport_latlon(icaos=icaos)
    idx_latlon = __convert_latlon_to_indices(latlon, layer=layer)

    for icao in icaos:
        for f in file_pair:
            grbs = pygrib.open(f)
            for grb in grbs:
                ft = grb.forecastTime
                if layer == "surface":
                    date = grb.validDate.strftime("%Y%m%d%H%M")
                    df_grbs[icao].loc[ft, "date"] = date
                pn = grb.parameterName
                if layer == "upper":
                    pn = pn[:4] + str(grb.level)
                lat = idx_latlon[icao][0]
                lon = idx_latlon[icao][1]
                df_grbs[icao].loc[ft, pn] = grb.values[lat, lon]

    return df_grbs


def concat_surface_upper(surface, upper):
    icaos = list(surface.keys())
    df_grbs = {icao: None for icao in icaos}
    for icao in icaos:
        df_grbs[icao] = pd.concat([surface[icao], upper[icao]], axis=1)

    return df_grbs


def read(layer, date, time):
    print(date)


def plot_forecast_map(file_path, layer, params, forecast_time, level=None, alpha=1., show=True, save_path=None):
    grbs = pygrib.open(file_path)

    lat1 = 22.4
    lat2 = 47.6
    lon1 = 120
    lon2 = 150

    if hasattr(forecast_time, "__iter__"):
        fcst = forecast_time
    else:
        fcst = [forecast_time]

    for ft in fcst:
        fig = plt.figure(figsize=(12, 12))
        fig.add_subplot()
        fig.subplots_adjust(top=1, bottom=0., right=1.0, left=0.)

        m = Basemap(projection="cyl", resolution="l", llcrnrlat=lat1, urcrnrlat=lat2, llcrnrlon=lon1, urcrnrlon=lon2)
        # m.drawcoastlines(color='lightgray')
        # m.drawcountries(color='lightgray')
        # m.fillcontinents(color="white", lake_color="white")
        # m.drawmapboundary(fill_color="white")
        m.bluemarble()

        if type(params) == str:
            params = [params]

        for param in params:
            if layer == "surface":
                if param in ["Wind speed", "Wind direction"]:
                    grb = grbs.select(forecastTime=ft,
                                      parameterName=["u-component of wind", "v-component of wind"])
                else:
                    grb = grbs.select(forecastTime=ft, parameterName=param)
                level = "surface"
            else:
                if param in ["Wind speed", "Wind direction"]:
                    grb = grbs.select(forecastTime=ft,
                                      parameterName=["u-component of wind", "v-component of wind"],
                                      level=level)
                else:
                    grb = grbs.select(forecastTime=ft, parameterName=param, level=level)

            lats, lons = grb[0].latlons()

            if param == "Wind direction":
                u = grb[0].values[::10, ::10].ravel()
                v = grb[1].values[::10, ::10].ravel()

                x = lons[::10, ::10].ravel()
                y = lats[::10, ::10].ravel()

                plt.quiver(x, y, u, v, color="lightgray")

            elif param == "Wind speed":
                u = grb[0].values
                v = grb[1].values

                val = np.sqrt(u ** 2 + v ** 2)

                interval = np.arange(val.min(), val.max())
                mi = np.trunc(val.min()).astype(int)
                ma = np.ceil(val.max()).astype(int)
                delta = int((ma - mi) / 5)
                ticks = np.arange(mi, ma, delta)
                plt.contourf(lons, lats, val, interval, latlon=True, cmap="jet", alpha=alpha)

                if len(params) == 1:
                    m.colorbar(location="bottom", ticks=ticks)

            elif param == "Total precipitation":
                val = grb[0].values

                interval = np.arange(1, 160)
                ticks = range(0, 160, 20)
                plt.contourf(lons, lats, val, interval, latlon=True, cmap="jet", alpha=alpha)

                if len(params) == 1:
                    m.colorbar(location="bottom", ticks=ticks)

            elif param in ["Pressure reduced to MSL", "Pressure"]:
                val = grb[0].values

                m.contour(lons, lats, val, latlon=True, cmap="jet")

                if len(params) == 1:
                    m.colorbar(location="bottom")
            else:
                val = grb[0].values

                interval = np.arange(val.min(), val.max())
                mi = np.trunc(val.min()).astype(int)
                ma = np.ceil(val.max()).astype(int)
                delta = int((ma - mi) / 5)
                ticks = np.arange(mi, ma, delta)
                plt.contourf(lons, lats, val, interval, latlon=True, cmap="jet", alpha=alpha)

                if len(params) == 1:
                    m.colorbar(location="bottom", ticks=ticks)

        plt.title("%s : level = %s : FT = %d" % (",".join(params), level, ft), fontsize=16)

        if save_path is not None:
            plt.savefig("%s/FT%02d.png" % (save_path, ft))

    if show:
        plt.show()


def animate_forecast_map(save_file, date, time, layer, params, level,
                         alpha=1., show=False, cache=True):
    os.makedirs(OUTPUT_PATH + "/image/tmp", exist_ok=True)

    if not cache:
        files = glob.glob(OUTPUT_PATH + "/image/tmp/FT*")
        for f in files:
            os.remove(f)
        if layer == "surface":
            if "Total precipitation" in params:
                fts = range(15), range(15, 33)
            else:
                fts = range(16), range(16, 34)
            tids = GRIB["surface"]["tag_id"]["FT_0-15"], GRIB["surface"]["tag_id"]["FT_16-33"]
        else:
            fts = range(0, 16, 3), range(18, 34, 3)
            tids = GRIB["upper"]["tag_id"]["FT_0-15"], GRIB["upper"]["tag_id"]["FT_16-33"]

        for tid, ft in zip(tids, fts):
            plot_forecast_map(
                file_path='%s/%s/%s_%s.grib2' % (DATA_PATH, tid, date, time),
                layer=layer, params=params, forecast_time=ft, level=level, alpha=alpha,
                show=False, save_path=OUTPUT_PATH + "/image/tmp"
            )
            for i in range(len(ft)):
                plt.close()

    files = glob.glob(OUTPUT_PATH + "/image/tmp/FT*")
    files.sort()

    fig = plt.figure(figsize=(6, 6))
    fig.add_subplot()
    fig.subplots_adjust(top=1., bottom=0., right=1.0, left=0.)
    imgs = []

    for f in files:
        img = plt.imread(f)

        plt.axis("off")
        imgs.append([plt.imshow(img)])

    ani = animation.ArtistAnimation(fig, imgs, interval=100)
    ani.save(save_file, writer="ffmpeg")

    if show:
        plt.show()


def __get_icao():
    airports_info = json.load(open(DATA_PATH + "/all_airport_data.json"))
    icaos = airports_info["airport"].keys()
    icaos = [icao for icao in icaos if icao[:2] == "RJ"]
    icaos.sort()

    lat1_grb = 22.4
    lat2_grb = 47.6
    lon1_grb = 120.
    lon2_grb = 150.

    latlon = {icao: (float(airports_info["airport"][icao][-1]["lat"]),
                     float(airports_info["airport"][icao][-1]["lon"]))
              for icao in icaos}

    icaos = {icao: (latlon[icao][0], latlon[icao][1]) for icao in icaos
             if (lat1_grb <= latlon[icao][0]) and (latlon[icao][0] <= lat2_grb)
             and (lon1_grb <= latlon[icao][1]) and (latlon[icao][1] <= lon2_grb)}

    return icaos


def __get_airport_latlon(icaos):
    airports_info = json.load(open(DATA_PATH + "/all_airport_data.json"))
    latlon = {icao: (float(airports_info["airport"][icao][-1]["lat"]),
                     float(airports_info["airport"][icao][-1]["lon"]))
              for icao in icaos}

    return latlon


def __convert_latlon_to_indices(latlon, layer):
    icaos = list(latlon.keys())

    lat1_grb = 22.4
    lat2_grb = 47.6
    lon1_grb = 120.
    lon2_grb = 150.

    if layer == "surface":
        n_lats = 505
        n_lons = 481
    else:
        n_lats = 253
        n_lons = 241

    icaos = {icao: (latlon[icao][0], latlon[icao][1]) for icao in icaos
             if (lat1_grb <= latlon[icao][0]) and (latlon[icao][0] <= lat2_grb)
             and (lon1_grb <= latlon[icao][1]) and (latlon[icao][1] <= lon2_grb)}

    idx_latlon = {icao: (round(n_lats * (latlon[icao][0] - lat1_grb) / (lat2_grb - lat1_grb)),
                         round(n_lons * (latlon[icao][1] - lon1_grb) / (lon2_grb - lon1_grb)))
                  for icao in icaos}

    return idx_latlon


def __get_file_pair(dir_pair, date):
    f1 = glob.glob(DATA_PATH + "/" + dir_pair[0] + "/%s*" % date)
    f2 = glob.glob(DATA_PATH + "/" + dir_pair[1] + "/%s*" % date)
    if len(f1) > 0 and len(f2) > 0:
        f1_tree = f1[0].split("/")
        f2_tree = f2[0].split("/")
        if f1_tree[-1][:8] == f2_tree[-1][:8]:
            file_pair = (f1[0], f2[0])
        else:
            raise Exception("Date of surface file is not different from upper layer file.")

    elif len(f1) == 0 and len(f2) > 0:
        raise Exception("There is not surface layer file.")
    elif len(f1) > 0 and len(f2) == 0:
        raise Exception("There is not upper layer file.")
    else:
        raise Exception("There are not surface and upper layer file.")

    return file_pair


def main():
    date = "20180704"
    time = "030000"

    # icaos = __get_icao()
    # icaos = ["RJFT", "RJFK", "RJOT", "RJCC"]
    icaos = ["RJFK"]

    # ポイントデータ抽出
    df_sf_grbs = read_airports(layer="surface", icaos=icaos, date=date)
    df_ul_grbs = read_airports(layer="upper", icaos=icaos, date=date)

    df_grbs = concat_surface_upper(df_sf_grbs, df_ul_grbs)

    print(df_grbs["RJFK"])

    # 面データ表示
    layer = "surface"
    params = [GRIB[layer]["Relative humidity"], GRIB[layer]["Wind direction"]]
    level = "surface"

    forecast_time = range(1)
    plot_forecast_map(
        file_path='/Users/makino/PycharmProjects/SkyCC/data/tss_sky_ml/%s/20180704_030000.grib2'
                  % GRIB[layer]["tag_id"]["FT_0-15"],
        layer=layer,
        params=params,
        forecast_time=forecast_time,
        level=level,
        alpha=0.5,
        show=True,
        save_path=None
    )

    """
    animate_forecast_map(OUTPUT_PATH + "/movie/%s_%s_%s_%s.mp4" % (params[0], level, date, time),
                         date, time, layer, params, level,
                         alpha=1., show=True, cache=False)
    """


if __name__ == "__main__":
    main()
