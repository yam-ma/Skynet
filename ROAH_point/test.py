import numpy as np
import pandas as pd
import pygrib
import os

import datetime

#=======================================
#  Main Routine
#=======================================
if __name__ == "__main__":

    # columns[surface]
    # 'ValidityDate/Time': first make as the index, and merge two (surface & upper) data later
    columns = ['Pressure reduced to MSL', 'Pressure'
        , 'u-component of wind', 'v-component of wind', 'Temperature', 'Relative humidity'
        , 'Low cloud cover', 'Medium cloud cover', 'High cloud cover', 'Total cloud cover', 'Total precipitation']


    # indices
    filename = "20150101_030000.000.156"
    fn = filename.split(".")[0]
    fn = fn.split("_")
    fn = fn[0]+fn[1]
    print(fn)

    fn_time = datetime.datetime.strptime(fn, '%Y%m%d%H%M%S')
    print(fn_time)

    #
    # make index from a 1-h sequence from the BaseTime given by filename
    #

    td = datetime.timedelta(hours=1)
    ft = [(fn_time + i*td).strftime('%Y%m%d%H%M%S') for i in range(24)]
    print("ft=", ft)



    df = pd.DataFrame(index=ft, columns=columns)  # empty df

    print(df)

    df.loc['20150101030000', 'Pressure reduced to MSL'] = 9999

    print(df)

    path = "/home/yamada-m/MSM_data/xrKAKsY/400220001/0000911000220001/2015/01/01/20150101_030000.000/"
    file_bt = path.split("/")[-2]
    print(file_bt)
    file_bt = file_bt.split(".")[0]
    file_bt = file_bt.split("_")
    file_bt = file_bt[0] + file_bt[1]
    print(file_bt)


    #
    # survey folders
    #
    years = [str(i) for i in range(2015, 2018)]
    months = [str('{:02}').format(i) for i in range(13)]
    days = [str('{:02}').format(i) for i in range(32)]
    print(months)
    print(days)

    fname = years[1]+months[1]+days[1]+"_030000.000/"
    path = "/home/yamada-m/MSM_data/xrKAKsY/400220001/0000911000220001/"+years[1]+"/"+months[1]+"/"+days[1]+"/"+fname
    print("fname=", fname)
    print("path=", path)

    import os.path
    print(os.path.isdir(path))

    #
    # test icaos position getting
    #
    import nwp2d as npd

    icaos = npd.msm.get_jp_icaos()
    print(icaos)
    latlon_icao = npd.msm.get_airport_latlon(icaos)
    idx_latlon = npd.msm.latlon_to_indices(latlon_icao, layer='surface')
    print(idx_latlon['ROAH'])

    idx_lat, idx_lon = idx_latlon['ROAH']
    print("idx_lat=", idx_lat, "idx_lon=", idx_lon)


    for i in range(1,2):
        print(i)