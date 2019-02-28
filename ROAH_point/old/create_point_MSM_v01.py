import numpy as np
import pandas as pd
import pygrib
import os
import datetime


def getNearestIndex(list, num):
    # returns index of a one-dimensional list having closet value to num
    # list: data list
    # num: a value scalar

    idx = np.abs(np.asarray(list)-num).argmin()

    return idx


def makedfFrame(filename):
    #
    # columns
    #
    columns = ['Pressure reduced to MSL','Pressure'
        ,'u-component of wind','v-component of wind','Temperature','Relative humidity'
        ,'Low cloud cover','Medium cloud cover','High cloud cover','Total cloud cover','Total precipitation']

    #
    # indices
    #

    # base time
    fn = filename
    fn_time = datetime.datetime.strptime(fn, '%Y%m%d%H%M%S')

    # make index from a 1-h sequence from the BaseTime given by filename
    td = datetime.timedelta(hours=1)
    ft = [(fn_time + i * td).strftime('%Y%m%d%H%M%S') for i in range(24)]


    #
    # make an empty data frame and return it
    #
    df = pd.DataFrame(index=ft, columns=columns)  # empty df

    return df



#=======================================
#  Main Routine
#=======================================
if __name__ == "__main__":

    #
    # make an empty data frame first (add new lines from a single data file)
    #

    df_base = pd.DataFrame(index=[0], columns=[])

    #
    # get a file list from a directory
    #
    path = "/home/yamada-m/MSM_data/xrKAKsY/400220001/0000911000220001/2015/01/01/20150101_030000.000/"
    #path = "/home/yamada-m/MSM_data/E9LCOrP6KQ/400220115/0000911000220115/2015/01/01/20150101_030000.000/"
    files = os.listdir(path)

    ## for debug
    FT = []

    file_bt = path.split("/")[-2]
    file_bt = file_bt.split(".")[0]
    file_bt = file_bt.split("_")
    file_bt = file_bt[0]+file_bt[1]
    print(file_bt)

    df = makedfFrame(file_bt)
    print("df")
    print(df.head())

    #------------
    # file loop
    #------------
    for file in files:
        #
        # open file
        #
        # filename = "/home/yamada-m/MSM_data/7p_uv7A/400220001/0000911000220001/2015/01/01/20150101_030000.000/20150101_030000.000.156"
        if file != "index.html":
            filename = path+file
            print("filename=", filename)
            grbs = pygrib.open(filename)

            for grb in grbs:
                print("grb", grb)
                print(grb.forecastTime, grb.validityTime, grb.validDate.strftime('%Y%m%d%H%M%S'))
                ftime = grb.forecastTime
                btime = grb.validDate.strftime('%Y%m%d%H%M%S')
                FT.append(ftime)

            grb = grbs.select(forecastTime=ftime)[0]
            # print(type(grbs), type(grb))
            # print(grb)

            param = grb.parameterName
            print("param= ", param)

            level = grb.level
            print("level=", level)

            mslp = grb.values
            print(mslp.shape)

            #
            # extract position of the airport in question
            #

            # obtain lat&lon metric
            lats, lons = grb.latlons()
            # print(lats)
            # print(lons)
            lat = [lats[i,0] for i in range(lats.shape[0])]
            lon = [lons[0,i] for i in range(lons.shape[1])]
            print(lat)
            print(lon)

            # search for closest lat and lon for a starting point
            lat_ROAH = 26.2
            lon_ROAH = 127.7
            ind_lat = getNearestIndex(lat, lat_ROAH)
            ind_lon = getNearestIndex(lon, lon_ROAH)
            print("ind_lat, ind_lon")
            print(ind_lat, ind_lon)
            print(lat[ind_lat], lon[ind_lon])


            #
            # extract data for the nearest point from a file
            #
            dat = mslp[ind_lat, ind_lon]
            print('dat=')
            print(dat)

            #
            # Make a column for the data frame
            #
            df.loc[btime, param] = dat




    print("FT=", FT)
    print(btime)

    print(df.columns)

    print(df.head())
    print(df[['Pressure', 'Temperature']])

    df.to_csv("test.csv", index=True)