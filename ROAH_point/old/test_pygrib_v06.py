import pygrib
import numpy as np
import pandas as pd
import os
import datetime

import nwp2d as npd

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
    fn_time = datetime.datetime.strptime(filename, '%Y%m%d%H%M%S')

    # make index from a 1-h sequence from the BaseTime given by filename
    td = datetime.timedelta(hours=1)
    ft = [(fn_time + i * td).strftime('%Y%m%d%H%M%S') for i in range(24)]


    #
    # make an empty data frame and return it
    #
    df = pd.DataFrame(index=ft, columns=columns)  # empty df

    return df

def main():

    #
    # make an empty data frame first (add new lines from a single data file)
    #

    columns = ['Pressure reduced to MSL','Pressure'
        ,'u-component of wind','v-component of wind','Temperature','Relative humidity'
        ,'Low cloud cover','Medium cloud cover','High cloud cover','Total cloud cover','Total precipitation']
    df_2015 = pd.DataFrame(index=[], columns=columns)
    df_2016 = pd.DataFrame(index=[], columns=columns)
    df_2017 = pd.DataFrame(index=[], columns=columns)

    print(df_2015.head())



    #
    # set ROAH location indices by hand
    #
    # ind_lat = 428
    # ind_lon = 122

    icaos = npd.msm.get_jp_icaos()
    latlon_icao = npd.msm.get_airport_latlon(icaos)
    idx_latlon = npd.msm.latlon_to_indices(latlon_icao, layer='surface')
    # print(idx_latlon['ROAH'])

    ind_lat, ind_lon = idx_latlon['ROAH']
    # print("idx_lat=", idx_lat, "idx_lon=", idx_lon)

    #
    # prepare for the folder survey
    #
    years = [str(i) for i in range(2015, 2018)]
    months = [str('{:02}').format(i) for i in range(13)]
    days = [str('{:02}').format(i) for i in range(32)]

    #
    # Directory loop
    #
    for i in range(3):
        for j in range(13):
            for k in range(32):
                #
                # get a file list from a directory
                #
                fname = years[i] + months[j] + days[k] + "_030000.000/"
                path = "/home/yamada-m/MSM_data/xrKAKsY/400220001/0000911000220001/" + years[i] + "/" + months[j] + "/" + days[k] + "/" + fname

                # print("path=", path)

                if os.path.isdir(path):   # if the directory exists:
                    print("Directory exists:")
                    files = os.listdir(path)

                    file_bt = path.split("/")[-2]
                    file_bt = file_bt.split(".")[0]
                    file_bt = file_bt.split("_")
                    file_bt = file_bt[0] + file_bt[1]
                    print(file_bt)

                    df = makedfFrame(file_bt)
                    print("df")
                    print(df.head())

                    #
                    # File loop
                    #
                    for file in files:
                        if file != "index.html":
                            filename = path+file
                            print("filename=", filename)
                            grbs = pygrib.open(filename)

                            # obtain times and parameter name
                            for grb in grbs:
                                print("params=",grb.parameterName)
                                param = grb.parameterName
                                ftime = grb.forecastTime
                                btime = grb.validDate.strftime('%Y%m%d%H%M%S')

                            slice = grbs.select(forecastTime=ftime)[0]
                            grbs.close()

                            mslp = grb.values

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


        #
        # add data rows
        #
        if i == 0:
            df_2015 = pd.concat([df_2015, df], axis=0)
        elif i == 1:
            df_2016 = pd.concat([df_2016, df], axis=0)
        else:
            df_2017 = pd.concat([df_2017, df], axis=0)


if __name__== "__main__":
    main()
