import pygrib
import numpy as np
import pandas as pd
import os

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
                                params = grb.parameterName
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


if __name__== "__main__":
    main()
