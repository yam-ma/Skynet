import numpy as np
import pandas as pd
import pygrib
import os
import datetime
import os.path

import nwp2d as npd

import gc
# from memory_profiler import profile


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
    columns = ['Geop1000','u-co1000','v-co1000','Temp1000','Vert1000','Rela1000'
        ,'Geop975','u-co975','v-co975','Temp975','Vert975','Rela975'
        ,'Geop950','u-co950','v-co950','Temp950','Vert950','Rela950'
        ,'Geop925','u-co925','v-co925','Temp925','Vert925','Rela925'
        ,'Geop900','u-co900','v-co900','Temp900','Vert900','Rela900'
        ,'Geop850','u-co850','v-co850','Temp850','Vert850','Rela850'
        ,'Geop800','u-co800','v-co800','Temp800','Vert800','Rela800'
        ,'Geop700','u-co700','v-co700','Temp700','Vert700','Rela700'
        ,'Geop600','u-co600','v-co600','Temp600','Vert600','Rela600'
        ,'Geop500','u-co500','v-co500','Temp500','Vert500','Rela500'
        ,'Geop400','u-co400','v-co400','Temp400','Vert400','Rela400'
        ,'Geop300','u-co300','v-co300','Temp300','Vert300','Rela300'
        ,'Geop250','u-co250','v-co250','Temp250','Vert250','Rela250'
        ,'Geop200','u-co200','v-co200','Temp200','Vert200','Rela200'
        ,'Geop150','u-co150','v-co150','Temp150','Vert150','Rela150'
        ,'Geop100','u-co100','v-co100','Temp100','Vert100','Rela100'
        ]

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



#=======================================
#  Main Routine
#=======================================
def main():

    #
    # make an empty data frame first (add new lines from a single data file)
    #

    columns = ['Geop1000','u-co1000','v-co1000','Temp1000','Vert1000','Rela1000'
        ,'Geop975','u-co975','v-co975','Temp975','Vert975','Rela975'
        ,'Geop950','u-co950','v-co950','Temp950','Vert950','Rela950'
        ,'Geop925','u-co925','v-co925','Temp925','Vert925','Rela925'
        ,'Geop900','u-co900','v-co900','Temp900','Vert900','Rela900'
        ,'Geop850','u-co850','v-co850','Temp850','Vert850','Rela850'
        ,'Geop800','u-co800','v-co800','Temp800','Vert800','Rela800'
        ,'Geop700','u-co700','v-co700','Temp700','Vert700','Rela700'
        ,'Geop600','u-co600','v-co600','Temp600','Vert600','Rela600'
        ,'Geop500','u-co500','v-co500','Temp500','Vert500','Rela500'
        ,'Geop400','u-co400','v-co400','Temp400','Vert400','Rela400'
        ,'Geop300','u-co300','v-co300','Temp300','Vert300','Rela300'
        ,'Geop250','u-co250','v-co250','Temp250','Vert250','Rela250'
        ,'Geop200','u-co200','v-co200','Temp200','Vert200','Rela200'
        ,'Geop150','u-co150','v-co150','Temp150','Vert150','Rela150'
        ,'Geop100','u-co100','v-co100','Temp100','Vert100','Rela100'
        ]
    df_2015 = pd.DataFrame(index=[], columns=columns)
    df_2016 = pd.DataFrame(index=[], columns=columns)
    df_2017 = pd.DataFrame(index=[], columns=columns)



    #
    # prepare for the folder survey
    #
    years = [str(i) for i in range(2015, 2018)]
    months = [str('{:02}').format(i) for i in range(13)]
    days = [str('{:02}').format(i) for i in range(32)]
    # print(months, days)

    #
    # set ROAH location indices by hand
    #
    # ind_lat = 428
    # ind_lon = 122

    icaos = npd.msm.get_jp_icaos()
    latlon_icao = npd.msm.get_airport_latlon(icaos)
    idx_latlon = npd.msm.latlon_to_indices(latlon_icao, layer='upper')
    # print(idx_latlon['ROAH'])

    ind_lat, ind_lon = idx_latlon['ROAH']
    print("idx_lat=", ind_lat, "idx_lon=", ind_lon)



    #---------------------------------
    #  survey folders and files
    #---------------------------------

    #
    # Can't solve the memory outflow problem: split the loop below by hand. about a half year data can be compiled
    # at once. (Feb.26.2019)
    #
    for i in range(1,3):   # year loop
        #for j in range(13):   # month loop
        for j in range(9,13):
            #for k in range(32):   # day loop
            for k in range(32):
                #
                # get a file list from a directory
                #
                fname = years[i] + months[j] + days[k] + "_030000.000/"
                path = "/home/yamada-m/MSM_data/5OW2Xj0g/400220009/0000911000220009/" + years[i] + "/" + months[j] + "/" + days[k] + "/" + fname
                #path = "/home/yamada-m/MSM_data/xrKAKsY/400220001/0000911000220001/2015/01/01/20150101_030000.000/"
                #path = "/home/yamada-m/MSM_data/E9LCOrP6KQ/400220115/0000911000220115/2015/01/01/20150101_030000.000/"
                print("path=", path, i, j, k)

                del fname
                gc.collect()

                if os.path.isdir(path):   # if the directory exists:
                    print("Directory exists:")
                    files = os.listdir(path)

                    ## for debug
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
                            ##FT.append(ftime)

                        grb = grbs.select(forecastTime=ftime)[0]
                        # print(type(grbs), type(grb))
                        # print(grb)

                        param = grb.parameterName
                        level = grb.level
                        param = param[0:4]+str(level)
                        print("param= ", param)


                        # print("level=", level)

                        mslp = grb.values
                        print(mslp.shape)

                        # grbs.close()

                        # for debug (memory error)
                        del grb
                        gc.collect()

                        #
                        # extract data for the nearest point from a file
                        #
                        dat = mslp[ind_lat, ind_lon]
                        print('dat=')
                        print(dat)

                        # for debug (memory error)
                        del mslp
                        gc.collect()

                        #
                        # Make a column for the data frame
                        #
                        df.loc[btime, param] = dat

                    # end of file loop
                    outname = years[i]+months[j]+days[k]+"_upper_tmp.csv"
                    df.to_csv(outname, index=True)

                    # for debug (memory error)
                    del df  # delete tmp
                    gc.collect()


        #
        # add data rows
        #
        # if i == 0:
        #     df_2015 = pd.concat([df_2015, df], axis=0)
        # elif i == 1:
        #    df_2016 = pd.concat([df_2016, df], axis=0)
        #else:
        #    df_2017 = pd.concat([df_2017, df], axis=0)


    # print(df.columns)

    # print(df_2015.head())
    # print(df[['Pressure', 'Temperature']])

    # df.to_csv("test.csv", index=True)
    # df_2015.to_csv("test_2015.csv", index=True)
    # df_2016.to_csv("test_2016.csv", index=True)
    # df_2017.to_csv("test_2017.csv", index=True)



if __name__ == "__main__":
    main()