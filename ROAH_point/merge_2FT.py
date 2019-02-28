import pandas as pd
import numpy as np

def merge_two(fname1, fname2):
    dat1 = pd.read_csv(fname1, index_col=0)
    dat2 = pd.read_csv(fname2, index_col=0)

    dat = pd.concat([dat1, dat2], axis=0)

    # sort index
    dat.sort_index(inplace=True)
    dat.index = dat.index.set_names(['ValidityDate/Time'])
    dat0 = dat.reset_index()

    return dat0


if __name__ == "__main__":

    years = range(2015,2018)
    levels = ["surface", "upper"]

    #
    # merge two FT-sets
    #
    for level in levels:
        for year in years:

            fname1 = "df_"+str(year)+"_"+level+"_ROAH.csv"
            fname2 = "df_"+str(year)+"_"+level+"_ROAH2.csv"

            dat0 = merge_two(fname1, fname2)

            print(dat0.head())

            fname = str(year)+level+"_ROAH.csv"
            print(fname)
            dat0.to_csv(fname, index=False)



    #
    # merge upper and surface
    #
    for year in years:
        fname1 = str(year)+levels[0]+"_ROAH.csv"
        fname2 = str(year)+levels[1] + "_ROAH.csv"

        dat1 = pd.read_csv(fname1)
        dat2 = pd.read_csv(fname2)

        dat = pd.merge(dat1,dat2, on="ValidityDate/Time", how='left')

        # dat.to_csv(str(year)+"_ROAH.csv", index=False)

        #
        # write out
        #

        # read header first
        with open("head.txt") as f:
            l = f.readlines()

        print(l)

        oname = str(year)+"_ROAH.csv"
        # write header
        with open(oname, mode="w") as f:
            for line in l:
                print(line[0:-1], file=f)

        dat.to_csv(oname, mode="a", header=False, index=False)
