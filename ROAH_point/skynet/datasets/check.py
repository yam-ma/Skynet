def check_opengirb2():
    import glob
    import pygrib
    from skynet import MSM_INFO, MSM_DATA_DIR
    tagid_list = [tagid for tagid in MSM_INFO.keys() if re.match(r'4002200', tagid)]
    tagid_list.sort()

    for tagid in tagid_list:
        meta = MSM_INFO[tagid]

        layer = meta['layer']

        path = '%s/%s/bt%s/vt%s%s' % (
            MSM_DATA_DIR,
            layer,
            meta['base time'],
            meta['first validity time'],
            meta['last validity time']
        )

        path_list = glob.glob('%s/2017*' % path)
        path_list.sort()

        for p in path_list:
            msm_files = glob.glob('%s/201*' % p)
            for f in msm_files:
                grbs = pygrib.open(f)
                grbs.select()
                print(f)
                grbs.close()


def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from skynet.nwp2d import msm
    from skynet import DATA_DIR, MSM_BBOX, AIRPORT_LATLON

    # jp_icaos = msm.get_jp_icaos()
    jp_icaos = [
        'RJOT',
        'RJAA',
        'RJSC',
        'RJSI',
        'RJSK',
        'RJSM',
        'RJSN',
        'RJSS',
        'RJTT',
        'ROAH',
        'RJOC',
        'RJOO',
        # 'RJBB',
        'RJCC',
        'RJCH',
        'RJFF',
        'RJFK',
        'RJGG',
        'RJNK',
        'RJOA',
    ]

    icao = 'RJAA'
    X = pd.read_csv('%s/msm_airport/%s.csv' % (DATA_DIR, icao))

    X.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    X.index = X['date'].values
    X.sort_index(inplace=True)

    lon1, lat1, lon2, lat2 = MSM_BBOX
    m = Basemap(projection="cyl", resolution="l", llcrnrlat=lat1, urcrnrlat=lat2, llcrnrlon=lon1, urcrnrlon=lon2)
    m.drawcoastlines(color='lightgray')
    m.drawcountries(color='lightgray')
    # m.fillcontinents(color="white", lake_color="white")
    m.drawmapboundary(fill_color="white")

    lon_ap, lat_ap = AIRPORT_LATLON[icao]['lon'], AIRPORT_LATLON[icao]['lat']
    plt.scatter(lon_ap, lat_ap)

    plt.show()


if __name__ == '__main__':
    main()
