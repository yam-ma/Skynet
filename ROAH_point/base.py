import os
import json
import configparser

__config = configparser.ConfigParser()
__config.read('%s/config.ini' % os.path.dirname(os.path.abspath(__file__)))

MY_DIR = __config.get('path', 'my_dir')
USER_DIR = __config.get('path', 'user_dir')
SHARE_DIR = __config.get('path', 'share_dir')
DATA_DIR = __config.get('path', 'data_dir')
MSM_DATA_DIR = __config.get('path', 'msm_dir')

__data = json.load(open('%s/data.json' % os.path.dirname(os.path.abspath(__file__)), 'r'))
MSM_INFO = __data['MSM']
AF_INFO = __data['AF']

MSM_BBOX = (120., 22.4, 150., 47.6,)
MSM_SHAPE = {'surface': (505, 481), 'upper': (253, 241)}

__airport_data = json.load(open('%s/all_airport_data.json' % os.path.dirname(os.path.abspath(__file__)), 'r'))
ICAOS = list(__airport_data['airport'].keys())
AIRPORT_LATLON = {
    icao: {
        'lat': float(__airport_data['airport'][icao][-1]['lat']),
        'lon': float(__airport_data['airport'][icao][-1]['lon'])
    } for icao in ICAOS
}
