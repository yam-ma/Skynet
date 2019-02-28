import pickle


def get_init_features(code='long'):
    from skynet import MSM_INFO
    fs = [
        'date',
        'Pressure reduced to MSL',
        'Pressure',
        'u-component of wind',
        'v-component of wind',
        'Temperature',
        'Relative humidity',
        'Low cloud cover',
        'Medium cloud cover',
        'High cloud cover',
        'Total cloud cover',
        'Total precipitation',
        'Geop1000',
        'u-co1000',
        'v-co1000',
        'Temp1000',
        'Vert1000',
        'Rela1000',
        'Geop975',
        'u-co975',
        'v-co975',
        'Temp975',
        'Vert975',
        'Rela975',
        'Geop950',
        'u-co950',
        'v-co950',
        'Temp950',
        'Vert950',
        'Rela950',
        'Geop925',
        'u-co925',
        'v-co925',
        'Temp925',
        'Vert925',
        'Rela925',
        'Geop900',
        'u-co900',
        'v-co900',
        'Temp900',
        'Vert900',
        'Rela900',
        'Geop850',
        'u-co850',
        'v-co850',
        'Temp850',
        'Vert850',
        'Rela850',
        'Geop800',
        'u-co800',
        'v-co800',
        'Temp800',
        'Vert800',
        'Rela800',
        'Geop700',
        'u-co700',
        'v-co700',
        'Temp700',
        'Vert700',
        'Rela700',
        'Geop600',
        'u-co600',
        'v-co600',
        'Temp600',
        'Vert600',
        'Rela600',
        'Geop500',
        'u-co500',
        'v-co500',
        'Temp500',
        'Vert500',
        'Rela500',
        'Geop400',
        'u-co400',
        'v-co400',
        'Temp400',
        'Vert400',
        'Rela400',
        'Geop300',
        'u-co300',
        'v-co300',
        'Temp300',
        'Vert300',
        'Rela300'
    ]
    if code == 'long':
        return fs
    elif code == 'wni':
        wni_code = {
            'surface': [
                'SSPRSS',
                'ARPRSS',
                'UWIND',
                'VWIND',
                'AIRTMP',
                'RHUM',
                'LOWCLD',
                'MIDCLD',
                'UPRCLD',
                'AMTCLD',
                'PRCRIN_1HOUR',
                'SOLRAD'
            ],
            'upper':
                ['GPHGT',
                 'UWIND',
                 'VWIND',
                 'AIRTMP',
                 'ASCCRR',
                 'RHUM'

                 ]
        }

        for layer in wni_code:
            params_long = MSM_INFO['parameter'][layer]
            params_wni = wni_code[layer]

            if layer == 'surface':
                for p_long, p_wni in zip(params_long, params_wni):
                    old_param = p_long
                    new_param = 'SFC-' + p_wni
                    fs = [f.replace(old_param, new_param) for f in fs]

            if layer == 'upper':
                for l in MSM_INFO['level'][layer]:
                    for p_long, p_wni in zip(params_long, params_wni):
                        old_param = p_long[:4] + l
                        new_param = l + '-' + p_wni
                        fs = [f.replace(old_param, new_param) for f in fs]

        return fs


def get_init_response():
    r = ["visibility_rank"]
    return r


def get_init_vis_level():
    vis_level = {0: 0, 1: 800, 2: 1600, 3: 2600, 4: 3600, 5: 4800, 6: 6000, 7: 7400, 8: 8800}
    return vis_level


def read_learning_data(path):
    features = get_init_features()
    response = get_init_response()
    data = pickle.load(open(path, "rb"))
    data = data[features + response].reset_index(drop=True)
    return data


def main():
    fets = get_init_features('wni')
    print(fets)


if __name__ == '__main__':
    main()
