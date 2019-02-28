import re
import os


def move_nwp_file(input_path, output_path, pattern):
    list_dir = os.listdir(input_path)
    list_dir.sort()
    for d in list_dir:
        if os.path.isdir(input_path + "/" + d):
            if re.match(pattern, d):
                print('mv -f %s/%s %s' % (input_path, d, output_path))
                os.system('mv %s/%s %s/%s' % (input_path, d, output_path, d))
            else:
                move_nwp_file(input_path + "/" + d, output_path, pattern)
        else:
            if not re.match(pattern, d):
                print('remove %s' % d)
                os.system('rm \'%s/%s\'' % (input_path, d))


def convert_dict_construction(old, new: dict, pwd: str, depth: int):
    new = __apply_convert_dict_construction(old, new, pwd)
    keys = list(new.keys())
    skeys = [key.split(pwd) for key in keys]
    tn = []
    for i, d in enumerate(skeys):
        if len(d) > 2:
            tn.append(len(d) - 1 - depth)
        else:
            tn.append(1)

    keys = ["/".join(key[d:]) for d, key in zip(tn, skeys)]

    new = {key: new[n] for key, n in zip(keys, new)}

    return new


def __apply_convert_dict_construction(old, new: dict, pwd: str):
    if type(old) == dict:
        for o in old:
            nkey = pwd
            if type(o) == str:
                if pwd == "/":
                    nkey += o
                else:
                    nkey += "/" + o
            if __check_iterable(old[o]):
                __apply_convert_dict_construction(old[o], new, nkey)
            else:
                if nkey in new.keys():
                    new[nkey].append(old[o])
                else:
                    new[nkey] = [old[o]]
    else:
        for o in old:
            nkey = pwd
            if type(o) == str:
                if pwd == "/":
                    nkey += o
                else:
                    nkey += "/" + o
            if __check_iterable(o):
                __apply_convert_dict_construction(o, new, nkey)

    return new


def __check_iterable(obj):
    if hasattr(obj, "__iter__"):
        if type(obj) == str:
            return False
        else:
            return True
    else:
        return False


def main():
    from skynet import MSM_INFO

    data_path = '/Users/makino/PycharmProjects/SkyCC/data'
    tagid_dirs = os.listdir('%s/legacy' % data_path)
    tagid_dirs.sort()

    for tagid in MSM_INFO:
        if tagid in tagid_dirs:
            bt = MSM_INFO[tagid]['base time']
            fvt = MSM_INFO[tagid]['first validity time']
            lvt = MSM_INFO[tagid]['last validity time']
            layer = MSM_INFO[tagid]['layer']

            input_path = '%s/legacy/%s' % (data_path, tagid)
            output_path = '%s/grib2/MSM/%s/bt%s/vt%s%s' % (data_path, layer, bt, fvt, lvt)

            print(output_path)
            date_dir_list = os.listdir(output_path)
            date_dir_list.sort()
            for date_dir in date_dir_list:
                if os.path.exists('%s/%s/%s' % (output_path, date_dir, date_dir)):
                    # os.system('mv -f %s/%s/%s/* %s/%s/' % (output_path, date_dir, date_dir, output_path, date_dir))
                    os.system('rm -r %s/%s/%s' % (output_path, date_dir, date_dir))
                    # print('mv -f %s/%s/%s/* %s/%s/' % (output_path, date_dir, date_dir, output_path, date_dir))
                    print('rm -r %s/%s/%s' % (output_path, date_dir, date_dir))
                    # print(date_dir)
            # os.makedirs(output_path, exist_ok=True)
            # move_nwp_file(input_path, output_path, pattern=r'\d{8}_\d{6}')


if __name__ == '__main__':
    main()
