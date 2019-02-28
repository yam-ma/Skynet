import warnings
import re
import numpy as np

from pandas import DataFrame


class NWPFrame(DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, clone=True):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

        if clone:
            self._init_data = None
            self._set_init_data()

    @property
    def init_data(self):
        return self._init_data

    def _set_init_data(self):
        from copy import deepcopy
        self._init_data = deepcopy(self)

    def append(self, other, axis=0, key=None, ignore_index=False, verify_integrity=False, inplace=False, **kwargs):
        from pandas.core.reshape.concat import concat
        if axis == 0:
            new_data = concat([self, DataFrame(other, index=key)], axis=axis)
        else:
            new_data = concat([self, DataFrame(other, columns=key)], axis=axis)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def sync(self, objs, sync_key, inplace=False):
        from copy import deepcopy
        from pandas.core.reshape.concat import concat

        def __set_sync_index():

            try:
                sync_index = new_data[sync_key].astype(int)
            except ValueError:
                sync_index = new_data[sync_key]

            try:
                sync_index = sync_index.astype(str)
            except ValueError:
                sync_index = new_data[sync_key]

            try:
                new_data.index = sync_index.values
            except ValueError:
                raise

            try:
                sync_index = objs[sync_key].astype(int)
            except ValueError:
                sync_index = objs[sync_key]

            try:
                sync_index = sync_index.astype(str)
            except ValueError:
                sync_index = objs[sync_key]

            try:
                objs.index = sync_index.values
            except ValueError:
                raise

        new_data = deepcopy(self)
        __set_sync_index()
        new_data = concat([new_data, objs], axis=1).reset_index(drop=True)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def match_keys(self, pattern):
        return [f for f in self.keys() if re.match(pattern, f)]

    def match_keyargs(self, pattern):
        exts = [f for f in self.keys() if re.match(pattern, f)]
        return [list(self.keys()).index(f) for f in exts]

    def calc_wind_speed(self, inplace=False):
        from pandas.core.reshape.concat import concat
        uco = self.match_keys('u-co')
        vco = self.match_keys('v-co')

        wspd = DataFrame()
        for uh, vh in zip(uco, vco):
            u = self[uh]
            v = self[vh]
            if uh == "u-component of wind":
                wh = "wind speed"
                wspd[wh] = np.sqrt(u ** 2 + v ** 2)

            else:
                wh = "wspd" + uh[4:]
                wspd[wh] = np.sqrt(u ** 2 + v ** 2)

        new_data = concat([self, wspd], axis=1)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def maxcol(self, pattern=r'\w+', inplace=False):
        from pandas.core.reshape.concat import concat
        exts = self.match_keys(pattern)
        ma = DataFrame(self[exts].abs().max(axis=1).values, columns=["max_%s" % pattern])

        new_data = concat([self, ma], axis=1)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def grad(self, axis=0, pattern=r'\w+', inplace=False):
        from pandas.core.reshape.concat import concat
        exts = self.match_keys(pattern)

        new_feats = ['grad_%s' % f for f in exts]
        grad = DataFrame(self[exts].diff(periods=1, axis=axis).values, columns=new_feats)

        new_data = concat([self, grad], axis=1)
        new_data.dropna(axis=axis, inplace=True)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def conv(self, kernel, mode='valid', axis=0, pattern=r'\w+', inplace=False):
        from pandas.core.reshape.concat import concat
        from scipy import signal
        exts = self.match_keys(pattern)

        if axis == 0:
            kernel = kernel.reshape(-1, 1)
        elif axis == 1:
            kernel = kernel.reshape(1, -1)

        conv = signal.convolve2d(self[exts].values, kernel, mode=mode)
        new_feats = ['conv_%s_%s' % (pattern, i) for i in range(conv.shape[1])]
        conv = DataFrame(conv, columns=new_feats)

        new_data = concat([self, conv], axis=1)
        new_data.dropna(axis=axis, inplace=True)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def sine(self, column, period, drop=False, inplace=False):
        from pandas.core.reshape.concat import concat
        val = self[column].astype(int).values
        s = DataFrame(np.cos(2 * np.pi * val / period), columns=['sine_%s' % column])

        if drop:
            self.drop(column, axis=1, inplace=True)

        new_data = concat([self, s], axis=1)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def merge_strcol(self, merged_columns, new_column, sep='', drop=False, inplace=False):
        from pandas.core.reshape.concat import concat
        objs = self[merged_columns].astype(str).values
        merged = DataFrame([sep.join(list(obj)) for obj in objs], columns=[new_column])

        if drop:
            self.drop(merged_columns, axis=1, inplace=True)

        new_data = concat([self, merged], axis=1)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def split_strcol(self, split_column, new_columns, pattern='', drop=False, inplace=False):
        from pandas.core.reshape.concat import concat
        objs = self[split_column].astype(str).values
        split = DataFrame([re.split(pattern, obj) for obj in objs])

        if len(new_columns) == split.shape[1]:
            split.columns = new_columns
        else:
            warnings.warn('new_columns length is not equal split-DataFrame shape[1]')

        if drop:
            self.drop(split_column, axis=1, inplace=True)

        new_data = concat([self, split], axis=1)

        if inplace:
            self._update_inplace(new_data)
        else:
            return new_data

    def strtime_to_datetime(self, date_key, fmt, inplace=False):
        import datetime
        try:
            date = self[date_key].astype(int)
        except ValueError:
            date = self[date_key]

        try:
            date = date.astype(str)
        except ValueError:
            date = self[date_key]

        if inplace:
            self[date_key] = [datetime.datetime.strptime(d, fmt) for d in date]
        else:
            return [datetime.datetime.strptime(d, fmt) for d in date]

    def datetime_to_strtime(self, date_key, fmt, inplace=False):
        if inplace:
            self[date_key] = [d.strftime(fmt) for d in self[date_key]]
        else:
            return [d.strftime(fmt) for d in self[date_key]]


def main():
    from skynet.datasets.learning_data import read_learning_data

    path = '/Users/makino/PycharmProjects/SkyCC/data/skynet/test_RJCC.pkl'
    data = read_learning_data(path)

    ndf = NWPFrame(data)
    ndf.calc_wind_speed(inplace=True)
    ndf.maxcol(pattern='Rela', inplace=True)
    ndf.grad(axis=1, pattern='u-co', inplace=True)
    ndf.conv(kernel=np.ones((1, 3)) / 3, axis=1, pattern='u-co', inplace=True)
    ndf.strtime_to_datetime(date_key='date', fmt="%Y%m%d%H%M")
    ndf.datetime_to_strtime(date_key='date', fmt="%Y-%m-%d %H:%M")
    ndf.split_strcol('date', new_columns=['year', 'month', 'day', 'hour', 'min'], pattern=r'[-\s:]', inplace=True)
    ndf.sine(column='hour', period=24)


if __name__ == '__main__':
    main()
