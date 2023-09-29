import copy


class AverageValue(object):
    """Computes and stores the average and current value"""

    def __init__(self, value_names=None):
        self.value_names = value_names if value_names is not None else [
            'value']
        self.num_value = 1 if value_names is None else len(value_names)
        self.reset()

    def reset(self):
        self._val = {vn: 0 for vn in self.value_names}
        self._sum = {vn: 0 for vn in self.value_names}
        self._count = {vn: 0 for vn in self.value_names}

    def update(self, values):
        if type(values) == dict:
            for vn, v in values.items():
                self._val[vn] = v
                self._sum[vn] += v
                self._count[vn] += 1
        else:
            self._val['value'] = values
            self._sum['value'] += values
            self._count['value'] += 1

    def val(self, vn=None):
        if vn is None:
            return self._val['value'] if self.value_names == ['value'] else copy.deepcopy(self._val)
        else:
            return self._val[vn]

    def count(self, vn=None):
        if vn is None:
            return self._count['value'] if self.value_names == ['value'] else copy.deepcopy(self._count)
        else:
            return self._count[vn]

    def avg(self, vn=None):
        if vn is None:
            return self._sum['value'] / self._count['value'] if self.value_names == ['value'] else {
                vn: self._sum[vn] / self._count[vn] for vn in self.value_names
            }
        else:
            return self._sum[vn] / self._count[vn]

    def avg_str(self, vn=None, format_str='%.4f'):
        avg_vs = self.avg(vn)
        if type(avg_vs) == dict:
            result = ''
            for k, v in avg_vs.items():
                result += ("%s= " + format_str + " ") % (k, v)
            return result
        else:
            return format_str % avg_vs
