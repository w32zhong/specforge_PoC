import json
import time
import torch
import statistics
from collections import defaultdict


class TimeStats():
    def __init__(self, disable=False):
        self.reset()
        self.disable = disable

    def reset(self):
        self._hist = defaultdict(list)
        self._start = defaultdict(float)

    def start(self, key='time'):
        if self.disable: return
        torch.cuda.synchronize()
        self._start[key] = time.perf_counter_ns()

    def stop(self, key='time', verbose=False):
        if self.disable: return
        torch.cuda.synchronize()

        dt = time.perf_counter_ns() - self._start[key]
        dt_ms = dt / 1_000_000
        self._hist[key].append(dt_ms)
        if verbose: print(key, dt_ms, 'ms')

    def f(self, func_name, hist):
        func = getattr(statistics, func_name)
        if func_name == 'stdev' and len(hist) < 2:
            return float('nan')
        else:
            return func(hist)

    def report(self, lst=None):
        use = lambda k: (lst is None or k in lst)
        return json.dumps({
            k: {
                f'cnt': len(self._hist[k]),
                f'sum': sum(self._hist[k]),
                f'max': max(self._hist[k]),
                f'min': min(self._hist[k]),
                f'mean': self.f('mean', self._hist[k]),
                f'stdev': self.f('stdev', self._hist[k]),
            }
            for k in self._hist.keys() if use(k)
        }, indent=2)
