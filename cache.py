import pickle
from pathlib import Path
from random import random

import attr

try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    def cache(user_function):
        'Simple lightweight unbounded cache.  Sometimes called "memoize".'
        return lru_cache(maxsize=None)(user_function)


@attr.s
class DiskCache:
    base_dir = attr.ib(converter=Path)
    _local_cache = attr.ib(factory=dict)

    def __attrs_post_init__(self):
        self.base_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _tuple_to_name(t):
        return '_'.join(map(str, t))

    def _tuple_path(self, t):
        return self.base_dir / self._tuple_to_name(t)

    def get(self, t, default=None):
        if res := self._local_cache.get(t):
            return res

        file_path = self._tuple_path(t)
        if not file_path.exists():
            return default

        res = self._tuple_path(t).read_bytes()
        res = pickle.loads(res)

        self._local_cache[t] = res
        return res

    def __setitem__(self, t, value):
        self._local_cache[t] = value
        if random() < .9:
            return
        file_path = self._tuple_path(t)
        if not file_path.exists():
            file_path.write_bytes(pickle.dumps(value))
