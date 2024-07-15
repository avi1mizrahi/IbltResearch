import itertools
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from fpfz import write_to_disk, read_from_disk, calc_recursive_memory_job

k = [
    None,
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
]
d = np.arange(2, 19)
b = 2**np.arange(4, 17)

cache_path = Path('recursive_dict.pkl')
dump = cache_path.with_suffix('.tmp')

workdir = Path('recursive_cache')
workdir.mkdir(parents=True, exist_ok=True)


def chop_to_chunks(it, size):
    it = iter(it)
    while 1:
        x = tuple(itertools.islice(it, size))
        if not x:
            return
        yield x


if __name__ == '__main__':
    rng = np.random.default_rng()
    params = list(itertools.product(b, d, k))
    # rng.shuffle(params)
    # params.reverse()
    jobs = list(chop_to_chunks(params, 10))
    print('#jobs=', len(jobs))

    futures = []
    n_workers = cpu_count()

    with Pool(maxtasksperchild=1) as pool:
        results = pool.starmap(calc_recursive_memory_job, (jobs.pop(), cache_path, workdir))

    cache = dict()
    if cache_path.exists():
        cache = read_from_disk(cache_path)
        print('cache loaded!')

    len_before = len(cache)
    for path in results:
        if path:
            cache.update(read_from_disk(path))

    if len_before < len(cache):
        write_to_disk(dump, cache)
        dump.rename(cache_path)

