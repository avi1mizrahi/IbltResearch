from itertools import starmap, repeat, product
from math import ceil
from multiprocessing import Pool
from pathlib import Path

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

from random import getrandbits

import numpy as np
import pandas as pd

from iblt.pyblt import PYBLT as Iblt

out_file = Path('IBLT_decoding_stats.csv')
work_dir = Path('IBLT_decoding_stats_workdir')

m = 2 ** 13
t = 100000


@cache
def old_get_optimal_iblt_params_df():
    df = pd.read_csv('iblt/param.export.csv', index_col=0)
    assert df['p'].nunique() == 1
    return df


@cache
def get_optimal_iblt_params_df():
    return (pd
            .read_csv('iblt/iblt_params.csv')
            .pivot_table(index=['items', 'size', 'keys'], values=['p'], aggfunc=np.mean)
            .pivot_table(index=['items', 'size'], values=['p'], aggfunc=np.min)
            )


@cache
def fetch_optimal_iblt_size_p(n_items, iblt_df=None):
    if iblt_df is None:
        iblt_df = old_get_optimal_iblt_params_df()

    return iblt_df.loc[n_items, 'size'], iblt_df.loc[n_items, 'p']


def empirical_rate(n):
    if n > 200:  # TODO: this is a special case to save time with the given params
        return n, 0

    Iblt.set_parameter_filename('iblt/param.export.csv')
    success = 0
    for _ in range(t):
        iblt = Iblt(entries=1, value_size=16, ibltSize=m)
        for item in np.random.randint(0, 2 ** 16, n):
            iblt.insert(item, item)
        success += int(iblt.list_entries()[0])

    return n, success / t


def try_decode(iblt, log_n, N):
    # This is a bug, it can be that an element is inserted twice
    for _ in range(N):
        iblt.insert(getrandbits(log_n))
    success = len(iblt.peel()) == N
    iblt.clear()
    return success


def calc_successes(r, k, log_n, N, reps):
    m = ceil(N*r)
    iblt = Iblt(value_size=0, num_hashes=k, m=m)
    res = m, k, log_n, N, reps, sum(starmap(try_decode, repeat((iblt, log_n, N), reps)))
    del iblt
    (work_dir / '_'.join(map(str, res))).touch()
    return res


if __name__ == "__main__":
    work_dir.mkdir(parents=True, exist_ok=True)

    # ms = np.unique(np.logspace(3, 10, num=20, base=2, dtype=int))
    rs = np.linspace(1.34, 1.7, 100)
    ks = [4,5,6]
    ns = np.arange(32, 32+1)  # log
    Ns = np.logspace(1, 4, num=20, base=10, dtype=int) # 2 ** np.arange(3, 14)
    reps = [2*10**4]

    params = list(product(rs, ks, ns, Ns, reps))
    print('param space: ', len(params))
    np.random.shuffle(params)

    with Pool(maxtasksperchild=1) as pool:
        print('starting pool')
        rates = pool.starmap(calc_successes, params, chunksize=4)

    print('finished, writing results')
    df = pd.DataFrame(rates, columns=['m', 'k', 'log_n', 'N', 'reps', 'n_success'])
    df.to_csv(out_file, index=False)
