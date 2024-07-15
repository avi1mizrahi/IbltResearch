from itertools import starmap, repeat, product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

out_file = Path('hash_collision.csv')
work_dir = Path('hash_collision_workdir')


def one(rng, n, N):
    return (np.unique(rng.integers(0, high=n, size=N), return_counts=True)[1]>1).any()


def calc_successes(n, N, reps):
    rng = np.random.default_rng()
    res = n, N, reps, sum(starmap(one, repeat((rng, n, N), reps)))
    (work_dir / '_'.join(map(str, res))).touch()
    return res


if __name__ == "__main__":
    work_dir.mkdir(parents=True, exist_ok=True)

    ns = [
        # 8,
        16,
        32,
    ]  # log
    ns = list(set.union(*[set(np.geomspace(3, 2**l, num=1024, dtype=int)) for l in ns]))
    Ns = np.arange(3, 100)
    reps = [10**5]

    params = list(product(ns, Ns, reps))
    print('param space: ', len(params))
    np.random.shuffle(params)

    with Pool(maxtasksperchild=1) as pool:
        print('starting pool')
        rates = pool.starmap(calc_successes, params, chunksize=4)

    print('finished, writing results')
    df = pd.DataFrame(rates, columns=['n', 'N', 'reps', 'n_fail'])
    df.to_csv(out_file, index=False)
