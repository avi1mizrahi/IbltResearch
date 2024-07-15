import multiprocessing
from fractions import Fraction
from itertools import product, starmap

import numpy as np

from model_search import Search

if __name__ == '__main__':
    items = range(1, 20)

    nominators = np.arange(1, 100)
    denominators = nominators + 1

    probabilities = starmap(Fraction, zip(nominators, denominators))

    searches = starmap(Search, product(items, probabilities))

    with multiprocessing.Pool() as p:
        p.map(Search.run, searches)
