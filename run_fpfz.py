"""
This file is only for making sure that the interpreter is reset between executions.
It could have been implemented as a call to fpfz file, or even do this loop there,
but I don't want to realize in the future that there were memory leaks or whatever.
"""

import os

for m in range(3, 1 + 5):
    for k in range(1, 1 + m // 2):
        for n in range(1, 1 + min(2 ** m, 5)):
            os.system(f'python fpfz.py -n {n} -m {m} -k {k} --skip-exist --skip-combs-gt 1000000000')
