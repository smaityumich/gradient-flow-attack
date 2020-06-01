import itertools
import sys
import os
import numpy as np

starts = np.arange(0, 9001, 200)
ends = np.arange(200, 9201, 200)
ends[-1] = 9045
expts = ['sensr', 'baseline_bal', 'reduction']
data_index = range(ends.shape[0])
iteration = range(10)
lrs = [1e-2, 1e-3, 4e-4, 1e-4]


def part_fluc(expt, d, i, lr):
    start = starts[d]
    end = ends[d]
    np.random.seed(1)
    if expt == 'reduction':
        seeds = np.random.randint(100000, size = (10, ))
        seed = seeds[i]
        os.system(f'{expt}/adv_ratio.py {start} {end} {seed} {lr}')
    elif expt == 'baseline_bal':
        seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'{expt}/adv_ratio.py {start} {end} {data_seed} {expt_seed} {lr}')

    else:
        seeds = np.random.randint(100000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'{expt}/adv_ratio.py {start} {end} {data_seed} {expt_seed} {lr}')

