import itertools
import sys
import os
import numpy as np

def part_summary(args):
    expt, i, lr = args
    np.random.seed(1)
    if expt == 'reduction':
        seeds = np.random.randint(100000, size = (10, ))
        seed = seeds[i]
        os.system(f'python3 ./{expt}/summary.py {seed} {lr}')
    elif expt == 'baseline_bal':
        seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')

    else:
        seeds = [20609, 49100, 21440, 98448, 31228,36230, 45413,35730, 50057,73349,51668,64395, 5192,98047, 7751,43757, 97561,21758,98539,77708]
        data_seed = seeds[2*i]
        expt_seed = seeds[2*i+ 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')


if __name__ == '__main__':
    
    expts = ['sensr'] 
    iteration = range(10)
    lrs = [1e-2, 1e-3, 4e-4, 1e-4]

    a = itertools.product(expts, iteration, lrs)
    b = [i for i in a]
    i = int(sys.argv[1])
    part_summary(b[i])





