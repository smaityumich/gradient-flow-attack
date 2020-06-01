import sys
import numpy as np
import os
import run_sensr_adult
np.random.seed(1)
seeds = np.random.randint(100000, size = (10, 2))

i = int(float(sys.argv[1]))
seed_data = seeds[i, 0]
seed_model = seeds[i, 1]
print(f'{type(i)} {seed_data} {seed_model}')
run_sensr_adult.fit_model(seed_data, seed_model)
