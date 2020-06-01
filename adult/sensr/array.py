import sys
import numpy as np
import os
np.random.seed(1)
seeds = np.random.randint(100000, size = (10, 2))

i = int(float(sys.argv[1]))
seed_data = seeds[i, 0]
seed_model = seeds[i, 1]
os.system(f"python3 run_sensr_adult.py {seed_data} {seed_model}")