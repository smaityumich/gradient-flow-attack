import os
import numpy as np
import itertools
 

job_file = 'submit.sbat'

np.random.seed(1)
seeds = np.random.randint(100000, size = (10, 2))
np.save('seeds.npy', seeds)




#os.system('touch summary/adult7.out')



for i in range(10):
    os.system(f'touch {job_file}')
    seed_data = seeds[i, 0]
    seed_model = seeds[i, 1]
        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=adult.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=1\n')
        fh.writelines('#SBATCH --mem-per-cpu=4gb\n')
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --account=stats_dept1\n")
        fh.writelines("#SBATCH --mail-type=NONE\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 run_sensr_adult.py {seed_data} {seed_model}")


    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')