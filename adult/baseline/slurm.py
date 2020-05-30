import os
import numpy as np
import itertools
 

job_file = 'submit.sbat'

# Experiment 1
start = np.arange(0, 9001, 200)
end = np.arange(200, 9201, 200)
end[-1] = 9045


#os.system('touch summary/adult7.out')



for s, e in zip(start, end):
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=adult.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=1\n')
        fh.writelines('#SBATCH --mem-per-cpu=2gb\n')
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --account=yuekai1\n")
        fh.writelines("#SBATCH --mail-type=NONE\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 adv_ratio.py {s} {e}")


    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}')
