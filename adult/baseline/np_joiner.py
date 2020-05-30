import numpy as np
ratios = []
start = np.arange(0, 9001, 200)
end = np.arange(200, 9201, 200)
end[-1] = 9045


#os.system('touch summary/adult7.out')



for s, e in zip(start, end):
    filename = f'outcome/perturbed_ratio_{start}_{end}.npy'
    ratio_part = np.load(filename)
    ratios.append(ratio_part)

np.save('outcome/perturbed_ratio.npy', np.concatenate(ratios))