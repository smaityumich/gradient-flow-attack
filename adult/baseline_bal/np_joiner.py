import numpy as np
ratios = []
start = np.arange(0, 9001, 200)
end = np.arange(200, 9201, 200)
end[-1] = 9045


#os.system('touch summary/adult7.out')



for s, e in zip(start, end):
    filename = f'outcome/perturbed_ratio_{s}_{e}.npy'
    ratio_part = np.load(filename)
    ratios.append(ratio_part)
a = np.concatenate(ratios)
np.save('outcome/perturbed_ratio.npy', a)
a = a[np.isfinite(a)]
lb = np.mean(a) - 1.96*np.std(a)/np.sqrt(a.shape[0])
print(f'Lower bound {lb}')


