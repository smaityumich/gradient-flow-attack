import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import classifier as cl
import utils
import random
import matplotlib.pyplot as plt
import scipy


seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)


expt = '_1'
filename = f'outcome/perturbed_loss{expt}.npy'
histplot = f'adversarial-points/perturbed-mean-entropy-hist{expt}.png'
qqplot = f'adversarial-points/perturbed-mean-entropy-qqplot{expt}.png'


test_ratio =  np.load(filename)




def ratio_mean(n = 9045):
    index = random.sample(range(n), 1000)
    srswr_ratio=[test_ratio[i] for i in index]
    return np.mean(srswr_ratio)

ratio_means = [ratio_mean() for _ in range(5000)]
plt.hist(ratio_means)
plt.title(f'Histogram of mean loss of ratios for expt{expt}')
plt.xticks(rotation = 35, fontsize = 'x-small')
plt.savefig(histplot)
plt.close()


scipy.stats.probplot(ratio_means, plot=plt)
plt.title(f'Normal qq-plot of mean loss of ratios for expt{expt}')
plt.savefig(qqplot)
plt.close()

expt = '_1_fair'
filename = f'outcome/perturbed_loss{expt}.npy'
histplot = f'adversarial-points/perturbed-mean-entropy-hist{expt}.png'
qqplot = f'adversarial-points/perturbed-mean-entropy-qqplot{expt}.png'


test_ratio =  np.load(filename)




def ratio_mean(n = 9045):
    index = random.sample(range(n), 1000)
    srswr_ratio=[test_ratio[i] for i in index]
    return np.mean(srswr_ratio)

ratio_means = [ratio_mean() for _ in range(5000)]
plt.hist(ratio_means)
plt.title(f'Histogram of mean loss of ratios for expt{expt}')
plt.xticks(rotation = 35, fontsize = 'x-small')
plt.savefig(histplot)
plt.close()


scipy.stats.probplot(ratio_means, plot=plt)
plt.title(f'Normal qq-plot of mean loss of ratios for expt{expt}')
plt.savefig(qqplot)
