import numpy as np

def get_data(minority_group_prop = 0.2, sample_size = 200, seed = 1):

    
    mu_minority = np.array([1.5, 0])
    mu_majority = np.array([-1.5, 0])
    w_minority  = np.array([1.5, np.sqrt(8)/3])
    w_majority  = np.array([-1.5, np.sqrt(8)/3])
    d = 2
    sigma = 0.5
    np.random.seed(seed)
    x = np.random.normal(0, scale=sigma, size=(sample_size,d))
    z = np.random.choice([False, True], size=sample_size, p=[minority_group_prop, 1-minority_group_prop])
    x[z,:] = x[z,:] + mu_majority
    x[~z,:] = x[~z,:] + mu_minority
    y = np.zeros(sample_size)
#    noise = np.random.normal(size = n)
    noise = np.zeros(sample_size)
    y[z] = np.sign(np.dot(x[z,:],w_majority) - np.dot(w_majority,mu_majority) + noise[z])
    y[~z] = np.sign(np.dot(x[~z,:],w_minority) - np.dot(w_minority,mu_minority) + noise[~z])
    np.save('data/x.npy', x)
    np.save('data/y.npy', (y+1)/2)