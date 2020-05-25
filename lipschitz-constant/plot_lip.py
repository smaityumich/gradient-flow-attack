import numpy as np
import matplotlib.pyplot as plt

steps = [20, 40, 80, 160, 320, 640, 1280]#[20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# Unfair lipschitz plot
mean_lip = np.load('output/mean-lipschitz.npy')
plt.plot(steps, mean_lip)
plt.title('Unfair classifier')
plt.xlabel('Grad-steps')
plt.ylabel('$L$')
#plt.xscale('log')
plt.savefig('output/mean-lipschitz.pdf')
plt.close()

# Fair lipschitz plot
mean_lip = np.load('output/mean-lipschitz-fair.npy')
plt.plot(steps, mean_lip)
plt.title('Fair classifier')
plt.xlabel('Grad-steps')
plt.ylabel('$L$')
#plt.xscale('log')
plt.savefig('output/mean-lipschitz-fair.pdf')
