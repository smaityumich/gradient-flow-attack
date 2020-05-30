import numpy as np
import matplotlib.pyplot as plt
theta1 = np.arange(0, 4.1, step = 0.2)
theta2 = np.arange(0, 4.1, step= 0.2)
T1, T2 = np.meshgrid(theta1, theta2)

fig, (ax0, ax1, ax2) = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 4])
b = np.load('data/test_stat_0.npy')
im0 = ax0.pcolormesh(T2, T1, b, shading='gouraud')
ax0.set_title('Rotation $0^O$')
ax0.set_xlabel('$\\theta_1$')
ax0.set_ylabel('$\\theta_2$')
fig.colorbar(im0, ax=ax0)
b = np.load('data/test_stat_10.npy')
im1 = ax1.pcolormesh(T2, T1, b, shading='gouraud')
ax1.set_title('Rotation $10^O$')
ax1.set_xlabel('$\\theta_1$')
ax1.set_ylabel('$\\theta_2$')
fig.colorbar(im1, ax=ax1)
b = np.load('data/test_stat_20.npy')
im2 = ax2.pcolormesh(T2, T1, b, shading='gouraud')
ax2.set_title('Rotation $20^O$')
ax2.set_xlabel('$\\theta_1$')
ax2.set_ylabel('$\\theta_2$')
fig.colorbar(im2, ax=ax2)

plt.savefig('plots/mean_ratios.pdf')

fig, (ax0, ax1, ax2) = plt.subplots(nrows = 1, ncols = 3, figsize = [15, 4])
b = np.load('data/mean_ratio_l2_0.npy')
im0 = ax0.pcolormesh(T2, T1, b, shading='gouraud')
ax0.set_title('Rotation $0^O$')
ax0.set_xlabel('$\\theta_1$')
ax0.set_ylabel('$\\theta_2$')
fig.colorbar(im0, ax=ax0)
b = np.load('data/mean_ratio_l2_10.npy')
im1 = ax1.pcolormesh(T2, T1, b, shading='gouraud')
ax1.set_title('Rotation $10^O$')
ax1.set_xlabel('$\\theta_1$')
ax1.set_ylabel('$\\theta_2$')
fig.colorbar(im1, ax=ax1)
b = np.load('data/mean_ratio_l2_20.npy')
im2 = ax2.pcolormesh(T2, T1, b, shading='gouraud')
ax2.set_title('Rotation $20^O$')
ax2.set_xlabel('$\\theta_1$')
ax2.set_ylabel('$\\theta_2$')
fig.colorbar(im2, ax=ax2)

plt.savefig('plots/mean_ratios_l2.pdf')