"""
This simple algorithm generate matplotlib diagram that shows a gaussian mixture model
Author: Pu Zhang(ml.puzh@gmail.com)
Date: 2023-08-16
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Gaussian function
def gaussian_2d(x, y, mu, sigma):
    return np.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2 * sigma ** 2))

# Define grid of points
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
x, y = np.meshgrid(x, y)

# Define 4 modes with different means and the same standard deviation
mu1 = [-2, -2]
mu2 = [2, 2]
mu3 = [-2, 2]
mu4 = [2, -2]
sigma = 1.0

# Calculate the sum of 4 Gaussian distributions at each point on the grid
z = (0.5 * gaussian_2d(x, y, mu1, sigma) + 
     gaussian_2d(x, y, mu2, sigma) + 
     1.2 * gaussian_2d(x, y, mu3, sigma) + 
     0.7 * gaussian_2d(x, y, mu4, sigma))

# Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='blue', edgecolor='none', linewidth=0, antialiased=True)
ax.set_axis_off()
plt.show()
