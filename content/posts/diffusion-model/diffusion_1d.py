"""
This simple algorithm generate matplotlib diagram that shows how diffusion process can transform a complex 1d distribution into standard
gaussian distribution.
Author: Pu Zhang(ml.puzh@gmail.com)
Date: 2023-07-27
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# Parameters of the two Gaussians in the mixture
mu1, sigma1 = -2, 0.5
mu2, sigma2 = 2, 1

# Number of samples and steps
n_samples = 10000
n_steps = 200

# Generate the mixture of two Gaussians
X0 = np.concatenate([np.random.normal(mu1, sigma1, n_samples//2),
                     np.random.normal(mu2, sigma2, n_samples//2)])

# Initialize the array to store the steps
X = np.zeros((n_samples, n_steps+1))
X[:, 0] = X0

# Iteratively add noise
for t in range(1, n_steps+1):
    X[:, t] = np.sqrt(0.99)*X[:, t-1] + np.sqrt(0.001)*np.random.normal(size=n_samples)

# Generate the standard Gaussian distribution
mu_t = np.random.normal(size=n_samples)

# Select 12 random points from -1 to 1 and track their transformation
points = [2.4, 2.5, 2.7, 0.5, 0, -0.5, -2.4, -2.5, -2.7]
# Generate a larger list of distinct colors
colors = plt.cm.viridis(np.linspace(0, 1, len(points)))

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))

# Plot the normal distributions
x_range = np.linspace(-10, 10, 1000)
y_range_0 = norm.pdf(x_range, mu1, sigma1)*0.5 + norm.pdf(x_range, mu2, sigma2)*0.5
y_range_200 = norm.pdf(x_range, 0, 1)
ax.fill_betweenx(x_range, 0, -y_range_0*50, color='lightgray', alpha=0.5)
ax.fill_betweenx(x_range, 200, y_range_200*50+200, color='lightgray', alpha=0.5)


# Plot the dashed lines at steps 0 and 200
ax.axvline(x=0, color='black', linestyle='dashed')
ax.axvline(x=200, color='black', linestyle='dashed')


# Plot the traces of the points
for i, point in enumerate(points):
    trace = X[np.abs(X[:, 0] - point).argmin(), :]
    plt.plot(trace, color=colors[i])

plt.xlim([-50, 250])
plt.ylim([-6, 6])
plt.xlabel('Steps')
plt.ylabel('Density')

ax.text(0.5, 0.95, r'$X_t = \sqrt{0.99}X_{t-1} + \sqrt{0.001}\mu_t$ where $\mu_t \sim \mathcal{N}(0, 1)$',
         horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)

plt.title('The diffusion process for 1D distribution', fontsize=20)
plt.grid(True)
plt.show()
