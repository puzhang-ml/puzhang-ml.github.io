"""
This simple algorithm generate matplotlib diagram that shows the vector fields for a guassian mixture model with two modes.
Author: Pu Zhang(ml.puzh@gmail.com)
Date: 2023-08-16
"""

import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the two Gaussian distributions
mean1 = [0.2, 0.2]  # bottom-left
mean2 = [0.8, 0.8]  # top-right
cov1 = [[0.01, 0], [0, 0.01]]
cov2 = [[0.01, 0], [0, 0.01]]

# Define the Gaussian mixture model
def gmm(x, y):
    z1 = (1/ (2 * np.pi * np.sqrt(np.linalg.det(cov1)))) * np.exp(-0.5 * (np.array([x, y]) - mean1) @ np.linalg.inv(cov1) @ (np.array([x, y]) - mean1).T)
    z2 = 0.5 * (1/ (2 * np.pi * np.sqrt(np.linalg.det(cov2)))) * np.exp(-0.5 * (np.array([x, y]) - mean2) @ np.linalg.inv(cov2) @ (np.array([x, y]) - mean2).T)
    return z1 + z2

# Create a grid of points
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(gmm)(X, Y)

# Compute the gradient
grad_x, grad_y = np.gradient(Z)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(Z, cmap="Blues", origin="lower", extent=[0, 1, 0, 1])
ax.quiver(X[::5, ::5], Y[::5, ::5], grad_x[::5, ::5], grad_y[::5, ::5], color="black", scale=30)
ax.axis("off")
plt.show()
