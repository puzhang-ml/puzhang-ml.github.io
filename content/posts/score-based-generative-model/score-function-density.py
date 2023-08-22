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
cov_noise = [[0.02, 0], [0, 0.02]]

# Define the Gaussian mixture model
def gmm(x, y):
    z1 = (1/ (2 * np.pi * np.sqrt(np.linalg.det(cov1)))) * np.exp(-0.5 * (np.array([x, y]) - mean1) @ np.linalg.inv(cov1) @ (np.array([x, y]) - mean1).T)
    z2 = 0.5 * (1/ (2 * np.pi * np.sqrt(np.linalg.det(cov2)))) * np.exp(-0.5 * (np.array([x, y]) - mean2) @ np.linalg.inv(cov2) @ (np.array([x, y]) - mean2).T)
    z1_noise = (1/ (2 * np.pi * np.sqrt(np.linalg.det(cov_noise)))) * np.exp(-0.5 * (np.array([x, y]) - mean1) @ np.linalg.inv(cov_noise) @ (np.array([x, y]) - mean1).T)
    z2_noise = (1/ (2 * np.pi * np.sqrt(np.linalg.det(cov_noise)))) * np.exp(-0.5 * (np.array([x, y]) - mean2) @ np.linalg.inv(cov_noise) @ (np.array([x, y]) - mean2).T)
    return z1 + z2 + 20 * (z1_noise + z2_noise)

# Create a grid of points
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(gmm)(X, Y)

# Compute the gradient
grad_x, grad_y = np.gradient(Z)

# Define the boundaries of the central square
square_min, square_max = 0, 1
num_samples = 500
samples_z1_noise = np.random.multivariate_normal(mean1, cov_noise, num_samples)
samples_z2_noise = np.random.multivariate_normal(mean2, cov_noise, num_samples)


# Filter samples to retain only those inside the central square
filtered_samples_z1_noise = samples_z1_noise[(samples_z1_noise[:, 0] > square_min) & 
                                            (samples_z1_noise[:, 0] < square_max) & 
                                            (samples_z1_noise[:, 1] > square_min) & 
                                            (samples_z1_noise[:, 1] < square_max)]

filtered_samples_z2_noise = samples_z2_noise[(samples_z2_noise[:, 0] > square_min) & 
                                            (samples_z2_noise[:, 0] < square_max) & 
                                            (samples_z2_noise[:, 1] > square_min) & 
                                            (samples_z2_noise[:, 1] < square_max)]

# Randomly select 20 samples from the filtered points
num_samples_final = 50
final_samples_z1_noise = filtered_samples_z1_noise[np.random.choice(filtered_samples_z1_noise.shape[0], num_samples_final, replace=False)]
final_samples_z2_noise = filtered_samples_z2_noise[np.random.choice(filtered_samples_z2_noise.shape[0], num_samples_final, replace=False)]

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(Z, cmap="Blues", origin="lower", extent=[0, 1, 0, 1])
#ax.quiver(X[::5, ::5], Y[::5, ::5], grad_x[::5, ::5], grad_y[::5, ::5], color="black", scale=30)
ax.scatter(final_samples_z1_noise[:, 0], final_samples_z1_noise[:, 1], c='blue', s=15, alpha=0.8)
ax.scatter(final_samples_z2_noise[:, 0], final_samples_z2_noise[:, 1], c='blue', s=15, alpha=0.8)
ax.axis("off")
plt.show()
