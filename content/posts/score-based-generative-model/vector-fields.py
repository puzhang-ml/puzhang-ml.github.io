"""
This simple algorithm generate matplotlib diagram that shows vector fields and their difference.
Author: Pu Zhang(ml.puzh@gmail.com)
Date: 2023-08-16
"""

import numpy as np
import matplotlib.pyplot as plt

scale = 15

# Existing Gaussian parameters
mean1 = [0.6, 0.3]
cov1 = [[0.04, 0], [0, 0.04]]

# New Gaussian parameters (shifted mode)
mean2 = [0.7, 0.4]  # slightly shifted
cov2 = [[0.04, 0], [0, 0.04]]

# Create a grid of points
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)

def gaussian(x, y, mean, cov):
    x_centered = np.array([x, y]) - mean
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * np.dot(x_centered.T, np.dot(inv_cov, x_centered))
    normalization = 1/ (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return normalization * np.exp(exponent)

# Compute gradient for the existing Gaussian
Z1 = np.vectorize(lambda x, y: gaussian(x, y, mean1, cov1))(X, Y)
grad_x1, grad_y1 = np.gradient(Z1)

# Compute gradient for the new Gaussian
Z2 = np.vectorize(lambda x, y: gaussian(x, y, mean2, cov2))(X, Y)
grad_x2, grad_y2 = np.gradient(Z2)

# Compute the difference gradient
diff_grad_x = grad_x2 - grad_x1
diff_grad_y = grad_y2 - grad_y1

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, grad_x1, grad_y1, color="blue", scale=scale)
ax.quiver(X, Y, grad_x2, grad_y2, color="red", scale=scale)
ax.quiver(X + grad_x1 / scale, Y + grad_y1 / scale, diff_grad_x, diff_grad_y, color="green", scale=scale)
ax.axis("off")
plt.legend()
plt.show()
