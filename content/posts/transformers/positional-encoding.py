"""
This simple algorithm generate matplotlib diagram that shows the positional encoding algorithm in transformer.
Author: Pu Zhang(ml.puzh@gmail.com)
Date: 2023-08-29
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_positional_encodings(pos, d_model):
    """Compute positional encodings for a given position and model dimension."""
    encoding = np.zeros(d_model)
    for i in range(0, d_model, 2):
        encoding[i] = np.sin(pos / (10000 ** (2 * i / d_model)))
        if i + 1 < d_model:  # To ensure we don't go out of bounds
            encoding[i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))
    return encoding

# Model dimension
d_model = 64

# Positions
positions = np.arange(10, -1, -1)  # From 10 to 0

# Compute positional encodings for each position
encodings = np.array([compute_positional_encodings(pos, d_model) for pos in positions])

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(encodings, cmap="viridis", aspect="auto", extent=[0, d_model, 0, 10])
plt.colorbar(label='Encoding Value')
plt.ylabel("Position")
plt.xlabel("Dimension")
plt.title("Positional Encoding")
plt.tight_layout()
plt.show()