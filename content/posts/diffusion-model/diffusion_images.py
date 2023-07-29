from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the image file
img = Image.open(r'images\cat.jpg')

# Convert the image to numpy array
img_array = np.array(img)
# Convert the image array to float type and normalize to [0, 1]
img_array = img_array.astype(np.float32) / 255.0

# Generate the noise once
noise = np.random.normal(0, 0.1, img_array.shape)

# Define a function to add noise to the image at a specific step
def add_noise_at_step(image, step):
    noise_weight = 0
    for i in range(step):
        noise_weight += (0.99**(step/2)) * np.sqrt(0.01)
    res = np.sqrt(0.99**step) * image + noise_weight * noise
    return np.clip(res, 0, 1)

# Show the noised image at step 0, 50, 100, 200, 250, 300
steps = [0, 50, 100, 200, 250, 300]
fig, axs = plt.subplots(1, 6, figsize=(30, 10))

for i, step in enumerate(steps):
    img_noised = add_noise_at_step(img_array, step)
    axs[i].imshow(img_noised)
    axs[i].set_title(f'Step {step}')
    axs[i].axis('off')

# Add right arrows and three dots
for i in range(5):
    axs[i].annotate('', xy=(1.2, 0.5), xytext=(1.0, 0.5), xycoords='axes fraction', 
                    arrowprops=dict(facecolor='black', width=0.5, headwidth=8, shrink=0.05), ha='right')

plt.show()
