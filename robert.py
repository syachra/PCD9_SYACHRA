import imageio
import numpy as np
import matplotlib.pyplot as plt

def apply_roberts_operator(image):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])

    gradient_x = convolve(image, roberts_x)
    gradient_y = convolve(image, roberts_y)

    return np.sqrt(gradient_x**2 + gradient_y**2)

def apply_sobel_operator(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)

    return np.sqrt(gradient_x**2 + gradient_y**2)

def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    output = np.zeros_like(image, dtype=float)

    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output

# Load the image and convert to grayscale
image = imageio.v2.imread("//content//tugas sesi9 PCD.jpeg", mode= 'F')

# Apply Roberts and Sobel operators
edges_roberts = apply_roberts_operator(image)
edges_sobel = apply_sobel_operator(image)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Roberts Edge Detection')
plt.imshow(edges_roberts, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Sobel Edge Detection')
plt.imshow(edges_sobel, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
