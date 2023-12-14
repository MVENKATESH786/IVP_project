import cv2
import numpy as np
from scipy.fft import fft2, fftshift, ifft2
import matplotlib.pyplot as plt

def frequency_spectrum(img):
    f = fft2(img)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def adjust_contrast(image, factor):
    mean = np.mean(image)
    return np.clip(factor * (image - mean) + mean, 0, 255)

# Load the images
image1 = cv2.imread('image.jpg', 0)  # Load as grayscale
image2 = cv2.imread('image2.jpg', 0)  # Load as grayscale

# Ensure both images are the same size
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Apply a Gaussian blur (low-pass filter) to the first image
low_pass = cv2.GaussianBlur(image1, (31, 31), 0)

# Apply a high-pass filter to the second image
blurred = cv2.GaussianBlur(image2, (31, 31), 0)
high_pass = image2 - blurred
high_pass = adjust_contrast(high_pass, 1.5)  # Increase contrast

# Frequency domain visualization
plt.subplot(121), plt.imshow(frequency_spectrum(low_pass), cmap='gray')
plt.title('Low-Pass Filter'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(frequency_spectrum(high_pass), cmap='gray')
plt.title('High-Pass Filter'), plt.xticks([]), plt.yticks([])
plt.show()

# Weighted blend
alpha = 0.6  # weight of the first image
hybrid_image = cv2.addWeighted(low_pass, alpha, high_pass, 1 - alpha, 0)

# Save or display the result
cv2.imwrite('advanced_hybrid_image.jpg', hybrid_image)
cv2.imshow('Advanced Hybrid Image', hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()