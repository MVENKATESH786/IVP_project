import cv2
import numpy as np

# Load the images
image1 = cv2.imread('image.jpg')
image2 = cv2.imread('image2.jpg')

# Ensure both images are the same size
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Apply a Gaussian blur (low-pass filter) to the first image
low_pass = cv2.GaussianBlur(image1, (21, 21), 0)

# Apply a high-pass filter to the second image
# First, blur the image
blurred = cv2.GaussianBlur(image2, (21, 21), 0)
# Subtract the blurred version from the original
high_pass = image2 - blurred

# Combine the two images
hybrid_image = low_pass + high_pass

# Save or display the result
cv2.imwrite('hybrid_image.jpg', hybrid_image)
cv2.imshow('Hybrid Image', hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
