import cv2
import numpy as np
import matplotlib.pyplot as plt

# Provide the full path to the image file
imgpath = 'C:/Users/RESHIHA/Downloads/car.jpg'

# Read the image
img = cv2.imread(imgpath)

# Check if the image was read successfully
if img is None:
    print("Error: Could not read the image")
    exit()

# Make a copy of the original image
original = img.copy()

# Define the mapping function for contrast stretching
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')

# Apply contrast stretching using lookup table
img = cv2.LUT(img, table)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarize the grayscale image using a threshold value
_, binarized_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

# Resize images for better visualization
b1 = cv2.resize(original, (900, 900))
b2 = cv2.resize(img, (900, 900))

# Plot images using subplots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(b1, cv2.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(b2, cv2.COLOR_BGR2RGB))
plt.title('Output')

plt.subplot(1, 3, 3)
plt.imshow(binarized_img, cmap='gray')
plt.title('Binarized')

plt.show()
