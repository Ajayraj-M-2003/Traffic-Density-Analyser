import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your single image
imgpath = "car.jpg"

# Read the image using OpenCV
img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

# Check if the image was loaded successfully
if img is not None:
    # Calculate the original histogram
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(img)

    # Calculate the histogram of the equalized image
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    # Display the original histogram
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(hist_original)
    plt.title("Original Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Display the histogram equalization
    plt.subplot(132)
    plt.imshow(equalized_image, cmap='gray')
    plt.title("Histogram Equalized Image")

    # Display the histogram of the equalized image
    plt.subplot(133)
    plt.plot(hist_equalized)
    plt.title("Histogram of Equalized Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Show all plots
    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not read the image")
