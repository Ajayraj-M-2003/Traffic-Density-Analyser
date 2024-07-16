import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your single image using raw string literal
imgpath = r'C:\Users\RESHIHA\Downloads\car.jpg'

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

    # Define the target histogram (adjust as needed)
    target_hist = np.array([10] * 256, dtype=np.uint8)

    # Perform histogram matching
    matched_image = cv2.LUT(equalized_image, target_hist)

    # Calculate the histogram of the matched image
    hist_matched = cv2.calcHist([matched_image], [0], None, [256], [0, 256])

    # Display the histograms and images
    plt.figure(figsize=(15, 4))
    
    # Original Histogram
    plt.subplot(131)
    plt.plot(hist_original)
    plt.title("Original Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Histogram Equalization
    plt.subplot(132)
    plt.plot(hist_equalized)
    plt.title("Histogram Equalization")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Histogram Matching
    plt.subplot(133)
    plt.plot(hist_matched)
    plt.title("Histogram Matching")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not read the image")
