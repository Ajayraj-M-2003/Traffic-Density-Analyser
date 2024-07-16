import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your image
imgpath = "C:/Users/RESHIHA/Downloads/car.jpg"

# Read the image using OpenCV
img = cv2.imread(imgpath)

# Check if the image was loaded successfully
if img is not None:
    # Log transformation
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, dtype=np.uint8)

    # Display the original and log-transformed images
    plt.figure()
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(log_image, cmap='gray')
    plt.title("Log-Transformed Image")
    plt.show()
else:
    print("Error: Could not read the image")
