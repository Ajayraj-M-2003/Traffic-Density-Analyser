import cv2
import numpy as np
import matplotlib.pyplot as plt

def augment_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image file at {image_path}")
        return None
    
    # Rotation
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Brightness adjustment
    brightness_factor = np.random.uniform(0.5, 1.5)
    bright_image = cv2.convertScaleAbs(rotated_image, alpha=brightness_factor, beta=0)
    
    return bright_image, image

# Example usage
image_path = 'car.jpg'
augmented_image, original_image = augment_image(image_path)
if augmented_image is not None:
    # Display input and output images separately in a single screen
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
    plt.title('Augmented Image')
    plt.axis('off')

    plt.show()
else:
    print("Failed to augment image.")
