import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('car.jpg', 0)
lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        lst.append(np.binary_repr(img[i][j], width=8))

bit_img_8 = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(img.shape[0], img.shape[1])
bit_img_7 = (np.array([int(i[1]) for i in lst], dtype=np.uint8) * 64).reshape(img.shape[0], img.shape[1])
bit_img_6 = (np.array([int(i[2]) for i in lst], dtype=np.uint8) * 32).reshape(img.shape[0], img.shape[1])
bit_img_5 = (np.array([int(i[3]) for i in lst], dtype=np.uint8) * 16).reshape(img.shape[0], img.shape[1])
bit_img_4 = (np.array([int(i[4]) for i in lst], dtype=np.uint8) * 8).reshape(img.shape[0], img.shape[1])
bit_img_3 = (np.array([int(i[5]) for i in lst], dtype=np.uint8) * 4).reshape(img.shape[0], img.shape[1])
bit_img_2 = (np.array([int(i[6]) for i in lst], dtype=np.uint8) * 2).reshape(img.shape[0], img.shape[1])
bit_img_1 = (np.array([int(i[7]) for i in lst], dtype=np.uint8) * 1).reshape(img.shape[0], img.shape[1])

# Plotting images using subplots
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

axs[0, 0].imshow(bit_img_8, cmap='gray')
axs[0, 0].set_title('Bit 8')

axs[0, 1].imshow(bit_img_7, cmap='gray')
axs[0, 1].set_title('Bit 7')

axs[0, 2].imshow(bit_img_6, cmap='gray')
axs[0, 2].set_title('Bit 6')

axs[0, 3].imshow(bit_img_5, cmap='gray')
axs[0, 3].set_title('Bit 5')

axs[1, 0].imshow(bit_img_4, cmap='gray')
axs[1, 0].set_title('Bit 4')

axs[1, 1].imshow(bit_img_3, cmap='gray')
axs[1, 1].set_title('Bit 3')

axs[1, 2].imshow(bit_img_2, cmap='gray')
axs[1, 2].set_title('Bit 2')

axs[1, 3].imshow(bit_img_1, cmap='gray')
axs[1, 3].set_title('Bit 1')

plt.tight_layout()
plt.show()
