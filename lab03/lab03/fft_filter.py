import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

def process_image(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print(f"Erro ao carregar {filename}")
        return

    avg_blur = cv2.blur(img, (5, 5))
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 1)
    median_blur = cv2.medianBlur(img, 5)

    kernel = np.float32([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    custom = cv2.filter2D(img, -1, kernel)

    kernel = np.float32([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt = cv2.filter2D(img, -1, kernel)

    plt.figure(figsize=(12, 8))
    plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title(f'Original ({filename})')
    plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(avg_blur, cmap='gray'), plt.title('Media')
    plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(gaussian_blur, cmap='gray'), plt.title('Gaussian')
    plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(median_blur, cmap='gray'), plt.title('Mediana')
    plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(custom, cmap='gray'), plt.title('Custom')
    plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt')
    plt.xticks([]), plt.yticks([])

filenames = ['imagens/salt_noise.png', 'imagens/pieces.jpg']

for f in filenames:
    process_image(f)

plt.show()
