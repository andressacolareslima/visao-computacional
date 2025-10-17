import cv2
import numpy as np
import matplotlib.pyplot as plt

imagens = ["brasil.jpg", "gamora_nebula.jpg"]

for filename in imagens:
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    height, width = im.shape[:2]

    M_rotation = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)
    im_rotated = cv2.warpAffine(im, M_rotation, (width, height))

    M_scaling = np.float32([[1, 0, 0], [0, 0.7, 0]])
    im_scaled = cv2.warpAffine(im, M_scaling, (width, height))

    M_translation = np.float32([[1, 0, 180], [0, 1, 100]])
    im_translated = cv2.warpAffine(im, M_translation, (width, height))

    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Transformações - {filename}")
    plt.subplot(221), plt.imshow(im), plt.title("Original")
    plt.subplot(222), plt.imshow(im_rotated), plt.title("Rotacionada 45°")
    plt.subplot(223), plt.imshow(im_scaled), plt.title("Escala Y 0.7")
    plt.subplot(224), plt.imshow(im_translated), plt.title("Transladação")
    plt.show()
