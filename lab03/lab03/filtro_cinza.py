import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color = cv2.imread('imagens/mugiwaras.png')
if img_color is None:
    print("Erro: imagem não encontrada em 'imagens/mugiwaras.png'")
    exit()

b, g, r = cv2.split(img_color)
gray = (0.3 * r + 0.59 * g + 0.11 * b).astype(np.uint8)

kernel_media = np.ones((5, 5), np.float32) / 25
filtro_media = cv2.filter2D(gray, -1, kernel_media)

filtro_gauss = cv2.GaussianBlur(gray, (7, 7), 1.5)

imagens = [
    cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB),
    gray,
    filtro_media,
    filtro_gauss
]

titulos = [
    'Original (Colorida)',
    'Tons de Cinza (Y = 0.3R + 0.59G + 0.11B)',
    'Filtro Média (5x5)',
    'Filtro Gaussiano (7x7, σ=1.5)'
]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    cmap = 'gray' if i > 0 else None
    plt.imshow(imagens[i], cmap=cmap)
    plt.title(titulos[i], fontsize=10)
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
