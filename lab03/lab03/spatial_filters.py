import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = 'imagens/halftone.png'

img = cv2.imread(filename, 0)
if img is None:
    print(f"Erro: Nao foi possivel carregar {filename}")
    sys.exit(1)

l, c = img.shape

img_fft = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT))

mask_notch = np.ones((l, c, 2), np.float32)
crow, ccol = l // 2, c // 2
r = 20
picos_coords_yx = [
    (crow - 55, ccol - 55), (crow + 55, ccol + 55),
    (crow - 55, ccol + 55), (crow + 55, ccol - 55),
    (crow, ccol - 55), (crow, ccol + 55),
    (crow - 55, ccol), (crow + 55, ccol)
]
for (py, px) in picos_coords_yx:
    cv2.circle(mask_notch, (px, py), r, (0, 0), -1)

fft_filtered = cv2.multiply(img_fft, mask_notch)

img_back = cv2.idft(np.fft.ifftshift(fft_filtered))
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)

mag_spectrum = 20 * np.log(1 + cv2.magnitude(img_fft[:, :, 0], img_fft[:, :, 1]))
mask_vis = mask_notch[:, :, 0]

imagens = [img, mag_spectrum, mask_vis, img_back]
titles = ['Original (Halftone)', 'Espectro FFT (Picos = Ruído)', 'Filtro Notch (Mascara)', 'Imagem Filtrada']

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(imagens[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
