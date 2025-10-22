import cv2
import numpy as np
import matplotlib.pyplot as plt

def filtros_espaciais(img):
    media = cv2.blur(img, (7, 7))
    gaussiano = cv2.GaussianBlur(img, (7, 7), 2)
    mediana = cv2.medianBlur(img, 7)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    return media, gaussiano, mediana, bilateral

def filtro_passa_baixa_freq(img, cutoff=60):
    l, c = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    x, y = np.ogrid[:l, :c]
    crow, ccol = l // 2, c // 2
    mask = np.exp(-((x - crow)**2 + (y - ccol)**2) / (2 * cutoff**2))

    fshift_filtered = fshift * mask
    img_back = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(img_back)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

def filtro_notch(img, dist=55, r=15):
    l, c = img.shape
    img_fft = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT))
    mask_notch = np.ones((l, c, 2), np.float32)
    crow, ccol = l // 2, c // 2

    coords = [
        (crow - dist, ccol - dist), (crow + dist, ccol + dist),
        (crow - dist, ccol + dist), (crow + dist, ccol - dist),
        (crow, ccol - dist), (crow, ccol + dist),
        (crow - dist, ccol), (crow + dist, ccol)
    ]
    for (py, px) in coords:
        cv2.circle(mask_notch, (px, py), r, (0, 0), -1)

    fft_filtered = cv2.multiply(img_fft, mask_notch)
    img_back = cv2.idft(np.fft.ifftshift(fft_filtered))
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

def processar_imagem(caminho, nome):
    img = cv2.imread(caminho, 0)
    if img is None:
        print(f"Erro ao carregar {caminho}")
        return

    media, gauss, mediana, bilateral = filtros_espaciais(img)
    freq_low = filtro_passa_baixa_freq(img)
    freq_notch = filtro_notch(img)

    imagens = [img, media, gauss, mediana, bilateral, freq_low, freq_notch]
    titulos = [
        f'{nome} - Original',
        'Média (7x7)',
        'Gaussiano (σ=2)',
        'Mediana (7x7)',
        'Bilateral (preserva bordas)',
        'Passa-Baixa (FFT)',
        'Notch (FFT)'
    ]

    plt.figure(figsize=(12, 8))
    for i in range(len(imagens)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(imagens[i], cmap='gray')
        plt.title(titulos[i], fontsize=9)
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

imagens = [
    ('imagens/halftone.png', 'Halftone'),
    ('imagens/pieces.png', 'Peças'),
    ('imagens/salt_noise.png', 'Ruído Sal e Pimenta')
]

for caminho, nome in imagens:
    processar_imagem(caminho, nome)
