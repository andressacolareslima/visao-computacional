import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Leitura da imagem colorida ---
img_color = cv2.imread('imagens/mugiwaras.png')
if img_color is None:
    print("Erro: imagem não encontrada em 'imagens/mugiwaras.png'")
    exit()

# --- 1. Filtro de Média (suavização) ---
kernel_media = np.ones((5, 5), np.float32) / 25
filtro_media = cv2.filter2D(img_color, -1, kernel_media)

# --- 2. Filtro Gaussiano ---
filtro_gauss = cv2.GaussianBlur(img_color, (7, 7), 1.5)

# --- Função para aplicar coloração sépia diretamente ---
def aplicar_sepia(img_bgr):
    img = img_bgr.astype(np.float32) / 255
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_img = cv2.transform(img, sepia_matrix)
    sepia_img = np.clip(sepia_img * 255, 0, 255).astype(np.uint8)
    return sepia_img

# --- 3. Aplica o tom sépia em cada imagem ---
sepia_original = aplicar_sepia(img_color)
sepia_media = aplicar_sepia(filtro_media)
sepia_gauss = aplicar_sepia(filtro_gauss)

# --- Converte todas as imagens para RGB antes de exibir ---
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
sepia_media_rgb = cv2.cvtColor(sepia_media, cv2.COLOR_BGR2RGB)
sepia_gauss_rgb = cv2.cvtColor(sepia_gauss, cv2.COLOR_BGR2RGB)
sepia_original_rgb = cv2.cvtColor(sepia_original, cv2.COLOR_BGR2RGB)

# --- Exibição dos resultados ---
imagens = [img_rgb, sepia_media_rgb, sepia_gauss_rgb, sepia_original_rgb]
titulos = [
    'Original (Colorida)',
    'Filtro Média + Sépia',
    'Filtro Gaussiano + Sépia',
    'Sépia Puro'
]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(imagens[i])
    plt.title(titulos[i], fontsize=10)
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
