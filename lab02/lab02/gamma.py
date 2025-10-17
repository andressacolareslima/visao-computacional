import cv2
import numpy as np

def yellow_gamma_correction_LUT(image, gamma=1.0, yellow_intensity=100):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    image_gamma = cv2.LUT(image, table)

    b, g, r = cv2.split(image_gamma)
    r = cv2.add(r, int(yellow_intensity * 1.2)) 
    g = cv2.add(g, int(yellow_intensity))        
    b = cv2.subtract(b, int(yellow_intensity * 0.8))  

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    return cv2.merge([b, g, r])

def nothing(x):
    pass

im = cv2.imread('imagens/jato.jpg')
if im is None:
    print("Erro: não foi possível carregar a imagem.")
    exit()

cv2.namedWindow('image')
cv2.createTrackbar('gamma', 'image', 100, 300, nothing)
cv2.createTrackbar('amarelo', 'image', 100, 300, nothing)

while True:
    g = cv2.getTrackbarPos('gamma', 'image') / 100
    y = cv2.getTrackbarPos('amarelo', 'image')

    im_yellow = yellow_gamma_correction_LUT(im, gamma=g, yellow_intensity=y)
    cv2.imshow('image', im_yellow)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
