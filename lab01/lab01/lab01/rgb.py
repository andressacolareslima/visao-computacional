import cv2
import matplotlib.pyplot as plt

imagens = ["brasil.jpg", "gamora_nebula.jpg"]

for filename in imagens:
    im = cv2.imread(filename)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    h_channel, s_channel, v_channel = cv2.split(im_hsv)

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Canais HSV - {filename}")
    plt.subplot(131), plt.imshow(h_channel, cmap="gray"), plt.title("H (Matiz)")
    plt.subplot(132), plt.imshow(s_channel, cmap="gray"), plt.title("S (Saturação)")
    plt.subplot(133), plt.imshow(v_channel, cmap="gray"), plt.title("V (Valor)")
    plt.show()










