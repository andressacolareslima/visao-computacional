import cv2
import matplotlib.pyplot as plt

imagens = ["brasil.jpg", "gamora_nebula.jpg"]

for filename in imagens:
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    im_neg = 255 - im

    new_w, new_h = im.shape[1] // 2, im.shape[0] // 2
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, im_thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Negativo, Resize e Threshold - {filename}")
    plt.subplot(131), plt.imshow(im_neg, cmap="gray"), plt.title("Negativo")
    plt.subplot(132), plt.imshow(im_resized, cmap="gray"), plt.title("Resize 50%")
    plt.subplot(133), plt.imshow(im_thresh, cmap="gray"), plt.title("Threshold 127")
    plt.show()
