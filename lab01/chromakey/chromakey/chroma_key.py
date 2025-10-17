import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_chroma(foreground_path, background_image, x, y, scale=1.0):
    im_fg = cv2.imread(foreground_path)
    im_fg = cv2.cvtColor(im_fg, cv2.COLOR_BGR2RGB)
    
    width = int(im_fg.shape[1] * scale)
    height = int(im_fg.shape[0] * scale)
    im_fg = cv2.resize(im_fg, (width, height), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(im_fg, cv2.COLOR_RGB2HSV)

    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    im_mask = cv2.inRange(hsv, lower_green, upper_green)
    im_mask_inv = cv2.bitwise_not(im_mask)

    im_mask_inv_3ch = cv2.cvtColor(im_mask_inv, cv2.COLOR_GRAY2BGR)
    im_mask_3ch = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2BGR)

    roi = background_image[y:y+height, x:x+width]

    im_fg_resized = cv2.resize(im_fg, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

    im_mask_inv_3ch_resized = cv2.resize(im_mask_inv_3ch, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
    im_mask_3ch_resized = cv2.resize(im_mask_3ch, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
    
    im_fg_cut = cv2.bitwise_and(im_fg_resized, im_mask_inv_3ch_resized)
    roi_bg = cv2.bitwise_and(roi, im_mask_3ch_resized)
    
    dst = cv2.add(roi_bg, im_fg_cut)
    
    background_image[y:y+height, x:x+width] = dst
    
    return background_image

background_file = 'fg.jpg'
foreground_file_1 = 'ck_fg.jpg'
foreground_file_2 = 'ck_fg2.jpg'

im_bg_orig = cv2.imread(background_file)
im_bg_orig = cv2.cvtColor(im_bg_orig, cv2.COLOR_BGR2RGB)

result_1 = apply_chroma(foreground_file_1, im_bg_orig.copy(), 100, 200, 0.5)

result_2 = apply_chroma(foreground_file_2, im_bg_orig.copy(), 500, 150, 0.7)

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.imshow(result_1)
plt.title('ck_fg')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result_2)
plt.title('ck_fg2')
plt.axis("off")

plt.show()