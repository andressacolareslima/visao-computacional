import cv2
import numpy as np
import sys

IMG_SIZE = 300
LINE_THICKNESS = 4

try:
    _ = cv2.imread('imagens/circle.jpg', cv2.IMREAD_GRAYSCALE)
    _ = cv2.imread('imagens/line.jpg', cv2.IMREAD_GRAYSCALE)
except (AttributeError, TypeError, FileNotFoundError):
    sys.exit()

D_HEAD = 50
H_TRUNK = 100
H_ARM = int(H_TRUNK * 0.75)
H_LEG = H_TRUNK

canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

y_head_center = cy - H_TRUNK // 2 - D_HEAD // 2
y_trunk_top = y_head_center + D_HEAD // 2
y_trunk_bottom = y_trunk_top + H_TRUNK
y_shoulders = y_trunk_top + int(H_TRUNK * 0.15)
y_hips = y_trunk_bottom

cv2.circle(canvas, (cx, y_head_center), D_HEAD // 2, 255, -1)
cv2.line(canvas, (cx, y_trunk_top), (cx, y_trunk_bottom), 255, LINE_THICKNESS)

def draw_limb(canvas, anchor_x, anchor_y, length, angle_degrees, thickness):
    angle_rad = np.deg2rad(angle_degrees)
    end_x = int(anchor_x + length * np.cos(angle_rad))
    end_y = int(anchor_y + length * np.sin(angle_rad))
    cv2.line(canvas, (anchor_x, anchor_y), (end_x, end_y), 255, thickness)

draw_limb(canvas, cx, y_shoulders, H_ARM, 135, LINE_THICKNESS)
draw_limb(canvas, cx, y_shoulders, H_ARM, 45, LINE_THICKNESS)
draw_limb(canvas, cx, y_hips, H_LEG, 135, LINE_THICKNESS)
draw_limb(canvas, cx, y_hips, H_LEG, 45, LINE_THICKNESS)

final_image = cv2.bitwise_not(canvas)
cv2.imshow("Boneco palito centralizado", final_image)

while True:
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
