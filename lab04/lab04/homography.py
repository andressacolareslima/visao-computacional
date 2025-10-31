import cv2
import numpy as np
import matplotlib.pyplot as plt

global_transformation_matrix_step1 = None
global_transformation_matrix_step2 = None
global_img_center_resized = None
global_img_right_resized = None
global_img_left_resized = None

def feather_blend(imgA, imgB, ksize=41):
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    mA = (grayA > 0).astype(np.uint8) * 255
    mB = (grayB > 0).astype(np.uint8) * 255
    mA = cv2.GaussianBlur(mA, (ksize, ksize), 0).astype(np.float32)
    mB = cv2.GaussianBlur(mB, (ksize, ksize), 0).astype(np.float32)
    denom = (mA + mB)
    denom[denom == 0] = 1.0
    wA = (mA / denom)[..., None]
    wB = (mB / denom)[..., None]
    blended = (imgA.astype(np.float32) * wA + imgB.astype(np.float32) * wB).astype(np.uint8)
    return blended

def stitch_images(image_left, image_right, is_step1=False, is_step2=False):
    global global_transformation_matrix_step1, global_transformation_matrix_step2
    global global_img_center_resized, global_img_right_resized, global_img_left_resized

    h1, w1 = image_left.shape[:2]
    image_left_resized = cv2.resize(image_left, (int(w1 * 0.75), int(h1 * 0.75)))
    h2, w2 = image_right.shape[:2]
    image_right_resized = cv2.resize(image_right, (int(w2 * 0.75), int(h2 * 0.75)))

    if is_step1:
        global_img_center_resized = image_left_resized
        global_img_right_resized = image_right_resized
    elif is_step2:
        global_img_left_resized = image_left_resized

    img_left_color = image_left_resized
    img_right_color = image_right_resized

    img1 = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img_right_color, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    pts1, pts2 = [], []
    for m in good:
        pts1.append(kp1[m[0].queryIdx].pt)
        pts2.append(kp2[m[0].trainIdx].pt)

    points1 = np.float32(pts1).reshape(-1, 1, 2)
    points2 = np.float32(pts2).reshape(-1, 1, 2)
    transformation_matrix, inliers = cv2.findHomography(points1, points2, cv2.RANSAC)

    if is_step1:
        global_transformation_matrix_step1 = transformation_matrix
    elif is_step2:
        global_transformation_matrix_step2 = transformation_matrix

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    corners_img1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    transformed_corners_img1 = cv2.perspectiveTransform(corners_img1, transformation_matrix)
    all_corners = np.concatenate(
        (transformed_corners_img1,
         np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)),
        axis=0
    )

    min_x = int(np.min(all_corners[:, :, 0]))
    min_y = int(np.min(all_corners[:, :, 1]))
    max_x = int(np.max(all_corners[:, :, 0]))
    max_y = int(np.max(all_corners[:, :, 1]))

    trans_mat = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    img_output_size = (max_x - min_x, max_y - min_y)

    left_on_canvas  = cv2.warpPerspective(img_left_color,  trans_mat.dot(transformation_matrix), img_output_size)
    right_on_canvas = np.zeros((img_output_size[1], img_output_size[0], 3), dtype=img_left_color.dtype)
    right_on_canvas[-min_y:height2 - min_y, -min_x:width2 - min_x] = img_right_color

    output_image = feather_blend(left_on_canvas, right_on_canvas, ksize=51)
    return output_image, trans_mat

def highlight_regions(final_panorama, trans_mat_step1, H_step1, trans_mat_step2, H_step2):
    global global_img_left_resized, global_img_center_resized, global_img_right_resized
    highlight_pano = final_panorama.copy()
    COLOR_LEFT = (255, 0, 0)
    COLOR_CENTER = (0, 255, 0)
    COLOR_RIGHT = (0, 0, 255)

    h_r, w_r = global_img_right_resized.shape[:2]
    corners_right_resized = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
    final_transform_right = trans_mat_step2.dot(trans_mat_step1)
    corners_right_in_final_pano = cv2.perspectiveTransform(corners_right_resized, final_transform_right)

    h_c, w_c = global_img_center_resized.shape[:2]
    corners_center_resized = np.float32([[0, 0], [0, h_c], [w_c, h_c], [w_c, 0]]).reshape(-1, 1, 2)
    final_transform_center = trans_mat_step2.dot(trans_mat_step1.dot(H_step1))
    corners_center_in_final_pano = cv2.perspectiveTransform(corners_center_resized, final_transform_center)

    h_l, w_l = global_img_left_resized.shape[:2]
    corners_left_resized = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
    final_transform_left = trans_mat_step2.dot(H_step2)
    corners_left_in_final_pano = cv2.perspectiveTransform(corners_left_resized, final_transform_left)

    mask_left = np.zeros_like(highlight_pano, dtype=np.uint8)
    cv2.fillPoly(mask_left, [np.int32(corners_left_in_final_pano)], COLOR_LEFT)
    mask_center = np.zeros_like(highlight_pano, dtype=np.uint8)
    cv2.fillPoly(mask_center, [np.int32(corners_center_in_final_pano)], COLOR_CENTER)
    mask_right = np.zeros_like(highlight_pano, dtype=np.uint8)
    cv2.fillPoly(mask_right, [np.int32(corners_right_in_final_pano)], COLOR_RIGHT)

    highlight_pano = cv2.addWeighted(highlight_pano, 0.7, mask_left, 0.3, 0)
    highlight_pano = cv2.addWeighted(highlight_pano, 0.7, mask_center, 0.3, 0)
    highlight_pano = cv2.addWeighted(highlight_pano, 0.7, mask_right, 0.3, 0)
    return highlight_pano

img_right = cv2.imread("imagens/paisagem_1.png")
img_left = cv2.imread("imagens/paisagem_2.png")
img_center = cv2.imread("imagens/paisagem_3.png")

pano_center_right, trans_mat_step1 = stitch_images(img_center, img_right, is_step1=True)
final_panorama, trans_mat_step2 = stitch_images(img_left, pano_center_right, is_step2=True)

highlighted_final_panorama = highlight_regions(
    final_panorama,
    trans_mat_step1, 
    global_transformation_matrix_step1,
    trans_mat_step2, 
    global_transformation_matrix_step2
)

cv2.imshow("Panorama Final", final_panorama)
cv2.imshow("Panorama Final com Destaques", highlighted_final_panorama)
cv2.imwrite("panorama_final_destacado.jpg", highlighted_final_panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
