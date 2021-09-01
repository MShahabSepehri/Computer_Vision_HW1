import cv2 as cv
from Utils import utils, config_reader
from Questions import matching
import numpy as np

params = config_reader.param_config("Q4")

img1 = utils.get_image("im03.jpg")
img2 = utils.get_image("im04.jpg")

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

img1_with_kps, kps1, des1 = matching.get_sift_key_points(img1, green)
img2_with_kps, kps2, des2 = matching.get_sift_key_points(img2, green)

ratio_tr = params["ratio_tr"]
matches = matching.get_match_points(des1, des2, ratio_tr)

src_points = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
des_points = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

img1_with_kps_and_matches = cv.drawKeypoints(img1_with_kps, [kps1[m.queryIdx] for m in matches],
                                             None, color=blue)
img2_with_kps_and_matches = cv.drawKeypoints(img2_with_kps, [kps2[m.trainIdx] for m in matches],
                                             None, color=blue)

inlier_tr = params["inlier_tr"]
confidence = params["confidence"]
status, idxs, H, counter = matching.my_RANSAC(kps1, kps2, matches, inlier_tr, confidence)

x_off = params["x_off"]
y_off = params["y_off"]
offset = np.array([[1, 0, x_off], [0, 1, y_off], [0, 0, 1]])
new_H = np.matmul(offset, np.linalg.inv(H))
img2_after_homography = cv.warpPerspective(img2, new_H, (5*img2.shape[1], 4*img2.shape[0]))
utils.plot_array("res20.jpg", utils.crop_zero_parts(img2_after_homography))

inliers, outliers = matching.get_inliers_and_outliers(status, matches)
no_single_inliers = cv.drawMatches(img1, kps1, img2, kps2, inliers, None, matchColor=(255, 0, 0),
                                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

utils.plot_array("inliers_Q4.jpg", no_single_inliers)

print("Homography matrix without offset:")
print(np.linalg.inv(H))
print("Homography matrix with offset:")
print(new_H)
print("number of iterations: " + str(counter))
print("number of inliers: " + str(np.sum(status)))
print("number of outliers: " + str(int(len(status) - np.sum(status))))
