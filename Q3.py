import cv2 as cv
from Utils import utils, config_reader
from Questions import matching
import numpy as np

params = config_reader.param_config("Q3")

img1 = utils.get_image("im03.jpg")
img2 = utils.get_image("im04.jpg")

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

img1_with_kps, kps1, des1 = matching.get_sift_key_points(img1, green)
img2_with_kps, kps2, des2 = matching.get_sift_key_points(img2, green)

utils.plot_double_arrays("res13_corners.jpg", img1_with_kps, img2_with_kps)

ratio_tr = params["ratio_tr"]
matches = matching.get_match_points(des1, des2, ratio_tr)

src_points = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
des_points = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

img1_with_kps_and_matches = cv.drawKeypoints(img1_with_kps, [kps1[m.queryIdx] for m in matches],
                                             None, color=blue)
img2_with_kps_and_matches = cv.drawKeypoints(img2_with_kps, [kps2[m.trainIdx] for m in matches],
                                             None, color=blue)

utils.plot_double_arrays("res14_correspondences.jpg", img1_with_kps_and_matches, img2_with_kps_and_matches)

all_matched_points = cv.drawMatches(img1, kps1, img2, kps2, matches, None,
                                    matchColor=blue, singlePointColor=green)

utils.plot_array("res15_matches.jpg", all_matched_points)

twenty_matched_points = cv.drawMatches(img1, kps1, img2, kps2, matches[:20], None,
                                       matchColor=blue, singlePointColor=green)

utils.plot_array("res16.jpg", twenty_matched_points)

N = int(params['n'])
x_off = params["x_off"]
y_off = params["y_off"]

new_H, H, status = matching.get_homography_opencv(src_points, des_points, N, x_off=x_off, y_off=y_off)
img2_after_homography = cv.warpPerspective(img2, new_H, (5*img2.shape[1], 4*img2.shape[0]))

utils.plot_array("res19.jpg", utils.crop_zero_parts(img2_after_homography))
inliers, outliers = matching.get_inliers_and_outliers(status, matches)

no_single_all_matches = cv.drawMatches(img1, kps1, img2, kps2, matches, None, matchColor=blue,
                                       flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

flags = (cv.DrawMatchesFlags_DRAW_OVER_OUTIMG + cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
no_single_all_matches_inliers = cv.drawMatches(img1, kps1, img2, kps2, inliers, no_single_all_matches.copy(),
                                               matchColor=red, flags=flags)

utils.plot_array("res17.jpg", no_single_all_matches_inliers)

no_single_inliers = cv.drawMatches(img1, kps1, img2, kps2, inliers, None, matchColor=red,
                                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

utils.plot_array("inliers_Q3.jpg", no_single_inliers)

print("Homography matrix without offset:")
print(H)
print("Homography matrix with offset:")
print(new_H)
print("number of inliers: " + str(np.sum(status)))
print("number of outliers: " + str(int(len(status) - np.sum(status))))
