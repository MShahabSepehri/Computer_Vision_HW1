from Questions import Harris
from Utils import utils, config_reader


path1 = "im01.jpg"
path2 = "im02.jpg"
Harris.plot_abs_grad(path1, "res01_grad.jpg")
Harris.plot_abs_grad(path2, "res02_grad.jpg")

params = config_reader.param_config("Q1")
# For the guassian filter's variance I used sigma = 2 based on figures that are in my notebook
sigma = params['sigma']
Sx2_01, Sy2_01, Sxy_01 = Harris.applying_guassian_filter(path1, sigma=sigma, save="01")
Sx2_02, Sy2_02, Sxy_02 = Harris.applying_guassian_filter(path2, sigma=sigma, save="02")

k = params['k']
score_01 = Harris.compute_score(Sx2_01, Sy2_01, Sxy_01, k)
score_02 = Harris.compute_score(Sx2_02, Sy2_02, Sxy_02, k)

utils.plot_array("res03_score.jpg", score_01)
utils.plot_array("res04_score.jpg", score_02)

tr = params['threshold']
utils.plot_array("res05_thresh.jpg", score_01 * (score_01 > tr))
utils.plot_array("res06_thresh.jpg", score_02 * (score_02 > tr))

radius = int(params['nms_radius'])
int_points1 = Harris.non_maximum_suppression(score_01, tr, radius)
int_points2 = Harris.non_maximum_suppression(score_02, tr, radius)
utils.plot_array("res07_harris.jpg", int_points1 != 0)
utils.plot_array("res08_harris.jpg", int_points2 != 0)

n = int(params['n'])
features1, int_points1 = Harris.get_int_points_features(int_points1, Harris.get_gray_image("im01.jpg"), n)
features2, int_points2 = Harris.get_int_points_features(int_points2, Harris.get_gray_image("im02.jpg"), n)

d1_d2_tr = params['d2_d1_ratio']
dic1 = Harris.check_d1_d2_tr(d1_d2_tr, features1, features2)
dic2 = Harris.check_d1_d2_tr(d1_d2_tr, features2, features1)

points = Harris.get_corresponding_points(dic1, dic2)

Harris.plot_final_interest_points(path1, points, 0, int_points1, "res09_corres.jpg")
Harris.plot_final_interest_points(path2, points, 1, int_points2, "res10_corres.jpg")

Harris.plot_corresponding_points("im01.jpg", "im02.jpg", points, int_points1, int_points2, "res11.jpg")
