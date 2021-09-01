import numpy as np
import cv2 as cv
import scipy.signal as sig
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from Utils import utils


def get_gray_image(path):
    im = cv.imread(path)
    gray_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    return gray_image


def get_grad(path):
    """"
    I used sobel filters to compute the derivations of the image
    """
    Gx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gy = Gx.transpose()
    gray_image = get_gray_image(path)
    Ix = sig.convolve2d(gray_image, Gx, mode='valid')
    Iy = sig.convolve2d(gray_image, Gy, mode='valid')
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    return Ix2, Iy2, Ixy


def plot_abs_grad(path, fig_name):
    Ix2, Iy2, _ = get_grad(path)
    abs_grad = np.sqrt(Ix2 + Iy2)
    utils.plot_array(fig_name, abs_grad)


def applying_guassian_filter(path, sigma, save=None):
    Ix2, Iy2, Ixy = get_grad(path)
    Sx2 = nd.gaussian_filter(Ix2, sigma=sigma)
    Sy2 = nd.gaussian_filter(Iy2, sigma=sigma)
    Sxy = nd.gaussian_filter(Ixy, sigma=sigma)
    if save:
        utils.plot_array("Sx2" + save, Sx2)
        utils.plot_array("Sy2" + save, Sy2)
        utils.plot_array("Sxy" + save, Sxy)
    return Sx2, Sy2, Sxy


def compute_score(Sx2, Sy2, Sxy, k):
    det = Sx2*Sy2 - (Sxy**2)
    tr = Sx2 + Sy2
    score = det - k*(tr**2)
    return score


def non_maximum_suppression(scores, tr, radius):
    lx, ly = scores.shape
    scores = scores * (scores > tr)
    new_scores = np.copy(scores)
    for x in range(lx):
        for y in range(ly):
            if new_scores[x, y] <= tr:
                new_scores[x, y] = 0
                continue

            new_scores = one_point_nms(new_scores, radius, x, y)
    return new_scores


def one_point_nms(scores, radius, x, y):
    lx, ly = scores.shape
    copy_scores = np.copy(scores)
    for dx in range(radius):
        if x + dx >= lx:
            break
        for dy in range(-radius, radius):
            if y + dy >= ly:
                break
            if (dy == 0 and dx == 0) or dx**2 + dy**2 >= radius**2 or y + dy < 0:
                continue
            if scores[x + dx, y + dy] <= scores[x, y]:
                copy_scores[x + dx, y + dy] = 0
            else:
                scores[x, y] = 0
                return scores
    return copy_scores


def get_feature_vector(gray_image, x, y, n):
    feature_vector = np.zeros((2*n+1)**2)
    lx, ly = gray_image.shape

    if x < n or y < n or x >= lx + n or y >= ly + n:
        feature_vector[0] = None
        return feature_vector
    i = 0
    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            feature_vector[i] = gray_image[x + dx, y + dy]
            i += 1
    return feature_vector


def get_nonzero_locs(array):
    locs_2d_list = np.nonzero(array)
    locs = []
    for i in range(locs_2d_list[0].shape[0]):
        x = locs_2d_list[0][i]
        y = locs_2d_list[1][i]
        locs.append([x, y])
    return locs


def get_int_points_features(int_points, gray_image, n):
    int_points = get_nonzero_locs(int_points)
    features_out = []
    int_points_out = []
    for i in range(len(int_points)):
        x = int_points[i][0]
        y = int_points[i][1]
        feature_vector = get_feature_vector(gray_image, x, y, n)
        if feature_vector[0] is not None:
            features_out.append(feature_vector)
            int_points_out.append([x, y])
    return features_out, int_points_out


def get_dist(feat1, feat2):
    return np.sqrt(np.sum((feat1 - feat2)**2))


def get_nearests(feature, features_list):
    d1 = get_dist(feature, features_list[0])
    d2 = get_dist(feature, features_list[1])
    if d1 < d2:
        p1 = 0
        p2 = 1
    else:
        p1 = 1
        p2 = 0
        tmp = d1
        d1 = d2
        d2 = tmp

    for i in range(2, len(features_list)):
        d = get_dist(feature, features_list[i])
        if d <= d1:
            d2 = d1
            p2 = p1
            p1 = i
            d1 = d
        elif d <= d2:
            d2 = d
            p2 = i
    return p1, d1, p2, d2


def check_d1_d2_tr(tr, features1, features2):
    dic = {}
    for i in range(len(features1)):
        p1, d1, p2, d2 = get_nearests(features1[i], features2)
        if d2/d1 > tr:
            dic[i] = p1
    return dic


def get_corresponding_points(dic1, dic2):
    points = []
    for p1 in dic1.keys():
        p2 = dic1[p1]
        if p2 in dic2.keys():
            if dic2[p2] == p1:
                points.append([p1, p2])
    return points


def plot_final_interest_points(path, points, num, inp, name, size=(40, 40)):
    plt.figure(figsize=size)
    plt.imshow(utils.get_image(path))
    for i in range(len(points)):
        xy = inp[points[i][num]]
        plt.plot(xy[1], xy[0], 'ro', markersize=15)
        plt.axis("off")
    utils.save_fig(name)


def plot_corresponding_points(path1, path2, points, inp1, inp2, name, num_points=20, size=(40, 40)):
    fig = plt.figure(figsize=size)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(utils.get_image(path1))
    ax2.imshow(utils.get_image(path2))
    cm = plt.get_cmap('gist_rainbow')
    for i in range(min(len(points), num_points)):
        xy1 = inp1[points[i][0]].copy()
        xy2 = inp2[points[i][1]].copy()

        tmp = xy1[0]
        xy1[0] = xy1[1]
        xy1[1] = tmp
        tmp = xy2[0]
        xy2[0] = xy2[1]
        xy2[1] = tmp

        color = cm(i / min(len(points), num_points))
        con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2, color=color, linewidth=3)
        ax2.add_artist(con)

        ax1.plot(xy1[0], xy1[1], 'ro', markersize=15, color=color)
        ax2.plot(xy2[0], xy2[1], 'ro', markersize=15, color=color)

    ax1.axis("off")
    ax2.axis("off")

    utils.save_fig(name)
