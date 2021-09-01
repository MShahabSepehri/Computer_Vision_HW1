import cv2 as cv
import numpy as np
from Utils import utils


def get_sift_key_points(image, color):
    sift = cv.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(image, None)
    out_image = cv.drawKeypoints(image, key_points, None, color=color)
    return out_image, key_points, descriptors


def get_inliers_and_outliers(status, matches):
    inliers = []
    outliers = []
    for i in range(len(status)):
        if status[i] == 1:
            inliers.append(matches[i])
        else:
            outliers.append(matches[i])
    return inliers, outliers


def get_match_points(des1, des2, ratio_tr):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches_alpha1 = bf.knnMatch(des1, des2, k=2)
    matches1 = []
    for p1, p2 in matches_alpha1:
        if p1.distance < ratio_tr * p2.distance:
            matches1.append(p1)

    matches_alpha2 = bf.knnMatch(des2, des1, k=2)
    matches2 = []
    for p1, p2 in matches_alpha2:
        if p1.distance < ratio_tr * p2.distance:
            matches2.append(p1)

    matches = []
    for m1 in matches1:
        q1 = m1.queryIdx
        t1 = m1.trainIdx
        for m2 in matches2:
            if t1 == m2.queryIdx and q1 == m2.trainIdx:
                matches.append(m1)
                break
    return matches


def get_homography_opencv(src_points, des_points, N, x_off=1, y_off=1):
    H, status = cv.findHomography(src_points, des_points, cv.RANSAC, maxIters=N)
    offset_mat = np.array([[1, 0, x_off], [0, 1, y_off], [0, 0, 1]])
    new_H = np.matmul(offset_mat, np.linalg.inv(H))
    return new_H, H, status


def compute_homography(src, des):
    A = np.zeros((8, 9))
    counter = 0
    for p_src, p_des in zip(src, des):
        x = p_src[0]
        y = p_src[1]
        x_p = p_des[0]
        y_p = p_des[1]
        A[2*counter, 0] = -x
        A[2*counter, 1] = -y
        A[2*counter, 2] = -1
        A[2*counter, 6] = x*y_p
        A[2*counter, 7] = y*y_p
        A[2*counter, 8] = y_p
        A[2*counter + 1, 3] = -x
        A[2*counter + 1, 4] = -y
        A[2*counter + 1, 5] = -1
        A[2*counter + 1, 6] = x*x_p
        A[2*counter + 1, 7] = y*x_p
        A[2*counter + 1, 8] = x_p
        counter += 1
    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape(3, 3)
    return utils.replace_rows(H, 0, 1)


def create_full_src_des(kps1, kps2, matches):
    full_src = np.zeros((3, len(matches)))
    full_des = np.zeros((3, len(matches)))
    counter = 0
    for m in matches:
        full_src[0, counter] = kps1[m.queryIdx].pt[0]
        full_src[1, counter] = kps1[m.queryIdx].pt[1]
        full_src[2, counter] = 1
        full_des[0, counter] = kps2[m.trainIdx].pt[0]
        full_des[1, counter] = kps2[m.trainIdx].pt[1]
        full_des[2, counter] = 1
        counter += 1
    return full_src, full_des


def compute_error(full_src, full_des, H):
    ans = np.matmul(H, full_src)
    d_vec = ans[2, :]
    if np.sum(d_vec == 0) != 0:
        return None
    src_trans = (ans.transpose() / d_vec[:, None]).squeeze().transpose()
    error = np.sqrt(np.sum((src_trans - full_des)**2, 0))
    return error


def my_RANSAC(kps1, kps2, matches, inlier_tr, confidence, max_itr=10000, min_itr=100):
    full_src, full_des = create_full_src_des(kps1, kps2, matches)
    max_sup = 0
    H_ans = np.zeros((3, 3))
    idxs = np.zeros(4)

    # for _ in tqdm.tqdm(range(num_itr)):
    counter = 0
    N = max_itr
    status = np.zeros(len(matches))
    while counter < max_itr and (N > counter or counter < min_itr):
        rand = np.random.choice(list(range(len(matches))), 4)

        src = [kps1[matches[rand[0]].queryIdx].pt,
               kps1[matches[rand[1]].queryIdx].pt,
               kps1[matches[rand[2]].queryIdx].pt,
               kps1[matches[rand[3]].queryIdx].pt]

        des = [kps2[matches[rand[0]].trainIdx].pt,
               kps2[matches[rand[1]].trainIdx].pt,
               kps2[matches[rand[2]].trainIdx].pt,
               kps2[matches[rand[3]].trainIdx].pt]

        H = compute_homography(src, des)
        error = compute_error(full_src, full_des, H)
        if error is None:
            continue
        sup = np.sum(error < inlier_tr)
        if sup > max_sup:
            max_sup = sup
            status = (error < inlier_tr)
            idxs = rand
            H_ans = H
            w = sup/len(matches)
            N = np.log(1 - confidence)/np.log(1 - w**4)

        counter += 1

    return status, idxs, H_ans, counter
