import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np


def crop_zero_parts(image):
    x, y, _ = np.nonzero(image)

    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)

    return image[x_min:x_max, y_min:y_max, :]


def plot_array(name, array, size=(40, 40)):
    if size is not None:
        plt.figure(figsize=size)
    plt.imshow(array, cmap='gray')
    plt.axis('off')
    if name is not None:
        save_fig(name)


def plot_double_arrays(name, array1, array2, size=(40, 40)):
    plt.figure(figsize=size)
    plt.subplot(1, 2, 1)
    plot_array(None, array1, size=None)
    plt.subplot(1, 2, 2)
    plot_array(None, array2, size=None)
    save_fig(name)


def save_fig(name):
    if not os.path.exists("Results/"):
        os.mkdir("Results/")
    plt.savefig("Results/" + name, bbox_inches='tight')


def get_image(path):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


def replace_rows(mat, r1, r2):
    out = mat.copy()
    tmp = out[r1, :].copy()
    out[r1, :] = out[1, :]
    out[r2, :] = tmp
    return out
