import numpy as np
from Utils import utils, config_reader

params = config_reader.param_config("Q2")

logo = utils.get_image(params['image_path'])
Px = logo.shape[0]/2
Py = logo.shape[1]/2
f = params['f']
d = params['d']
z = params['z']
n1 = params['n1']
n2 = params['n2']
n3 = params['n3']
n = np.array([[n1, n2, n3]])
K = np.array([[f, 0, Px], [0, f, Py], [0, 0, 1]])
theta = -np.arctan(z/d)
sin_th = np.sin(theta)
cos_th = np.cos(theta)
R = np.array([[cos_th, 0, -sin_th], [0, 1, 0], [sin_th, 0, cos_th]])
C = np.array([[z, 0, 0]]).transpose()
t = -np.matmul(R, C)

H = np.matmul(np.matmul(K, R - np.matmul(t, n)/d), np.linalg.inv(K))

i = 0
j = 0
x = np.array([[i, j, 1]]).transpose()
x_p = np.matmul(H, x)
min_i = int(x_p[0]/x_p[2])
min_j = int(x_p[1]/x_p[2])
max_i = min_i
max_j = min_j
for i in range(logo.shape[0]):
    for j in range(logo.shape[1]):
        x = np.array([[i, j, 1]]).transpose()
        x_p = np.matmul(np.linalg.inv(H), x)
        i_p = int(x_p[0]/x_p[2])
        j_p = int(x_p[1]/x_p[2])
        min_i = min(min_i, i_p)
        min_j = min(min_j, j_p)
        max_i = max(max_i, i_p)
        max_j = max(max_j, j_p)

out = np.zeros((max_i - min_i + 1, max_j - min_j + 1, 3))

for i in range(out.shape[0]):
    for j in range(out.shape[1]):
        x = np.array([[i + min_i, j + min_j, 1]]).transpose()
        x_p = np.matmul(H, x)
        i_p = int(x_p[0]/x_p[2])
        j_p = int(x_p[1]/x_p[2])
        if logo.shape[0] > i_p > 0 and logo.shape[1] > j_p > 0:
            out[i, j, :] = logo[i_p, j_p, :]

utils.plot_array('res12.jpg', out.astype(np.int32))
