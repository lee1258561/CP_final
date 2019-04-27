import numpy as np
import cv2
import math

from scipy.spatial.distance import cdist

def SIFTDescriptor(image, x, y, feature_width, scales=None):
    blur = cv2.GaussianBlur(image, (5, 5), 1, 1)
    blur = (blur - np.mean(blur)) / np.std(blur)

    Ix = cv2.Sobel(image,-1,1,0,ksize=3)
    Iy = cv2.Sobel(image,-1,0,1,ksize=3)

    orientation = np.arctan2(Iy, Ix)
    magnitude = np.hypot(Iy, Ix)

    bins = np.linspace(-math.pi, math.pi, num=9)
    sub_grid_width = int(feature_width / 4)

    fv = []
    for x_cor, y_cor in zip(x, y):
        x_cor = int(x_cor); y_cor = int(y_cor)

        #find patch range
        x_range = (int(x_cor - feature_width / 2), int(x_cor + feature_width / 2))
        if x_range[0] < 0: x_range = (0, feature_width)
        elif x_range[1] > image.shape[1]: x_range = (image.shape[1] - feature_width, image.shape[1])
        y_range = (int(y_cor - feature_width / 2), int(y_cor + feature_width / 2))
        if y_range[0] < 0: y_range = (0, feature_width)
        elif y_range[1] > image.shape[0]: y_range = (image.shape[0] - feature_width, image.shape[0])

        grid_orient = orientation[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        #TODO: Gaussian normalize grid mag
        grid_mag = magnitude[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        
        feature = np.array([])

        #get sub grid histogram
        for i in range(4):
            for j in range(4):
                sub_grid_orient = grid_orient[i * sub_grid_width:i * sub_grid_width + 4, j * sub_grid_width:j * sub_grid_width + 4]
                sub_grid_mag = grid_mag[i * sub_grid_width:i * sub_grid_width + 4, j * sub_grid_width:j * sub_grid_width + 4]
                hist, _ = np.histogram(sub_grid_orient, bins, weights=sub_grid_mag)
                if (hist < 0).any():
                    print('somthing bad happened.')

                feature = np.concatenate((feature, hist))

        #normalize
        feature = feature / np.sum(feature)
        feature = np.clip(feature, 0., 0.2)
        feature = feature / np.sum(feature)
        fv.append(feature)

    fv = np.array(fv)

    return fv


def ratioTestMatching(features1, features2, x1, y1, x2, y2):

    dist_mat = cdist(features1, features2)
    sort_idx_mat = np.argsort(dist_mat, axis=0)
    images2_NN_idx = sort_idx_mat[0,:]
    sort_mat = np.sort(dist_mat, axis=0)

    threshold = 0.8
    confidence = sort_mat[0] / sort_mat[1]
    match_idx = np.argsort(confidence)

    matches = []
    for image2_idx in match_idx:
        if confidence[image2_idx] < threshold:
            idx1, idx2 = images2_NN_idx[image2_idx], image2_idx
            matches.append([x1[idx1], y1[idx1], x2[idx2], y2[idx2], 1 / confidence[idx2]])

    return np.array(matches)


