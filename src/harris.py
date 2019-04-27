import cv2
import numpy as np

def harrisKeyPoint(image, feature_width, alpha=0.06, n=1500):
    confidences, scales, orientations = None, None, None

    Ix = cv2.Sobel(image,-1,1,0,ksize=5)
    Iy = cv2.Sobel(image,-1,0,1,ksize=5)

    kernel_size, std_x, std_y = (11, 11), 2, 2
    Sxx = cv2.GaussianBlur(Ix ** 2, kernel_size, std_x, std_y)
    Sxy = cv2.GaussianBlur(Ix * Iy, kernel_size, std_x, std_y)
    Syy = cv2.GaussianBlur(Iy ** 2, kernel_size, std_x, std_y)

    R = (Sxx * Syy) - (Sxy ** 2) - alpha * ((Sxx + Syy) ** 2)


    point_list = []
    mean_R = np.mean(R)
    for x in range(R.shape[1]):
        for y in range(R.shape[0]):
            if R[y, x] > mean_R: point_list.append([x, y, R[y, x]])

    #thresholding and sort
    point_list = np.array(sorted(point_list, key=lambda x: x[2], reverse=True))

    # Non-Maximum Suppression
    radius_list = []
    radius_list.append([point_list[0,0], point_list[0,1], float('inf'), 0])

    for i in range(1, point_list.shape[0]):
        stronger_x, stronger_y = point_list[:i, 0], point_list[:i, 1]
        stronger_radius = np.sqrt((stronger_x - point_list[i, 0]) ** 2 + (stronger_y - point_list[i, 1]) ** 2)
        radius_list.append([point_list[i, 0], point_list[i, 1], np.min(stronger_radius), i])
        

    radius_list.sort(key=lambda x: x[2], reverse=True)

    radius_list = np.array(radius_list)[:n]
    print ("last index: ", radius_list[-1, 3])
    x = radius_list[:, 0]
    y = radius_list[:, 1]

    return x, y, confidences, scales, orientations