import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from src.warp import *
from src.config import *

def create_path(*arg, filename=None):
    path = os.getcwd()
    for directory in arg:
        path = os.path.join(path, directory)
        if not os.path.exists(path):
            print('%s doesn\'t exist, creating...' % path)
            os.mkdir(path)

    if filename:
        path = os.path.join(path, filename)
    return path

def save_image(image, filename):
    plt.figure()
    plt.imshow(image)
    plt.savefig(filename, dpi=1000)
    plt.clf()

def normalize_image(image):
    return cv2.normalize(image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

def prepare_image_data(file_dir, focal_length=None, data_dir='data'):
    image_datas = {'suffix': file_dir, 'sequences': []}
    if VERBOSE and focal_length is not None: print ("Do warping after loading images.")

    idx = 0
    path = os.path.join(data_dir, file_dir, '1.jpg')
    while os.path.isfile(path):
        image_datas['sequences'].append({})
        img = cv2.imread(path)
        if focal_length is not None:
            img, mask = warpSpherical(img, focal_length)
            image_datas['sequences'][idx]['mask'] = mask
        norm_img = normalize_image(img)
        image_datas['sequences'][idx]['color'] = norm_img[:, :, ::-1]
        image_datas['sequences'][idx]['gray'] = cv2.cvtColor(norm_img, cv2.COLOR_RGB2GRAY)

        idx += 1
        path = os.path.join(data_dir, file_dir, str(idx + 1) + '.jpg')

    return image_datas

def to_homo(coordinates):
    return np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))

def from_homo(homo_coordinates):
    return (homo_coordinates[:, 0:2].T / homo_coordinates[:, 2]).T


def hstack_images(imgA, imgB):
    Height = max(imgA.shape[0], imgB.shape[0])
    Width  = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB

    return newImg

def show_correspondence_circles(imgA, imgB, X1, Y1, X2, Y2):
    newImg = hstack_images(imgA, imgB)
    shiftX = imgA.shape[1]
    X1 = X1.astype(np.int)
    Y1 = Y1.astype(np.int)
    X2 = X2.astype(np.int)
    Y2 = Y2.astype(np.int)

    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        cur_color = np.random.rand(3)
        green = (0, 1, 0)
        newImg = cv2.circle(newImg, (x1, y1), 10, cur_color, -1, cv2.LINE_AA)
        newImg = cv2.circle(newImg, (x1, y1), 10, green, 2, cv2.LINE_AA)
        newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, cur_color, -1, cv2.LINE_AA)
        newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, green, 2, cv2.LINE_AA)

    return newImg

def show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
    newImg = hstack_images(imgA, imgB)
    shiftX = imgA.shape[1]
    X1 = X1.astype(np.int)
    Y1 = Y1.astype(np.int)
    X2 = X2.astype(np.int)
    Y2 = Y2.astype(np.int)

    dot_colors = np.random.rand(len(X1), 3)
    if line_colors is None:
        line_colors = dot_colors

    for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors,
            line_colors):
        newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
        newImg = cv2.circle(newImg, (x2+shiftX, y2), 5, dot_color, -1)
        newImg = cv2.line(newImg, (x1, y1), (x2+shiftX, y2), line_color, 2,
                                            cv2.LINE_AA)
    return newImg