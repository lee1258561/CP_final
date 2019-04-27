import os
import cv2
import numpy as np

def warpLocal(src, uv):
    width = src.shape[1]
    height  = src.shape[0]
    mask = cv2.inRange(uv[:,:,1],0,height-1.0) & cv2.inRange(uv[:,:,0],0,width-1.0)
    warped = cv2.remap(src, uv[:, :, 0].astype(np.float32), uv[:, :, 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    image = cv2.bitwise_and(warped, warped, mask = mask)
    return image

def computeSphericalWarpMappings(shape, f, k1, k2):
    
    mat = np.zeros(3)
    mat[0] = np.sin(0.0) * np.cos(0.0)
    mat[1] = np.sin(0.0)
    mat[2] = np.cos(0.0) * np.cos(0.0)
    y_min = mat[1] # get min y value

    # calculate spherical coordinates
    # (x,y) is the spherical image coordinates.
    # (theta,phi) is the spherical coordinates, e.g., theta is the angle theta
    # and phi is the angle phi
    one = np.ones((shape[0], shape[1]))
    theta = one * np.arange(shape[1])
    phi = one.T * np.arange(shape[0])
    phi = phi.T

    theta = ((theta - 0.5 * shape[1]) / f)
    phi = ((phi - 0.5 * shape[0]) / f - y_min)

    # projection on the sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(phi)
    z = np.cos(theta) * np.cos(phi)

    # normalized image coordinates
    x /= z
    y /= z

    # radial distortion
    r_sqr = x ** 2 + y ** 2
    coef = (1 + k1 * r_sqr + k2 * r_sqr ** 2)
    xt = x * coef
    yt = y * coef

    # Convert back to regular pixel coordinates
    xn = 0.5 * shape[1] + xt * f
    yn = 0.5 * shape[0] + yt * f
    uv = np.dstack((xn,yn))

    return uv

def warpSpherical(image, f):
    #     image:filename(string)
    #     f:focal length in pixel as int
    #     output image in a numpy array with values from 0 to 255. 
    #     The dimensions are (rows, cols, color bands BGR).

    # compute spherical warp by computing the mapping from source to dest image 
    k1 = -0.21
    k2 = 0.26
    uv = computeSphericalWarpMappings(np.array(image.shape), f, k1, k2)

    return warpLocal(image, uv)
