import cv2
import numpy as np
from src.utils import * 
from src.config import *

def ransac(matches, e=0.9, s=4, p=0.99, threshold=1.5, useHomo=True):

    if HARD_ITER is not None: iter_N = HARD_ITER
    else: iter_N = int(np.log(1 - p) / np.log(1 - ((1 - e) ** s)))

    match_N = matches.shape[0]
    src = matches[:, 0:2]
    target = matches[:,2:4]
    homo_src = to_homo(src)
    homo_target = to_homo(target)

    best_M, best_match_index, best_choice_index = None, np.array([]), np.array([])

    if VERBOSE: 
        print("Start RANSAC with %d iteration" % iter_N)
        if useHomo: print ("Use homography motion model.")
        else: print ("Use translation motion model.")

    for i in range(iter_N):
        choice = np.random.choice(match_N, size=s, replace=False)
        random_matches = np.take(matches, choice, axis=0).astype(np.float32)
        ref = np.float32(random_matches[:, 0:2].tolist())
        dst = np.float32(random_matches[:, 2:4].tolist())

        if useHomo: M = cv2.getPerspectiveTransform(ref, dst)
        else: M = getTranslateModel(ref, dst)

        homo_src_transform = np.matmul(M, homo_src.T).T
        match_errors = np.sum((from_homo(homo_src_transform) - target) ** 2, axis=1)

        agree_match_index = np.arange(match_N)[match_errors < threshold]

        #print (agree_match_index.shape[0])
        if agree_match_index.shape[0] > best_match_index.shape[0]:
            if VERBOSE: print ("New best model found. Number of inliers: %d" % agree_match_index.shape[0])
            best_M = M
            best_choice_index = choice
            best_match_index = agree_match_index

    inliers = np.take(matches, best_match_index, axis=0)
    choices = np.take(matches, best_choice_index, axis=0)

    return best_M, inliers, choices

def getTranslateModel(ref, dst):
    M = np.eye(3)
    translation = np.mean(dst - ref, axis=0)

    M[0, 2] = translation[0]
    M[1, 2] = translation[1]
    return M
