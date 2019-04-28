VERBOSE = True

# Camera Focal Length Parameter
# Use the following expression to calcuate focal length in pixel unit
# 		FOCAL_LENGTH = camera focal length (mm) / camera sensor width (mm) * image width in pixel	
FOCAL_LENGTH = 20.0 / 23.5 * 750 

# Harris Corner Detector Parameter
ALPHA = 0.06
N_KEYPOINTS = 1500
NUM_CORNER_CONSIDERED = 10000

# SIFT Descriptor Parameter
SIFT_SIZE = 16

# Ratio Test Parameter
THRESHOLD = 0.8

# RANSAC Parameter
ERROR_RATE = 0.9
SAMPLE_NUM = 4
CORRECT_PROP = 0.99
INLIERS_THRESHOLD = 1.5

