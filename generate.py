import argparse

from src.harris import harrisKeyPoint
from src.SIFT import SIFTDescriptor, ratioTestMatching
from src.RANSAC import ransac
from src.panorama import Panorama
from src.utils import *
from src.config import *


def parse_argument():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--input_dir', help='Input directory in data/ which contains three\
											 images (left_image.jpg, reference_image.jpg,\
											 and right_image.jpg) that is needed for \
											 generating panorama.')
	parser.add_argument("--warp", help="Do spherical warping if specified.", action="store_true")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_argument()

	focal_length = None
	if args.warp: 
		focal_length = FOCAL_LENGTH

	image_datas = prepare_image_data(args.input_dir, focal_length=focal_length)
	filename = create_path('results', image_datas['suffix'], filename='reference_warped.jpg')
	save_image(image_datas['reference']['color'], filename)

	# Key point detection
	x_left, y_left = harrisKeyPoint(image_datas['left']['gray'], alpha=ALPHA, n=N_KEYPOINTS)
	x_ref, y_ref = harrisKeyPoint(image_datas['reference']['gray'], alpha=ALPHA, n=N_KEYPOINTS)
	x_right, y_right = harrisKeyPoint(image_datas['right']['gray'], alpha=ALPHA, n=N_KEYPOINTS)

	# SIFT feature extraction
	left_features = SIFTDescriptor(image_datas['left']['gray'], x_left, y_left, SIFT_SIZE)
	ref_features = SIFTDescriptor(image_datas['reference']['gray'], x_ref, y_ref, SIFT_SIZE)
	right_features = SIFTDescriptor(image_datas['right']['gray'], x_right, y_right, SIFT_SIZE)

	# SIFT feature matching with ratio test
	left_ref_matches = ratioTestMatching(left_features, ref_features, x_left, y_left, x_ref, y_ref, threshold=THRESHOLD)
	right_ref_matches = ratioTestMatching(right_features, ref_features, x_right, y_right, x_ref, y_ref, threshold=THRESHOLD)

	# print('%d matches from %d corners for reference and left images'.format(len(left_ref_matches), len(x_ref)))
	# print('%d matches from %d corners for reference and right images'.format(len(right_ref_matches), len(x_ref)))

	# Visualize corresondence without RANSAC
	num_pts_to_visualize = len(left_ref_matches)
	vis = show_correspondence_lines(image_datas['left']['color'], 
								  	image_datas['reference']['color'],
								   	left_ref_matches[:num_pts_to_visualize, 0],
								   	left_ref_matches[:num_pts_to_visualize, 1],
	                    		   	left_ref_matches[:num_pts_to_visualize, 2], 
	                    		   	left_ref_matches[:num_pts_to_visualize, 3])

	filename = create_path('results', image_datas['suffix'], filename='vis_lines_left_ref.jpg')
	save_image(vis, filename)

	num_pts_to_visualize = len(right_ref_matches)
	vis = show_correspondence_lines(image_datas['reference']['color'], 
								  	image_datas['right']['color'],
								   	right_ref_matches[:num_pts_to_visualize, 2],
								   	right_ref_matches[:num_pts_to_visualize, 3],
	                    		   	right_ref_matches[:num_pts_to_visualize, 0], 
	                    		   	right_ref_matches[:num_pts_to_visualize, 1])

	filename = create_path('results', image_datas['suffix'], filename='vis_lines_ref_right.jpg')
	save_image(vis, filename)

	#RANSAC for best correspondences
	left_best_H, left_inliers, left_choices = ransac(left_ref_matches,
													 e=ERROR_RATE, 
													 s=SAMPLE_NUM, 
													 p=CORRECT_PROP, 
													 threshold=INLIERS_THRESHOLD)

	right_best_H, right_inliers, right_choices = ransac(right_ref_matches,
													 e=ERROR_RATE, 
													 s=SAMPLE_NUM, 
													 p=CORRECT_PROP, 
													 threshold=INLIERS_THRESHOLD)

	# Visualize corresondence with RANSAC
	vis = show_correspondence_lines(image_datas['left']['color'], 
								  	image_datas['reference']['color'],
								   	left_choices[:, 0],
								   	left_choices[:, 1],
	                    		   	left_choices[:, 2], 
	                    		   	left_choices[:, 3])
	filename = create_path('results', image_datas['suffix'], filename='vis_lines_ransac_left_ref.jpg')
	save_image(vis, filename)

	vis = show_correspondence_lines(image_datas['reference']['color'], 
								  	image_datas['right']['color'],
								   	right_choices[:, 2],
								   	right_choices[:, 3],
	                    		   	right_choices[:, 0], 
	                    		   	right_choices[:, 1])
	filename = create_path('results', image_datas['suffix'], filename='vis_lines_ransac_ref_right.jpg')
	save_image(vis, filename)

	# Panorama stitching
	Hs = {'left': left_best_H, 'right': right_best_H}
	p = Panorama(image_datas, Hs)
	stitched_image = p.stitch()

	filename = create_path('results', image_datas['suffix'], filename='panorama_result.jpg')
	save_image(stitched_image, filename)

