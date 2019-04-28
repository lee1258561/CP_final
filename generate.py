import argparse

from src.harris import harrisKeyPoint
from src.SIFT import SIFTDescriptor, ratioTestMatching
from src.RANSAC import ransac
from src.panorama import Panorama
from src.utils import *
from src.config import *


def parse_argument():
	parser = argparse.ArgumentParser(description='Automated Panorama with Warping')
	parser.add_argument('--input_dir', 
						help='Input directory in data/ which contains images\
							  that is needed for generating panorama. The \
							  filename of the images should be 1.jpg ~ N.jpg\
							  if you want to use N images to generate panorama.\
							  The order of the image should be from left to right\
							  with sufficient overlap between two adjacent images.')

	parser.add_argument("--warp", 
						action="store_true",
						help="Do spherical warping if specified. The FOCAL_LENGTH\
							  parameter need to be changed with repect to your \
							  camera setting in order to get the correct result of\
							  warping.")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_argument()

	focal_length = None
	if args.warp: 
		focal_length = FOCAL_LENGTH

	image_datas = prepare_image_data(args.input_dir, focal_length=focal_length)
	center_idx = int(len(image_datas['sequences']) / 2)
	# center_idx = 0

	print (len(image_datas['sequences']))
	filename = create_path('results', image_datas['suffix'], filename='reference_warped.jpg')
	save_image(image_datas['sequences'][center_idx]['color'], filename)

	cur_H = np.eye(3)
	X, Y, feats, Hs = [], [], [], [cur_H]
	for i in range(len(image_datas['sequences'])):
		# Key point detection
		x, y = harrisKeyPoint(image_datas['sequences'][i]['gray'], alpha=ALPHA, n=N_KEYPOINTS)
		X.append(x)
		Y.append(y)
		# SIFT feature extraction
		features = SIFTDescriptor(image_datas['sequences'][i]['gray'], x, y, SIFT_SIZE)
		feats.append(features)
		if i != 0:
			# SIFT feature matching with ratio test
			matches = ratioTestMatching(features, feats[i - 1], x, y, X[i - 1], Y[i - 1], threshold=THRESHOLD)
			
			# Visualize corresondence without RANSAC
			num_pts_to_visualize = len(matches)
			vis = show_correspondence_lines(image_datas['sequences'][i - 1]['color'], 
										  	image_datas['sequences'][i]['color'],
										   	matches[:num_pts_to_visualize, 2],
										   	matches[:num_pts_to_visualize, 3],
			                    		   	matches[:num_pts_to_visualize, 0], 
			                    		   	matches[:num_pts_to_visualize, 1])

			filename = create_path('results', image_datas['suffix'], 'match_vis', filename='vis_lines_%d_%d.jpg' % (i - 1, i))
			save_image(vis, filename)

			# RANSAC for best correspondences
			best_H, inliers, choices = ransac(matches,
											  e=ERROR_RATE, 
											  s=SAMPLE_NUM, 
											  p=CORRECT_PROP, 
											  threshold=INLIERS_THRESHOLD,
											  useHomo=USE_HOMO)

			vis = show_correspondence_lines(image_datas['sequences'][i - 1]['color'], 
										  	image_datas['sequences'][i]['color'],
										   	choices[:, 2],
										   	choices[:, 3],
			                    		   	choices[:, 0], 
			                    		   	choices[:, 1])
			filename = create_path('results', image_datas['suffix'], 'RANSAC_vis', filename='vis_lines_%d_%d.jpg' % (i - 1, i))
			save_image(vis, filename)

			cur_H = np.matmul(cur_H, best_H)
			Hs.append(cur_H)

	# Panorama stitching
	p = Panorama(image_datas, Hs)
	p.adjust_H_to_center(center=center_idx)
	stitched_image = p.stitch()

	filename = create_path('results', image_datas['suffix'], filename='panorama_result.jpg')
	save_image(stitched_image, filename)


