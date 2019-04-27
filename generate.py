import matplotlib.pyplot as plt
from src.harris import harrisKeyPoint
from src.SIFT import SIFTDescriptor, ratioTestMatching
from src.RANSAC import ransac
from src.utils import *

if __name__ == '__main__':
	# image1 = cv2.imread('data/reference_image.jpg')
	# image1 = cv2.normalize(image1.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)[:, :, ::-1]
	# image2 = cv2.imread('data/right_image.jpg')
	# image2 = cv2.normalize(image2.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)[:, :, ::-1]
	# # image1 = load_image('data/reference_image.jpg')
	# # image2 = load_image('data/right_image.jpg')
	# file_suffix = '_panorama'
	                    
	# image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
	# image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
	image_datas = prepare_image_data('tech_green')

	feature_width = 16

	x1, y1, _, scales1, _ = harrisKeyPoint(image_datas['right_gray'], feature_width)
	x2, y2, _, scales2, _ = harrisKeyPoint(image_datas['reference_gray'], feature_width)

	image1_features = SIFTDescriptor(image_datas['right_gray'], x1, y1, feature_width, 0)
	image2_features = SIFTDescriptor(image_datas['reference_gray'], x2, y2, feature_width, 0)

	matches = ratioTestMatching(image1_features, image2_features, x1, y1, x2, y2)

	print('{:d} matches from {:d} corners'.format(len(matches), len(x1)))

	num_pts_to_visualize = len(matches)
	c = show_correspondence_lines(image_datas['right'], 
								   image_datas['reference'],
								   matches[:num_pts_to_visualize, 0],
								   matches[:num_pts_to_visualize, 1],
	                    		   matches[:num_pts_to_visualize, 2], 
	                    		   matches[:num_pts_to_visualize, 3])
	plt.figure()
	plt.imshow(c)
	plt.savefig('results/vis_lines' +image_datas['suffix'] + '.jpg', dpi=1000)
	plt.clf()

	best_H, inliers, choices = ransac(matches)
	c = show_correspondence_lines(image_datas['right'], 
								   image_datas['reference'],
								   choices[:, 0],
								   choices[:, 1],
	                    		   choices[:, 2], 
	                    		   choices[:, 3])
	plt.figure()
	plt.imshow(c)
	plt.savefig('results/vis_lines_ransac' +image_datas['suffix'] + '.jpg', dpi=1000)
	plt.clf()

	result = cv2.warpPerspective(image_datas['right'], best_H,
			(image_datas['right'].shape[1] + image_datas['reference'].shape[1], image_datas['reference'].shape[0]))
	result[0:image_datas['reference'].shape[0], 0:image_datas['reference'].shape[1]] = image_datas['reference']

	plt.figure()
	plt.imshow(result)
	plt.savefig('results/panorama' +image_datas['suffix'] + '.jpg', dpi=1000)
	plt.clf()
