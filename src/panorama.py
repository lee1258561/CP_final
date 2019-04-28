import cv2
import numpy as np
from src.utils import *

class Panorama(object):
	def __init__(self, image_datas, Hs):
		self.image_datas = image_datas
		self.Hs = Hs

		self.Hs['reference'] = np.eye(3)

	def stitch(self):
		min_x, min_y, max_x, max_y = self.get_canvas_range()
		h, w = max_y - min_y, max_x - min_x
		tM = np.array([[1, 0, -min_x],
					   [0, 1, -min_y],
					   [0, 0, 1]])

		stitched_image = np.zeros((h, w, 3))
		warped_images, masks = [], []
		for pos in ['left', 'reference', 'right']:
			MtoCanvas = np.matmul(tM, self.Hs[pos])
			warp_img = cv2.warpPerspective(self.image_datas[pos]['color'], MtoCanvas, (w, h))
			warped_images.append(warp_img)

			warp_gray = cv2.cvtColor(warp_img, cv2.COLOR_RGB2GRAY)
			_, mask = cv2.threshold(warp_gray, 0, 1, cv2.THRESH_BINARY)
			mask = np.stack((mask, mask, mask), axis=2).astype('bool')
			masks.append(mask)


			np.copyto(stitched_image, warp_img, where=mask)

		return stitched_image


	def get_canvas_range(self):
		rows, cols = self.image_datas['reference']['gray'].shape
		corner = np.array([[0, 0, 1],
						   [0, rows, 1],
						   [cols, 0, 1],
						   [cols, rows, 1]])
		left_corner = np.matmul(self.Hs['left'], corner.T).T
		right_corner = np.matmul(self.Hs['right'], corner.T).T

		all_corner = from_homo(np.vstack((corner, left_corner, right_corner)))

		min_x = int(np.floor(np.min(all_corner[:, 0])))
		min_y = int(np.floor(np.min(all_corner[:, 1])))
		max_x = int(np.ceil(np.max(all_corner[:, 0])))
		max_y = int(np.ceil(np.max(all_corner[:, 1])))

		return min_x, min_y, max_x, max_y