import cv2
import numpy as np
from src.utils import *

class Panorama(object):
	def __init__(self, image_datas, Hs):
		self.image_datas = image_datas
		self.Hs = Hs

	def stitch(self):
		min_x, min_y, max_x, max_y = self.get_canvas_range()
		h, w = max_y - min_y, max_x - min_x
		tM = np.array([[1, 0, -min_x],
					   [0, 1, -min_y],
					   [0, 0, 1]])

		stitched_image = np.zeros((h, w, 3))
		warped_images, masks = [], []
		for i in range(len(self.Hs)):
			MtoCanvas = np.matmul(tM, self.Hs[i])
			warp_img = cv2.warpPerspective(self.image_datas['sequences'][i]['color'], MtoCanvas, (w, h))
			warped_images.append(warp_img)

			warp_gray = cv2.cvtColor(warp_img, cv2.COLOR_RGB2GRAY)
			_, mask = cv2.threshold(warp_gray, 0, 1, cv2.THRESH_BINARY)
			mask = np.stack((mask, mask, mask), axis=2).astype('bool')
			masks.append(mask)

			np.copyto(stitched_image, warp_img, where=mask)

		return stitched_image

	def adjust_H_to_center(self, center=0):
		adjust_M = np.linalg.inv(self.Hs[center])
		for i in range(len(self.Hs)):
			self.Hs[i] = np.matmul(adjust_M, self.Hs[i])

	def get_canvas_range(self):
		rows, cols = self.image_datas['sequences'][0]['gray'].shape
		corner = np.array([[0, 0, 1],
						   [0, rows, 1],
						   [cols, 0, 1],
						   [cols, rows, 1]])

		all_corner = []
		for i in range(len(self.Hs)):
			all_corner.append(np.matmul(self.Hs[i], corner.T).T)

		all_corner = from_homo(np.vstack(all_corner))

		min_x = int(np.floor(np.min(all_corner[:, 0])))
		min_y = int(np.floor(np.min(all_corner[:, 1])))
		max_x = int(np.ceil(np.max(all_corner[:, 0])))
		max_y = int(np.ceil(np.max(all_corner[:, 1])))

		return min_x, min_y, max_x, max_y
