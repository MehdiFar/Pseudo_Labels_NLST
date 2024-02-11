import numpy as np
from matplotlib import pyplot as plt
import os
import random
import pandas as pd
import SimpleITK as sitk
import csv
from operator import itemgetter
import sys
from pathlib import Path
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut


candidate_csv = "./Files/nlst_pseudoLabels_scores.csv"

class IndexTracker(object):
		def __init__(self, ax, X):
				self.ax = ax
				ax.set_title('use scroll wheel to navigate images')

				self.X = X
				rows, cols, self.slices = X.shape
				self.ind = self.slices//2

				self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=-1000, vmax=400)
				self.update()

		def onscroll(self, event):
				if event.button == 'up':
						self.ind = (self.ind + 1) % self.slices
				else:
						self.ind = (self.ind - 1) % self.slices
				self.update()

		def update(self):
				self.im.set_data(self.X[:, :, self.ind])
				self.ax.set_ylabel('slice %s' % self.ind)
				self.im.axes.figure.canvas.draw()



def read_dicom_series(file_dir):
	dicom_set = []
	for root, _, filenames in os.walk(file_dir):
		for filename in sorted(filenames):
			dcm_path = Path(root, filename)
			if dcm_path.suffix == ".dcm":
				try:
					dicom = pydicom.dcmread(dcm_path, force=True)
				except IOError as e:
					print(f"Can't import {dcm_path.stem}")
				else:
					hu = dicom.pixel_array
					dicom_set.append(dicom.RescaleIntercept+(dicom.RescaleSlope*hu))
	dicom_set = np.dstack(dicom_set)
	return dicom_set




def visualize_dicoms(scan_dir, time_point, prob_lb = 0.7, prob_ub = 1.0, visualize = True):
	patient_id = scan_dir.split('/')[-3]	
	candidates_df = pd.read_csv(candidate_csv).values.tolist()
	scan_detections = [x for x in candidates_df if (str(x[0]) == patient_id and str(x[1]) == str(time_point))] # list of pseudo labels for the patient 
	if not scan_detections:
		print('This patient does not have any detections ...')
	numpy_scan = read_dicom_series(scan_dir)
	i = 0
	detections = []
	while i < len(scan_detections):
			x_coord = scan_detections[i][3]
			y_coord = scan_detections[i][4]
			z_coord = scan_detections[i][5]
			prob = scan_detections[i][6]
			if prob < prob_lb or prob > prob_ub:
				i += 1
				continue


			offset_slices = 3
			offset = 15
			for xx in range(offset_slices):
				numpy_scan[y_coord-offset-1:y_coord-offset,x_coord-offset:x_coord+offset,z_coord-1-xx] = 400
				numpy_scan[y_coord-offset:y_coord+offset,x_coord-offset-1:x_coord-offset,z_coord-1-xx] = 400
				numpy_scan[y_coord+offset-1:y_coord+offset,x_coord-offset:x_coord+offset,z_coord-1-xx] = 400
				numpy_scan[y_coord-offset:y_coord+offset,x_coord+offset-1:x_coord+offset,z_coord-1-xx] = 400
				numpy_scan[y_coord-offset-1:y_coord-offset,x_coord-offset:x_coord+offset,z_coord-1+xx] = 400
				numpy_scan[y_coord-offset:y_coord+offset,x_coord-offset-1:x_coord-offset,z_coord-1+xx] = 400
				numpy_scan[y_coord+offset-1:y_coord+offset,x_coord-offset:x_coord+offset,z_coord-1+xx] = 400
				numpy_scan[y_coord-offset:y_coord+offset,x_coord+offset-1:x_coord+offset,z_coord-1+xx] = 400

			i += 1
			detections.append([x_coord, y_coord, z_coord])
	
	if visualize:
		fig, ax = plt.subplots(1, 1, figsize=(10, 10))
		fig.tight_layout(pad=4)
		tracker = IndexTracker(ax, numpy_scan)
		fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
		plt.show()
	
	return numpy_scan, detections
	

if __name__ == "__main__":
	scan_dir = "./Files/NLST_Scan/100026/01-02-2000-NA-NLSTLSS-32628/2.000000-1OPAGELS16B3972.51200.00.0null-57666"
	time_point = 1
	prob_lb = 0.1
	prob_ub = 1.0
	numpy_scan, detections = visualize_dicoms(scan_dir, time_point, prob_lb, prob_ub, True)
