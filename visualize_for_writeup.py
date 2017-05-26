import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import pickle
import glob
import collections

from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from search_classify import *
from pipeline import *
from random import randint

class VisualizeProject:
	def __init__(self):
		print('VisualizeProject object created ...')
		self.pipeline = Pipeline()

		self.cars = []
		self.notcars = []

		fns = self.pipeline.get_image_files()
		self.cars = fns['cars']
		self.notcars = fns['notcars']


	def visualize_input_images(self):
		p = self.pipeline

		rc = randint(0, len(self.cars))
		print('Car', rc)

		rnc = randint(0, len(self.notcars))
		print('Not car', rnc)

		print('Example of a car image:', self.cars[rc])
		print('Example of a non-car image:', self.notcars[rnc])	

		car_img = mpimg.imread(self.cars[rc])
		notcar_img = mpimg.imread(self.notcars[rnc])

		fig = plt.figure()
		plt.subplot(121)
		plt.imshow(car_img)
		plt.title('Car')
		plt.subplot(122)
		plt.imshow(notcar_img)
		plt.title('Not car')
		fig.tight_layout()
		#plt.show()
		fig.savefig('./examples/vis-car_not_car.png')

	def visualize_hog(self):

		rc = randint(0, len(self.cars))
		print('Car', rc)

		rnc = randint(0, len(self.notcars))
		print('Not car', rnc)

		car_img = mpimg.imread(self.cars[rc])
		notcar_img = mpimg.imread(self.notcars[rnc])

		colorspaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YCrCb']
		for color_space in colorspaces:
			#color_space = 'YCrCb'
			#color_space = 'HLS'
			print('Extracting hog -', color_space)
			self.display_hog(car_img, color_space, 'Car')
			self.display_hog(notcar_img, color_space, 'Not-Car')


	def display_hog(self, input_img, color_space, title):
		orient = 9  # HOG orientations
		pix_per_cell = 8 # HOG pixels per cell
		cell_per_block = 2 # HOG cells per block
		scale = 1
		img_tosearch = input_img.astype(np.float32)/255

		#img_tosearch = img[ystart:ystop,:,:]
		conv = 'RGB2' + color_space
		ctrans_tosearch = convert_color(img_tosearch, conv=conv)
		if scale != 1:
		    imshape = ctrans_tosearch.shape
		    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		    
		ch1 = ctrans_tosearch[:,:,0]
		ch2 = ctrans_tosearch[:,:,1]
		ch3 = ctrans_tosearch[:,:,2]

		# Define blocks and steps as above
		nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
		nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
		nfeat_per_block = orient*cell_per_block**2

		# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
		window = 64
		nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
		cells_per_step = 2  # Instead of overlap, define how many cells to step
		nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
		nysteps = (nyblocks - nblocks_per_window) // cells_per_step

		# Compute individual channel HOG features for the entire image
		f1, hog_img1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
		f2, hog_img2  = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
		f3, hog_img3  = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

		fig = plt.figure()
		fig.set_figheight(5)
		fig.set_figwidth(20)
		plt.subplot(141)
		plt.imshow(input_img)
		plt.title(title)
		plt.subplot(142)
		plt.imshow(hog_img1)
		plt.title('Ch1 - ' + color_space)
		plt.subplot(143)
		plt.imshow(hog_img2)
		plt.title('Ch2 - ' + color_space)
		plt.subplot(144)
		plt.imshow(hog_img3)
		plt.title('Ch3 - ' + color_space)

		fig.tight_layout()
		#plt.show()
		fn = './examples/vis-hog-' + color_space + '-' + title + '.png'
		print(fn)
		fig.savefig(fn)

	def visualize_search_windows(self, image_file):
		image = mpimg.imread(image_file)
		draw_image = np.copy(image)

		dist_pickle = self.get_param_dict("svc_pickle_RGB.p")
		svc = dist_pickle["svc"]
		X_scaler = dist_pickle["scaler"]
		orient = dist_pickle["orient"]
		pix_per_cell = dist_pickle["pix_per_cell"]
		cell_per_block = dist_pickle["cell_per_block"]
		spatial_size = dist_pickle["spatial_size"]
		hist_bins = dist_pickle["hist_bins"]
		y_start_stop = [None, None] 
		color_space = 'HLS'
		hog_channel = 'ALL'
		spatial_feat = True # Spatial features on or off
		hist_feat = True # Histogram features on or off
		hog_feat = True # HOG features on or off

		# Uncomment the following line if you extracted training
		# data from .png images (scaled 0 to 1 by mpimg) and the
		# image you are searching is a .jpg (scaled 0 to 255)
		image = image.astype(np.float32)/255

		windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
		                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

		hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
		                        spatial_size=spatial_size, hist_bins=hist_bins, 
		                        orient=orient, pix_per_cell=pix_per_cell, 
		                        cell_per_block=cell_per_block, 
		                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
		                        hist_feat=hist_feat, hog_feat=hog_feat)                       

		window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

		plt.imshow(window_img)
		plt.show()

	def visualize_identify_cars(self):
		image_file = './test_images/test6.jpg'
		image = mpimg.imread(image_file)

		colorspaces = ['RGB', 'HLS', 'HSV', 'YCrCb']

		for color_space in colorspaces:

			dist_pickle = self.pipeline.get_param_dict("svc_pickle_" + color_space + ".p")
			svc = dist_pickle["svc"]
			X_scaler = dist_pickle["scaler"]
			orient = dist_pickle["orient"]
			pix_per_cell = dist_pickle["pix_per_cell"]
			cell_per_block = dist_pickle["cell_per_block"]
			spatial_size = dist_pickle["spatial_size"]
			hist_bins = dist_pickle["hist_bins"]
			test_accuracy = dist_pickle["test_accuracy"]
			color_space = dist_pickle["color_space"]

			print('Using:',orient,'orientations',pix_per_cell,
			    'pixels per cell and', cell_per_block,'cells per block')

			print('color_space:', color_space, ' accuracy =', test_accuracy)
			ystart = 400
			ystop = 656
			scale = 1.5
			boxes = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
			draw_img = np.copy(image)

			for box in boxes:
				cv2.rectangle(draw_img,box[0],box[1],(0,0,255),6) 

				fig = plt.figure()
				#plt.subplot(121)
				plt.imshow(draw_img)
				plt.title('Detected Cars - ' + color_space)

				fig.savefig('./examples/vis-detectcars-' + color_space + '.png')

	
	def visualize_on_test_images(self):
		test_image_files = glob.glob('./test_images/*.jpg')

		fig = plt.figure()

		x = 1
		for image_file in test_image_files:
			print(image_file, x)
			image = mpimg.imread(image_file)
			heat = np.zeros_like(image[:,:,0]).astype(np.float)	
			draw_image = np.copy(image)

			boxes = self.pipeline.detect_cars(image)

			for box in boxes:
				cv2.rectangle(draw_image, box[0],box[1],(0,0,255),6) 

			plt.subplot(3, 2, x)
			plt.imshow(draw_image)
			plt.title(image_file)
			x = x + 1

		fig.tight_layout()
		fig.savefig('./examples/vis-test-images.png')

	def visualize_heatmap(self, heat, box_list, image):

		# Add heat to each box in box list
		heat = add_heat(heat, box_list)
		    
		# Apply threshold to help remove false positives
		heat = apply_threshold(heat, 2)

		# Visualize the heatmap when displaying    
		heatmap = np.clip(heat, 0, 255)

		# Find final boxes from heatmap using label function
		labels = label(heatmap)
		draw_img = draw_labeled_bboxes(np.copy(image), labels)

		fig = plt.figure()
		plt.subplot(121)
		plt.imshow(draw_img)
		plt.title('Car Positions')
		plt.subplot(122)
		plt.imshow(heatmap, cmap='hot')
		plt.title('Heat Map')
		fig.tight_layout()
		plt.show()

	def visualize_heat(self):
		
		test_image_files = glob.glob('./test_images/*.jpg')

		for image_file in test_image_files:
			#image_file = './test_images/test6.jpg'
			#pipeline.visualize_search_windows(image_file)

			image = mpimg.imread(image_file)
			heat = np.zeros_like(image[:,:,0]).astype(np.float)	
			#draw_image = np.copy(image)

			bbox_list = self.pipeline.detect_cars(image)

			self.visualize_heatmap(heat, bbox_list, image)
		

if __name__ == "__main__":
	vis = VisualizeProject()
	#vis.visualize_input_images()
	#vis.visualize_hog()
	#vis.visualize_identify_cars()
	#vis.visualize_on_test_images()
	vis.visualize_heat()
