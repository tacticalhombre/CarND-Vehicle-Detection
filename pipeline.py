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
#from hog_subsample import *

class Pipeline:

	def __init__(self):
		print('Pipeline object created ...')
		self.lastN_bbox = []
		self.frame = 1
		self.heatmaps = collections.deque(maxlen=10) 
		self.pickle_cache = {}

	def get_image_files(self):
		fn_cars = glob.glob('./data/vehicles/*/*.png')
		fn_notcars = glob.glob('./data/non-vehicles/*/*.png')

		print('Number of car images:', len(fn_cars))
		print('Number of non-car images:', len(fn_notcars))

		return {'cars':fn_cars, 'notcars':fn_notcars}

	def train_classifier(self):

		fns = self.get_image_files()
		cars = fns['cars']
		notcars = fns['notcars']

		# Reduce the sample size because
		# The quiz evaluator times out after 13s of CPU time
		#sample_size = 500
		#cars = cars[0:sample_size]
		#notcars = notcars[0:sample_size]

		### TODO: Tweak these parameters and see how the results change.
		colorspaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
		#color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
		orient = 9  # HOG orientations
		pix_per_cell = 8 # HOG pixels per cell
		cell_per_block = 2 # HOG cells per block
		hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
		spatial_size = (16, 16) # Spatial binning dimensions
		hist_bins = 16    # Number of histogram bins
		spatial_feat = True # Spatial features on or off
		hist_feat = True # Histogram features on or off
		hog_feat = True # HOG features on or off
		y_start_stop = [None, None] # Min and max in y to search in slide_window()

		for color_space in colorspaces:
			print('Extracting features for color_space', color_space)
			car_features = extract_features(cars, color_space=color_space, 
			                        spatial_size=spatial_size, hist_bins=hist_bins, 
			                        orient=orient, pix_per_cell=pix_per_cell, 
			                        cell_per_block=cell_per_block, 
			                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
			                        hist_feat=hist_feat, hog_feat=hog_feat)
			notcar_features = extract_features(notcars, color_space=color_space, 
			                        spatial_size=spatial_size, hist_bins=hist_bins, 
			                        orient=orient, pix_per_cell=pix_per_cell, 
			                        cell_per_block=cell_per_block, 
			                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
			                        hist_feat=hist_feat, hog_feat=hog_feat)

			X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
			# Fit a per-column scaler
			X_scaler = StandardScaler().fit(X)
			# Apply the scaler to X
			scaled_X = X_scaler.transform(X)

			# Define the labels vector
			y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


			# Split up data into randomized training and test sets
			rand_state = np.random.randint(0, 100)
			X_train, X_test, y_train, y_test = train_test_split(
			    scaled_X, y, test_size=0.2, random_state=rand_state)

			print('Using:',orient,'orientations',pix_per_cell,
			    'pixels per cell and', cell_per_block,'cells per block')
			print('Feature vector length:', len(X_train[0]))
			
			# Use a linear SVC 
			svc = LinearSVC()
			# Check the training time for the SVC
			t=time.time()
			svc.fit(X_train, y_train)
			t2 = time.time()

			print(round(t2-t, 2), 'Seconds to train SVC...')
			# Check the score of the SVC
			test_accuracy = round(svc.score(X_test, y_test), 4)
			print('Test Accuracy of SVC = ', test_accuracy)
			# Check the prediction time for a single sample
			t=time.time()

			dist_pickle = {}
			dist_pickle["svc"] = svc
			dist_pickle["scaler"] = X_scaler
			dist_pickle["orient"] = orient
			dist_pickle["pix_per_cell"] = pix_per_cell
			dist_pickle["cell_per_block"] = cell_per_block
			dist_pickle["spatial_size"] = spatial_size
			dist_pickle["hist_bins"] = hist_bins
			dist_pickle["test_accuracy"] = test_accuracy
			dist_pickle["color_space"] = color_space

			pickle_file = 'svc_pickle_' + color_space + '.p'
			pickle.dump( dist_pickle, open(pickle_file, 'wb') )


	def get_param_dict(self, pickle_file):
		if (pickle_file not in self.pickle_cache):
			dist_pickle = pickle.load( open(pickle_file, "rb" ) )
			self.pickle_cache[pickle_file] = dist_pickle
		else:
			dist_pickle = self.pickle_cache[pickle_file]

		return dist_pickle


	def detect_cars(self, img):
		ystart = 400
		ystop = 656
		scales = [1.0, 1.5, 1.75, 2.0]

		color_space = 'YCrCb'

		dist_pickle = self.get_param_dict("svc_pickle_" + color_space + ".p")
		svc = dist_pickle["svc"]
		X_scaler = dist_pickle["scaler"]
		orient = dist_pickle["orient"]
		pix_per_cell = dist_pickle["pix_per_cell"]
		cell_per_block = dist_pickle["cell_per_block"]
		spatial_size = dist_pickle["spatial_size"]
		hist_bins = dist_pickle["hist_bins"]
		test_accuracy = dist_pickle["test_accuracy"]
		color_space = dist_pickle["color_space"]
	
		bbox_list = []
		for scale in scales:
			boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
			bbox_list.extend(boxes)
	
		return bbox_list

	def process_image(self, image):
		heat = np.zeros_like(image[:,:,0]).astype(np.float)

		# get bounding boxes for all detections
		bbox_list = pipeline.detect_cars(image)	

		# Add heat to each box in box list
		heat = add_heat(heat, bbox_list)
		    
		# Apply threshold to help remove false positives
		heat = apply_threshold(heat, 2)
		
		self.heatmaps.append(heat)

		if len(self.heatmaps) == 10:
			
			heat = sum(list(self.heatmaps))

		heat = apply_threshold(heat, 5)

		# Visualize the heatmap when displaying    
		heatmap = np.clip(heat, 0, 255)

		# Find final boxes from heatmap using label function
		labels = label(heatmap)
		draw_img = draw_labeled_bboxes(np.copy(image), labels)

		# uncomment for visualizing
		"""
		if (self.frame <= 10):
			vis_image = np.copy(image)
			for box in bbox_list:
				cv2.rectangle(vis_image,box[0],box[1],(0,0,255),6)


			fig = plt.figure(figsize=(9,3))
			plt.subplot(121)
			plt.imshow(vis_image)
			#plt.title('Car Positions')
			plt.subplot(122)
			plt.imshow(heatmap, cmap='hot')
			#plt.title('Heat Map')
			fig.tight_layout()
			fig.savefig('./output_images/vis-frame-' + str(self.frame) + '.png')

			if (self.frame == 10):
				fig = plt.figure()
				plt.imshow(draw_img)
				plt.title('Detected Cars')
				fig.tight_layout()
				fig.savefig('./output_images/vis-frame-' + str(self.frame) + '-detected.png')

				fig = plt.figure()
				plt.imshow(labels[0], cmap='gray')
				plt.title('Labels - ' + str(labels[1]) + ' cars found')
				fig.tight_layout()
				fig.savefig('./output_images/vis-frame-' + str(self.frame) + '-labels.png')
		"""

		self.frame = self.frame + 1


		return draw_img


	def create_output_video(self):
	
		input_video_file = './project_video.mp4'
		output_video_file = './my-project_video.mp4'

		start = 21
		end = 31
		#clip1 = VideoFileClip(input_video_file).subclip(start, end)
		clip1 = VideoFileClip(input_video_file)
		proj_clip = clip1.fl_image(self.process_image) #NOTE: this function expects color images!!
		
		proj_clip.write_videofile(output_video_file , audio=False)
	

if __name__ == "__main__":
	pipeline = Pipeline()
	#pipeline.train_classifier()
	pipeline.create_output_video()


