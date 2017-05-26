##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/vis-car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/vis-hog-RGB-Car.png
[image9]: ./examples/vis-hog-RGB-Not-Car.png
[image10]: ./examples/vis-hog-HSV-Car.png
[image11]: ./examples/vis-hog-HSV-Not-Car.png
[image12]: ./examples/vis-hog-LUV-Car.png
[image13]: ./examples/vis-hog-LUV-Not-Car.png
[image14]: ./examples/vis-hog-HLS-Car.png
[image15]: ./examples/vis-hog-HLS-Not-Car.png
[image16]: ./examples/vis-hog-YCrCb-Car.png
[image17]: ./examples/vis-hog-YCrCb-Not-Car.png
[image18]: ./examples/vis-detectcars-HLS.png
[image19]: ./examples/vis-detectcars-RGB.png
[image20]: ./examples/vis-detectcars-HSV.png
[image21]: ./examples/vis-detectcars-YCrCb.png
[image22]: ./examples/vis-test-images.png
[image23]: ./examples/vis-frame-1.png
[image24]: ./examples/vis-frame-2.png
[image25]: ./examples/vis-frame-3.png
[image26]: ./examples/vis-frame-4.png
[image27]: ./examples/vis-frame-5.png
[image28]: ./examples/vis-frame-6.png
[image29]: ./examples/vis-frame-7.png
[image30]: ./examples/vis-frame-8.png
[image31]: ./examples/vis-frame-9.png
[image32]: ./examples/vis-frame-10.png
[image33]: ./examples/vis-frame-10-labels.png
[image34]: ./examples/vis-frame-10-detected.png

[video1]: ./my-project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Noted below is the number of images (this includes all the GTI* and KITTI images).

Number of car images: 8792
Number of non-car images: 8968

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I created a `train_classifier()` method in Pipeline class (pipeline.py, line 32).  Within this method, I used the `extract_features()` method that was used in the lessons.
This method uses another method `get_hog_features()` that invokes `skimage.hog()`.
```
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
``` 

I grabbed a random image from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are some examples with different color spaces and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image8]
![alt text][image10]
![alt text][image12]
![alt text][image14]
![alt text][image16]

![alt text][image9]
![alt text][image11]
![alt text][image13]
![alt text][image15]
![alt text][image17]

####2. Explain how you settled on your final choice of HOG parameters.

I started with the values below and tested with several color spaces to see what kind of performance I could get.
9 orientations 8 pixels per cell and 2 cells per block

The resuts are below:

Pipeline object created ...
Using: 9 orientations 8 pixels per cell and 2 cells per block
color_space: RGB  accuracy = 0.98
Using: 9 orientations 8 pixels per cell and 2 cells per block
color_space: HSV  accuracy = 0.9876
Using: 9 orientations 8 pixels per cell and 2 cells per block
color_space: HLS  accuracy = 0.9896
Using: 9 orientations 8 pixels per cell and 2 cells per block
color_space: YCrCb  accuracy = 0.9873

I think the accuracies are decent enough.  But the final test is if this is good enough to predict car or not car images.  I tried identifying the cars in images using a 1.5 scale. Here are the results:

![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]

It is clear that for all the color spaces I used, the car/s are identified correctly.  They only differ on the amount of false positives.  Holding everything else the same, the YCrCb color space shows the least amount of false positives for that 1 specific scale I used.  

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a classifier in the `train_pipeline()` method in class Pipeline (pipeline.py, line 33)
I trained using a combination of spatial, color histogram, and HOG features.

Training was done on 4 different colorspaces and ended up selecting 'YCrCb' since this has the best performance in detecting cars with minimal false positives. Accuracy of the detection is 98.73%.

colorspaces = ['RGB', 'HSV', 'HLS', 'YCrCb']

The extracted features are standardized by using `sklearn.preprocessing.StandardScaler()`. First, determine the mean and std to be used for scaling by calling `fit()`.  The standardization is then performed by the `transform()` function.

Finally I trained a linear SVM using `sklearn.svm.LinearSVC()` by invoking it's `fit()` method.

The final parameters used in training are below. 
``` 
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16) # Spatial binning dimensions
        hist_bins = 16    # Number of histogram bins
        spatial_feat = True # Spatial features on or off
        hist_feat = True # Histogram features on or off
        hog_feat = True # HOG features on or off
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As suggested in the lessons, it is more efficient to extract the HOG features for the entire region of interest.  This way we only perform the HOG extraction once and not for every window in our sliding window search.  
We then use different scaling factors to subsample the features array. This has the effect of multiple overlaying windows, in that instead of creating windows with different sizes, we essentailly have a single size window but we change the image shape based on the scale factor.
I ended up using these scales (pipeline.py line 135):
`scales = [1.0, 1.5, 1.75, 2.0]` 

Here is an example detection

![alt text][image19]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To improve my detection, I experimented with different scales.  For scale=1, not all cars are detected on the test images.  The detection improved as I increase the scaling, but this also increased the number of false positives.  
Ultimately I searched on 4 scales, [1.0, 1.5, 1.75, 2.0], using ALL channels of YCrCb HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:



![alt text][image22]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./my-project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Each frame of the input video is passed to my pipeline (method process_image() - pipeline.py line 157).  The first thing performed in the pipeline is to get all bounding boxes for all of the detections (line 161).  For each image, our classifier will detect the same car every time. But since we used different scales, for the same detected car there will have multiple bounding boxes within or around it.  There will also be sporadic false detections. In order to distinguish a real detection from a false one we need to create a heatmap and then apply a threshold.  For the heatmap, I used a queue structure (line 21) then add bounding boxes for detections in each image if there are at least 2 bounding boxes for that detection (line 167). This would drop some of the false detection for the frame. This heatmap is carried from frame to frame.

After 10 frames I added all the heatmaps and apply another threshold of 5.  This way if the same car is detected in 5 frames then we know that it is really a car. 

### Here are first 10 frames of the test video (test-video.mp4) and their corresponding heatmaps:

![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]

I then assumed each blob from the images above corresponded to a vehicle. I then used `scipy.ndimage.measurements.label()` to label each blob.

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 10 frames:
![alt text][image33]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image34]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Vehicle detection using HOG features fed into an SVM classifier is good enough for this project.  However, I do not think this technique is fast enough. It takes my pipeline 1.6 - 2.0 seconds to process each frame. This will be a problem if we are to apply this in realtime vehicle detection in autonomous cars.  Other techniques would be more appropriate if speed of detection is a concern.  I would like to explore a deep learning approach to vehicle detection if I were to pursue this project further.




