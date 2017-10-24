import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='BGR', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'BGR'
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='BGR', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


with open('objs.pickle') as f:  # Python 3: open(..., 'rb')
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, X_scaler = pickle.load(f)
svc = joblib.load('model.pkl') 

capture = cv2.VideoCapture();
print(capture.open('project_video.mp4'))
retval,image = capture.read()
print(retval)
#print(img)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video_writer = cv2.VideoWriter("output.avi", fourcc, 25, (image.shape[1], image.shape[0]))

i = 0

while(retval == True):

	#image = cv2.imread('test_images/test1.jpg')
	y_start_stop = [300, None] # Min and max in y to search in slide_window() 350
	draw_image = np.copy(image)

	# Uncomment the following line if you extracted training
	# data from .png images (scaled 0 to 1 by mpimg) and the
	# image you are searching is a .jpg (scaled 0 to 255)
	#image = image.astype(np.float32)/255

	windows = slide_window(image, x_start_stop=[400, None], y_start_stop=y_start_stop, 
		            xy_window=(96, 96), xy_overlap=(0.75, 0.75))
	windows += slide_window(image, x_start_stop=[400, None], y_start_stop=y_start_stop, 
		            xy_window=(128, 128), xy_overlap=(0.75, 0.75))
	#windows += slide_window(image, x_start_stop=[400, None], y_start_stop=y_start_stop, 
	#	            xy_window=(64, 64), xy_overlap=(0.75, 0.75))

	hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
		                spatial_size=spatial_size, hist_bins=hist_bins, 
		                orient=orient, pix_per_cell=pix_per_cell, 
		                cell_per_block=cell_per_block, 
		                hog_channel=hog_channel, spatial_feat=spatial_feat, 
		                hist_feat=hist_feat, hog_feat=hog_feat)                       

	window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

	video_writer.write(window_img)
	i = i + 1
	print(i)
	retval,image = capture.read()
capture.release() 

