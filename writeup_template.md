# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### 1.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. How HOG features were extracted from from the training images.

I started with the lesson_functions.py file used in the lecture material. I experimented with various other new color spaces including grayscale to see what worked the best. I also needed to change everything to work in the BGR space instead of the RGB space as I was using cv2 to load images and it works in the BGR space. 


#### 2. Decision for Final HOG parameters

I originally slected grayscale so to capture the effects of all the chanels while still having a relatively fast computation time. Ultamitly I went to back changed it to use all color channels individually after viewing the detecction results on the final video feed because I was getting too many false positives.

#### 3. Classifier Training

In search_classify.py I train a linear SVC. I used all the pictures to train it thinking that more data would give me a better result even though I would no longer have any idea of what it's final accuracy is. I only have a new model trained when a specific flag is set to true saving time on future uses. If the flag is set to false it loads the last trained model from a pickle file with all te parameters that were used to make that specific filter.

### Sliding Window Search

#### 1. Sliding Window Implementation
I used the slide and search windows functions given as examples in the lecture materail. I limited to the range for squares to search to be a little above the horizon and onl the right side of the road. Throug trial and error I determened what sizes and over lapping made sense. I added a smaller square search size to help detect cars when they apear smaller because they are further away. I also added more overlap is it seemed like detections were missed because the window was not lining up well with the vehicles in the screeen.

#### 2. Developing Classifier Robustness
I used all 3 color channels to maximise the accuracy of the classifier. Even so there were still false positive detections. To filter out the false positives I used a rolling heatmap which added 1 for every overlapping detection but subtracted a tuneable value every frame. I thresholded the update every frame to prevent negative values from building up. After test I realized I also needed to put in a satruation step to prevent the heatmap from over building and leaving ghost objects detected on the screen long after they moved one.


I had trouble saving the heatmap to an imager/video but [here](./boxes.avi) is a video fo the raw detections. 

### Video Implementation

#### 1. Final Video Output
Here's a [link to my video result](./output.avi)


#### 2. False Positive Filter

To filter out the false positives I used a rolling heatmap which added 1 for every overlapping detection but subtracted a tuneable value every frame. I thresholded the update every frame to prevent negative values from building up. After test I realized I also needed to put in a satruation step to prevent the heatmap from over building and leaving ghost objects detected on the screen long after they moved one. I used the `scipy.ndimage.measurements.label()` group blobs of detections in the heatmap into bounding boxes to be drawn.

lesson_functions.py
```
def update_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heatmap[:,:] -= 2

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def apply_saturation(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap > threshold] = threshold
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```

processVideo.py
```
hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                  spatial_size=spatial_size, hist_bins=hist_bins,
                  orient=orient, pix_per_cell=pix_per_cell,
                  cell_per_block=cell_per_block, 
                  hog_channel=hog_channel, spatial_feat=spatial_feat, 
                  hist_feat=hist_feat, hog_feat=hog_feat)                       

heat = update_heat(heat,hot_windows)
heat = apply_threshold(heat,0)
heat = apply_saturation(heat,9)
heatcheck = apply_threshold(heat,3)

labels = label(heatcheck)

draw_img = draw_labeled_bboxes(np.copy(image), labels)
```
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Computation time was definitly a big problem running this 50s video took several hours on my laptop. This would be challenging to run realtime on affordable hardware. I think this software is most likely to fail as vehicles get smaller and further away resutling in less total overlapping detections on the vehicle.

