## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image_input_calibration]: ./camera_cal/calibration1.jpg "Distorted Chessboard"
[image_output_calibration]: ./output_images/002_Undistorted_calibration1.jpg "Undistorted Chessboard"
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test3.jpg "Road Original"
[road_undistorted]: ./output_images/004_undistorted_test3.jpg "Road Transformed"
[thresholding_original]: ./test_images/test2.jpg "Road Original"
[image3]: ./output_images/006_binary_test2.jpg "Binary Thresholded"
[perspective_original]: ./test_images/test6.jpg "Road Original"
[image4]: ./output_images/008_perspective_transformed_test6.jpg "Warped"
[image5]: ./output_images/009_polyfitted_test6.jpg "Fit Visual"
[image6]: ./output_images/test_images_output/test6.jpg "Output"
[video1]: ./output_images/challenge_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I was facing some issues executing the code in notebook, so I have provided the code located in "./examples/pipeline.py" with different section headers marked in it
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the section "calibration of camera" located in "./examples/pipeline.py"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image_input_calibration]
![alt text][image_output_calibration]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
![alt text][road_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color, gradient, and saturation colorspace thresholds to generate a binary image (sections Colorspace Separation, Gradient calculation in "./examples/pipeline.py")

![alt text][thresholding_original]
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp(img)`, which appears under section Perspective transform in "./examples/pipeline.py".  The `warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dest = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dest` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective_original]
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I fitted a second order polynomial on the left and right lanes by using sliding window method.
Below example shows the fitted polynomial on the above perspective transformed image

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Assuming that the camera and vehicle are in center of the left and right lane lines, I took average of x positions of left and right lane line, and computed second degree polynomial to get the radius of curvature with the given expression and scaling factors. This is implemented in function `get_radius()` in "./examples/pipeline.py"

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I unwarped the plotted polynomial function using inverse perspective transform, and then superimposed it on the original image to get the final image in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](video1)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#####Challenges faced:
1. Initially was facing challange with calibrating the camera and getting the undistorted images which took lot of time
2. Being a beginner to python, had to go through the documentation for the functions provided by numpy and opencv from the code provided in solutions

#####Improvements:
1. can try optimizing the performance of the algorithms by trying to initially compute a centered polynomial and then look for lane lines around it
2. can try using the convolution method provided in the chapter to fit the polynomial
3. can try using higher order polynomial functions to accomodate multiple curvatures if detected in image