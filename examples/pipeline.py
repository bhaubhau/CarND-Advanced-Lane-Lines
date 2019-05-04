import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip

###################################################################################################
####################################calibration of camera
###################################################################################################
calibration_input_folder='../camera_cal/'
#chessboard_corners_output_folder='../camera_cal_output/'
#undistort_output_folder='../undistort_output/'

# if not os.path.exists(chessboard_corners_output_folder):
#     os.mkdir(chessboard_corners_output_folder)

# if not os.path.exists(undistort_output_folder):
#     os.mkdir(undistort_output_folder)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

nx=9
ny=6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Make a list of calibration images
images = glob.glob(calibration_input_folder + 'calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners        
        # img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        # outputfile=fname.replace(calibration_input_folder, chessboard_corners_output_folder)
        # cv2.imwrite(outputfile, img)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)      
    else:
        print('Chessboard corner points not found for ' + fname)
#cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
def get_undistorted_image(image):
    return cv2.undistort(image, mtx, dist, None, mtx)

# for fname in images:
#     img = cv2.imread(fname)    
#     undist = get_undistorted_image(img)
#     outputfile=fname.replace(calibration_input_folder, undistort_output_folder)
#     cv2.imwrite(outputfile, undist)

###################################################################################################
###################################################################################################
test_images_input_folder='../test_images/'
test_images_output_folder='../test_images_output/'

if not os.path.exists(test_images_output_folder):
    os.mkdir(test_images_output_folder)

# test_file = 'test3.jpg'
# img = cv2.imread(test_images_input_folder + test_file)
# undist = get_undistorted_image(img)
# cv2.imwrite(test_images_output_folder + 'undistorted_' + test_file, undist)

###################################################################################################
####################################Colorspace Separation
###################################################################################################
def s_thresh(img, mag_thresh=(0, 255)):
    hls_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)  
    S = hls_img[:,:,2]    
    binary_output = np.zeros_like(S)
    binary_output[(S >= mag_thresh[0]) & (S <= mag_thresh[1])] = 255
    return binary_output

# images = glob.glob(test_images_input_folder + '*.jpg')
# for fname in images:
#     img = cv2.imread(fname)
#     hls_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)  
#     outputfile=fname.replace(test_images_input_folder, test_images_output_folder)  

#     H = hls_img[:,:,0]
#     L = hls_img[:,:,1]
#     S = hls_img[:,:,2]
#     outputfile_H=outputfile.replace('.jpg','_H.jpg')
#     outputfile_L=outputfile.replace('.jpg','_L.jpg')
#     outputfile_S=outputfile.replace('.jpg','_S.jpg')
    
#     #cv2.imwrite(outputfile, hls_img)
#     #cv2.imwrite(outputfile_H, H)
#     #cv2.imwrite(outputfile_L, L)
#     cv2.imwrite(outputfile_S, S)

###################################################################################################
####################################Gradient calculation
###################################################################################################
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    # Return the binary image
    return binary_output

def comb_grad(img):
    grad_x_bin_img=abs_sobel_thresh(img,'x',3,(50,150))
    grad_y_bin_img=abs_sobel_thresh(img,'y',3,(50,150))
    grad_mag_bin_img=mag_thresh(img,3,(20,200))
    grad_dir_bin_img=dir_thresh(img,3,(0.7,1.3))
    combined = np.zeros_like(grad_x_bin_img)
    combined[((grad_x_bin_img == 255) & (grad_y_bin_img == 255)) | ((grad_mag_bin_img == 255) & (grad_dir_bin_img == 255))] = 255
    return combined

def combined_binary(img):    
    grad_bin_img=comb_grad(img)
    s_bin_img=s_thresh(img,(170,255))   
    combined = np.zeros_like(grad_bin_img) 
    combined[(grad_bin_img==255)|(s_bin_img==255)]=255
    return combined

# images = glob.glob(test_images_input_folder + '*.jpg')
# for fname in images:
#     img = cv2.imread(fname)
  
#     outputfile=fname.replace(test_images_input_folder, test_images_output_folder)
#     # outputfile_SX=outputfile.replace('.jpg','_SX.jpg')
#     # outputfile_SY=outputfile.replace('.jpg','_SY.jpg')
#     # outputfile_MAG=outputfile.replace('.jpg','_MAG.jpg')
#     # outputfile_DIR=outputfile.replace('.jpg','_DIR.jpg')
#     outputfile_COMB=outputfile.replace('.jpg','_COMB.jpg')
    
#     # grad_x_bin_img=abs_sobel_thresh(img,'x',3,(20,200))
#     # grad_y_bin_img=abs_sobel_thresh(img,'y',3,(20,200))
#     # grad_mag_bin_img=mag_thresh(img,3,(20,200))
#     # grad_dir_bin_img=dir_thresh(img,3,(0.7,1.3))
    
#     # cv2.imwrite(outputfile_SX, grad_x_bin_img)
#     # cv2.imwrite(outputfile_SY, grad_y_bin_img)
#     # cv2.imwrite(outputfile_MAG, grad_mag_bin_img)
#     # cv2.imwrite(outputfile_DIR, grad_dir_bin_img)

#     # combined = np.zeros_like(grad_x_bin_img)
#     # combined[((grad_x_bin_img == 255) & (grad_y_bin_img == 255)) | ((grad_mag_bin_img == 255) & (grad_dir_bin_img == 255))] = 255

#     # cv2.imwrite(outputfile_COMB, combined)

#     #mpimg.imsave(outputfile_SX, abs_sobel_thresh(img,'x',20,200))
#     #mpimg.imsave(outputfile_SY, abs_sobel_thresh(img,'y',20,200))

#     cv2.imwrite(outputfile_COMB, combined_binary(img))

###################################################################################################
####################################Perspective transform
###################################################################################################
def warp(img): 

    img_size = (img.shape[1], img.shape[0])
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

    undistort=get_undistorted_image(img)
    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(undistort, M, img_size)
    return warped

def unwarp(img): 

    img_size = (img.shape[1], img.shape[0])
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

    Minv = cv2.getPerspectiveTransform(dest, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size)
    return unwarped

# images = glob.glob(test_images_input_folder + '*.jpg')
# for fname in images:
#     img = cv2.imread(fname)
#     outputfile=fname.replace(test_images_input_folder, test_images_output_folder)
#     comb=combined_binary(warp(img))
#     cv2.putText(comb,'Binary warped, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255),2)
#     cv2.imwrite(outputfile, comb)

###################################################################################################
####################################Curve fitting
###################################################################################################
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# test_file = 'test3.jpg'
# img = cv2.imread(test_images_input_folder + test_file)
# img_warped=warp(img)
# binary_warped = combined_binary(img_warped)
#binary_warped= mpimg.imread('warped_example.jpg')

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[(binary_warped.shape[0]*2)//3:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

left_fit=np.array([0,0,0])
right_fit=np.array([0,0,0])

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    global left_fit,right_fit
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='red')
    #plt.plot(right_fitx, ploty, color='blue')

    return out_img

#out_img = fit_polynomial(binary_warped)

#plt.imshow(out_img)
#plt.show()

######Polyfit in next frame########
# test_file = 'test4.jpg'
# img = cv2.imread(test_images_input_folder + test_file)
# img_warped=warp(img)
# binary_warped = combined_binary(img_warped)
# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
#left_fit = np.array([ 1.39704016e-04,-2.94429437e-01,4.80061572e+02])
#right_fit = np.array([2.13760772e-04,-4.36511278e-01,1.17539007e+03])

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    global left_fit,right_fit

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    pts = np.hstack((left_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    
    cv2.fillPoly(window_img, np.int_([pts]), (0,255, 0))
    cv2.polylines(window_img,np.int_([left_line_window1]),False,(0,0,255),thickness = 10)
    cv2.polylines(window_img,np.int_([right_line_window1]),False,(255,0,0),thickness = 10)
    #result = cv2.addWeighted(img_warped, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='red')
    #plt.plot(right_fitx, ploty, color='blue')
    ## End visualization steps ##
    
    #return result
    return left_fitx, right_fitx, ploty, window_img

# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
#result = search_around_poly(binary_warped)

# View your output
#plt.imshow(result)
#plt.show()

###################################################################################################
####################################Radius of Curvature
###################################################################################################
def get_radius(left_fitx, right_fitx, ploty, center_point):    
    
    global ym_per_pix,xm_per_pix
    center_fitx=(left_fitx+right_fitx)/2
    center_fit = np.polyfit(ploty*ym_per_pix, center_fitx*xm_per_pix, 2)        
    center_offsetx=center_point-center_fitx[-1]        
    
    y_eval = np.max(ploty)
    center_curverad = ((1 + (2*center_fit[0]*y_eval*ym_per_pix + center_fit[1])**2)**1.5) / np.absolute(2*center_fit[0])
    
    center_offset=round(abs(center_offsetx)*xm_per_pix,2)
    position_str=""
    if round(center_offsetx)==0:
        position_str="Vehicle is on center"
    elif center_offsetx<0:
        position_str="Vehicle is " + str(center_offset) + "m left of center"
    elif center_offsetx>0:
        position_str="Vehicle is " + str(center_offset) + "m right of center"        
    
    return str(int(center_curverad)),position_str

###################################################################################################
####################################Image processing
###################################################################################################
def get_lanes_image_warped(img, nextFrame=False):
    img_warped=warp(img)
    binary_warped = combined_binary(img_warped)
    if nextFrame==False:
        fit_polynomial(binary_warped)
    left_fitx, right_fitx, ploty, result = search_around_poly(binary_warped)    
    return left_fitx, right_fitx, ploty, result

def get_lanes_image(img, nextFrame=False):
    left_fitx, right_fitx, ploty, lines_warped=get_lanes_image_warped(img, nextFrame)
    return left_fitx, right_fitx, ploty, unwarp(lines_warped)

def process_image(img, nextFrame=False):
    undistort=get_undistorted_image(img)
    
    img_morph=cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_morph[:,:,0] = cv2.equalizeHist(img_morph[:,:,0])
    img_morph = cv2.cvtColor(img_morph, cv2.COLOR_YUV2BGR)
    
    left_fitx, right_fitx, ploty, lines_image=get_lanes_image(img, nextFrame)
    result=cv2.addWeighted(undistort, 0.8, lines_image, 1, 0)
    radius, center_offset=get_radius(left_fitx, right_fitx, ploty, (undistort.shape[1])/2)
    cv2.putText(result,'Radius of Curvature = ' + radius + '(m)', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result,center_offset, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    return result

images = glob.glob(test_images_input_folder + '*.jpg')
for fname in images:
    img = cv2.imread(fname)
    outputfile=fname.replace(test_images_input_folder, test_images_output_folder)    
    cv2.imwrite(outputfile, process_image(img))

###################################################################################################
####################################Video processing
###################################################################################################
initialised=False
def process_video_frame(img):    
    global left_fit,right_fit,initialised
    if initialised==False:
        left_fit=np.array([0,0,0])
        right_fit=np.array([0,0,0])   
        initialised=True        
        return process_image(img)
    else:
        return process_image(img, True)

def process_video(input_video_path,output_video_path):
    global initialised
    initialised=False    
    clip1 = VideoFileClip(input_video_path)
    #clip1=clip1.subclip(0,5)
    white_clip = clip1.fl_image(process_video_frame)
    white_clip.write_videofile(output_video_path, audio=False)

videos = glob.glob(test_images_input_folder + 'project_video.mp4')
for fname in videos:    
    outputfile=fname.replace(test_images_input_folder, test_images_output_folder)    
    process_video(fname,outputfile)