import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os

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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
test_file = 'test3.jpg'
img = cv2.imread(test_images_input_folder + test_file)
warped = combined_binary(warp(img))

# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
	    # Add graphic points from window mask here to total pixels found 
	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

# Display the final results
plt.imshow(warped)
plt.show()
plt.imshow(output)
plt.title('window fitting results')
plt.show()