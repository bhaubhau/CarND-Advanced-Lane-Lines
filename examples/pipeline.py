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
#     cv2.imwrite(outputfile, combined_binary(warp(img)))

###################################################################################################
####################################Curve fitting
###################################################################################################
