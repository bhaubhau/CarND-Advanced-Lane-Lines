import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

###################################################################################################
####################################calibration of camera
###################################################################################################
calibration_input_folder='../camera_cal/'
#chessboard_corners_output_folder='../camera_cal_output/'
undistort_output_folder='../undistort_output/'

# if not os.path.exists(chessboard_corners_output_folder):
#     os.mkdir(chessboard_corners_output_folder)

if not os.path.exists(undistort_output_folder):
    os.mkdir(undistort_output_folder)

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

for fname in images:
    img = cv2.imread(fname)    
    undist = get_undistorted_image(img)
    outputfile=fname.replace(calibration_input_folder, undistort_output_folder)
    cv2.imwrite(outputfile, undist)

###################################################################################################
###################################################################################################
test_images_input_folder='../test_images/'
test_images_output_folder='../test_images_output/'

if not os.path.exists(test_images_output_folder):
    os.mkdir(test_images_output_folder)

test_file = 'test3.jpg'
img = cv2.imread(test_images_input_folder + test_file)
undist = get_undistorted_image(img)
cv2.imwrite(test_images_output_folder + 'undistorted_' + test_file, undist)

###################################################################################################
####################################Colorspace Separation
###################################################################################################
images = glob.glob(test_images_input_folder + '*.jpg')
for fname in images:
    img = cv2.imread(fname)
    hls_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)  
    outputfile=fname.replace(test_images_input_folder, test_images_output_folder)  

    H = hls_img[:,:,0]
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    outputfile_H=outputfile.replace('.jpg','_H.jpg')
    outputfile_L=outputfile.replace('.jpg','_L.jpg')
    outputfile_S=outputfile.replace('.jpg','_S.jpg')
    
    cv2.imwrite(outputfile, hls_img)
    cv2.imwrite(outputfile_H, H)
    cv2.imwrite(outputfile_L, L)
    cv2.imwrite(outputfile_S, S)