import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

calibration_input_folder='../camera_cal/'
calibration_output_folder='../camera_cal_output/'
undistort_output_folder='../undistort_output/'

if not os.path.exists(calibration_output_folder):
    os.mkdir(calibration_output_folder)

if not os.path.exists(undistort_output_folder):
    os.mkdir(undistort_output_folder)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob(calibration_input_folder + 'calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img_corners=np.copy(img)
        img_corners = cv2.drawChessboardCorners(img_corners, (9,6), corners, ret)
        outputfile=fname.replace(calibration_input_folder, calibration_output_folder)
        cv2.imwrite(outputfile, img_corners)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        outputfile=fname.replace(calibration_input_folder, undistort_output_folder)
        cv2.imwrite(outputfile, img)


#cv2.destroyAllWindows()