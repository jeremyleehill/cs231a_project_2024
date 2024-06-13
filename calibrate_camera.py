# imports 
import numpy as np 
import cv2 as cv 
import glob
from scipy.spatial.transform import Rotation as Rot
  
def calibrate_camera(images_folder):
    
    # termination criteria 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # # Real world coordinates of circular grid 
        
    dots_H = 12
    dots_V = 8
    spacing = 7 #mm
    
    a = np.arange(dots_H, dtype=np.float32) * spacing
    b = np.arange(dots_V, dtype=np.float32) * spacing

    X, Y = np.meshgrid(a, b)

    # Reshaping for broadcasting (optional but recommended for efficiency)
    X_reshaped = X.reshape(-1, 1)  # Reshape X to have one column
    Y_reshaped = Y.reshape(-1, 1)  # Reshape Y to have one column

    # Interleaving elements using broadcasting
    combined_array = np.concatenate([X_reshaped, Y_reshaped], axis=1)

    # Zeros as a column vector
    zeros_to_add = np.zeros((combined_array.shape[0], 1))

    # Stack horizontally
    obj3d = np.column_stack([combined_array, zeros_to_add])
    obj3d = np.array([obj3d], dtype = np.float32)
    # print(obj3d)
    
    # Vector to store 3D points 
    obj_points = [] 
    # Vector to store 2D points 
    img_points = [] 
    
    # Extracting path of individual image stored in a given directory 
    images = glob.glob(images_folder)

    for f in images: 
        # Loading image 
        img = cv.imread(f)
        # print(f)
        
        # Conversion to grayscale image 
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        #thresholding and blur parameters can be changed
        blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
        ret, thresh = cv.threshold(blur, 50, 255, cv.THRESH_TOZERO)
        
        #replace with gray
        ret, corners = cv.findCirclesGrid(thresh, (dots_H,dots_V), None, flags = cv.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid
    
        # If true is returned,  
        # then 3D and 2D vector points are updated and corner is drawn on image 
        if ret == True:
            # print('TRUE!')
    
            obj_points.append(obj3d)
            # print(obj_points)
    
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) 
            # In case of circular grids,  
            # the cornerSubPix() is not always needed, so alternative method is: 
            # corners2 = corners 
            img_points.append(corners2) 
    
            # Drawing the corners, saving and displaying the image 
            cv.drawChessboardCorners(img, (dots_H,dots_V), corners2, ret) 
            cv.imwrite('output.jpg', img) #To save corner-drawn image 
            cv.imshow('img', img) 
            cv.waitKey(250)
            
    cv.destroyAllWindows()
    cv.waitKey(1)

        
    # """Camera calibration:  
    # Passing the value of known 3D points (obj points) and the corresponding pixel coordinates  
    # of the detected corners (img points)"""
    ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv.calibrateCamera( 
        obj_points, img_points, gray.shape[::-1], None, None)
    
    # flags = cv.CALIB_ZERO_TANGENT_DIST

    print("Error in projection : \n", ret) 
    print("\nCamera matrix : \n", camera_mat)
    # (k1, k2, p1, p2, k3) 
    print("\nDistortion coefficients : \n", distortion) 
    print("\nRotation vector : \n", rotation_vecs) 
    print("\nTranslation vector : \n", translation_vecs)

    return camera_mat, distortion

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    #read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)

    # #This particular dataset used 1 for left camera and 0 for right camera
    # #define c1 for left camera and c2 for right camera
    # c1_images_names = images_names[1::2]
    # c2_images_names = images_names[::2]
    
    c1_images_names = images_names[::2]
    c2_images_names = images_names[1::2]
    # print('\nleft camera: \n',c1_images_names)
    # print('\nright camera: \n',c2_images_names)
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    
    dots_H = 14
    dots_V = 9
    spacing = 7 #mm
    
    a = np.arange(dots_H, dtype=np.float32) * spacing
    b = np.arange(dots_V, dtype=np.float32) * spacing

    X, Y = np.meshgrid(a, b)

    # Reshaping for broadcasting (optional but recommended for efficiency)
    X_reshaped = X.reshape(-1, 1)  # Reshape X to have one column
    Y_reshaped = Y.reshape(-1, 1)  # Reshape Y to have one column

    # Interleaving elements using broadcasting
    combined_array = np.concatenate([X_reshaped, Y_reshaped], axis=1)

    # Zeros as a column vector
    zeros_to_add = np.zeros((combined_array.shape[0], 1))

    # Stack horizontally
    objp = np.column_stack([combined_array, zeros_to_add])
    objp = np.array([objp], dtype = np.float32)
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
    
    count = 0
    
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        #thresholding and blur parameters can be changed
        blur = cv.GaussianBlur(gray1, (5, 5), cv.BORDER_DEFAULT)
        ret, thresh = cv.threshold(blur, 50, 255, cv.THRESH_TOZERO)
        c_ret1, corners1 = cv.findCirclesGrid(thresh,(dots_H,dots_V), None, flags = cv.CALIB_CB_SYMMETRIC_GRID)
        
        #thresholding and blur parameters can be changed
        blur = cv.GaussianBlur(gray2, (5, 5), cv.BORDER_DEFAULT)
        ret, thresh = cv.threshold(blur, 50, 255, cv.THRESH_TOZERO)
        c_ret2, corners2 = cv.findCirclesGrid(thresh,(dots_H,dots_V), None, flags = cv.CALIB_CB_SYMMETRIC_GRID)
        
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (dots_H,dots_V), corners1, c_ret1)
            # cv.imshow('img_L', frame1)
            # cv.waitKey(250)
 
            cv.drawChessboardCorners(frame2, (dots_H,dots_V), corners2, c_ret2)
            # cv.imshow('img_R', frame2)
            # cv.waitKey(250)
            
            # concatenate image Horizontally 
            Hori = np.concatenate((frame1, frame2), axis=1)
            cv.imshow('HORIZONTAL', Hori)
            cv.waitKey(0)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            
            count+=1
            print(count)
            
    cv.destroyAllWindows()
    cv.waitKey(1)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    # stereocalibration_flags = cv.CALIB_USE_INTRINSIC_GUESS
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, gray1.shape[::-1], criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T, CM1, dist1, CM2, dist2

if __name__ == "__main__":
    
    # #This particular dataset used 1 for left camera and 0 for right camera
    # mtx0, dist0 = calibrate_camera(images_folder = 'images/single_cam_1/*.tif')
    # mtx1, dist1 = calibrate_camera(images_folder = 'images/single_cam_0/*.tif')
    
    mtx0, dist0 = calibrate_camera(images_folder = 'images/single_cam_0/*.tif')
    mtx1, dist1 = calibrate_camera(images_folder = 'images/single_cam_1/*.tif')
    
    # mtx0, dist0 = calibrate_camera(images_folder = 'images/gom_cal/L/*.png')
    # mtx1, dist1 = calibrate_camera(images_folder = 'images/gom_cal/R/*.png')
    
    R, T, CM1, d1, CM2, d2 = stereo_calibrate(mtx0, dist0, mtx1, dist1, 'images/ExperimentalCal_14x10-7mm/*.tif')
    print("\nRotation Matrix: \n", R)
    print("\nTranslation Vector: \n", T)
    print("\nCamera matrix 0: \n", CM1)
    print("\nDist Coeff 0: \n", d1)
    print("\nCamera matrix 1: \n", CM2)
    print("\nDist Coeff 1: \n", d2)
    
    r = Rot.from_matrix(R)
    print("Euler Angles: \n", r.as_euler('zyx', degrees=True))