'''
* *****************************************************************************
* @author	ChangMin An
* @Mod		2024 - 01 - 26
* @brief	Brick line detection
* *****************************************************************************
'''

'''
* *****************************************************************************
* Module Import
* *****************************************************************************
'''
import cv2
import numpy as np
import os

'''
* *****************************************************************************
* CONSTANT
* *****************************************************************************
'''

GREEN_RANGE = [np.array([40, 40, 40]), np.array([120, 255, 255])]
THRESHOLD = [140, 255]
THRESHOLD_GREEN = [190, 255]
ROI_RANGE = []
THRED_SOBEL = [30, 255]
THRED_GRAY = [150, 210]
# Color
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_thickness = 10
font_color = WHITE 


# Brick area
BRICK_MINIMUM_AREA = 200000

brick_height_mm = []

# Mode
SOBEL_MODE = 1
BINARY_MODE = 2

FLAG= 2
Flag_roi = 0

'''
* *****************************************************************************
* Function
* *****************************************************************************
''' 
# TAN Function
def Brick_Length_FUNC(_theta, _dY, _dZ, _noize):
    return _dY/np.tan(_theta - _noize) - _dZ

# Intrinsic Calibration
def Intrinsic_Calibration(_src):
    h, w = _src.shape[:2]
    distortion_file = "../Parameter/Distort_Mat.txt"
    intrinsic_file = "../Parameter/Intrinsic_Mat.txt"
    
    distortion_coeffs = np.loadtxt(distortion_file)
    intrinsic_matrix = np.loadtxt(intrinsic_file).reshape((3, 3))
    
    global FOCAL_LENGTH_H, imSensor_H
    FOCAL_LENGTH_H = intrinsic_matrix[1,1]
    
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, (w, h), 1, (w, h))

    _dst = cv2.undistort(_src, intrinsic_matrix, distortion_coeffs, None, new_camera_matrix)

    # Crop the image based on the region of interest (ROI)
    x, y, w, h = roi
    
    _dst = _dst[y:y+h, x:x+w]
    
    # imSensor_H getting
    imSensor_H = h
    
    return _dst

# ROI Configuration
def Get_ROI(event, x, y, flags, param):
    global ROI_RANGE
    if event == cv2.EVENT_LBUTTONDOWN:
        ROI_RANGE.append((x, y))

        if len(ROI_RANGE) == 4:
            ROI_RANGE = np.array(ROI_RANGE, np.int32)
            ROI_RANGE.reshape((-1, 1, 2))
            
            # Close
            cv2.destroyAllWindows()
            
# ROI Range Show
def ROI_Show(_src):
    cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("roi", _src.shape[1], _src.shape[0])
    cv2.imshow("roi", _src)
    cv2.setMouseCallback("roi", Get_ROI)
    cv2.waitKey(0)

# Image Show
def Image_Show(*_src):
    # idx name Initialization
    idx = 1
    for src_data in _src:
        cv2.namedWindow(f"{idx}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{idx}", src_data.shape[1], src_data.shape[0])
        cv2.imshow(f"{idx}", src_data)
        
        # idx Update
        idx += 1
    
# Image Pause
def Image_Pause():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Image Crop
# Image Crop
def Image_Crop(_src):
    # ROI masking
    global ROI_RANGE
    mask = np.zeros(_src.shape[:2], dtype=np.uint8)
    # with open("parameter\\ROI_RANGE.txt", 'r') as file:
    #     for line in file:
    #         x, y = map(int, line.strip().split(','))
    #         ROI_RANGE.append((x, y))
    # ROI_RANGE = np.array([ROI_RANGE], dtype=np.int32)
    # print(ROI_RANGE)
    ROI_RANGE = [[[1738,5276],[7322,5278],[8643,6293],[475,6345]]]
    ROI_RANGE = np.array([ROI_RANGE], dtype=np.int32)
    cv2.fillPoly(mask, ROI_RANGE, (255, 255, 255))

    # ROI Extract
    _dst = cv2.bitwise_and(_src, _src, mask=mask)
    
    return _dst
    
# Thresholding
def Thresholding(_src, _thred):
    # Thresholding
    _, _dst = cv2.threshold(_src, _thred[0], _thred[1], cv2.THRESH_BINARY)
    return _dst

# Contour Extract
def Contour_Extract(_src, _area):
    # Initialization
    _contour_data = []
    
    # Find Contour
    _contours, _ = cv2.findContours(_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sorting
    _contours = sorted(_contours, key=cv2.contourArea, reverse=True)
    
    # Select ONLY brick using contourArea
    for _contour in _contours:
        area = cv2.contourArea(_contour)
        if area <= _area:
            break
        else:
            _contour_data.append(_contour)
    
    # Sorting x axis
    _contour_data = sorted(_contour_data, key=lambda x: cv2.minEnclosingCircle(x)[0][0])
    
    return _contour_data
    
def Draw_Contour(_src, _contours):
    _dst = np.zeros_like(_src)
    cv2.drawContours(_dst, _contours, -1, WHITE, thickness=cv2.FILLED)
    
    return _dst
# Green Image Extract
def Laser_Extract_InRange(_src):
    # BGR to HSV
    _src_hsv = cv2.cvtColor(_src, cv2.COLOR_BGR2HSV)

    # inRange Extract
    green_mask = cv2.inRange(_src_hsv, GREEN_RANGE[0], GREEN_RANGE[1])
    _src_green = cv2.bitwise_and(_src, _src, mask=green_mask)
    
    # split green
    _, _dst, _ = cv2.split(_src_green)

    return _dst

# Brick + Laser Merging
def Brick_Laser_Merge(_src_laser, _src_brick):
    # Gray to Green
    src_laser = np.zeros_like(_src_laser)
    src_laser[:, :] = cv2.bitwise_and(_src_laser, _src_brick)
    src_laser_green = cv2.merge([np.zeros_like(_src_laser), src_laser, np.zeros_like(_src_laser)])

    # Merge Brick and Laser
    _src_brick_buf = cv2.cvtColor(_src_brick, cv2.COLOR_GRAY2BGR)
    _green_px  = np.where((src_laser_green[:, :, 1] == 255))
    _src_brick_buf[_green_px[0], _green_px[1]] = src_laser_green[_green_px[0], _green_px[1]]
    
    return _src_brick_buf, src_laser

# Kernel
def Kernel(_size):
    return np.ones((_size, _size), dtype=np.uint8)

# Brick Point Extract
def Brick_Point_Extract(_src, _dst):
    laser_point = np.where(_src == 255)
    # print(np.max(laser_point[0]))
    y_max_px = np.max(laser_point[0])
    y_min_px = np.min(laser_point[0])
    x_max_px = np.max(laser_point[1][laser_point[0] == y_max_px])
    x_min_px = np.max(laser_point[1][laser_point[0] == y_min_px])

    if abs(x_max_px - x_min_px) < 100:
        cv2.circle(_dst, (x_max_px, y_max_px), 20, RED, -1)
        cv2.circle(_dst, (x_min_px, y_min_px), 20, RED, -1)
        cv2.line(_dst, (x_max_px, y_max_px), (x_min_px, y_min_px), RED, 10)
    return [y_min_px, y_max_px], _dst

# Get birck height [px] -> [mm]
def Get_Brick_Height(_height_y_px):
    # Get dY, dZ, noise
    with open('../Parameter/Extrinsic_Mat.txt', 'r') as file:
         dY, dZ, noise = [float(line.strip()) for line in file]

    # y_max, y_min get
    y_max_px = _height_y_px[1]
    y_min_px = _height_y_px[0]

    # Plate height
    plate_px = y_max_px - (int)(imSensor_H/2)
    
    # theta
    theta = np.arctan(plate_px/(FOCAL_LENGTH_H))
    
    # Get Zc using (dY/np.tan(theta - noize) - dZ)
    Zc = dZ + Brick_Length_FUNC(theta, dY, dZ, noise)

    # Get Brick Height
    brick_height_px = y_max_px - y_min_px 
    _brick_height_mm = Zc * brick_height_px / FOCAL_LENGTH_H

    return _brick_height_mm

# Extract Brick & Laser data 
def Brick_Laser_Extract(_src, _each_brick, _contour_bricks, _contour_lasers):
    # image buf
    _src_brick_filtered = np.zeros_like(_src)
    _src_laser_filtered = np.zeros_like(_src)
    
    # Draw contour
    cv2.drawContours(_src_brick_filtered, [_contour_bricks[_each_brick]], -1, WHITE, thickness=cv2.FILLED)
    cv2.drawContours(_src_laser_filtered, _contour_lasers, -1, WHITE, thickness=cv2.FILLED)
    
    # Filtered laser info
    _src_laser_filtered = cv2.bitwise_and(_src_brick_filtered, _src_laser_filtered)
    
    return _src_brick_filtered, _src_laser_filtered

# Thresholding_Process
def Thresholding_Process(_src, _threshold):
    _src_thred = Thresholding(_src, _threshold)
    _dst = cv2.adaptiveThreshold(_src_thred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    
    return _dst
# Morphology Process(Blackhat, Close)
def Morphology_Process(_src, _kernel, _iter):
    for _ in range(_iter):
        src_blackhat = cv2.morphologyEx(_src, cv2.MORPH_BLACKHAT, _kernel)

    _dst = cv2.morphologyEx(src_blackhat, cv2.MORPH_CLOSE, Kernel(3))
    
    return _dst

def Brick_Segmentation(_src, _flag):
    if _flag == SOBEL_MODE:
        # Soble Algorithm
        src_sobel = cv2.Sobel(_src, cv2.CV_8U, 1, 0, 3)
        # Thresholding Algorithm
        src_thred = Thresholding_Process(src_sobel, THRED_SOBEL)

        # Morphology (BLACKHAT, Close)
        dst = Morphology_Process(src_thred, Kernel(21), 3)
        
    
    elif _flag == BINARY_MODE:
        dst = Thresholding(_src, THRED_GRAY)
        # dst = cv2.adaptiveThreshold(_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        

    # Image_Show(dst)
    # Image_Pause()
    
    return dst