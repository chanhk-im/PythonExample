'''
* *****************************************************************************
* @author	Industrial Intelligence Lab
* @brief	Calibration Function
* *****************************************************************************
'''

'''
* *****************************************************************************
* Import module
* *****************************************************************************
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
* *****************************************************************************
* CONSTANT
* *****************************************************************************
'''
data_path = "../SampleDataset/Calibration/Calibration_Image.jpg"

# Color
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

# ROI Range
ROI_RANGE = []

# DISTANCE
DISTANCE = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0])

# Threshold
THRESHOLD = [175, 255]

# Contour Area
CONTOUR_AREA = 100

imSensor_H = 0.0
FOCAL_LENGTH_H = 0.0

'''
* *****************************************************************************
* Function
* *****************************************************************************
'''  
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
    
    return _dst, imSensor_H, FOCAL_LENGTH_H

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
def Image_Crop(_src):
    # ROI masking
    mask = np.zeros(_src.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [ROI_RANGE], (255, 255, 255))
    with open('../Parameter/ROI_RANGE.txt', 'w') as file:
        for point in ROI_RANGE:
            file.write(f"{point[0]},{point[1]}\n")

    # ROI Extract
    _dst = cv2.bitwise_and(_src, _src, mask=mask)
    
    return _dst

# Thresholding
def Thresholding(_src, _thred):
    # Thresholding
    _, _dst = cv2.threshold(_src, _thred[0], _thred[1], cv2.THRESH_BINARY)
    return _dst

# Contour Extract
def Contour_Extract(_src, _area, max_contours=24):
    # Initialization
    _contour_data = []
    _dst = np.zeros_like(_src)
    
    # Find Contour
    _contours, _ = cv2.findContours(_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sorting
    _contours = sorted(_contours, key=cv2.contourArea, reverse=True)
    
    # Select ONLY brick using contourArea
    for _contour in _contours[:max_contours]:
        area = cv2.contourArea(_contour)
        if area <= _area:
            break
        else:
            _contour_data.append(_contour)
    
    cv2.drawContours(_dst, _contour_data, -1, WHITE, thickness=cv2.FILLED)
    
    return _dst, _contour_data

# Ymax Calculate
def Contour_Ymax_Calculate(_contours):
    # Extract x value
    x_values = [point[0][0] for point in _contours]
    
    # Calculate median
    x_median = np.median(x_values)
    
    # Maximum Y Extract
    y_max = np.max([point[0][1] for point in _contours if x_median - 50 <= point[0][0] <= x_median + 50])
    
    # Mean y_max

    y_max_mean = y_max
    
    return [x_median, y_max_mean]

# Y Position Extract
def Y_Pose_Extract(_contours):
    points = np.array([Contour_Ymax_Calculate(contour) for contour in _contours])
    y_values = points[:, 1]
    
    y_values = y_values[np.argsort(-y_values)]
    
    y_data = np.mean(y_values.reshape(-1, 3), axis=1)
    
    y_data = np.array(np.round(y_data).astype(int))
    
    return y_data

# Theta Extract
def Theta_Extract(_h_data):
    return np.array(np.arctan(_h_data / FOCAL_LENGTH_H))

# ATAN Function(y: theta, x: distance)
def ATAN_FUNC(_x, _dY, _dZ, _noize):
    return np.arctan(_dY / (_x + _dZ)) + _noize

# Curve Fitting Parameter
def Extrinsic_Calibration(_theta):
    _params, _ = curve_fit(ATAN_FUNC, DISTANCE, _theta)
    dY, dZ, noize = _params
    theta_fit = ATAN_FUNC(DISTANCE, dY, dZ, noize)
    
    plt.scatter(DISTANCE, _theta, label='Raw Data', color='red')
    plt.plot(DISTANCE, theta_fit, label='Fitted Curve', color='magenta')  # Convert radians to degrees for better visualization
    plt.xlabel('Distance[mm]')
    plt.ylabel(r"${\theta}$[rad]")
    plt.legend()
    plt.show()
    
    return _params

# Parameter txt Save
def Save_Extrinsic_Parameter(_params):
    with open('../Parameter/Extrinsic_Mat.txt', 'w') as file:
        for _data in _params:
            file.write(str(_data) + '\n')
