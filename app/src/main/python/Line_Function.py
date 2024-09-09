'''
* *****************************************************************************
* @author	Industrial Intelligence Lab
* @brief	Brick line detection - Line
* *****************************************************************************
'''

'''
* *****************************************************************************
* Import module
* *****************************************************************************
'''
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import argparse


'''
* *****************************************************************************
* CONSTANT
* *****************************************************************************
'''



# data_path = "data\\20"
# Color
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_thickness = 10
font_color = WHITE  # 흰색


# ROI Range
ROI_RANGE = []

# Thresholding
THRED_SOBEL = [55, 255]
THRED_GRAY = [100, 255]
BRICK__CONTOUR_AREA = 200000
LINE_WEIGHT = 15
EDGE_KERNEL = 9

# Hough Line
HOUGH_BRICK_ANGLE_RANGE_V = [np.deg2rad(-1),np.deg2rad(1)]
HOUGH_THRED_V = 50
HOUGH_RHO_V = 100000

HOUGH_BRICK_ANGLE_RANGE_H = [np.deg2rad(89),np.deg2rad(91)]
HOUGH_THRED_H = 100
HOUGH_RHO_H = 100000


# Segmentation
NUM_CLUSTERS = 2
KMEANS_ANGLE_THRED = np.radians(5)

HOUGH_BRICK_ANGLE_RANGE = [HOUGH_BRICK_ANGLE_RANGE_V, HOUGH_BRICK_ANGLE_RANGE_H]
HOUGH_THRED = [HOUGH_THRED_V, HOUGH_THRED_H]
HOUGH_RHO = [HOUGH_RHO_V, HOUGH_RHO_H]


LINE_LENGTH = 100000

brick_height_mm = []

brick_line_points = []

# Mode
SOBEL_MODE = 1
BINARY_MODE = 2

FLAG= 1


'''
* *****************************************************************************
* Function
* *****************************************************************************
''' 
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
    
# Image Crop
def Image_Crop(_src):
    # ROI masking
    global ROI_RANGE
    mask = np.zeros(_src.shape[:2], dtype=np.uint8)
    # with open('parameter\\ROI_RANGE.txt', 'w') as file:
    #     for point in ROI_RANGE:
    #         file.write(f"{point[0]},{point[1]}\n")
    # with open("../Parameter/ROI_RANGE.txt", 'r') as file:
    #     for line in file:
    #         x, y = map(int, line.strip().split(','))
    #         ROI_RANGE.append((x, y))
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

# Kernel
def Kernel(_size):
    return np.ones((_size, _size), dtype=np.uint8)

# Intrinsic Calibration
def Intrinsic_Calibration(_src):
    h, w = _src.shape[:2]
    distortion_file = "/data/data/com.example.myapplication/files/chaquopy/AssetFinder/app/Parameter/Distort_Mat.txt"
    intrinsic_file = "/data/data/com.example.myapplication/files/chaquopy/AssetFinder/app/Parameter/Intrinsic_Mat.txt"
    
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

# Draw Contour
def Draw_Contour(_src, _contours):
    _dst = np.zeros_like(_src)
    cv2.drawContours(_dst, _contours, -1, WHITE, thickness=cv2.FILLED)
    
    return _dst

# Brick Edge Detection
def Brick_Edge(_src, _kernel):
    _src_eroded = cv2.erode(_src, _kernel, iterations=1)
    _dst = _src - _src_eroded
    
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
        
    
    return dst
    

# Hough Line Algorithm
def Brick_Line_Detector(_src, _src_line=None):
    # Hough line
    line_range = 0
    data_theta, data_rho = [], []
    max_x_line = None
    min_x_line = None
    max_x_value = float('-inf')
    min_x_value = float('inf')
    # Convert Gray to BGR
    _dst = None
    if _src_line is not None:
        _dst = cv2.cvtColor(_src_line, cv2.COLOR_GRAY2BGR)
        
    for line_range in range(len(HOUGH_BRICK_ANGLE_RANGE)):
        brick_lines = cv2.HoughLines(_src, rho=1, theta=np.pi/180, threshold=HOUGH_THRED[line_range], min_theta=HOUGH_BRICK_ANGLE_RANGE[line_range][0], max_theta=HOUGH_BRICK_ANGLE_RANGE[line_range][1])
        
        if brick_lines is not None:
            for brick_line in brick_lines:
                rho, theta = brick_line[0][0], brick_line[0][1]
                
                # if abs(rho) < HOUGH_RHO[line_range]:
                if abs(rho) < HOUGH_RHO[line_range] and abs(theta) < 0.01:
                    # data_theta.append(theta)
                    # data_rho.append(rho)
                    a, b = np.cos(theta), np.sin(theta)
                    x0, y0 = a*rho, b*rho
                    
                    x1 = int(x0 + LINE_LENGTH * -b)
                    y1 = int(y0 + LINE_LENGTH * a)
                    x2 = int(x0 - LINE_LENGTH * -b)
                    y2 = int(y0 - LINE_LENGTH * a)
                    
                    if _src_line is not None:
                        cv2.line(_dst, (x1, y1), (x2, y2), GREEN, 10)
                             
                    if x1 > max_x_value:
                        max_x_value = x1
                        max_x_line = brick_line
                    if x2 > max_x_value:
                        max_x_value = x2
                        max_x_line = brick_line
                        
                    if x1 < min_x_value:
                        min_x_value = x1
                        min_x_line = brick_line
                    if x2 < min_x_value:
                        min_x_value = x2
                        min_x_line = brick_line
                        
    rho, theta = min_x_line[0][0], min_x_line[0][1]
    data_theta.append(theta)
    data_rho.append(rho)
    rho, theta = max_x_line[0][0], max_x_line[0][1]
    data_theta.append(theta)
    data_rho.append(rho)
    line_data = np.column_stack((data_theta, data_rho))
    
    return line_data, _dst

# K-means Clustering
def KMeans_Brick_Clustering(_line_data):
    # Min-Max scaler
    minmax_scaler = MinMaxScaler()
    _scaled_data = minmax_scaler.fit_transform(_line_data)
    
    # K-means clustering
    _brick_kmeans = KMeans(n_clusters=NUM_CLUSTERS)
    _brick_kmeans.fit(_scaled_data)
    
    # Label
    _brick_labels = _brick_kmeans.labels_
    
    # Centroid
    _scaled_centroids = _brick_kmeans.cluster_centers_
    
    # Inverse transformation to raw data scale
    _kmean_line_data = minmax_scaler.inverse_transform(_scaled_data)
    _centroids_data = minmax_scaler.inverse_transform(_scaled_centroids)
    
    return _kmean_line_data, _centroids_data

# Point Extract
def Brick_Point_Extract(_src, _line_data):
    _point_y = []
    _points = []
    _src_buf = cv2.cvtColor(_src, cv2.COLOR_GRAY2BGR)
    # Draw Brick Line
    dst_line = np.copy(_src_buf)
    brick_line_px = []
    for centroid in _line_data:
        dst_line = np.copy(_src_buf)
        rho, theta = centroid[1], centroid[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        
        x1 = int(x0 + LINE_LENGTH * -b)
        y1 = int(y0 + LINE_LENGTH * a)
        x2 = int(x0 - LINE_LENGTH * -b)
        y2 = int(y0 - LINE_LENGTH * a)
        cv2.line(dst_line, (x1, y1), (x2, y2), GREEN, LINE_WEIGHT)
        # Image_Show(dst_line)
        # Image_Pause()
        # Brick & line filtering
        dst_line = cv2.bitwise_and(dst_line, _src_buf)

        
        # Brick line check
        brick_line_px = np.column_stack(np.where((dst_line == [0, 255, 0]).all(axis=-1)))
        
        # Sorting brick line info
        brick_line_px = brick_line_px[np.argsort(brick_line_px[:, 0])]
        
        # Brick Point Extract
        _point_y.append([brick_line_px[0,0], brick_line_px[-1,0]])
        _points.append([brick_line_px[0], brick_line_px[-1]])
        # print(_points)
    return _point_y, dst_line, _points

# TAN Function
def Brick_Length_FUNC(_theta, _dY, _dZ, _noize):
    return _dY/np.tan(_theta - _noize) - _dZ

# Get birck height [px] -> [mm]
def Get_Brick_Height(_height_y_px):
    # Get dY, dZ, noise
    with open('/data/data/com.example.myapplication/files/chaquopy/AssetFinder/app/Parameter/Extrinsic_Mat.txt', 'r') as file:
         dY, dZ, noise = [float(line.strip()) for line in file]

    _brick_height_mm = []
    
    # left, right height
    for idx in range(len(_height_y_px)):
        
        # y_max, y_min get
        y_max_px = _height_y_px[idx][1]
        y_min_px = _height_y_px[idx][0]

        # Plate height
        plate_px = y_max_px - (int)(imSensor_H/2)
        
        # theta
        theta = np.arctan(plate_px/(FOCAL_LENGTH_H))
        
        # Get Zc using (dY/np.tan(theta - noize) - dZ)
        Zc = dZ + Brick_Length_FUNC(theta, dY, dZ, noise)

        # Get Brick Height
        brick_height_px = y_max_px - y_min_px 
        _brick_height_mm.append(Zc * brick_height_px / FOCAL_LENGTH_H)

    return _brick_height_mm

def Draw_Brick_Line(_src, _points):
    for point in _points:
        y1, x1 = point[0]
        y2, x2 = point[-1]
        cv2.line(_src, (x1, y1), (x2, y2), RED, 20)
        cv2.circle(_src, (x1, y1), 20, RED, -1)
        cv2.circle(_src, (x2, y2), 20, RED, -1)
    
    return _src
        