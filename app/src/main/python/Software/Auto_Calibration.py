'''
* *****************************************************************************
* @author	Industrial Intelligence Lab
* @brief	Auto Calibration
* *****************************************************************************
'''

'''
* *****************************************************************************
* Import Module
* *****************************************************************************
'''
from Calibration_Function import *

'''
* *****************************************************************************
* Main Loop
* *****************************************************************************
'''
def main():
    # Load the image
    src = cv2.imread(data_path)
    
    # Image Intrinsic Calibration
    src_intrinsic, imSensor_H, FOCAL_LENGTH_H = Intrinsic_Calibration(src)
    
    # ROI Configuration
    ROI_Show(src_intrinsic)
    
    # Image Crop
    src_crop = Image_Crop(src_intrinsic)
    
    '''
    * *****************************************************************************
    * Image Processing
    * *****************************************************************************
    '''
    # Blue Extract
    src_blue = src_crop[:, :, 0]
    
    # Thresholding
    src_thred = Thresholding(src_blue, THRESHOLD)
    
    # Contour Area
    src_contour, contour_data = Contour_Extract(src_thred, CONTOUR_AREA)
    Image_Show(src_contour)
    Image_Pause()
    # Extract h_data
    h_data = Y_Pose_Extract(contour_data) - (int)(imSensor_H/2)
    print(h_data, FOCAL_LENGTH_H)
    # Extract Theta
    theta_data = Theta_Extract(h_data)
    
    # Extrinsic Calibration(Curve Fitting)
    params = Extrinsic_Calibration(theta_data)
    
    # Export
    Save_Extrinsic_Parameter(params)
    

if __name__ == "__main__":
    main()