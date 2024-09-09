'''
* *****************************************************************************
* @author	Industrial Intelligence Lab
* @brief	Brick line detection - Laser
* *****************************************************************************
'''

'''
* *****************************************************************************
* Import module
* *****************************************************************************
'''
from Laser_Function import *

'''
* *****************************************************************************
* Main Loop
* *****************************************************************************
'''
def main():
    directory_path = "../SampleDataset/Laser"
    files = [file for file in os.listdir(directory_path) if file.endswith('.jpg')]
    

    brick_height_mm_result = []
    for file_name  in files:
        data_path = os.path.join(directory_path, file_name)
        
        # Image read
        src = cv2.imread(data_path)
        
        # Image Intrinsic Calibration
        src_intrinsic = Intrinsic_Calibration(src)
        
        # Image Crop
        src_crop = Image_Crop(src_intrinsic)

        '''
        * *****************************************************************************
        * Brick Processing
        * *****************************************************************************
        '''
        # BGR to Gray
        src_gray = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY)
        # src_brick_thred = Thresholding(src_gray, THRESHOLD)
        src_brick_thred = Brick_Segmentation(src_gray, FLAG)

        # Extract Brick Contour Data Sorting
        contour_brick_data = Contour_Extract(src_brick_thred, BRICK_MINIMUM_AREA)
        
        src_contour = Draw_Contour(src_brick_thred, contour_brick_data)

        '''
        * *****************************************************************************
        * Green Laser Processing
        * *****************************************************************************
        '''
        # Green Range Extraction
        src_laser = Laser_Extract_InRange(src_crop)
        
        # Thresholding
        src_laser_thred = Thresholding(src_laser, THRESHOLD_GREEN)

        # Extract Laser Contour Data
        contour_laser_data = Contour_Extract(src_laser_thred, 3000)
        dst = Draw_Contour(src_laser_thred, contour_laser_data)

        '''
        * *****************************************************************************
        * Calculate Height of each brick
        * *****************************************************************************
        '''

        # Each Brick 
        for brick in range(len(contour_brick_data)):

            src_brick_filtered, src_laser_filtered = Brick_Laser_Extract(src_gray, brick, contour_brick_data, contour_laser_data)

            # brick height pixel extraction
            height_y_px, src_intrinsic = Brick_Point_Extract(src_laser_filtered, src_intrinsic)

            # Save Brick Height[mm]
            brick_height_mm.append(Get_Brick_Height(height_y_px))
    
    brick_height_mm_result = brick_height_mm

    # Visualization Brick Height
    height, width, _ = src_intrinsic.shape
    rect_width = int(width // 4)
    rect_height = int(height // 7)
    top_left_point = (width - rect_width, 0)
    bottom_right_point = (width, rect_height)
    
    # # Image read
    # data_path = f"../SampleDataset/Laser/{6}.jpg"
    # src = cv2.imread(data_path)
    
    # # Image Intrinsic Calibration
    # src_intrinsic = Intrinsic_Calibration(src)

    cv2.rectangle(src_intrinsic, top_left_point, bottom_right_point, (0, 0, 0), -1)


    brick_height_mm_result = np.reshape(brick_height_mm_result, (-1, 2))

    for brick_idx in range(len(brick_height_mm_result)):
        text = f"Brick {brick_idx + 1}: {brick_height_mm_result[brick_idx][0]:.2f}/{brick_height_mm_result[brick_idx][1]:.2f} mm"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = top_left_point[0] + 20
        text_y = top_left_point[1] + text_size[1] + 100 + brick_idx * 200
        cv2.putText(src_intrinsic, text, (text_x, text_y), font, font_scale, font_color, font_thickness)     

    # Image Show & brick height show
    print(brick_height_mm)
    Image_Show(src_intrinsic)
    Image_Pause()    



if __name__ == "__main__":
    main()
