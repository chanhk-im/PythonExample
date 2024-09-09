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
from Line_Function import *
import os
from os.path import dirname, join

'''
* *****************************************************************************
* Main Loop
* *****************************************************************************
'''
def main(args):
    filename = join(dirname(__file__), args)
    print(filename)
    # Image Read
    data_path = f"{filename}"
    print(data_path)
    src = cv2.imread(data_path)
    if src is None:
        raise ValueError("The specified file was not found or could not be opened.")
    
    
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
    
    src_mode = Brick_Segmentation(src_gray, FLAG)
    
    # Contour Extraction(findContour)
    contour_brick_data = Contour_Extract(src_mode, BRICK__CONTOUR_AREA)

    '''
    * *****************************************************************************
    * Brick Selection
    * *****************************************************************************
    '''
    for brick in range(len(contour_brick_data)):
        # Draw Contour
        src_brick = Draw_Contour(src_mode, [contour_brick_data[brick]])

        # Brick Edge Detection
        src_edge = Brick_Edge(src_brick, Kernel(EDGE_KERNEL))

        # Line Data
        brick_line_data, src_line = Brick_Line_Detector(src_edge, src_edge)
        
        # K-means Algorithm
        kmean_line_data, centroids_data = KMeans_Brick_Clustering(brick_line_data)
        
        brick_height_y_px, dst_line, brick_line_points = Brick_Point_Extract(src_edge, centroids_data)

        # Save Brick Height[mm]
        brick_height_mm.append(Get_Brick_Height(brick_height_y_px))

        # Draw Brick Line
        src_intrinsic = Draw_Brick_Line(src_intrinsic, brick_line_points)
    
    '''
    * *****************************************************************************
    * Image Show
    * *****************************************************************************
    '''
    # Visualization Brick Height
    height, width, _ = src_intrinsic.shape
    rect_width = int(width // 4)
    rect_height = height // 7
    top_left_point = (width - rect_width, 0)
    bottom_right_point = (width, rect_height)

    # Draw Rectangle
    cv2.rectangle(src_intrinsic, top_left_point, bottom_right_point, (0, 0, 0), -1)
    
    for brick_idx in range(len(brick_height_mm)):
        text = f"Brick {brick_idx + 1}: {brick_height_mm[brick_idx][0]:.2f} / {brick_height_mm[brick_idx][1]:.2f} mm"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = top_left_point[0] + 20 
        text_y = top_left_point[1] + text_size[1] + 100 + brick_idx * 200
        cv2.putText(src_intrinsic, text, (text_x, text_y), font, font_scale, font_color, font_thickness)     


    print(brick_height_mm)
    # Image_Show(src_intrinsic)
    # Image_Pause()
    
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Brick Line Detection Script')
#     parser.add_argument('--fileName', default='Line_Data.jpg', required=False, help='Name of the file to process', )
#     args = parser.parse_args()
#     main(args)
