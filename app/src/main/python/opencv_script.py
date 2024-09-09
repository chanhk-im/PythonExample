# app/src/main/python/opencv_script.py
import cv2
import numpy as np
import base64
from test import myFun

def process_image(img_str):
    myFun()
    img_bytes = base64.b64decode(img_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # sobel = cv2.Sobel(gray, cv2.CV_8U, ksize=3)
    # laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    canny = cv2.Canny(image, 100, 255)

    _, buffer = cv2.imencode('.jpg', canny)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64