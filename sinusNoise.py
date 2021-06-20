import math

import cv2 as cv
import numpy as np
import argparse
import sys
from matplotlib import pyplot as plt

def nothing(value):
    pass

listImage = [
    "images/1_wIXlvBeAFtNVgJd49VObgQ.png",
    "images/1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png",
    "images/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png",
    "images/1_zd6ypc20QAIFMzrbCmJRMg.png",
    "images/objets1.jpg",
    "images/objets2.jpg",
    "images/objets3.jpg",
    "images/objets4.jpg"
]

defaultImage = listImage[2]

parser = argparse.ArgumentParser(description='Object counting tool')
parser.add_argument('--image', help='Path to the input image.', default=defaultImage)
args = parser.parse_args()

originImg = cv.imread(args.image)
if originImg is None:
    sys.exit("Could not read the image.")

cv.imshow("Origin image", originImg)

grayImg = cv.cvtColor(originImg, cv.COLOR_BGR2GRAY)
cv.imshow("Gray image", grayImg)

f = np.fft.fft2(grayImg)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(1+np.abs(fshift))
magnitude_spectrum_origin = cv.normalize(src=magnitude_spectrum, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
magnitude_spectrum_img = magnitude_spectrum_origin.copy()

rows, cols = fshift.shape
mask = np.ones((rows,cols),np.uint8)

ix, iy = -1,-1
drawing = False

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,magnitude_spectrum_img
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            r = int(math.sqrt((ix-x)**2 + (iy-y)**2))//2
            xc = (x+ix)//2
            yc = (y+iy)//2
            magnitude_spectrum_img = magnitude_spectrum_origin.copy()
            cv.circle(magnitude_spectrum_img,(xc,yc),r,0,1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        r = int(math.sqrt((ix-x)**2 + (iy-y)**2))//2
        xc = (x + ix) // 2
        yc = (y + iy) // 2
        cv.circle(magnitude_spectrum_origin, (xc, yc), r, 0, -1)
        magnitude_spectrum_img = magnitude_spectrum_origin.copy()
        cv.circle(mask,(xc,yc),r,0,-1)

magnitude_spectrum_window = "magnitude spectrum"
cv.namedWindow(magnitude_spectrum_window)
cv.setMouseCallback(magnitude_spectrum_window, draw_circle)

while(True):
    cv.imshow(magnitude_spectrum_window, magnitude_spectrum_img)
    f_ishift = np.fft.ifftshift(fshift*mask)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    cv.imshow("After", np.uint8(img_back))

    k = cv.waitKey(20)
    if k == 13:  # press enter to exit
        break

cv.destroyAllWindows()




