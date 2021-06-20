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

config013 = {
    "BlurKsize": 1, #2*value+1
    "BlockSize": 200, #2*value+1
    "C": 0,
    "OpeningKsize": 4,
    "Invert": False,
    "Sinus" : False,
}

config2 = {
    "BlurKsize": 1, #2*value+1
    "BlockSize": 200, #2*value+1
    "C": 0,
    "OpeningKsize": 4,
    "Invert": False,
    "Sinus" : True,
    "SinusThreshold": 230,
}

config4 = {
    "BlurKsize": 0, #2*value+1
    "BlockSize": 150, #2*value+1
    "C": 20,
    "OpeningKsize": 16,
    "Invert": True,
    "Sinus" : False,
}

config56 = {
    "BlurKsize": 1, #2*value+1
    "BlockSize": 200, #2*value+1
    "C": 20,
    "OpeningKsize": 16,
    "Invert": True,
    "Sinus" : False,
}

config7 = {
    "BlurKsize": 0, #2*value+1
    "BlockSize": 200, #2*value+1
    "C": 21,
    "OpeningKsize": 16,
    "Invert": True,
    "Sinus" : False,
}

defaultImage = listImage[2]
config = config2
parser = argparse.ArgumentParser(description='Object counting tool')
parser.add_argument('--image', help='Path to the input image.', default=defaultImage)
args = parser.parse_args()

originImg = cv.imread(args.image)
if originImg is None:
    sys.exit("Could not read the image.")

cv.imshow("Origin image", originImg)

grayImg = cv.cvtColor(originImg, cv.COLOR_BGR2GRAY)
cv.imshow("Gray image", grayImg)

if (config["Sinus"]):
    f = np.fft.fft2(grayImg)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(1 + np.abs(fshift))
    magnitude_spectrum = cv.normalize(src=magnitude_spectrum, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv.imshow("magitude_spectrum", np.uint8(magnitude_spectrum))
    h, w = fshift.shape[:2]
    temp = fshift[h//2][w//2]
    fshift = np.where(magnitude_spectrum > config["SinusThreshold"], 0 , fshift)
    magnitude_spectrum = np.where(magnitude_spectrum > config["SinusThreshold"], 0, magnitude_spectrum)
    cv.imshow("magitude_spectrum_after", np.uint8(magnitude_spectrum))
    fshift[h//2][w//2] = temp
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    cv.imshow("After", np.uint8(img_back))
    grayImg = np.uint8(img_back)

blurWindowName = "Blur Image"
cv.namedWindow(blurWindowName)
cv.createTrackbar("Blur level", blurWindowName, config["BlurKsize"], 15, nothing)

binaryWindowName = "Binary Image"
cv.namedWindow(binaryWindowName)
cv.createTrackbar("C", binaryWindowName, config["C"], 255, nothing)
cv.createTrackbar("BlockSize", binaryWindowName, config["BlockSize"], 1000, nothing)

openingWindowName = "Opening/Closing Image"
cv.namedWindow(openingWindowName)
cv.createTrackbar("Ksize", openingWindowName, config["OpeningKsize"], 100, nothing)

regionWindowName = "Region"
cv.namedWindow(regionWindowName)

colors = []
for i in range(10000):
    colors.append(list(np.random.random(size=3) * 256))

while True:
    blurKsize = cv.getTrackbarPos("Blur level", blurWindowName)
    blurKsize = blurKsize*2+1
    blurImg = cv.medianBlur(grayImg, blurKsize)
    cv.imshow(blurWindowName, blurImg)

    C = cv.getTrackbarPos("C", binaryWindowName)
    BlockSize = cv.getTrackbarPos("BlockSize", binaryWindowName)
    if(BlockSize == 0):
        BlockSize = 1
    thresholdType = cv.THRESH_BINARY_INV if config["Invert"] else cv.THRESH_BINARY
    binaryImg = cv.adaptiveThreshold(blurImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType, 2*BlockSize+1, C)
    cv.imshow(binaryWindowName, binaryImg)

    OpeningKsize = cv.getTrackbarPos("Ksize", openingWindowName)
    kernel = np.ones((OpeningKsize, OpeningKsize), np.uint8)
    openImg = cv.morphologyEx(binaryImg, cv.MORPH_CLOSE, kernel) if config["Invert"] else cv.morphologyEx(binaryImg, cv.MORPH_OPEN, kernel)
    cv.imshow(openingWindowName, openImg)

    contours, hierarchy = cv.findContours(openImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    regionImg = originImg.copy()
    h, w = regionImg.shape[:2]
    imgArea = (h-1)*(w-1)

    count = 0
    for contour in contours:
        #if(cv.contourArea(contour) > imgArea /3): continue
        #cv.drawContours(regionImg, [contour], -1, colors[count], cv.FILLED)

        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv.boundingRect(approx)
        cv.rectangle(regionImg, (x, y), (x + w, y + h), (255, 0, 0), 1)

        cv.imshow(regionWindowName, regionImg)
        count +=1
        # cv.waitKey()

    cv.putText(regionImg, "Count: " + str(count), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow(regionWindowName, regionImg)
    k = cv.waitKey(100)
    if k == 13:# press enter to exit
        break

cv.destroyAllWindows()
