import cv2 as cv
import numpy as np
import argparse
import sys

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

config = {
    "blurKennelSize": 1, #value*2 +1
    "cannyMinThreshold": 100,
    "cannyMaxThreshold": 200,
    "SinusThreshold": 180
}

cv.imshow("Origin image", originImg)

grayImg = cv.cvtColor(originImg, cv.COLOR_BGR2GRAY)

f = np.fft.fft2(grayImg)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(1 + np.abs(fshift))
magnitude_spectrum = cv.normalize(src=magnitude_spectrum, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                  dtype=cv.CV_8U)
cv.imshow("magitude_spectrum", np.uint8(magnitude_spectrum))
h, w = fshift.shape[:2]
temp = fshift[h // 2][w // 2]
fshift = np.where(magnitude_spectrum > config["SinusThreshold"], 0, fshift)
magnitude_spectrum = np.where(magnitude_spectrum > config["SinusThreshold"], 0, magnitude_spectrum)
cv.imshow("magitude_spectrum_after", np.uint8(magnitude_spectrum))
fshift[h // 2][w // 2] = temp
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)
cv.imshow("After", np.uint8(img_back))
grayImg = np.uint8(img_back)

cv.imshow("Gray image", grayImg)

blurWindowName = "Blur image(medianBlur)"
cv.namedWindow(blurWindowName)
cv.createTrackbar("Blur lever", blurWindowName, config["blurKennelSize"], 15, nothing)

detectEdgeWindowName = "Detect edge(canny)"
cv.namedWindow(detectEdgeWindowName)
cv.createTrackbar("Min", detectEdgeWindowName, config["cannyMinThreshold"], 255, nothing)
cv.createTrackbar("Max", detectEdgeWindowName, config["cannyMaxThreshold"], 255, nothing)

# binaryWindowName = "Binary Image"
# cv.namedWindow(binaryWindowName)
# ret, th = cv.threshold(grayImg, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# cv.createTrackbar("Threshold", binaryWindowName, int(ret), 255, nothing)

regionWindowName = "Region"
cv.namedWindow(regionWindowName)

while True:
    # threshold = cv.getTrackbarPos("Threshold", binaryWindowName)
    # binaryImg = cv.adaptiveThreshold(grayImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, threshold)
    # cv.imshow(binaryWindowName, binaryImg)

    blurKennelSize = cv.getTrackbarPos("Blur lever", blurWindowName)
    blurKennelSize = blurKennelSize*2 + 1  #Ksize = 2*value+1
    blurImg = cv.medianBlur(grayImg, blurKennelSize)
    cv.imshow(blurWindowName, blurImg)

    cannyMinThreshold = cv.getTrackbarPos("Min", detectEdgeWindowName)
    cannyMaxThreshold = cv.getTrackbarPos("Max", detectEdgeWindowName)
    cannyImg = cv.Canny(blurImg, cannyMinThreshold, cannyMaxThreshold)
    cv.imshow(detectEdgeWindowName, cannyImg)

    contours, hierarchy = cv.findContours(cannyImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    regionImg = originImg.copy()
    h, w = regionImg.shape[:2]
    imgArea = (h-1)*(w-1)
    # print(imgArea)
    count = 0
    for contour in contours:
        # region = cv.convexHull(contour)
        area = cv.contourArea(contour)
        # print(area)
        if area >= imgArea/3:
            continue

        cv.drawContours(regionImg, [contour], -1, 255, cv.FILLED)
        # cv.drawContours(regionImg, [contour], -1, 255, cv.FILLED, offset=(-1, -1))

        cv.imshow(regionWindowName, regionImg)
        count += 1

        #cv.waitKey()

    cv.putText(regionImg, "Count: " + str(count), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow(regionWindowName, regionImg)
    k = cv.waitKey(100)
    if k == 13: #press enter to exit
        break

cv.destroyAllWindows()




