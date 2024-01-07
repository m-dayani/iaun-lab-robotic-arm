import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

img_dir = os.getenv("IMG_PATH")

img = cv2.imread(os.path.join(img_dir, 'npr.brightspotcdn.webp'))
img1 = cv2.imread(os.path.join(img_dir, '1665339569_241056_url.jpeg'))
img2 = cv2.imread(os.path.join(img_dir, 'creative-learning-objects-on-a-wooden-table-GND52K.jpg'))


def img_resize(img, scale):
    width = img.shape[1]
    height = img.shape[0]
    scaled_dim = (int(width * scale), int(height * scale))
    return cv2.resize(img, scaled_dim, cv2.INTER_AREA)


img_resized = img_resize(img, 0.5)
img_resized1 = img_resize(img1, 0.5)
img_resized2 = img_resize(img2, 0.5)

# cv2.imshow("Image", img_resized)
# cv2.imshow("Image1", img_resized1)
# cv2.imshow("Image2", img_resized2)

gray = cv2.cvtColor(img_resized1, cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)

canny = cv2.Canny(gauss, 127, 255)

dilated = cv2.dilate(canny, (3, 3), iterations=5)
eroded = cv2.erode(dilated, (3, 3), iterations=5)

# cv2.imshow("Gray", gray)
# cv2.imshow("Canny", canny)

ret, thresh = cv2.threshold(gauss, 127, 255, cv2.THRESH_BINARY)

contours, h = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

blank = np.zeros(img_resized1.shape)

blank = cv2.drawContours(blank, contours, -1, (0, 0, 255), 2)


# for each of the contours detected, the shape of the contours is approximated using approxPolyDP()
# function and the contours are drawn in the image using drawContours() function
def detect_objects(contours, blank):
    font_scale = 0.5
    f_thick = 1
    for count in contours:
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        cv2.drawContours(blank, [approximations], 0, (0, 255, 0), 3)
        # the name of the detected shapes are written on the image
        i, j = approximations[0][0]
        if len(approximations) == 3:
            cv2.putText(blank, "Triangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif len(approximations) == 4:
            cv2.putText(blank, "Rectangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif len(approximations) == 5:
            cv2.putText(blank, "Pentagon", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif 6 < len(approximations) < 15:
            cv2.putText(blank, "Ellipse", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        else:
            cv2.putText(blank, "Circle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        # displaying the resulting image as the output on the screen
        cv2.imshow("Resulting_image", blank)
        cv2.waitKey(0)

# cv2.imshow('Thresh', thresh)
cv2.imshow('Contours', blank)


gx = cv2.Sobel(gauss, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(gauss, cv2.CV_32F, 0, 1)
mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0.0)
mag1 = np.uint8(cv2.normalize(np.sqrt(gx * gx + gy * gy), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255)

# cv2.imshow('Gradient Mag (linear)', mag)
# cv2.imshow('Gradient Mag (rms)', mag1)


img = cv2.imread(os.path.join(img_dir, 'dice-on-a-craps-table-CX6M3W.jpg'))
assert img is not None, "file could not be read, check with os.path.exists()"
img = img_resize(img, 0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# cv.imshow('Thresh', thresh)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# cv.imshow("Sure_fg", unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

cv2.imshow('final', img)

cv2.waitKey(0)
