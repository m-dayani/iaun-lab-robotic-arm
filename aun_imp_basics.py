import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys


def img_resize(img, scale):
    width = img.shape[1]
    height = img.shape[0]
    scaled_dim = (int(width * scale), int(height * scale))
    return cv2.resize(img, scaled_dim, cv2.INTER_AREA)


def calc_img_gradient(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # mag = cv2.addWeighted(gx, 0.5, gy, 0.5, 0.0)
    m = np.sqrt(gx * gx + gy * gy)
    mag = np.uint8(cv2.normalize(m, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255)
    ang = np.arctan2(gx, gy)
    # cv2.imshow('Gradient Mag (linear)', mag)
    # cv2.imshow('Gradient Mag (rms)', mag1)
    return mag, ang, gx, gy


def point_dist(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def calc_object_center(patch):
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(patch, ddepth, 1, 0)
    dy = cv2.Sobel(patch, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)

    th = 100
    mask = mag > th
    w, h = mask.shape

    X = np.repeat(np.arange(0, h).reshape((1, h)), w, axis=0)
    Y = np.repeat(np.arange(0, w).reshape((w, 1)), h, axis=1)
    x_sel = X[mask]
    y_sel = Y[mask]
    obj_x = np.sum(x_sel) / len(x_sel)
    obj_y = np.sum(y_sel) / len(y_sel)

    return np.array([obj_x, obj_y])


def smooth(img, gk_size=(3, 3), mk_size=5):
    # make sure images are grayscale
    gray = img
    if len(gray.shape) >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform gaussian denoising followed by median filtering
    gauss = cv2.GaussianBlur(gray, gk_size, cv2.BORDER_DEFAULT)
    med = cv2.medianBlur(gauss, mk_size)

    return med


def get_mesh(size):
    xsz = size[0]
    ysz = size[1]
    x = np.arange(ysz)
    y = np.arange(xsz)
    X = np.repeat(x.reshape((1, len(x))), xsz, axis=0)
    Y = np.repeat(y.reshape((len(y), 1)), ysz, axis=1)

    return X, Y


def get_mask(point, img_sz, r=30):
    x = point[0]
    y = point[1]
    w = img_sz[1]
    h = img_sz[0]

    xm_min = max(0, int(x - r))
    xm_max = min(w, int(x + r))
    ym_min = max(0, int(y - r))
    ym_max = min(h, int(y + r))

    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (xm_min, ym_min), (xm_max, ym_max), 255, -1)

    return mask


def test_img_resize(img, scale):
    img_resized = img_resize(img, scale)
    cv2.imshow("Resized Image", img_resized)
    cv2.waitKey(0)


def test_color_space(img, action):
    img_new = img
    if action == 'gray':
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Result", img_new)
    cv2.waitKey(0)


def test_im_filters(img, action):
    img_new = img
    if action == 'gauss':
        img_new = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    cv2.imshow("Result", img_new)
    cv2.waitKey(0)


def test_canny(img):
    canny = cv2.Canny(img, 127, 255)
    cv2.imshow("Canny", canny)
    cv2.waitKey(0)


def test_morph(img):
    dilated = cv2.dilate(img, (3, 3), iterations=5)
    eroded = cv2.erode(dilated, (3, 3), iterations=5)


def test_tresh(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Thresh', thresh)
    cv2.waitKey(0)


def test_contours(img):
    contours, h = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(img.shape)
    blank = cv2.drawContours(blank, contours, -1, (0, 0, 255), 2)
    cv2.imshow('Contours', blank)
    cv2.waitKey(0)


def main():
    from glob import glob
    for fn in glob(os.getenv("DATA_PATH") + '/another-square.jpg'):
        img = cv.imread(fn)
        squares = find_squares(img)
        cv.drawContours(img, squares, -1, (0, 255, 0), 3)
        cv.imshow('squares', img)
        ch = cv.waitKey()
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':

    img_dir = os.getenv("IMG_PATH")

    img = cv2.imread(os.path.join(img_dir, 'npr.brightspotcdn.webp'))
    img1 = cv2.imread(os.path.join(img_dir, '1665339569_241056_url.jpeg'))
    img2 = cv2.imread(os.path.join(img_dir, 'creative-learning-objects-on-a-wooden-table-GND52K.jpg'))
    img = cv2.imread(os.path.join(img_dir, 'dice-on-a-craps-table-CX6M3W.jpg'))
    assert img is not None, "file could not be read, check with os.path.exists()"

    main()
    cv.destroyAllWindows()
