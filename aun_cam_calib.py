#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as scipyR
import matplotlib.pyplot as plt


class CamCalib:
    def __init__(self, calib_path):

        self.img_path = calib_path

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (6, 9)
        self.CHECKERBOARD = CHECKERBOARD
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Defining the world coordinates for 3D points
        self.objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        # prev_img_shape = None

        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpoints = []

        self.img_size = []
        self.n_images = 0

    def set_images_path(self, path):
        self.img_path = path

    def add_image(self, gray):

        ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        corners2 = corners
        if ret:
            self.objpoints.append(self.objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            self.imgpoints.append(corners2)

            self.n_images += 1

        return ret, corners2

    def load_images(self, img_ext='png', img_show=False):

        images = glob.glob(self.img_path + '/*.' + img_ext)
        for fname in images:

            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = self.add_image(gray)

            # Draw and display the corners
            if img_show and ret:
                img = cv2.drawChessboardCorners(img, self.CHECKERBOARD, corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(0)

            if len(self.img_size) <= 0:
                self.img_size = gray.shape[::-1]
                # print(self.img_size)

    def calibrate(self, K=None, D=None):
        # Always load images before calling this
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, K, D)
        return ret, mtx, dist, rvecs, tvecs

    def recalib(self, cam_obj, new_img, ws=1.0):

        if cam_obj is None:
            return False

        # include old images
        self.load_images()

        if self.n_images < 9:
            return False

        # add the new image
        corners = []
        if len(new_img) > 0:
            new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            ret, corners = self.add_image(new_gray)
            if not ret:
                return False

        # calibrate
        self.objpoints = ws * np.float32(self.objpoints)
        ret, mtx, dist, rvecs, tvecs = self.calibrate()

        # the last estimate is what we want
        robj = scipyR.from_rotvec(rvecs[-1].reshape((1, 3)).squeeze())
        R_cw = robj.as_matrix()
        t_cw = tvecs[-1].reshape((1, 3)).squeeze()

        cam_obj.update_pose(R_cw, t_cw)
        cam_obj.update_intrinsics(mtx)
        cam_obj.D = dist.squeeze()
        cam_obj.img_size = self.img_size
        cam_obj.calibrated = True

        cam_obj.update_homography()

        if len(corners) > 0:
            pts_img = corners.squeeze()
            cam_obj.ax_x_img = [pts_img[0], pts_img[1]]
            cam_obj.ax_y_img = [pts_img[0], pts_img[self.CHECKERBOARD[0]]]

        return True

        # pts_w = ws * np.float32(objp).squeeze()
        # calc_reprojection_error(cam_obj, pts_w, pts_img)
        # test_homography(cam_obj, pts_w, pts_img)

    def refine_intrinsics(self, cam_obj, img, show_img=False):

        K = cam_obj.K
        D = cam_obj.D
        img_size = cam_obj.img_size
        # Refining the camera matrix using parameters obtained by calibration
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)

        # Method 1 to undistort the image
        dst = cv2.undistort(img, K, D, None, K_new)

        # Method 2 to undistort the image
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K_new, img_size, 5)

        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        if show_img:
            # Displaying the undistorted image
            cv2.imshow("undistorted image", dst)
            cv2.waitKey(0)


def calc_reprojection_error(cam_model, pts_w, pts_img):
    if cam_model is None:
        return -1

    # TEST 1: Reprojection Error
    T_cw = cam_model.T_cw
    robj = scipyR.from_matrix(cam_model.R_cw)
    rvec = robj.as_rotvec()
    tvec = cam_model.t_cw
    K = cam_model.K
    D = cam_model.D

    # use opencv project method
    res, other = cv2.projectPoints(pts_w, rvec, tvec, K, D, pts_img)

    # project the hard way!
    pts_w_h = np.concatenate([pts_w, np.ones((len(pts_w), 1))], axis=1).T

    Pc = T_cw[0:3, :] @ pts_w_h
    depth_pc = np.copy(Pc[2, :])
    Pc /= depth_pc

    xy = K @ Pc
    xy = xy[0:2, :].T

    Pc_p = cv2.undistortPoints(pts_img, K, D).squeeze()

    err = abs(res.squeeze() - pts_img)
    sum_err = np.sum(np.sqrt(np.sum(err * err, axis=1)))
    err1 = abs(Pc[0:2, :].T - Pc_p)
    sum_err1 = np.sum(np.sqrt(np.sum(err1 * err1, axis=1)))
    print('sum projection error (OpenCV): %.4f' % sum_err)
    print('sum projection error: %.4f' % sum_err1)

    return sum_err1


def test_homography(cam_model, pts_w, pts_img):
    if cam_model is None:
        return

    T_cw = cam_model.T_cw
    K = cam_model.K
    D = cam_model.D

    # TEST 2: Homographies
    H = np.eye(3)
    H[:, 0:2] = T_cw[0:3, 0:2]
    H[:, 2] = T_cw[0:3, 3]
    Hp = K @ H

    H_1 = np.linalg.inv(H)
    Hp_1 = np.linalg.inv(Hp)

    pts_img_norm = cv2.undistortPoints(pts_img, K, D).squeeze()
    upts_homo = np.concatenate([pts_img_norm, np.ones((len(pts_img_norm), 1))], axis=1).T

    XY_orig = np.concatenate([pts_w[:, 0:2], np.ones((len(pts_w), 1))], axis=1)

    # Projection
    xy_p = Hp @ XY_orig.T
    dd = np.copy(xy_p[2, :])
    xy_pp = xy_p / dd  # these are image points if you distort them with D

    # Unprojection
    # remember the points from cv.undistortPoints are normalized by K
    # (so don't use K_1 in the following equation)
    pc = H_1 @ upts_homo

    s_pc = np.copy(pc[2, :])

    XY_homo = pc / s_pc
    XY_homo = XY_homo.T

    err = abs(XY_homo[:, 0:2] - pts_w[:, 0:2])


def show_img_axis(img, cam_obj):
    img = cv2.line(img, np.int32(cam_obj.ax_x_img[0]), np.int32(cam_obj.ax_x_img[1]), (0, 0, 255))
    img = cv2.line(img, np.int32(cam_obj.ax_y_img[0]), np.int32(cam_obj.ax_y_img[1]), (0, 255, 0))
    img = cv2.circle(img, np.int32(cam_obj.ax_x_img[0]), 3, (255, 0, 0))

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.waitKey(0)


def draw_camera_and_wpts(cam_obj, pts_w, ws=1.0):
    ax = plt.figure().add_subplot(projection='3d')

    cs = 3.0
    X = np.concatenate([cam_obj.t_cw, cam_obj.t_cw + cs * cam_obj.R_cw[:, 0]]).reshape((2, 3))
    Y = np.concatenate([cam_obj.t_cw, cam_obj.t_cw + cs * cam_obj.R_cw[:, 1]]).reshape((2, 3))
    Z = np.concatenate([cam_obj.t_cw, cam_obj.t_cw + cs * cam_obj.R_cw[:, 2]]).reshape((2, 3))

    ss = 5.0
    x = np.array([[0, 0, 0], [1., 0, 0]]) * ss
    y = np.array([[0, 0, 0], [0, 1., 0]]) * ss
    z = np.array([[0, 0, 0], [0, 0, 1.]]) * ss

    pts_w = ws * np.float32(pts_w)

    ax.plot(X[:, 0], X[:, 1], X[:, 2], c="r")
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], c="g")
    ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], c="b")

    ax.scatter(pts_w[:, 0], pts_w[:, 1], pts_w[:, 2])

    ax.plot(x[:, 0], x[:, 1], x[:, 2], c="r")
    ax.plot(y[:, 0], y[:, 1], y[:, 2], c="g")
    ax.plot(z[:, 0], z[:, 1], z[:, 2], c="b")

    ax.set_aspect('equal', adjustable='box')

    plt.show()


# this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    # right-click event value is 2
    if event == cv2.EVENT_LBUTTONDOWN:
        print([x, y])


if __name__ == "__main__":

    data_dir = os.getenv('DATA_PATH')
    image_path = os.path.join(data_dir, 'aun-lab-calib')
    sample_image = os.path.join(data_dir, 'image0.png')
    img_show = cv2.imread(sample_image)
    new_img = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)

    camCalib = CamCalib(image_path)
    camCalib.load_images()
    camCalib.add_image(new_img)
    camCalib.calibrate()

    # camCalib.recalib(cam_obj, new_img, ws=2.5)
    # Visualizations
    # show_img_axis(img_show, cam_obj)
    # draw_camera_and_wpts(cam_obj, objp.squeeze())
