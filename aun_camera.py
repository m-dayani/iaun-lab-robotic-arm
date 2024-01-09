import os
import numpy as np
from scipy.spatial.transform import Rotation as scipyR
import cv2
import pickle

from aun_cam_calib import recalib_camera


class PinholeCamera:
    def __init__(self):
        self.fx = 625.24678685
        self.fy = 624.90293937
        self.cx = 297.50306391
        self.cy = 251.99629151
        self.K = np.array([[self.fx, 0., self.cx],
                           [0., self.fy, self.cy],
                           [0., 0., 1.]])
        self.K_1 = np.linalg.inv(self.K)

        self.D = np.array([-0.1043762, 0.37733355, -0.00270735, -0.01037959, -0.44566628])

        self.R_cw = np.eye(3)
        self.t_cw = np.zeros((3, 1))
        self.T_cw = np.eye(4)
        self.T_cw[0:3, 0:3] = self.R_cw
        self.T_cw[0:3, 3] = self.t_cw.reshape((3,))
        self.T_wc = np.linalg.inv(self.T_cw)

        self.calibrated = False

        self.H = np.eye(3)
        self.H_1 = np.eye(3)
        self.Hp = self.K @ self.H
        self.Hp_1 = np.linalg.inv(self.Hp)

        self.ax_x_img = []
        self.ax_y_img = []
        self.img_size = np.array([640, 480])

    def update_pose(self, R_cw, t_cw):
        self.R_cw = np.copy(R_cw)
        self.t_cw = np.copy(t_cw)
        self.T_cw[0:3, 0:3] = self.R_cw
        self.T_cw[0:3, 3] = self.t_cw.reshape((3,))
        self.T_wc = np.linalg.inv(self.T_cw)

    def update_intrinsics(self, K):
        self.K = K
        self.K_1 = np.linalg.inv(K)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def update_homography(self):
        try:
            self.H[:, 0:2] = self.T_cw[0:3, 0:2]
            self.H[:, 2] = self.T_cw[0:3, 3]
            self.Hp = self.K @ self.H
            self.H_1 = np.linalg.inv(self.H)
            self.Hp_1 = np.linalg.inv(self.Hp)
        except np.linalg.LinAlgError:
            print('WARNING: Cannot calculate H^-1 because H is singular')

    def project(self, pt_w):
        if len(pt_w) < 3:
            return None
        Pw = np.array([pt_w[0], pt_w[1], pt_w[2], 1.0]).reshape((4, 1))
        Pc = self.K @ self.T_cw[0:3, :] @ Pw
        depth = Pc[2][0]
        return Pc / depth, depth

    def unproject(self, pt_img):
        if len(pt_img) < 2:
            return None
        upts = cv2.undistortPoints(np.array([pt_img[0], pt_img[1]], dtype=np.float32), self.K, self.D)
        upts = upts.squeeze()
        Pc = np.array([upts[0], upts[1], 1.0]).reshape((3, 1))
        Pw = self.R_cw.T @ (Pc - self.t_cw.reshape((3, 1)))
        return Pw

    def project_homo(self, pt_w):
        if len(pt_w) < 2:
            return None
        pt_w = np.array([pt_w[0], pt_w[1], 1.0]).reshape((3, 1))
        xy_p = self.Hp @ pt_w
        depth = np.copy(xy_p[2, :])
        xy_p /= depth
        return np.array(xy_p[0], xy_p[1])

    def unproject_homo(self, pt_img):
        if len(pt_img) < 2:
            return None
        upts = cv2.undistortPoints(np.array([pt_img[0], pt_img[1]], dtype=np.float32), self.K, self.D)
        upts = upts.squeeze()
        upts_homo = np.array([upts[0], upts[1], 1.0])
        pc = self.H_1 @ upts_homo.reshape((3, 1))
        s_pc = np.copy(pc[2, :])
        pc /= s_pc
        return np.array([pc[0], pc[1]])

    def save_params(self, file_name):
        with open(file_name, 'wb') as cam_file:
            pickle.dump({'K': self.K, 'D': self.D, 'T_cw': self.T_cw, 'calibrated': self.calibrated,
                         'ax_x_img': self.ax_x_img, 'ax_y_img': self.ax_y_img, 'img_size': self.img_size}, cam_file)

    def load_params(self, file_name):
        if not os.path.exists(file_name):
            return
        with open(file_name, 'rb') as cam_file:
            obj = pickle.load(cam_file)

            self.calibrated = obj['calibrated']

            if self.calibrated:
                self.D = obj['D']
                T_cw = obj['T_cw']
                self.update_pose(T_cw[0:3, 0:3], T_cw[0:3, 3])
                self.update_intrinsics(obj['K'])
                self.update_homography()
                self.ax_x_img = obj['ax_x_img']
                self.ax_y_img = obj['ax_y_img']
                self.img_size = obj['img_size']


class LabCamera(PinholeCamera):

    def __init__(self, port, img_size):
        super().__init__()
        self.port = port
        self.img_size = img_size
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.capture_duration = 10
        self.usb_cam = None
        self.init()

    def init(self):
        self.usb_cam = cv2.VideoCapture(self.port)

        # usb_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('U', 'Y', 'V', 'Y'))
        self.usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        self.usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)

        if not self.usb_cam.isOpened():
            print("Error reading video file")

        img_width = int(usb_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(usb_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("image size: (%d x %d)" % (img_width, img_height))

    def get_next(self):
        if self.usb_cam is not None:
            return self.usb_cam.read()
        return False, None

    def calibrate(self, calib_path):
        # calibrate_camera(cam, frame, tm_patch)
        res = recalib_camera(calib_path, frame, my_cam, ws=2.5)
        return res

    def close(self):
        if self.usb_cam is not None:
            usb_cam.release()


if __name__ == "__main__":
    base_dir = os.getenv('DATA_PATH')
    param_file = os.path.join(base_dir, 'lab-cam-params.pkl')
    my_cam = PinholeCamera()
    my_cam.save_params(param_file)
    my_cam.load_params(param_file)
