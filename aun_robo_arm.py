import os
import sys
import numpy as np
import cv2
import time

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as scipyR

from aun_arduino import MyArduino
from aun_objdt_classic import TrackerKLT
from aun_imp_basics import calc_object_center, get_mask
from aun_camera import LabCamera


px_loc = []
point_updated = False


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def mouse_callback(event, x, y, flags, params):

    global px_loc
    global point_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        px_loc = np.array([x, y])
        # print(cam.unproject_homo(px_loc))
        point_updated = True


def detect_object(frame, patch, obj_center):
    # Some methods to locate an object in the image:
    #   1. Template matching/registration
    #   2. Feature extraction/matching (e.g. ORB features)
    #   3. Deep learning

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if patch is None or len(patch) <= 0:
        points = cv2.selectROI(frame, False)
        xy0 = np.array(points[0:2])
        wh0 = xy0 + np.array(points[2:4])
        patch = gray[xy0[1]:wh0[1], xy0[0]:wh0[0]]

    if len(obj_center) <= 0:
        obj_center = calc_object_center(patch)

    # Template Matching (find ROI)
    tm_res = cv2.matchTemplate(gray, patch, cv2.TM_SQDIFF_NORMED)
    _, _, tm_loc, _ = cv2.minMaxLoc(tm_res)

    return np.array(tm_loc) + obj_center, patch, obj_center


def track_klt_of(trackerKLT, frame):

    global point_updated
    global px_loc

    if point_updated:
        point_updated = False
        mask = get_mask(px_loc, frame.shape[:2])
        trackerKLT.initialize(frame, mask)
        trackerKLT.updatePoints(np.concatenate([np.array([[px_loc]]), trackerKLT.p0], axis=0))

        return px_loc
    else:
        if trackerKLT.is_initialized:
            trackerKLT.track(frame)
            good_new = trackerKLT.p1
            if trackerKLT.p1 is not None:
                st = trackerKLT.st
                good_new = trackerKLT.p1[st == 1]

                if good_new is None or len(good_new) <= 0:
                    return []

            trackerKLT.updatePoints(good_new.reshape(-1, 1, 2))
            return good_new[0]
        else:
            return []


class Robustify:
    def __init__(self):
        self.xLim = [0.0, 200.0]
        self.yLim = [0.0, 200.0]
        self.limAction = 1  # output current values
        self.last_point = np.array([0.0, 0.0])

        self.nAvgPoints = 20
        self.xyQueue = [(-1.0, -1.0) for v in range(self.nAvgPoints)]
        self.xyAverage = True
        self.idx = 0

    def check_limits(self, xy_point):
        return self.xLim[0] <= xy_point[0] <= self.xLim[1] and self.yLim[0] <= xy_point[1] <= self.yLim[1]

    @staticmethod
    def average_points(xy_list):

        N = len(xy_list)
        xac = 0.0
        yac = 0.0

        for x, y in xy_list:
            if x < 0 or y < 0:
                continue
            xac += x
            yac += y

        return xac / N, yac / N

    def process(self, w_loc):
        if not self.check_limits(w_loc):
            w_loc = self.last_point

        self.xyQueue[self.idx % self.nAvgPoints] = (w_loc[0], w_loc[1])

        if self.xyAverage:
            w_loc = np.array(Robustify.average_points(self.xyQueue))

        self.last_point = w_loc
        self.idx += 1

        return w_loc


if __name__ == "__main__":

    # initialize
    data_dir = os.getenv('DATA_PATH')
    if data_dir is None:
        data_dir = '.'
    img_ext = '.png'

    calib_path = os.path.join(data_dir, 'images')
    param_file = os.path.join(data_dir, 'lab-cam-params.pkl')
    dt_patch_file = os.path.join(data_dir, 'ball_patch.png')

    # try to load camera parameters
    my_cam = LabCamera(np.array([640, 480]), 2)
    my_cam.load_params(param_file)

    dt_patch = cv2.imread(dt_patch_file, cv2.IMREAD_GRAYSCALE)
    obj_center = []
    px_sel_flag = False
    dt_patch_flag = dt_patch is not None and len(dt_patch) > 0

    # Open serial port (arduino)
    arduino = MyArduino('/dev/ttyACM0', 9600)

    # KLT optical flow tracker
    trackerKLT = TrackerKLT()

    # Limits and Averaging
    rbst = Robustify()

    # Main Loop
    while True:  # int(time.time() - start_time) < capture_duration:
        try:
            ret, frame = my_cam.get_next()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret:
                img_show = np.copy(frame)

                # If camera is not calibrated, calibrate the camera
                if not my_cam.calibrated:
                    res = my_cam.calibrate(calib_path)
                    if res:
                        my_cam.save_params(param_file)
                else:
                    # Else, track object
                    # px_loc = track_klt_of(trackerKLT, gray)
                    # if len(px_loc) <= 0:
                    px_loc, dt_patch, obj_center = detect_object(frame, dt_patch, obj_center)

                    if not dt_patch_flag:
                        cv2.imwrite(dt_patch_file, dt_patch)

                    w_loc = my_cam.unproject_homo(px_loc)

                    w_loc = rbst.process(w_loc)

                    img_txt = f'(%.2f, %.2f)' % (w_loc[0], w_loc[1])
                    # NOTE: first space is essential for Arduino to correctly parse the first number
                    loc_txt = f' %.2f %.2f' % (w_loc[0], w_loc[1])
                    cv2.putText(img_show, img_txt, np.int32(px_loc), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0))
                    cv2.drawMarker(img_show, np.int32(px_loc), (0, 0, 255), cv2.MARKER_CROSS, 2, 2)

                    # todo: Publish a pose message to the listeners
                    msg_ret = arduino.transmit(loc_txt)
                    # print(msg_ret)

                cv2.imshow("USB_CAM", img_show)

                if not px_sel_flag:
                    cv2.setMouseCallback('USB_CAM', mouse_callback)
                    px_sel_flag = True

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        except RuntimeError:
            break

    arduino.close()
    my_cam.close()
    cv2.destroyAllWindows()
