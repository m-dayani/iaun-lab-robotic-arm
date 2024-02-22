import os
import sys
import time

import numpy as np
import cv2
import threading

# import concurrent.futures

sys.path.append('..')
from my_serial.aun_arduino import MyArduino
from tracking.aun_obj_tracking import TrackerCV, TrackerMS
from tracking.aun_dl_special import TrackerYOLO
from cam_calib.aun_cam_model import LabCamera
from cam_calib.aun_cam_calib import CamCalib

px_loc = []
w_loc = []
mouse_cb_flag = False
point_updated = False
proc_finished = False
ret = False
frame = None
img_show = None
my_cam = None


# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)


def mouse_callback(event, x, y, flags, params):
    global px_loc
    global point_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        px_loc = np.array([x, y])
        # print(cam.unproject_homo(px_loc))
        point_updated = True


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


class CamCapture:
    def __init__(self, fps):
        self.is_running = False
        self.fps = fps
        self.T = 1.0 / float(fps)

    def run(self):

        global proc_finished
        global ret
        global frame
        global img_show
        global my_cam

        self.is_running = True
        while self.is_running:

            t0 = time.perf_counter()

            try:
                if my_cam is not None:
                    ret, frame = my_cam.get_next()
                    if ret:
                        img_show = np.copy(frame)
            except RuntimeError:
                break

            if proc_finished:
                self.is_running = False
                break

            t1 = time.perf_counter()
            dt = t1 - t0
            if dt < self.T:
                time.sleep(self.T - dt)


class ArduinoThread(MyArduino):
    def __init__(self, port, bd):
        super().__init__(port, bd)
        self.is_running = False

    def run(self):

        global proc_finished
        global w_loc
        self.is_running = True

        while self.is_running:

            if len(w_loc) >= 2:
                # NOTE: first space is essential for Arduino to correctly parse the first number
                loc_txt = f' %.2f %.2f' % (w_loc[0], w_loc[1])
                # todo: Publish a pose message to the listeners
                msg_ret = self.transmit(loc_txt)

            if proc_finished:
                self.is_running = False
                break


class VizThread:
    def __init__(self):
        self.is_running = False

    def run(self):

        global proc_finished
        global mouse_cb_flag
        global img_show
        global w_loc
        global px_loc

        self.is_running = True
        while self.is_running:

            if img_show is None:
                continue

            if len(w_loc) <= 0:
                img_txt = 'Object not detected'
            else:
                img_txt = f'(%.2f, %.2f)' % (w_loc[0], w_loc[1])

            loc_int = (0, 0)
            if len(px_loc) >= 2:
                loc_int = np.int32(px_loc[0:2])

            cv2.putText(img_show, img_txt, loc_int, cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0))
            cv2.drawMarker(img_show, loc_int, (0, 0, 255), cv2.MARKER_CROSS, 8, 2)

            cv2.imshow("USB_CAM", img_show)
            if not mouse_cb_flag and cv2.getWindowProperty('USB_CAM', 0) >= 0:
                cv2.setMouseCallback('USB_CAM', mouse_callback)
                mouse_cb_flag = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                proc_finished = True
                self.is_running = False
                break


class TrackingThread:
    def __init__(self, calib_path):

        self.is_running = False
        self.initialized = False
        # Load calibrator
        self.myCalib = CamCalib(calib_path)
        # Load Tracker
        self.tracker = TrackerCV()
        # Limits and averaging
        self.rbst = Robustify()
        self.ini_delay = 50

    def run(self):

        global ret
        global frame
        global proc_finished
        global my_cam
        global w_loc
        global px_loc

        max_cnt = 100
        cnt = 0
        t_cols = 2
        time_stat = np.zeros((max_cnt, t_cols))

        self.is_running = True
        while self.is_running:
            if ret:
                t0 = time.perf_counter()

                if my_cam.calibrated and self.tracker.initialized:
                    ret1, px_loc = self.tracker.update(frame)

                t1 = time.perf_counter()

                w_loc = my_cam.unproject_homo(px_loc)
                # w_loc = self.rbst.process(w_loc)

                t2 = time.perf_counter()

                time_stat[(cnt % max_cnt), :] = (t1 - t0, t2 - t1)
                if (cnt % max_cnt) == 0:
                    print(np.mean(time_stat, axis=0))
                    if cnt > 0:
                        cnt = 0
                cnt += 1

            if proc_finished:
                self.is_running = False
                break

        # tracker.save_patch(dt_patch_file)

    def init(self):

        global my_cam
        global px_loc
        global mouse_cb_flag

        delay = 0

        while not self.initialized:
            try:
                ret, frame = my_cam.get_next()
                if ret:
                    img_show = np.copy(frame)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # give time to open and initialize the camera
                    if delay > self.ini_delay:
                        if not my_cam.calibrated:
                            # Camera Calibration
                            res = self.myCalib.recalib(my_cam, gray, ws=2.5)
                            if res:
                                my_cam.save_params(param_file)
                            else:
                                # show a message
                                print('ERROR: Camera is not calibrated')
                        else:
                            if not self.tracker.initialized:
                                ret1, px_loc = self.tracker.init(frame, [])
                                if self.tracker.initialized:
                                    self.initialized = True

                    delay += 1
                    cv2.imshow("USB_CAM", img_show)
                    if not mouse_cb_flag:
                        cv2.setMouseCallback('USB_CAM', mouse_callback)
                        mouse_cb_flag = True

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except RuntimeError:
                break


if __name__ == "__main__":

    # initialize
    data_dir = os.getenv('DATA_PATH')
    if data_dir is None:
        data_dir = '.'
    img_ext = '.png'

    calib_path = os.path.join(data_dir, 'calib', 'images')
    param_file = os.path.join(data_dir, 'calib', 'params', 'lab-cam-params.pkl')
    dt_patch_file = os.path.join(data_dir, 'obj_det', 'obj_lid', 'ball_patch.png')
    fps = 24

    # Load camera and parameters
    my_cam = LabCamera(2, (640, 480))
    my_cam.load_params(param_file)

    cam_capture = CamCapture(fps)
    cam_view = VizThread()

    # Open serial port (arduino)
    arduino = ArduinoThread('/dev/ttyACM0', 9600)

    # Computation thread:
    tracking = TrackingThread(calib_path)

    # Must initialize before tracking, because
    # it seems you can't open OpenCV windows from different threads
    tracking.init()

    # Spawn Threads
    thCamCapture = threading.Thread(target=cam_capture.run, args=())
    # thImgView = threading.Thread(target=cam_view.run, args=())
    thTracking = threading.Thread(target=tracking.run, args=())
    thArduino = threading.Thread(target=arduino.run, args=())

    thCamCapture.start()
    # thImgView.start()
    thTracking.start()
    thArduino.start()

    # Only viewer runs from the main thread
    cam_view.run()

    thCamCapture.join()
    # thImgView.join()
    thTracking.join()
    thArduino.join()

    arduino.close()
    my_cam.close()
    cv2.destroyAllWindows()
