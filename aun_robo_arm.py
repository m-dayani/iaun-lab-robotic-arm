import os
import sys
import numpy as np
import cv2

from aun_arduino import MyArduino
from aun_obj_tracking import TrackerCV, TrackerMS
from aun_cam_model import LabCamera
from aun_cam_calib import CamCalib


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

    calib_path = os.path.join(data_dir, 'calib', 'images')
    param_file = os.path.join(data_dir, 'calib', 'params', 'lab-cam-params.pkl')
    dt_patch_file = os.path.join(data_dir, 'obj_det', 'obj_lid', 'ball_patch.png')

    # Load camera and parameters
    my_cam = LabCamera(2, (640, 480))
    my_cam.load_params(param_file)

    # Load calibrator
    myCalib = CamCalib(calib_path)

    # Open serial port (arduino)
    arduino = MyArduino('/dev/ttyACM0', 9600)

    # Load Tracker
    tracker = TrackerMS()

    # Limits and averaging
    rbst = Robustify()

    mouse_cb_flag = False

    # Main Loop
    while True:
        try:
            ret, frame = my_cam.get_next()
            if ret:
                img_show = np.copy(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if not my_cam.calibrated:
                    # Camera Calibration
                    res = myCalib.recalib(my_cam, gray, ws=2.5)
                    if res:
                        my_cam.save_params(param_file)
                    else:
                        # show a message
                        print('ERROR: Camera is not calibrated')
                else:
                    if not tracker.initialized:
                        ret, px_loc = tracker.init(frame, [])
                        # px_loc = tracker.last_point
                    else:
                        ret, px_loc = tracker.update(frame)

                    w_loc = my_cam.unproject_homo(px_loc)
                    w_loc = rbst.process(w_loc)

                    img_txt = f'(%.2f, %.2f)' % (w_loc[0], w_loc[1])
                    # NOTE: first space is essential for Arduino to correctly parse the first number
                    loc_txt = f' %.2f %.2f' % (w_loc[0], w_loc[1])
                    loc_int = np.int32(px_loc[0:2])
                    cv2.putText(img_show, img_txt, loc_int, cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0))
                    cv2.drawMarker(img_show, loc_int, (0, 0, 255), cv2.MARKER_CROSS, 8, 2)

                    # todo: Publish a pose message to the listeners
                    msg_ret = arduino.transmit(loc_txt)

                cv2.imshow("USB_CAM", img_show)
                if not mouse_cb_flag:
                    cv2.setMouseCallback('USB_CAM', mouse_callback)
                    mouse_cb_flag = True

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        except RuntimeError:
            break

    # tracker.save_patch(dt_patch_file)

    arduino.close()
    my_cam.close()
    cv2.destroyAllWindows()
