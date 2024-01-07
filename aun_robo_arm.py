import os
import sys
import numpy as np
import cv2
import time
import serial
import matplotlib.pyplot as plt

from hough_transform import my_detect_lines
from aun_pattern_detection import detect_black_square, calc_object_center, TrackerKLT
from scipy.spatial.transform import Rotation as scipyR

from squares import find_squares

sys.path.append('./calib')
from cameraCalibration import recalib_camera
from cameraModels import LabCamera


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


def calibrate_camera(cam, frame, patch):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Detect the reference square
    if len(patch) <= 0:
        # prompt user to select an area
        patch = cv2.selectROI(gray, False)

    squares = detect_black_square(gray, patch)

    if len(squares) != 4:
        return

    # 2. calculate homography: r0, r1, t, s
    print('calibrated')


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


def calc_obj_pose(px_loc, cam):
    print('calculate camera pose')


def calc_obj_speed(last_loc, curr_loc, dt, cam):
    print('calculate speed')


def open_arduino(port, bd):
    arduino = serial.Serial(port=port, baudrate=bd, timeout=0.1)
    return arduino


def check_limits(xy_point, lim_x, lim_y):
    return lim_x[0] <= xy_point[0] <= lim_x[1] and lim_y[0] <= xy_point[1] <= lim_y[1]


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


if __name__ == "__main__":

    # initialize
    capture_duration = 10
    cam_port = 2
    img_width = 640
    img_height = 480

    # global px_loc
    # global point_updated

    usb_cam = cv2.VideoCapture(cam_port)

    # usb_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('U', 'Y', 'V', 'Y'))
    usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

    if not usb_cam.isOpened():
        print("Error reading video file")

    img_width = int(usb_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(usb_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("image size: (%d x %d)" % (img_width, img_height))

    data_dir = os.getenv('DATA_PATH')
    if data_dir is None:
        data_dir = '.'
    img_ext = '.png'

    # also need a patch for template matching
    # tm_patch = cv2.imread(os.path.join(data_dir, 'patch0.png'), cv2.IMREAD_GRAYSCALE)

    calib_path = os.path.join(data_dir, 'images')

    i = 0
    start_time = time.time()

    cam = LabCamera()

    param_file = os.path.join(data_dir, 'lab-cam-params.pkl')
    # try to load camera parameters
    cam.load_params(param_file)

    dt_patch_file = os.path.join(data_dir, 'ball_patch.png')
    dt_patch = cv2.imread(dt_patch_file, cv2.IMREAD_GRAYSCALE)
    obj_center = []
    px_sel_flag = False
    dt_patch_flag = dt_patch is not None and len(dt_patch) > 0

    # Open serial port (arduino)
    arduino = None
    try:
        arduino = open_arduino('/dev/ttyACM0', 9600)
        time.sleep(2)
    except serial.SerialException:
        print("Error: cannot open Arduino")

    # Limits and Averaging
    xLim = [0.0, 200.0]
    yLim = [0.0, 200.0]
    limAction = 1   # output current values
    last_point = np.array([0.0, 0.0])

    nAvgPoints = 20
    xyQueue = [(-1.0, -1.0) for v in range(nAvgPoints)]
    xyAverage = True

    # KLT optical flow tracker
    trackerKLT = TrackerKLT()

    # Main Loop
    while True:  # int(time.time() - start_time) < capture_duration:
        try:
            ret, frame = usb_cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret:
                img_show = np.copy(frame)

                # If camera is not calibrated, calibrate the camera
                if not cam.calibrated:
                    # calibrate_camera(cam, frame, tm_patch)
                    res = recalib_camera(calib_path, frame, cam, ws=2.5)
                    if res:
                        cam.save_params(param_file)
                else:
                    # Else, track object
                    px_loc = track_klt_of(trackerKLT, gray)
                    if len(px_loc) <= 0:
                        px_loc, dt_patch, obj_center = detect_object(frame, dt_patch, obj_center)

                    if not dt_patch_flag:
                        cv2.imwrite(dt_patch_file, dt_patch)

                    w_loc = cam.unproject_homo(px_loc)

                    if not check_limits(w_loc, xLim, yLim):
                        w_loc = last_point

                    xyQueue[i % nAvgPoints] = (w_loc[0], w_loc[1])
                    i += 1

                    if xyAverage:
                        w_loc = np.array(average_points(xyQueue))

                    last_point = w_loc

                    img_txt = f'(%.2f, %.2f)' % (w_loc[0], w_loc[1])
                    # NOTE: first space is essential for Arduino to correctly parse the first number
                    loc_txt = f' %.2f %.2f' % (w_loc[0], w_loc[1])
                    cv2.putText(img_show, img_txt, np.int32(px_loc), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0))
                    cv2.drawMarker(img_show, np.int32(px_loc), (0, 0, 255), cv2.MARKER_CROSS, 2, 2)

                    # todo: Publish a pose message to the listeners
                    if arduino is not None:
                        arduino.write(bytes(loc_txt, 'utf-8'))
                        time.sleep(0.05)
                        msg = arduino.readline()
                        print(msg)

                # cv2.imwrite(os.path.join(data_dir, 'image' + str(i) + img_ext), frame)
                # i += 1

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

    if arduino is not None:
        arduino.close()

    usb_cam.release()
    cv2.destroyAllWindows()
