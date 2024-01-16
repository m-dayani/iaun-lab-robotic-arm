import os
import cv2
import glob
import numpy as np
# import matplotlib.pyplot as plt

from aun_imp_basics import calc_object_center

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

# Some methods to locate an object in the image:
#   1. Template matching/registration
#   2. Feature extraction/matching (e.g. ORB features)
#   3. Deep learning


def get_loc_from_bbox(bbox):
    return np.array([bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5])


class Tracker:
    def __init__(self):
        # print('base tracker')
        self.last_point = [-1, -1]
        self.initialized = False
        self.patch = []

    def inject_point(self, point):
        # print('base inject point ' + str(self))
        self.last_point = point

    @staticmethod
    def detect_roi(frame, roi):
        # print('base detect roi ' + str(self))
        if roi is None or len(roi) <= 0:
            roi = cv2.selectROI(frame, False)
        return roi

    def init(self, frame, bbox):
        print('base init ' + str(self))

    def update(self, frame):
        print('base update ' + str(self))

    def get_patch(self):
        # print('base get patch: ' + str(self))
        return self.patch

    def save_patch(self, file_name):
        cv2.imwrite(file_name, self.patch)


class TrackerLK(Tracker):

    def __init__(self):
        super().__init__()
        print('Locus-Kenade Tracker')
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0 = None
        self.p1 = None
        self.old_gray = None
        self.is_initialized = False

    def init(self, frame, mask=None):
        self.p0 = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)
        self.old_gray = frame.copy()
        self.is_initialized = True

    def update(self, frame):
        self.p1, self.st, self.err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame, np.float32(self.p0),
                                                              None, **self.lk_params)
        self.old_gray = frame.copy()
        self.p0 = self.p1

    def updatePoints(self, pt):
        self.p0 = pt


class TrackerCV(Tracker):
    def __init__(self, tracker_idx=3):
        super().__init__()
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
        tracker_type = self.tracker_types[tracker_idx]
        self.tracker_type = tracker_type

        tracker = None
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                tracker = cv2.legacy.TrackerBoosting_create()
            if tracker_type == 'MIL':
                tracker = cv2.legacy.TrackerMIL_create()
            if tracker_type == 'KCF':
                tracker = cv2.legacy.TrackerKCF_create()
            if tracker_type == 'TLD':
                tracker = cv2.legacy.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                tracker = cv2.legacy.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
            if tracker_type == "CSRT":
                tracker = cv2.legacy.TrackerCSRT_create()

        self.tracker = tracker

    def init(self, frame, bbox):
        if self.tracker is not None:
            bbox = Tracker.detect_roi(frame, bbox)
            ok = self.tracker.init(frame, bbox)
            self.initialized = ok
            px_loc = get_loc_from_bbox(bbox)
            self.last_point = px_loc
            return ok, px_loc
        return False

    def update(self, frame):
        if self.tracker is not None:
            ok, bbox = self.tracker.update(frame)
            px_loc = get_loc_from_bbox(bbox)
            return ok, px_loc
        return False, self.last_point


class TrackerTM(Tracker):
    def __init__(self, patch_path=''):
        super().__init__()
        # print('template matching')
        self.patch = None
        self.set_patch(patch_path)

    def set_patch(self, patch_path):
        self.patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)

    def init(self, frame, bbox):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = Tracker.detect_roi(frame, bbox)
        self.patch = gray[y:h, x:w]

        self.last_point = calc_object_center(self.patch)

        return True, self.last_point

    def update(self, frame):

        # Template Matching (find ROI)
        tm_res = cv2.matchTemplate(frame, self.patch, cv2.TM_SQDIFF_NORMED)
        _, _, tm_loc, _ = cv2.minMaxLoc(tm_res)

        return True, np.array([tm_loc[0], tm_loc[1], 30, 30])


class TrackerMS(Tracker):
    def __init__(self, mode='MEAN_SHIFT'):
        super().__init__()
        # print("mean shift")
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.roi_hist = []
        self.track_window = (0, 0, 30, 30)
        self.mode = mode

    def init(self, frame, bbox):

        bbox = Tracker.detect_roi(frame, bbox)

        (x, y, w, h) = bbox
        track_window = bbox
        roi = frame[y:y + h, x:x + w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        # plt.plot(roi_hist)

        self.roi_hist = roi_hist
        self.track_window = track_window
        self.initialized = True

        return True, np.array([(x+w)*0.5, (y+h)*0.5])

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply Mean-Shift or CAM-Shift to get the new location

        if self.mode == 'CAM_SHIFT':
            ret, track_window = cv2.CamShift(dst, self.track_window, self.term_crit)
        else:
            ret, track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

        self.track_window = track_window

        return ret, track_window


def track_klt_of(trackerKLT, frame):

    global point_updated
    global px_loc

    if point_updated:
        point_updated = False
        mask = []   # get_mask(px_loc, frame.shape[:2])
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


def test_klt_of(base_dir):
    video_file = os.path.join(base_dir, '..', 'slow_traffic_small.mp4')
    video_file1 = os.path.join(base_dir, 'obj_detection', 'images', 'output.avi')
    video_file2 = os.path.join(base_dir, '..', '20231103_084334.mp4')

    cap = cv2.VideoCapture(video_file1)

    if not cap.isOpened():
        print("Error opening video file")
        return

    kltTracker = TrackerLK()

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    kltTracker.initialize(old_gray)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kltTracker.track(frame_gray)

        # Select good points
        if kltTracker.p1 is not None:
            st = kltTracker.st
            good_new = kltTracker.p1[st == 1]
            good_old = kltTracker.p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update previous points
        kltTracker.updatePoints(good_new.reshape(-1, 1, 2))

    cap.release()
    cv2.destroyAllWindows()


def test_tracker_cv(base_dir):
    tracker = TrackerCV()
    tracker_type = tracker.tracker_type

    images_path = os.path.join(base_dir, 'obj_det', 'obj_lid')
    images = glob.glob(images_path + '/image*.png')

    if len(images) <= 0:
        print('No images in path: ' + images_path)
        exit(1)

    # Read first frame.
    frame = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    if not ok:
        print('Error initializing tracker')
        exit(1)

    for img_file in images:
        # Read a new frame
        # ok, frame = video.read()
        frame = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        # time.sleep(0.033)


def test_mean_shift(base_dir):
    input_path = os.path.join(base_dir, "usain_bolt.mp4")
    output_path = os.path.join(base_dir, 'output.avi')

    # Creating a VidoCapture and VideoWriter object
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_path, fourcc, 29, (1280, 720))

    body_rects = (620, 310, 40, 40)

    tracker = TrackerMS()
    tracker.init(frame, body_rects)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ret, track_window = tracker.update(frame)

            # Draw updated rectangle
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            out.write(frame)
        else:
            break

    out.release()
    cap.release()


def test_cam_shift(base_dir):
    input_path = os.path.join(base_dir, "usain_bolt.mp4")
    output_path = os.path.join(base_dir, 'output.avi')

    # Creating a VidoCapture and VideoWriter object
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_path, fourcc, 25, (1280, 720))

    body_rects = (500, 300, 30, 60)
    tracker = TrackerMS('CAM_SHIFT')
    tracker.init(frame, body_rects)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ret, track_window = tracker.update(frame)

            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            out.write(frame)
        else:
            break

    out.release()
    cap.release()


if __name__ == '__main__':
    data_dir = os.getenv('DATA_DIR')

    # test_tracker_cv(os.path.join(data_dir, 'aun-lab'))
    test_mean_shift(data_dir)
    # test_cam_shift(data_dir)

