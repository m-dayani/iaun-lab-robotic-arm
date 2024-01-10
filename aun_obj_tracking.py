import os
import cv2
import sys
import glob
import time
import sklearn

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')


class TrackerCV:
    def __init__(self, tracker_idx=3):
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
        if tracker is not None:
            ok = self.tracker.init(frame, bbox)
            return ok
        return False

    def update(self, frame):
        if tracker is not None:
            ok, bbox = self.tracker.update(frame)
            return ok, bbox
        return False, []


if __name__ == '__main__':

    tracker = TrackerCV()
    tracker_type = tracker.tracker_type

    data_dir = os.getenv('DATA_DIR')
    images_path = os.path.join(data_dir, 'obj_det', 'obj_lid')
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
