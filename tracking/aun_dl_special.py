import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from aun_obj_tracking import Tracker


class TrackerYOLO(Tracker):
    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov8n.pt')

    def annotate_frame(self, res, frame):
        img = None
        for r in res:
            annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
            img = annotator.result()
        return img

    def init(self, frame, bbox):
        res = self.model.predict(source=frame, show=False, save=False, conf=0.5)
        # Tracking with default tracker
        # res = self.model.track(source=frame, show=False, save=False, conf=0.5)
        img = self.annotate_frame(res, frame)
        return res, img

    def update(self, frame):
        res = self.model.predict(source=frame, show=False, save=False, conf=0.5)
        # res = self.model.track(source=frame, show=False, save=False, conf=0.5)
        img = self.annotate_frame(res, frame)
        return res, img


if __name__ == "__main__":

    data_dir = os.getenv('DATA_PATH')
    img_file = os.path.join(data_dir, '21420839085_e2fa3bffa8_c.jpg')
    video_file = os.path.join(data_dir, 'video.mp4')

    tracker = TrackerYOLO()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            if not tracker.initialized:
                res, img_show = tracker.init(frame, None)
                tracker.initialized = True
            else:
                res, img_show = tracker.update(frame)

            if img_show is not None:
                cv2.imshow("Image", img_show)
            else:
                cv2.imshow("Image", frame)
            print(res)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
