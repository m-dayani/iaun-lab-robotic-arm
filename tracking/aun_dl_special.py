import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from aun_obj_tracking import Tracker


class TrackerYOLO(Tracker):
    def __init__(self):
        super().__init__()
        self.model = YOLO('../models/m_omoumi_cap.pt')

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

    def get_bboxes(self, res):
        bboxes = dict()
        for r in res:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                if self.model.names[int(c)] not in bboxes.keys():
                    bboxes[self.model.names[int(c)]] = []
                bboxes[self.model.names[int(c)]].append(b.numpy())
        return bboxes

    def get_last_point(self, list_bbox):
        if 'Cap' in list_bbox.keys():
            bbox = list_bbox['Cap'][0]
            self.last_point = [bbox[0] * 0.5 + bbox[2] * 0.5, bbox[1] * 0.5 + bbox[3] * 0.5]
        return self.last_point

    def init(self, frame, bbox):
        res = self.model.predict(source=frame, show=False, save=False, conf=0.5)
        # Tracking with default tracker
        # res = self.model.track(source=frame, show=False, save=False, conf=0.5)
        list_bbox = self.get_bboxes(res)
        return self.get_last_point(list_bbox)

    def update(self, frame):
        res = self.model.predict(source=frame, show=False, save=False, conf=0.5)
        # res = self.model.track(source=frame, show=False, save=False, conf=0.5)
        list_bbox = self.get_bboxes(res)
        return self.get_last_point(list_bbox)


if __name__ == "__main__":

    data_dir = os.getenv('DATA_PATH')
    img_file = os.path.join(data_dir, '21420839085_e2fa3bffa8_c.jpg')
    video_file = os.path.join(data_dir, 'video.mp4')

    tracker = TrackerYOLO()
    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()

        if ret:
            if not tracker.initialized:
                res, img_show = tracker.init(frame, None)
                tracker.initialized = True
            else:
                res = tracker.update(frame)

            img_show = None
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
