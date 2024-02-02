import os
import cv2
from ultralytics import YOLO

from aun_obj_tracking import Tracker


class TrackerYOLO(Tracker):
    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov8n.pt')

    def init(self, frame, bbox):
        res = self.model.predict(source=frame, show=True, save=False, conf=0.5)
        # print(res)

    def update(self, frame):
        res = self.model.predict(source=frame, show=True, save=False, conf=0.5)


if __name__ == "__main__":

    data_dir = os.getenv('DATA_PATH')
    img_file = os.path.join(data_dir, '21420839085_e2fa3bffa8_c.jpg')
    video_file = os.path.join(data_dir, 'video.mp4')

    image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

    tracker = TrackerYOLO()
    tracker.init(image, [])

    for i in range(10):
        tracker.update(image)

