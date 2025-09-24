import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, conf_threshold, person_class_id):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

    def detect(self, frame):
        """Run YOLOv8 detection on a frame, return person bounding boxes."""
        results = self.model(frame, classes=[self.person_class_id], conf=self.conf_threshold)
        boxes = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()
                boxes.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
        return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 5), dtype=np.float32)