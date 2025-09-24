# detector.py
from ultralytics import YOLO
import numpy as np


class Detector:
def __init__(self, model_path='yolov8n.pt', conf=0.35, imgsz=640, device='cpu'):
self.model = YOLO(model_path)
self.conf = conf
self.imgsz = imgsz
# ultralytics picks device automatically; you can pass device arg in predict() if needed


def detect(self, frame):
# frame: BGR numpy array (OpenCV)
# returns list of [x1,y1,x2,y2,score]
results = self.model(frame, imgsz=self.imgsz, conf=self.conf, classes=[0])
out = []
# results may contain multiple result objects depending on batch; handle generically
for r in results:
boxes = r.boxes
if boxes is None:
continue
# boxes.xyxy, boxes.conf
xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else boxes.data.cpu().numpy()
confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
for i, b in enumerate(xyxy):
x1, y1, x2, y2 = map(int, b[:4])
score = float(b[4]) if b.shape[0] > 4 else (float(confs[i]) if confs is not None else 1.0)
out.append([x1, y1, x2, y2, score])
return out