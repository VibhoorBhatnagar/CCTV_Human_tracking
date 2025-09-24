# main.py
import threading
import time
import json
import argparse
import yaml
import cv2
import numpy as np
import redis
from detector import Detector
from tracker import Sort
from reid import ReIDExtractor


# utility: crop and clamp


def crop_and_clamp(frame, bbox):
h, w = frame.shape[:2]
x1,y1,x2,y2 = bbox
x1 = max(0, x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
if x2<=x1 or y2<=y1:
return None
return frame[y1:y2, x1:x2]


class CameraWorker(threading.Thread):
def __init__(self, cam_conf, detector, tracker, reid, redis_client, cfg, events_file_handle=None):
threading.Thread.__init__(self)
self.cam_id = cam_conf['id']
self.url = cam_conf['url']
self.detector = detector
self.tracker = tracker
self.reid = reid
self.redis = redis_client
self.stream_key = cfg['redis']['stream_key']
self.device = cfg['runtime']['device']
self.stop_flag = False
self.events_file = events_file_handle
self.emb_every = cfg['reid']['embedding_every_n_frames']
self.frame_idx = 0
self.min_track_len = cfg['tracking']['min_track_len']


def run(self):
cap = cv2.VideoCapture(self.url)
if not cap.isOpened():
print(f"Failed to open camera {self.cam_id} url={self.url}")
return
print(f"Started camera {self.cam_id}")
while not self.stop_flag:
ret, frame = cap.read()
if not ret:
time.sleep(0.1)
continue