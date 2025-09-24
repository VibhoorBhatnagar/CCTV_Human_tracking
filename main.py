# main.py
import cv2
import yaml
import redis
import json
import threading
import numpy as np
from detector import Detector
from tracker import Tracker
from reid import ReID

def process_camera(camera_config, detector, tracker, reid, redis_client):
    """Process a single camera stream, detect, track, and emit events to Redis."""
    camera_id = camera_config['id']
    rtsp_url = camera_config['rtsp_url']
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream {rtsp_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of stream or error reading frame from {rtsp_url}")
            break

        # Detect people
        detections = detector.detect(frame)

        # Update tracker
        tracks = tracker.update(detections)

        # Process tracks and emit events
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            embedding = reid.extract_embedding(crop)
            event = {
                'camera_id': camera_id,
                'local_id': track_id,
                'timestamp': cv2.getTickCount() / cv2.getTickFrequency(),
                'bbox': [x1, y1, x2, y2],
                'embedding': embedding.tolist()
            }
            redis_client.publish('tracking_events', json.dumps(event))

            # Draw bbox and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display frame
        cv2.imshow(f"Camera {camera_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {camera_id}")

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    detector = Detector(
        model_path=config['detector']['model_path'],
        conf_threshold=config['detector']['confidence_threshold'],
        person_class_id=config['detector']['person_class_id']
    )
    tracker = Tracker(
        max_age=config['tracker']['max_age'],
        min_hits=config['tracker']['min_hits'],
        iou_threshold=config['tracker']['iou_threshold']
    )
    reid = ReID(model_name=config['reid']['model_name'])
    redis_client = redis.Redis(
        host=config['redis']['host'],
        port=config['redis']['port'],
        db=config['redis']['db']
    )

    # Start a thread for each camera
    threads = []
    for camera_config in config['cameras']:
        thread = threading.Thread(
            target=process_camera,
            args=(camera_config, detector, tracker, reid, redis_client),
            daemon=True
        )
        threads.append(thread)
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
    redis_client.close()

if __name__ == "__main__":
    main()