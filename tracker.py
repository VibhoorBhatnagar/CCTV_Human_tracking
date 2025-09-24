import numpy as np
from scipy.spatial.distance import cdist

def iou_batch(bb_test, bb_gt):
    """Compute IoU between two sets of bounding boxes."""
    bb_test = bb_test[:, None]
    bb_gt = bb_gt[None, :]
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

class Tracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections):
        """Update tracks with new detections, return active tracks."""
        self.frame_count += 1
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections)

        # Update matched tracks
        for t, d in matched:
            self.tracks[t]['bbox'] = detections[d, :4]
            self.tracks[t]['age'] = 0
            self.tracks[t]['hits'] += 1

        # Create new tracks for unmatched detections
        for d in unmatched_dets:
            self.tracks.append({
                'id': self.next_id,
                'bbox': detections[d, :4],
                'age': 0,
                'hits': 1
            })
            self.next_id += 1

        # Increment age of unmatched tracks
        for t in unmatched_trks:
            self.tracks[t]['age'] += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        # Return active tracks with enough hits
        return [(t['id'], t['bbox']) for t in self.tracks if t['hits'] >= self.min_hits]

    def _associate_detections_to_trackers(self, detections):
        """Associate detections to existing tracks using IoU."""
        if not self.tracks or detections.shape[0] == 0:
            return np.empty((0, 2), dtype=int), np.arange(detections.shape[0]), np.arange(len(self.tracks))

        track_boxes = np.array([t['bbox'] for t in self.tracks])
        iou_matrix = iou_batch(detections[:, :4], track_boxes)

        if iou_matrix.size > 0:
            matches = []
            unmatched_dets = list(range(detections.shape[0]))
            unmatched_trks = list(range(len(self.tracks)))

            for d, det in enumerate(iou_matrix):
                for t, trk in enumerate(det):
                    if trk > self.iou_threshold:
                        matches.append((t, d))
                        if d in unmatched_dets:
                            unmatched_dets.remove(d)
                        if t in unmatched_trks:
                            unmatched_trks.remove(t)

            return np.array(matches, dtype=int), np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)
        else:
            return np.empty((0, 2), dtype=int), np.arange(detections.shape[0]), np.arange(len(self.tracks))
