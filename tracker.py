# tracker.py
t.predict()
trks.append(t.to_bbox())
dets = [d[:4] for d in detections]


if len(trks) == 0:
# create new tracks for all detections
for d in dets:
self.tracks.append(Track(d, self._next_id, max_age=self.max_age))
self._next_id += 1
else:
iou_mat = np.zeros((len(trks), len(dets)), dtype=np.float32)
for t_idx, trk in enumerate(trks):
for d_idx, det in enumerate(dets):
iou_mat[t_idx, d_idx] = iou(trk, det)
# Hungarian on -iou (maximize)
if iou_mat.size > 0:
row_ind, col_ind = linear_sum_assignment(-iou_mat)
else:
row_ind, col_ind = np.array([]), np.array([])


matched_trks = set()
matched_dets = set()
for r, c in zip(row_ind, col_ind):
if iou_mat[r, c] < self.iou_threshold:
continue
self.tracks[r].update(dets[c])
matched_trks.add(r)
matched_dets.add(c)


# unmatched detections -> new tracks
for d_idx, det in enumerate(dets):
if d_idx not in matched_dets:
self.tracks.append(Track(det, self._next_id, max_age=self.max_age))
self._next_id += 1


# remove dead tracks
alive_tracks = []
outputs = []
for t in self.tracks:
if t.time_since_update < self.max_age and (t.hits >= self.min_hits or t.time_since_update==0):
outputs.append({'track_id': t.track_id, 'bbox': t.to_bbox()})
alive_tracks.append(t)
self.tracks = alive_tracks
return outputs


# Helper


def iou(bb_test, bb_gt):
xx1 = max(bb_test[0], bb_gt[0])
yy1 = max(bb_test[1], bb_gt[1])
xx2 = min(bb_test[2], bb_gt[2])
yy2 = min(bb_test[3], bb_gt[3])
w = max(0., xx2-xx1)
h = max(0., yy2-yy1)
inter = w*h
area1 = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
area2 = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
o = inter/(area1+area2-inter+1e-6)
return o