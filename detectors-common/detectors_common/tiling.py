"""Tiling-bounds arithmetic + RF-DETR NMS merge, shared by
particle-tracking/track.py and verification/benchmark.py (both the RF-DETR and
YOLO12n inference paths). Uses the bounds-guarded tile_starts (the
`if length <= tile_size: return [0]` guard) — verification's prior standalone
copy was missing this guard, which produced a negative tile-start index
(silently wrapping into a slice from the end of the array) whenever a frame
was tiling-eligible overall but fit within tile_size in exactly one
dimension.
"""


def tile_starts(length, tile_size, stride):
    """Compute tile start offsets covering `length` with `tile_size`-wide
    tiles spaced `stride` apart, always including a final tile flush with the
    end of the axis."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size, stride))
    starts.append(length - tile_size)
    return starts


def detect_with_tiling(model, frame, threshold, tile_size, overlap, nms_threshold):
    """Run RF-DETR on overlapping tiles and merge detections with NMS.

    Adapts to any frame size at runtime. Falls back to a single predict() call
    when the frame fits within tile_size in both dimensions.
    """
    import numpy as np
    import supervision as sv

    H, W = frame.shape[:2]

    if H <= tile_size and W <= tile_size:
        return model.predict(frame, threshold=threshold)

    stride = tile_size - overlap

    all_xyxy, all_conf, all_class_id = [], [], []
    for y0 in tile_starts(H, tile_size, stride):
        for x0 in tile_starts(W, tile_size, stride):
            tile = frame[y0 : y0 + tile_size, x0 : x0 + tile_size]
            dets = model.predict(tile, threshold=threshold)
            if len(dets) > 0:
                boxes = dets.xyxy.copy()
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                all_xyxy.append(boxes)
                all_conf.append(dets.confidence)
                all_class_id.append(dets.class_id)

    if not all_xyxy:
        return sv.Detections.empty()

    merged = sv.Detections(
        xyxy=np.concatenate(all_xyxy),
        confidence=np.concatenate(all_conf),
        class_id=np.concatenate(all_class_id),
    )
    return merged.with_nms(threshold=nms_threshold)
