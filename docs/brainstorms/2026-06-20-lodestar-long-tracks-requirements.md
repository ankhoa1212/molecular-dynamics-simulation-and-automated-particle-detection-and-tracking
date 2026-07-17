# Longer Particle Tracks from LodeSTAR Detection

**Date:** 2026-06-20  
**Status:** Ready for implementation

---

## Problem

LodeSTAR detection produces ~800 detections/frame (max 2557) for a scene with ~500+ real particles. The tracker cannot be fixed by raising the detection threshold — sensitivity is a hard requirement because brightness varies dramatically across frames, and dim frames need `threshold: 0.1` to detect real particles.

The result: 25,065 tracks with mean length 7 frames (median 5), max 141. Tracks fragment because:
1. Real particles occasionally fall below the detection threshold in dim frames, creating gaps
2. 300+ false-positive detections/frame compete for links, causing identity switches
3. Gap-closing and memory are tuned conservatively, so fragmented track pieces are discarded rather than stitched

---

## Core Insight

Real particles persist across hundreds of frames at slowly-varying positions (Brownian motion). Noise detections at `threshold=0.1` appear at random isolated locations and do not persist across frames. The solution is **temporal filtering**: tolerate missed detections and reconnect fragments, then discard anything that doesn't survive long enough to be a real particle.

---

## Requirements

### 1. Increase `memory` for longer gap tolerance

**Current:** `memory: 5` (track ends if particle missing >5 consecutive frames)  
**Target:** `memory: 15–20`

Microscopy videos have dim frames where real particles fall below detection threshold. A particle should be allowed to "disappear" for up to ~20 frames before its track is terminated. Noise detections don't persist, so raising memory does not extend noise track lifetime significantly.

### 2. Enable `bridge_gap` to reconnect track fragments

**Current:** `bridge_gap` not set (disabled)  
**Target:** `bridge_gap: 15–20`, `bridge_radius: 20`

After trackpy linking, `bridge_track_gaps` in `track.py` reconnects pairs of track endpoints that are spatially close and temporally close. This converts multiple short fragments from the same real particle into a single long track. `bridge_radius` should be approximately 2× `search_range` (current `search_range: 10.0`, so `bridge_radius: 20`).

### 3. Raise `stub_filter` after gap bridging

**Current:** `stub_filter: 3`  
**Target:** `stub_filter: 30–50`

After gap bridging, real particle tracks will span many more frames. Noise tracks — random, non-persistent — will remain short. Setting `stub_filter` to 30–50 frames eliminates noise while preserving real particle tracks. This reduces the 25K track count to a number closer to the actual particle count (~500).

### 4. Investigate and fix `detect_lodestar` sigma scaling (code change)

**File:** `particle-tracking/track.py`, `detect_lodestar` function (line ~167)

**Issue:** `det[2]` from LodeSTAR's `model.detect()` is the sigma (characteristic size) in normalized [0, 1] coordinates. The current code uses it as-is for bounding box radius:

```python
r = abs(det[2]) if len(det) >= 3 else box_size / 2
```

With mean sigma ≈ 0.0047, boxes are ~0.0047px radius — essentially point detections. This is cosmetically wrong (bounding boxes don't represent particle size) but doesn't affect centroid-based trackpy linking.

However, for NMS: all detections have nearly identical confidence values (~0.0047 ± small variation), making NMS ordering essentially random. The suppression itself (within `nms_distance`) still works correctly since it operates on distances.

**Fix:** Scale sigma to pixel coordinates using the frame dimensions:

```python
sigma = abs(det[2])
frame_scale = max(frame.shape[:2])
r = sigma * frame_scale if sigma < 1.0 else sigma
confidences.append(sigma)  # keep sigma as relative NMS ordering score
```

This gives ~9–15px radius boxes for 2um particles in a 2000px frame, which is physically plausible. Verify the output after the fix by checking that `mean_confidence` in `metrics.json` changes to a scaled value (e.g., ~0.005 × 2000 = ~10.0).

**Note:** If `det[2]` turns out to be in pixel coordinates already (not normalized), the `< 1.0` guard preserves the original behavior. Confirm by printing `det[:5]` for a sample frame during a test run.

---

## Recommended Config for `basic_lodestar_config.yaml`

```yaml
tracking:
  tracker: trackpy
  search_range: 10.0      # unchanged — 10px is appropriate for particle motion
  memory: 20              # was 5; allow 20-frame gaps before ending a track
  stub_filter: 40         # was 3; discard tracks shorter than 40 frames after gap bridging
  adaptive_stop: 1.0
  adaptive_step: 0.95
  bridge_gap: 15          # reconnect fragments separated by up to 15 frames
  bridge_radius: 20       # reconnect fragments within 20px (2× search_range)
```

Detection parameters (`threshold`, `alpha`, `nms_distance`) are **unchanged**.

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| `n_tracks` | 25,065 | 500–3,000 |
| `track_length_mean` | 6.99 frames | 30+ frames |
| `track_length_max` | 141 frames | 141+ frames (unchanged or better) |
| `detection_rate` | 1.0 | 1.0 (unchanged) |

---

## Scope Boundaries

**In scope:**
- `particle-tracking/basic_lodestar_config.yaml` parameter changes
- `particle-tracking/track.py` sigma scaling fix in `detect_lodestar`

**Out of scope:**
- Changing `threshold`, `alpha`, or `nms_distance` (sensitivity is a hard constraint)
- Retraining the LodeSTAR model
- Switching tracker (trackpy → ByteTrack)

---

## Outstanding Questions

- What is `search_range: 10.0` in physical units (nm/pixel × 10px)? If particles move faster than ~10px/frame, `search_range` may need to increase before gap bridging can help.
- Does `det[2]` from `deeplay.LodeSTAR.detect()` confirm to normalized [0,1] coordinates? A single print statement in a test run will confirm.
