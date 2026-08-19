"""Point-to-box synthesis: a centroid plus a fixed `box_size` becomes an
`xyxy` box (`r = box_size / 2`, `xyxy = [x-r, y-r, x+r, y+r]`).

This is detection-side logic, not tracking-side -- it consumes `box_size`
(already a detection-side parameter, see `scale_derivation.py`'s docstring
for the `box_size`/`nms_distance`/`tile_size` vs. `search_range`/`diameter`/
`memory` split) to produce a raw detection box *before* any tracker sees it.
It therefore lives here in `detectors-common`, not in `trackers-common` --
adding it there would create a new `detectors-common -> trackers-common`
dependency edge that doesn't exist today and contradicts both packages'
"independent siblings" design.

Extracted from two previously independent, near-identical implementations:
`detectors_common.lodestar_loader.detect_lodestar` and
`verification/benchmark.py`'s `detect_trackpy`.
"""

import numpy as np


def points_to_xyxy(centers, box_size):
    """Convert `(x, y)` centroids into `xyxy` boxes of a fixed `box_size`.

    `centers` is array-like of shape `(N, 2)` (or anything `np.asarray`
    accepts in that shape), each row an `(x, y)` centroid. Returns an
    `(N, 4)` `float32` array of `[x1, y1, x2, y2]` rows in the same order as
    the input, with `x1 = x - r`, `y1 = y - r`, `x2 = x + r`, `y2 = y + r`,
    and `r = box_size / 2`. An empty `centers` input returns an empty
    `(0, 4)` array rather than raising.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    r = box_size / 2
    x = centers[:, 0]
    y = centers[:, 1]
    return np.stack([x - r, y - r, x + r, y + r], axis=1).astype(np.float32)
