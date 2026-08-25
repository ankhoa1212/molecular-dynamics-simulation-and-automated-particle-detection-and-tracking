"""Composite the main-body qualitative tracking figure for wacv2027-paper.

Panel (a): RF-DETR detections (boxes only, no traces) on one real 5um frame,
cropped to a legible central region.
Panel (b): the existing supplementary fig11_tracking.png trajectory plot.

One-off script for this figure; not part of the production track.py pipeline.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import tifffile

FRAME_IDX = 6
RAW_TIFF = "data/raw/scratch/real_5um_trajectory_analysis/frames_000-150.tif"
TRACKS_CSV = "output/qualitative_fig_scratch/real_rfdetr_preview/frames_000-150/tracks.csv"
FIG11_PATH = "/home/ankhoa1212/git/wacv2027-paper/figures/fig11_tracking.png"
OUT_PATH = "output/qualitative_fig_scratch/fig_qualitative_tracking.png"

# Crop window (px) in the full 3200x2200 frame -- central region with dense,
# individually-legible boxes. Aspect matches fig11_tracking.png (1999x1316,
# ~1.519) so both panels render at the same height with no dead whitespace.
CROP = (550, 260, 1990, 1207)  # x0, y0, x1, y1


def load_raw_frame(idx):
    with tifffile.TiffFile(RAW_TIFF) as tf:
        frame = tf.asarray(key=idx)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    frame = frame.astype(np.float32)
    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
    return (frame * 255).astype(np.uint8)


def main():
    frame = load_raw_frame(FRAME_IDX)
    df = pd.read_csv(TRACKS_CSV)
    frame_df = df[df["frame"] == FRAME_IDX]

    x0, y0, x1, y1 = CROP
    crop = frame[y0:y1, x0:x1]

    im_b = plt.imread(FIG11_PATH)
    aspect_a = (x1 - x0) / (y1 - y0)
    aspect_b = im_b.shape[1] / im_b.shape[0]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.4),
        dpi=200,
        gridspec_kw={"width_ratios": [aspect_a, aspect_b]},
    )

    ax = axes[0]
    ax.imshow(crop)
    n_boxes = 0
    for _, row in frame_df.iterrows():
        cx, cy, w, h = row["x"], row["y"], row["w"], row["h"]
        bx0, by0 = cx - w / 2 - x0, cy - h / 2 - y0
        if bx0 < -w or by0 < -h or bx0 > (x1 - x0) or by0 > (y1 - y0):
            continue
        ax.add_patch(
            patches.Rectangle(
                (bx0, by0),
                w,
                h,
                linewidth=1.1,
                edgecolor="#e8388a",
                facecolor="none",
            )
        )
        n_boxes += 1
    ax.set_title("(a) RF-DETR detections (real footage)", fontsize=9)
    ax.set_axis_off()
    print(f"Panel (a): {n_boxes} boxes drawn in crop window")

    ax2 = axes[1]
    ax2.imshow(im_b)
    ax2.set_title("(b) trackpy trajectories (same clip)", fontsize=9)
    ax2.set_axis_off()

    plt.tight_layout(pad=0.6, w_pad=1.0)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
