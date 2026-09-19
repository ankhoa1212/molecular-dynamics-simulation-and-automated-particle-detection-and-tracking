#!/usr/bin/env python3
"""
Preview script to visualize LodeSTAR data augmentations.
Use this to verify if Gaussian blur and Poisson noise levels are realistic.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import deeptrack as dt


def _pad_to_square(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop oversized axes then zero-pad to exactly size×size."""
    img_width, img_height = img.size
    if img_width > size or img_height > size:
        left = max((img_width - size) // 2, 0)
        top = max((img_height - size) // 2, 0)
        img = img.crop((left, top, left + min(img_width, size), top + min(img_height, size)))
        img_width, img_height = img.size
    pad_l = (size - img_width) // 2
    pad_t = (size - img_height) // 2
    return ImageOps.expand(
        img, border=(pad_l, pad_t, size - img_width - pad_l, size - img_height - pad_t), fill=0
    )


def _load_single_crop(fpath, crop_size=64):
    """Load, pad, and normalise a single crop image."""
    img = _pad_to_square(Image.open(fpath).convert("L"), crop_size)
    arr = np.array(img, dtype=np.float32)
    arr_ptp = np.ptp(arr)
    return (arr - arr.min()) / (arr_ptp if arr_ptp else 1.0)


def _make_pipeline(
    image_files,
    size,
    brightness_range=(-0.05, 0.05),
    contrast_range=(0.25, 1.0),
    noise_range=(0.001, 0.01),
):
    def load_random_image():
        img_path = random.choice(image_files)
        arr = _load_single_crop(img_path, size)
        return np.expand_dims(arr, axis=-1)

    return (
        dt.Value(load_random_image)
        >> dt.Multiply(lambda: np.random.uniform(*contrast_range))
        >> dt.Add(lambda: np.random.uniform(*brightness_range))
        >> dt.Gaussian(sigma=lambda: np.random.uniform(*noise_range))
        >> dt.Affine(
            rotation=lambda: np.random.uniform(0, 2 * np.pi),
            scale=lambda: np.random.uniform(0.8, 1.2),
            translate=lambda: (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)),
        )
        >> dt.FlipLR(p=0.5)
        >> dt.FlipUD(p=0.5)
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize LodeSTAR augmentations.")
    parser.add_argument("input", type=str, help="Path to a crop image or a folder of crops.")
    parser.add_argument(
        "--output", type=str, default="augmentation_preview.png", help="Output filename."
    )
    parser.add_argument("--count", type=int, default=12, help="Number of samples to generate.")
    parser.add_argument("--size", type=int, default=64, help="Crop size used during training.")
    parser.add_argument(
        "--brightness",
        type=float,
        nargs=2,
        default=(-0.05, 0.05),
        help="Brightness range (offset).",
    )
    parser.add_argument(
        "--contrast", type=float, nargs=2, default=(0.25, 1.0), help="Contrast range (multiplier)."
    )
    parser.add_argument(
        "--noise", type=float, nargs=2, default=(0.001, 0.01), help="Gaussian noise range (sigma)."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]
        if not image_files:
            print(f"Error: No images found in {input_path}")
            return
    else:
        image_files = [input_path]

    print(f"Found {len(image_files)} potential source images.")
    pipeline = _make_pipeline(image_files, args.size, args.brightness, args.contrast, args.noise)

    # Create plot (aim for a square grid)
    cols = int(np.ceil(np.sqrt(args.count)))
    rows = int(np.ceil(args.count / cols))

    # Scale figsize so the images don't get too tiny or too massive
    fig_w = min(cols * 3, 20)
    fig_h = fig_w * (rows / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).flatten()

    for i in range(args.count):
        augmented = pipeline.update().resolve()
        # Remove channel dim if present for imshow
        if augmented.ndim == 3:
            augmented = augmented[:, :, 0]

        axes[i].imshow(augmented, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Preview saved to {args.output}")


if __name__ == "__main__":
    main()
