import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
OUTPUT_DIR = "processed_data"
IMAGES_EXT = (".jpg", ".jpeg", ".png")
SPLIT_DIRS = ["train", "test", "validation"]


def copy_files(img_list, split_name):
    for img_path in img_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(DATA_DIR, base + ".txt")
        shutil.copy(
            img_path, os.path.join(OUTPUT_DIR, split_name, "images", os.path.basename(img_path))
        )
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(OUTPUT_DIR, split_name, "labels", base + ".txt"))


def run(config_path=None):
    found_split = any(
        os.path.isdir(os.path.join(DATA_DIR, split_dir_name)) for split_dir_name in SPLIT_DIRS
    )
    if found_split:
        print("Data directory already contains train/test/validation splits. No processing needed.")
        return

    processing_needed = True
    for split_name in SPLIT_DIRS:
        images_dir = os.path.join(DATA_DIR, split_name, "images")
        labels_dir = os.path.join(DATA_DIR, split_name, "labels")
        has_images_dir = os.path.isdir(images_dir)
        has_labels_dir = os.path.isdir(labels_dir)
        has_images = len(glob(os.path.join(images_dir, "*"))) > 0
        has_labels = len(glob(os.path.join(labels_dir, "*"))) > 0
        if not (has_images_dir and has_labels_dir and has_images and has_labels):
            processing_needed = False
            break
    if processing_needed:
        print("Data directory already in YOLO format. No processing needed.")
        return

    for split_name in ["train", "validation"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, "labels"), exist_ok=True)

    image_files = []
    for ext in IMAGES_EXT:
        image_files.extend(glob(os.path.join(DATA_DIR, f"*{ext}")))

    train_imgs, validation_imgs = train_test_split(image_files, test_size=0.2, random_state=42)

    copy_files(train_imgs, "train")
    copy_files(validation_imgs, "validation")

    print("Preprocessing complete. Data is ready for YOLOv12 fine-tuning.")


if __name__ == "__main__":
    run()
