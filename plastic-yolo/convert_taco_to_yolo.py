# convert_taco_to_yolo.py
"""
Convert the TACO dataset (single COCO JSON) to YOLO format while keeping only
plastic‑relevant classes, and create a train/val split.

Assumptions:
* TACO was downloaded/unzipped into a folder named TACO-master/
* Inside TACO-master/data/ we have:
 images/          ← raw .jpg/.png files (may be partially downloaded)
 annotations.json ← COCO format with all images/annotations
* Output will be written to:
      data/taco/images/train/
      data/taco/images/val/
      data/taco/labels/train/
      data/taco/labels/val/
We will *not* copy images; we will use them in place (they already live under
TACO-master/data/images). If an image is missing, we skip its annotations.
"""

import json
import os
import shutil
import random
from pathlib import Path

# ==================== USER SETTINGS ====================
COCO_ROOT = Path("TACO-master/data")          # folder containing images/ and annotations.json
OUTPUT_ROOT = Path("data/taco")               # where YOLO‑ready data will go

# List the plastic classes you want to keep (order determines YOLO ID)
PLASTIC_CLASSES = [
    "Plastic bottle",
    "Plastic bag",
    "Plastic wrapper",
    "Plastic container",
    "Other plastic"
]

# Train/val split ratio (e.g., 0.8 = 80% train, 20% val)
SPLIT_RATIO = 1.0
# Set a seed for reproducible splits
RANDOM_SEED = 42
# =======================================================


def build_id_mapping(coco_data: dict) -> dict[int, int]:
    """
    Build a map from original COCO category_id -> new YOLO class_id (0..N-1).
    Categories that contain any string in PLASTIC_CLASSES (as substring, case-insensitive) are kept.
    The first matching PLASTIC_CLASSES entry determines the YOLO class_id.
    Categories with no match map to -1 (ignore).
    """
    mapping = {}
    for cat in coco_data["categories"]:
        matched = False
        cat_name_lower = cat["name"].lower()
        for idx, pc in enumerate(PLASTIC_CLASSES):
            if pc.lower() in cat_name_lower:
                mapping[cat["id"]] = idx
                matched = True
                break
        if not matched:
            mapping[cat["id"]] = -1
    return mapping


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] (pixels) to YOLO normalized string."""
    x, y, w, h = bbox
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    return f"{xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def process_split(image_ids: list[int], split_name: str,
                  coco_data: dict, id_map: dict[int, int]):
    """
    Create YOLO label files for a given list of image IDs.
    Images are NOT copied; we expect them to already exist under
    COCO_ROOT / "images" / <file_name>.
    """
    print(f"[DEBUG] Processing {split_name}: {len(image_ids)} image IDs")

    # Build lookup: image_id -> (file_name, width, height)
    img_lookup = {
        im["id"]: (im["file_name"], im["width"], im["height"])
        for im in coco_data["images"]
        if im["id"] in image_ids
    }
    print(f"[DEBUG] Built lookup for {len(img_lookup)} images")

    img_out_dir = OUTPUT_ROOT / "images" / split_name
    lbl_out_dir = OUTPUT_ROOT / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare a dict to accumulate label lines per image
    labels_accum = {imid: [] for imid in image_ids}
    print(f"[DEBUG] Prepared labels_accum for {len(labels_accum)} images")

    plastic_annotations = 0
    skipped_non_plastic = 0
    skipped_wrong_split = 0
    skipped_missing_image = 0

    for anno in coco_data["annotations"]:
        orig_cid = anno["category_id"]
        new_cid = id_map.get(orig_cid, -2)
        if new_cid == -1:          # ignore non‑plastic classes
            skipped_non_plastic += 1
            continue
        imid = anno["image_id"]
        if imid not in image_ids:  # annotation belongs to an image not in this split
            skipped_wrong_split += 1
            continue

        file_name, img_w, img_h = img_lookup[imid]
        # Verify image file exists
        src_img = COCO_ROOT / "images" / file_name
        if not src_img.is_file():
            # Skip this annotation because image missing
            skipped_missing_image += 1
            continue

        yolo_str = coco_bbox_to_yolo(anno["bbox"], img_w, img_h)

        # Copy image to flat directory structure (no subdirectories)
        # Use image ID as filename to avoid collisions and ensure uniqueness
        dst_img = img_out_dir / f"{imid:012d}{Path(file_name).suffix}"
        if not dst_img.exists():
            # Ensure destination directory exists
            img_out_dir.mkdir(parents=True, exist_ok=True)
            # Try to create a hard link; if fails, copy
            try:
                os.link(src_img, dst_img)
            except (OSError, PermissionError):
                # Fallback to copy
                shutil.copy2(src_img, dst_img)

        # Append label line
        labels_accum[imid].append(f"{new_cid} {yolo_str}")
        plastic_annotations += 1

    print(f"[DEBUG] Finished looping through annotations:")
    print(f"  Plastic annotations: {plastic_annotations}")
    print(f"  Skipped non-plastic: {skipped_non_plastic}")
    print(f"  Skipped wrong split: {skipped_wrong_split}")
    print(f"  Skipped missing image: {skipped_missing_image}")

    # Write label files in flat directory structure (matching image naming)
    written_labels = 0
    for imid, lines in labels_accum.items():
        if lines:  # Only write if there are labels
            label_path = lbl_out_dir / f"{imid:012d}.txt"
            lbl_out_dir.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines))
            written_labels += 1

    print(f"[DEBUG] Wrote {written_labels} label files")

    # Summary
    img_count = len([p for p in img_out_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"[INFO] {split_name.upper()}: {img_count} images processed (with labels)")


def main():
    # Load COCO JSON
    annotations_path = COCO_ROOT / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Cannot find {annotations_path}")

    with open(annotations_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    print(f"[INFO] Loaded COCO with {len(coco_data['images'])} images and "
          f"{len(coco_data['annotations'])} annotations.")

    # Build ID mapping
    id_mapping = build_id_mapping(coco_data)
    print("\n[INFO] Category ID mapping (COCO -> YOLO):")
    for cocoid, yoid in id_mapping.items():
        if yoid >= 0:
            name = PLASTIC_CLASSES[yoid]
            print(f"       COCO {cocoid:>3} -> YOLO {yoid} ({name})")

    # Determine image IDs that have at least one plastic annotation
    plastic_image_ids = set()
    for anno in coco_data["annotations"]:
        orig_cid = anno["category_id"]
        new_cid = id_mapping.get(orig_cid, -2)
        if new_cid != -1:          # plastic class
            plastic_image_ids.add(anno["image_id"])
    plastic_image_ids = list(plastic_image_ids)
    print(f"[INFO] Found {len(plastic_image_ids)} images with at least one plastic annotation.")

    # Split plastic images into train/val
    random.seed(RANDOM_SEED)
    random.shuffle(plastic_image_ids)
    split_idx = int(len(plastic_image_ids) * SPLIT_RATIO)
    train_ids = plastic_image_ids[:split_idx]
    val_ids   = plastic_image_ids[split_idx:]

    print(f"\n[INFO] Splitting: {len(train_ids)} train images, {len(val_ids)} val images (plastic-only)")

    # Process splits
    process_split(train_ids, "train", coco_data, id_mapping)
    process_split(val_ids,   "val",   coco_data, id_mapping)

    print("\n[INFO] Conversion complete! Your YOLO-ready data lives under:")
    print(f"       {OUTPUT_ROOT.resolve()}")
    print("\nNext steps:")
    print("  1. Create a data.yaml file (see instructions below).")
    print("  2. Run `yolo train` (or your preferred trainer).")


if __name__ == "__main__":
    main()