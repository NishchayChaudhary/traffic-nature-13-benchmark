from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import json
import csv
import os
import shutil

COCO_ROOT = Path("/home/nishchay/datasets/coco_14")
FLOWER_ROOT = Path("/home/nishchay/datasets/traffic_nature_audit/raw/flower")
AUDIT_ROOT = Path("/home/nishchay/datasets/traffic_nature_audit")
OUT = Path("/home/nishchay/datasets/traffic_nature_13")

COCO_TRAIN_JSON = COCO_ROOT / "annotations/instances_train2017.json"
COCO_VAL_JSON = COCO_ROOT / "annotations/instances_val2017.json"

COCO_TRAIN_IMG = COCO_ROOT / "train2017"
COCO_VAL_IMG = COCO_ROOT / "val2017"

EXCLUSIONS = AUDIT_ROOT / "flower_exclusions.csv"

FINAL_CLASSES = [
    "person",
    "car",
    "dog",
    "cat",
    "bird",
    "cow",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "traffic_light",
    "stop_sign",
    "flower",
]

FINAL_ID = {name: i for i, name in enumerate(FINAL_CLASSES)}

COCO_NAME_MAP = {
    "person": "person",
    "car": "car",
    "dog": "dog",
    "cat": "cat",
    "bird": "bird",
    "cow": "cow",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
}

TARGET_PER_COCO_CLASS = 3000


def reset_output():
    if OUT.exists():
        shutil.rmtree(OUT)

    for split in ["train", "val", "test"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)

    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)


def symlink(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def target_coco_categories(data):
    id_to_name = {x["id"]: x["name"] for x in data["categories"]}

    return {
        cid: COCO_NAME_MAP[name]
        for cid, name in id_to_name.items()
        if name in COCO_NAME_MAP
    }


def select_coco_train(data):
    target_cats = target_coco_categories(data)

    image_classes = defaultdict(set)

    for ann in data["annotations"]:
        if ann["category_id"] in target_cats:
            image_classes[ann["image_id"]].add(
                target_cats[ann["category_id"]]
            )

    availability = Counter()

    for classes in image_classes.values():
        for c in classes:
            availability[c] += 1

    targets = {
        c: min(TARGET_PER_COCO_CLASS, availability[c])
        for c in COCO_NAME_MAP.values()
    }

    selected = set()
    counts = Counter()

    # Hard-preserve classes whose complete available pool is below target.
    rare_classes = {
        c for c in targets
        if availability[c] <= TARGET_PER_COCO_CLASS
    }

    for image_id in sorted(image_classes):
        classes = image_classes[image_id]

        if classes & rare_classes:
            selected.add(image_id)
            for c in classes:
                counts[c] += 1

    remaining = set(image_classes) - selected

    while True:
        deficits = {
            c: max(0, targets[c] - counts[c])
            for c in targets
        }

        if all(v == 0 for v in deficits.values()):
            break

        best_id = None
        best_score = 0

        for image_id in remaining:
            score = 0.0

            for c in image_classes[image_id]:
                if deficits[c] > 0:
                    score += deficits[c] / targets[c]

            if score > best_score:
                best_score = score
                best_id = image_id

            elif score == best_score and score > 0:
                if best_id is None or image_id < best_id:
                    best_id = image_id

        if best_id is None or best_score <= 0:
            break

        selected.add(best_id)
        remaining.remove(best_id)

        for c in image_classes[best_id]:
            counts[c] += 1

    return selected, counts, availability, targets


def add_coco_split(data, source_dir, split, selected_ids=None):
    target_cats = target_coco_categories(data)

    images_by_id = {x["id"]: x for x in data["images"]}

    anns_by_image = defaultdict(list)

    for ann in data["annotations"]:
        if ann["category_id"] in target_cats:
            anns_by_image[ann["image_id"]].append(ann)

    if selected_ids is None:
        selected_ids = set(images_by_id)

    output_images = []
    output_anns = []
    manifest = []

    next_img_id = 1
    next_ann_id = 1

    for old_image_id in sorted(selected_ids):
        src_info = images_by_id[old_image_id]
        src = source_dir / src_info["file_name"]

        new_name = f"coco_{src_info['file_name']}"
        dst = OUT / "images" / split / new_name

        symlink(src, dst)

        output_images.append({
            "id": next_img_id,
            "file_name": new_name,
            "width": src_info["width"],
            "height": src_info["height"],
        })

        for ann in anns_by_image[old_image_id]:
            class_name = target_cats[ann["category_id"]]

            output_anns.append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": FINAL_ID[class_name],
                "bbox": ann["bbox"],
                "area": ann.get(
                    "area",
                    ann["bbox"][2] * ann["bbox"][3]
                ),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
            })

            next_ann_id += 1

        manifest.append({
            "split": split,
            "output_filename": new_name,
            "source": "coco2017",
            "source_split": source_dir.name,
            "source_filename": src_info["file_name"],
        })

        next_img_id += 1

    return output_images, output_anns, manifest


def load_flower_exclusions():
    excluded = set()

    with open(EXCLUSIONS, newline="") as f:
        for row in csv.DictReader(f):
            if row["action"].startswith("exclude"):
                excluded.add((row["split"], row["filename"]))

    return excluded


def add_flower_split(split, output_split, start_img_id, start_ann_id):
    img_dir = FLOWER_ROOT / split / "images"
    label_dir = FLOWER_ROOT / split / "labels"

    excluded = load_flower_exclusions()

    images = []
    anns = []
    manifest = []

    img_id = start_img_id
    ann_id = start_ann_id

    for src in sorted(img_dir.iterdir()):
        if not src.is_file():
            continue

        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        if (split, src.name) in excluded:
            continue

        with Image.open(src) as im:
            width, height = im.size

        new_name = f"flower_{src.name}"
        dst = OUT / "images" / output_split / new_name
        symlink(src, dst)

        images.append({
            "id": img_id,
            "file_name": new_name,
            "width": width,
            "height": height,
        })

        label = label_dir / f"{src.stem}.txt"

        if label.exists():
            for line in label.read_text(errors="ignore").splitlines():
                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                source_class = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])

                bw = w * width
                bh = h * height
                x0 = (x - w / 2) * width
                y0 = (y - h / 2) * height

                anns.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": FINAL_ID["flower"],
                    "bbox": [x0, y0, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "segmentation": [],
                })

                ann_id += 1

        manifest.append({
            "split": output_split,
            "output_filename": new_name,
            "source": "roboflow_flower_v2",
            "source_split": split,
            "source_filename": src.name,
        })

        img_id += 1

    return images, anns, manifest, img_id, ann_id


def save_coco(split, images, anns):
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(FINAL_CLASSES)
    ]

    data = {
        "images": images,
        "annotations": anns,
        "categories": categories,
    }

    path = OUT / "annotations" / f"instances_{split}.json"

    with open(path, "w") as f:
        json.dump(data, f)

    return path


def main():
    reset_output()

    train_data = load_coco(COCO_TRAIN_JSON)
    val_data = load_coco(COCO_VAL_JSON)

    selected, selected_counts, availability, targets = select_coco_train(
        train_data
    )

    print("Selected COCO train images:", len(selected))

    coco_train_imgs, coco_train_anns, train_manifest = add_coco_split(
        train_data,
        COCO_TRAIN_IMG,
        "train",
        selected,
    )

    coco_val_imgs, coco_val_anns, val_manifest = add_coco_split(
        val_data,
        COCO_VAL_IMG,
        "val",
        None,
    )

    # Flower TRAIN
    f_train_imgs, f_train_anns, f_train_manifest, _, _ = add_flower_split(
        "train",
        "train",
        len(coco_train_imgs) + 1,
        len(coco_train_anns) + 1,
    )

    # Flower VALID
    f_val_imgs, f_val_anns, f_val_manifest, _, _ = add_flower_split(
        "valid",
        "val",
        len(coco_val_imgs) + 1,
        len(coco_val_anns) + 1,
    )

    # Flower TEST
    f_test_imgs, f_test_anns, f_test_manifest, _, _ = add_flower_split(
        "test",
        "test",
        1,
        1,
    )

    train_images = coco_train_imgs + f_train_imgs
    train_anns = coco_train_anns + f_train_anns

    val_images = coco_val_imgs + f_val_imgs
    val_anns = coco_val_anns + f_val_anns

    save_coco("train", train_images, train_anns)
    save_coco("val", val_images, val_anns)
    save_coco("test_flower", f_test_imgs, f_test_anns)

    all_manifest = (
        train_manifest +
        f_train_manifest +
        val_manifest +
        f_val_manifest +
        f_test_manifest
    )

    manifest_path = OUT / "manifests/source_manifest.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "output_filename",
                "source",
                "source_split",
                "source_filename",
            ]
        )
        writer.writeheader()
        writer.writerows(all_manifest)

    summary = {
        "classes": FINAL_CLASSES,
        "coco_train_available_image_presence": dict(availability),
        "coco_train_targets": dict(targets),
        "coco_train_selected_image_presence": dict(selected_counts),
        "coco_train_selected_unique_images": len(selected),
        "flower_train_images": len(f_train_imgs),
        "train_total_images": len(train_images),
        "train_total_annotations": len(train_anns),
        "val_total_images": len(val_images),
        "val_total_annotations": len(val_anns),
        "flower_test_images": len(f_test_imgs),
        "flower_test_annotations": len(f_test_anns),
    }

    with open(OUT / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 72)
    print("UNIFIED 13-CLASS DATASET BUILT")
    print("=" * 72)
    print("COCO train selected :", len(selected))
    print("Flower train        :", len(f_train_imgs))
    print("TOTAL train images  :", len(train_images))
    print("TOTAL train boxes   :", len(train_anns))
    print("TOTAL val images    :", len(val_images))
    print("TOTAL val boxes     :", len(val_anns))
    print("Flower test images  :", len(f_test_imgs))
    print()
    print("Output:", OUT)
    print("Raw datasets were NOT modified.")


if __name__ == "__main__":
    main()
