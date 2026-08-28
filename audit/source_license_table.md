# Dataset Source and License Table

| Source | Active? | Classes Used | Format | License / Rights Status | Project Decision |
|---|---|---|---|---|---|
| COCO 2017 | Yes | person, car, dog, cat, bird, cow, bus, truck, motorcycle, bicycle, traffic_light, stop_sign | COCO JSON | COCO annotations are licensed under CC BY 4.0. The COCO Consortium does not own the copyright of the images; image use must comply with the applicable Flickr/original image license and terms. | PASS |
| Roboflow Flower v2 | Yes | daisy + dandelion -> flower | YOLOv8 | CC BY 4.0. Attribution required. | PASS WITH DOCUMENTED CAVEAT |
| Tree detection by season/type v4 | No | tree | YOLOv8 | CC BY 4.0 | REJECTED: baked augmentation + 640x640 Stretch |
| Roboflow Tree Detection Dataset v5 | No | tree | YOLOv8 | CC BY 4.0 | REJECTED: baked 240x240 Stretch |
| Kaggle Tree Yolo annotated | No | tree | YOLO | Dataset page declared Apache 2.0; underlying stock/web image provenance was not independently verified. | REJECTED: inconsistent bbox semantics |

## Active benchmark licensing notes

The interim benchmark contains 13 classes.

Flower redistribution/derived-dataset documentation must include CC BY 4.0 attribution.

COCO annotations are licensed under CC BY 4.0.

COCO image copyrights remain with the original image owners. The COCO Consortium
does not grant one uniform license covering all images; use of each image remains
subject to its applicable Flickr/original image license and terms.

Therefore the project must not describe all COCO images as Apache 2.0,
Public Domain, or uniformly CC BY 4.0.

Rejected/deferred Tree sources are audit evidence only and are not part of the active
training/evaluation dataset.
