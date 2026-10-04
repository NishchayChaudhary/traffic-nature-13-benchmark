# RT-DETRv2-S vs YOLOX-M — Traffic + Nature Object Detection

A controlled transfer-learning benchmark comparing a transformer-based detector
(RT-DETRv2-S) with a CNN-based detector (YOLOX-M) on the same unified
13-class object-detection dataset.

> The original project targeted 14 classes. Tree is currently deferred, so
> the benchmark reported here contains 13 classes.



## Dataset

- Train images: 21,990
- Train boxes: 139,045
- Validation images: 5,364
- Validation boxes: 17,782
- Canonical annotation format: COCO JSON
- Input resolution: 640x640
- Training budget: 120 epochs
- Official benchmark GPU: NVIDIA GeForce RTX 5060 Ti 16GB

## Final Accuracy

| Metric | RT-DETRv2-S | YOLOX-M |
|---|---:|---:|
| Best epoch | 22 | 108 |
| mAP50:95 | **0.5167** | 0.4940 |
| AP50 | **0.707** | 0.698 |
| AP75 | **0.547** | 0.520 |
| AP Small | **0.294** | 0.269 |
| AP Medium | **0.540** | 0.528 |
| AP Large | **0.729** | 0.698 |

RT-DETRv2-S achieved approximately +2.27 AP points higher mAP50:95.

## Model Size

| Model | Parameters |
|---|---:|
| RT-DETRv2-S | 20,098,436 |
| YOLOX-M | 25,287,702 |

## Matched RTX 5060 Ti Benchmark

Batch size 1, 640x640 input, 50 warmup iterations, 300 measured
iterations, CUDA-event timing, forward-pass only.

| Metric | RT-DETRv2-S | YOLOX-M |
|---|---:|---:|
| FP32 median latency | **6.805 ms** | 6.828 ms |
| FP32 FPS | **146.96** | 146.47 |
| FP16 median latency | 4.978 ms | **4.662 ms** |
| FP16 FPS | 200.90 | **214.51** |
| FP32 peak allocated VRAM | **8524 MB** | 9953 MB |
| FP16 peak allocated VRAM | **345 MB** | 465 MB |

## Main Finding

RT-DETRv2-S provided the stronger accuracy-memory trade-off in this
benchmark, while YOLOX-M had a modest FP16 throughput advantage.

This experiment should be interpreted as a controlled model-recipe
benchmark, not as proof that one architecture family universally
outperforms another.

## Reproducibility

The repository contains:

- training configurations
- dataset construction script
- dataset hashes
- model checkpoint hashes
- training and evaluation logs
- per-class metrics
- hardware benchmark scripts
- environment information

Model checkpoints and datasets are not stored directly in this repository.

## Data Sources

The unified dataset uses COCO 2017 for the shared object categories and a
Roboflow flower dataset for the flower category. See `audit/source_license_table.md`
for source and licensing information.
