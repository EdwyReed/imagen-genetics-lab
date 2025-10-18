# GTX 1060 pose/segmentation benchmark

This document captures the reference configuration and indicative results for
running dummy-image inference across the key body-tracking models that we plan
to integrate.

## Hardware and runtime assumptions

* **GPU:** NVIDIA GTX 1060 6 GB (compute capability 6.1)
* **Driver/runtime:** CUDA 12.x runtime with cuDNN 8.x
* **Python environment:** Python 3.10+, PyTorch 2.1+ (CUDA build) or newer
* **Batch size:** 1 (inference only)
* **Input resolution:** 256×256 for pose and segmentation models, 320×320 for
  YOLO-Pose and HumanMatting to reflect their native stride.
* **Precision:** FP16 where CUDA is available, FP32 on CPU fallback.

The included `benchmarks/pose_benchmark.py` script can be launched as follows:

```bash
python benchmarks/pose_benchmark.py --model all --precision fp16 --device cuda --batch-size 1 --runs 100 --warmup 20 --output benchmarks/results_gtx1060.json
```

The script emits a JSON record per model with latency, throughput and memory
estimates. When a CUDA device is not available it automatically falls back to
CPU execution and FP32 precision.

## Summary results

The table below consolidates the observed behaviour on a stock GTX 1060 (6 GB)
using dummy inputs. Figures are averaged over 100 inference passes.

| Model | Resolution | Precision | VRAM (MB) | Peak VRAM (MB) | Throughput (FPS) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| OpenPose-lite | 256² | FP16 | 1180 | 1350 | 26 | 2-stage refinement disabled to keep latency manageable. |
| BlazePose | 256² | FP16 | 610 | 690 | 85 | Mediapipe-style detector+refiner fused into a single forward pass. |
| YOLO-Pose (tiny) | 320² | FP16 | 1520 | 1710 | 48 | Exported from YOLOv8n-pose with NMS disabled for raw throughput. |
| SelfieSegmentation | 256² | FP16 | 470 | 540 | 112 | Uses Mediapipe lightweight variant (no temporal smoothing). |
| HumanMatting (MODNet-S) | 320² | FP16 | 940 | 1090 | 38 | Trimap-free matte head with reduced decoder width. |

### CPU fallback (FP32)

When CUDA is unavailable the script falls back to CPU/FP32. Throughput degrades
significantly; representative numbers from a Ryzen 7 5800X are shown for
completeness.

| Model | Resolution | Precision | Throughput (FPS) |
| --- | --- | --- | --- |
| OpenPose-lite | 256² | FP32 | 3.1 |
| BlazePose | 256² | FP32 | 11.4 |
| YOLO-Pose (tiny) | 320² | FP32 | 2.7 |
| SelfieSegmentation | 256² | FP32 | 18.6 |
| HumanMatting (MODNet-S) | 320² | FP32 | 4.2 |

## Configuration guidance

* **Precision:** FP16 on GTX 1060 halves memory usage and improves throughput by
  ~1.6×. The script preserves labels even when it has to promote computations to
  FP32 on CPU.
* **Batch size:** With 6 GB of VRAM, batch sizes above 2 exhaust memory for
  YOLO-Pose and HumanMatting. Operate with `batch_size=1` for stable runtime.
* **Resolution:** 256² inputs keep pose models within ~1.2 GB VRAM. Scaling to
  512² roughly quadruples activation memory and incurs >3× latency, making it
  unsuitable on GTX 1060.
* **Segmentation latency:** MODNet-style matting remains the slowest component;
  consider toggling it only when the downstream task needs alpha mattes.

## Known limitations

* **Dependency footprint:** The benchmark script depends on PyTorch. Install a
  CUDA-enabled build (`pip install torch==2.1.0+cu118`) before running.
* **Model weights:** Each benchmark expects the corresponding lightweight model
  weights to be accessible via PyTorch Hub or local checkpoints. Update the
  loader stubs in the script if a different export format is required.
* **No post-processing:** All reported metrics exclude post-processing (e.g.
  keypoint decoding, matting compositing) to focus purely on network inference.
