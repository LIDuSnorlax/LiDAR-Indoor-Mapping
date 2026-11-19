# LiDAR Indoor Mapping — Pose-Free Point Cloud Registration

> Incrementally build dense indoor maps from raw LiDAR scans **without any initial pose estimate** (no IMU, no wheel odometry, no quaternion file required).

---

## Background

Most LiDAR SLAM pipelines assume access to an initial pose guess — typically from IMU integration or wheel encoders — to seed the frame-to-frame registration. This works well when sensor fusion hardware is available, but fails in minimal setups (handheld scanners, low-cost robots, retrofitted systems).

This project answers a simpler question:

> *Given only a sequence of raw LiDAR frames, how accurately can we reconstruct an indoor map?*

The approach relies entirely on **point cloud geometry** — no external pose information at any stage.

---

## What Makes This Different

### 1. Cold-start via FPFH + Fast Global Registration
Each frame pair is initialized using FPFH feature descriptors and Fast Global Registration (FGR), eliminating the need for an initial pose guess. When FGR confidence is low (fitness < threshold), the system falls back to an identity transform — a safe assumption for consecutive frames with small motion.

### 2. Plane-Constrained ICP (Mode B / C)
Indoor scenes are rich in flat surfaces (walls, floors, ceilings). After GICP coarse alignment, dominant planes are detected via RANSAC and injected as virtual point constraints. This keeps the registration geometrically grounded and significantly reduces drift along featureless corridors.

### 3. Submap-Based Local Registration + Loop Closure (Mode C)
Instead of aligning each frame against only its predecessor, each new frame is registered against a **local submap** built from the K most recent frames. This reduces accumulated noise per step. A global FPFH descriptor triggers loop closure candidates; confirmed loops feed into a lightweight pose-graph optimisation to correct long-range drift.

---

## Results

The following comparison uses the [DynamicMap Benchmark semindoor sequence](https://zenodo.org/records/10886629) (960 frames, VLP-16).

| | No Registration (raw stacked) | Mode A (FGR+GICP) | Mode C (Submap+Loop) |
|---|---|---|---|
| Chamfer Distance ↓ | — | ~0.08m | ~0.03m |
| F1 @ 0.1m ↑ | — | ~0.71 | ~0.89 |

*(replace with your actual numbers after running eval_pcd.py)*

**Visual comparison:**

| Raw frames stacked | Mode C output |
|---|---|
| ![before](assets/before.png) | ![after](assets/after.png) |

---

## Quick Start

```bash
pip install open3d numpy scipy
```

**Training (map building):**
```bash
python lidar_train.py
# Select PCD folder → choose Mode C → Start
```

**Visualisation & Analysis:**
```bash
python lidar_visualize.py
# Load the output .pcd → Viewer / Editor / Analysis tabs
```

**Evaluate against ground truth:**
```bash
python eval_pcd.py aligned_map.pcd gt_map.pcd --heatmap --histogram
```

---

## File Structure

```
.
├── lidar_train.py        # Map building GUI (Modes A / B / C)
├── lidar_visualize.py    # Viewer + Editor + Structural Analysis GUI
├── eval_pcd.py           # Quantitative evaluation between two .pcd file
├── rename_pcd.py         # Utility: rename arbitrary PCD files to expected format
└── assets/               # Screenshots for README
```

---

## References

- Zhou et al., *Fast Global Registration*, ECCV 2016
- Segal et al., *Generalized ICP*, RSS 2009
- DynamicMap Benchmark — https://github.com/KTH-RPL/DynamicMap_Benchmark
