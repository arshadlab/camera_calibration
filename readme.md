# Camera Calibration + ArUco Pose Tracking (OpenCV)

This project provides a complete workflow for:

1. **Camera calibration** using a printed chessboard  
2. **Camera pose tracking** using an ArUco marker  

A new user should be able to follow this document and reproduce the full setup from scratch.

---

# 1. Requirements

- Python 3.8+
- `opencv-contrib-python`
- `numpy`

Install:

```bash
pip install numpy opencv-contrib-python
```

Verify OpenCV:

```bash
python -c "import cv2; print(cv2.__version__)"
```

---

# 2. Step A — Print and Measure the Chessboard

## 2.1 Print the chessboard

- Print at **100% scale**
- Disable “Fit to page”
- Use thick paper if possible
- Keep it flat

---

## 2.2 Count Inner Corners

OpenCV uses **inner corners**, not number of squares.

Example:

If the board has:

- 10 squares horizontally → 9 inner corners → `--cols 9`
- 7 squares vertically → 6 inner corners → `--rows 6`

So a “9x6” board means:

```bash
--cols 9 --rows 6
```

---

## 2.3 Measure Square Size

Measure one square side using a ruler or caliper.

Convert to meters:

| Measured | Use |
|----------|------|
| 24 mm | `0.024` |
| 25 mm | `0.025` |
| 30 mm | `0.030` |

Pass this to:

```bash
--square <meters>
```

### Why this matters

- Intrinsics (K matrix) are unaffected by scale.
- **Pose distances (meters) depend on this value.**
- Wrong square size → wrong real-world scale.

---

# 3. Step B — Camera Calibration

## 3.1 Capture and Calibrate (Live Camera)

```bash
python calib_chessbrd.py --mode camera_calib --cols 9 --rows 6 --square 0.024 --manual
```

Controls:

| Key | Action |
|-----|--------|
| c | Save image (only if chessboard detected) |
| q | Stop capture and run calibration |

---

## 3.2 Calibration Quality Guidelines

After calibration, you will see:

```
Calibration RMS: ...
Reprojection RMSE (pixels): ...
```

### What is a good calibration?

| RMSE (pixels) | Quality |
|---------------|----------|
| < 0.3 px | Excellent |
| 0.3 – 0.5 px | Very Good |
| 0.5 – 1.0 px | Acceptable |
| > 1.0 px | Poor (retake images) |

For stable ArUco pose tracking:

```
Reprojection RMSE < 0.5 pixels
```

If RMSE is high:

- Capture more diverse poses
- Include edges and corners of image
- Avoid blur
- Ensure consistent resolution

---

## 3.3 Calibration from Existing Images

```bash
python calib_chessbrd.py --mode images_calib --pattern "calib_images/session/*.jpg" --cols 9 --rows 6 --square 0.024
```

---

# 4. Step C — Generate and Print ArUco Marker

## 4.1 Recommended Dictionary

For most setups:

```
4x4_50
```

This means:

- 4x4 binary grid
- 50 possible unique markers
- IDs: 0–49

You only need to print **one marker**.

---

## 4.2 Generate Marker Image

```python
import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_id = 0
size_px = 800

marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
cv2.imwrite("aruco_4x4_50_id0.png", marker)
print("Saved aruco_4x4_50_id0.png")
```

Print at 100% scale.

Alternatively, marker can be generated from online
  https://chev.me/arucogen/

---

## 4.3 Measure Marker Length

Measure the **black square width only** (not white border).

Convert to meters:

| Measured | Use |
|----------|------|
| 50 mm | `0.05` |
| 40 mm | `0.04` |

Pass as:

```bash
--marker_length <meters>
```

This determines pose scale.

---

# 5. Step D — Run ArUco Pose Tracking

```bash
python calib_chessbrd.py --mode aruco \
  --calib_file camera_calib.npz \
  --aruco_dict 4x4_50 \
  --marker_length 0.05 \
  --target_id 0 \
  --draw_axis
```

Controls:

| Key | Action |
|-----|--------|
| q | Quit |

---

# 6. Coordinate System Details

- Pose is relative to the **center of the marker**
- Units are meters
- Camera coordinate system:
  - +X → right
  - +Y → down
  - +Z → forward

Displayed value:

```
CamPos[m]: x=... y=... z=...
```

Means:

Camera position in marker coordinate frame.

---

# 7. Best Practices for Stable Results

## Calibration

- Use 20–40 diverse images
- Cover entire image area
- Tilt board at angles
- Avoid near-duplicate frames
- Keep resolution fixed

## ArUco

- Use 4x4_50 for robustness
- Use sufficiently large marker (≥ 5 cm recommended)
- Keep marker flat and rigid

---

# 8. Quick Start Summary

1. Print chessboard  
2. Measure square size  
3. Run camera calibration  
4. Print ArUco marker  
5. Measure marker length  
6. Run ArUco tracking  

---

If needed, the system can be extended to:

- Multi-marker world boards
- Pose smoothing
- Charuco-based higher precision calibration
- Multi-camera calibration

