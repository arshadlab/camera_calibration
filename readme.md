# Camera Calibration + ArUco Pose Tracking (OpenCV + ROS2)

This project provides a complete workflow for:

1. **Camera calibration** using a printed chessboard  
2. **Camera pose tracking** using an ArUco marker  
3. Support for:
   - OpenCV camera
   - ROS2 image topics
   - Image files (calibration only)

---

# 1. Requirements

## Core (Always Required)

- Python 3.8+
- numpy
- opencv-contrib-python

Install:

```bash
pip install numpy opencv-contrib-python
```

---

## Optional (Only if using ROS2)

- ROS2 Jazzy (or compatible distro)
- rclpy
- sensor_msgs
- cv_bridge

Install cv_bridge:

```bash
sudo apt install ros-jazzy-cv-bridge
```

Source ROS2 environment:

```bash
source /opt/ros/jazzy/setup.bash
```

---

# 2. Step A — Print and Measure Chessboard

## 2.1 Print Chessboard

- Print at **100% scale**
- Disable “Fit to page”
- Use rigid paper
- Avoid reflections

---

## 2.2 Count Inner Corners

OpenCV uses **inner corners**, not number of squares.

Example:

If the board has:

- 10 squares across → 9 inner corners → `--cols 9`
- 7 squares down → 6 inner corners → `--rows 6`

So:

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

Pass as:

```bash
--square 0.024
```

⚠ Important: This defines real-world scale.

---

# 3. Camera Calibration (`camera_calib.py`)

Supports 3 sources:

- `cam` → OpenCV camera
- `ros` → ROS2 topic
- `images` → existing image files

---

## 3.1 Calibration Using OpenCV Camera

Manual save (recommended):

```bash
python3 camera_calib.py \
  --source cam \
  --cols 9 --rows 6 \
  --square 0.024 \
  --manual
```

Controls:

| Key | Action |
|-----|--------|
| c | Save image (only if chessboard detected) |
| q | Stop capture and run calibration |

---

## 3.2 Calibration Using ROS2 Topic

```bash
python3 camera_calib.py \
  --source ros \
  --topic /image_raw \
  --cols 9 --rows 6 \
  --square 0.024 \
  --manual
```

If ROS2 is not installed:
- Script automatically falls back to camera mode
- Use `--strict_ros` to force ROS-only mode

---

## 3.3 Calibration From Images

```bash
python3 camera_calib.py \
  --source images \
  --pattern "calib_images/session/*.jpg" \
  --cols 9 --rows 6 \
  --square 0.024
```

---

# 4. Calibration Quality Guidelines

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

For stable ArUco pose:

```
RMSE < 0.5 pixels recommended
```

Tips:
- Capture 20–40 diverse poses
- Include edges of frame
- Tilt board
- Avoid motion blur
- Keep resolution fixed

Calibration file saved as:

```
camera_calib.npz
```

---

# 5. Generate and Print ArUco Marker

## 5.1 Recommended Dictionary

For most use:

```
4x4_50
```

Or more robust:

```
6x6_250
```

---

## 5.2 Generate Marker

Example for `4x4_50`:

```python
import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_id = 0
size_px = 800

marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
cv2.imwrite("aruco_4x4_50_id0.png", marker)
```

Print at 100%.

---

## 5.3 Measure Marker Length

Measure the black square side only.

Convert to meters:

| Measured | Use |
|----------|------|
| 50 mm | `0.05` |
| 40 mm | `0.04` |

---

# 6. ArUco Pose Estimation (`aruco_pose.py`)

Supports:

- `cam`
- `ros`

---

## 6.1 Using OpenCV Camera

```bash
python3 aruco_pose.py \
  --source cam \
  --calib_file camera_calib.npz \
  --aruco_dict 4x4_50 \
  --marker_length 0.05 \
  --target_id 0 \
  --draw_axis
```

---

## 6.2 Using ROS2 Topic

```bash
python3 aruco_pose.py \
  --source ros \
  --topic /image_raw \
  --calib_file camera_calib.npz \
  --aruco_dict 4x4_50 \
  --marker_length 0.05 \
  --target_id 0 \
  --draw_axis
```

If ROS2 not installed:
- Falls back to camera mode
- Use `--strict_ros` to enforce ROS only

---

# 7. Coordinate System

- Pose is relative to **marker center**
- Units are meters
- Camera coordinate frame:
  - +X → right
  - +Y → down
  - +Z → forward

Displayed:

```
CamPos[m]: x=... y=... z=...
```

This is camera position in marker frame.

---

# 8. Publishing Camera to ROS2 (Quick Test)

## 8.1 Install usb_cam

```bash
sudo apt install ros-jazzy-usb-cam
```

---

## 8.2 Run Camera Publisher

```bash
ros2 run usb_cam usb_cam_node_exe
```

Check topics:

```bash
ros2 topic list
```

Common topic names:

```
/image_raw
/usb_cam/image_raw
```

Check type:

```bash
ros2 topic info /image_raw
```

Should be:

```
sensor_msgs/msg/Image
```

---

## 8.3 View Camera

```bash
ros2 run rqt_image_view rqt_image_view
```

Select `/image_raw`.

---

## 8.4 Use With Calibration Script

```bash
python3 camera_calib.py --source ros --topic /image_raw --manual
```

---

## 8.5 Use With ArUco Script

```bash
python3 aruco_pose.py --source ros --topic /image_raw --marker_length 0.05
```

---

# 9. Best Practices

## Calibration

- 20–40 images
- Wide pose variation
- Stable lighting
- RMSE < 0.5 px

## ArUco

- Marker ≥ 5 cm recommended
- Use rigid surface
- Avoid motion blur
- Ensure good contrast

---

# 10. Typical Workflow

1. Print chessboard
2. Measure square size
3. Run `camera_calib.py`
4. Verify RMSE < 0.5 px
5. Print ArUco marker
6. Measure marker size
7. Run `aruco_pose.py`
8. Move camera and observe real-world coordinates

---

# 11. Notes

- Scripts automatically detect ROS2 availability.
- If ROS2 is not installed, system still works using OpenCV camera.
- Calibration scale depends on correct square measurement.
- Pose scale depends on correct marker measurement.

---

# Bonus: Using Estimated Pose to Make a Robot Reach the ArUco Marker

Once the system can estimate the pose of an ArUco marker, the next step is using this information to move a robot toward it.

This section explains the concept at a high level.

---

## 1. What the Script Provides

From `aruco_pose.py`, we obtain:

- Marker pose relative to camera
  OR
- Camera pose relative to marker (after inversion)

In both cases, we have a 3D transform between:

```
Camera frame  <->  Marker frame
```

This is typically represented as:

```
T_camera_marker
```

(4x4 transformation matrix)

---

## 2. What the Robot Needs

Robots do not move in the camera frame.

They move in their own coordinate frame, usually:

```
base_link
```

Therefore, we must know the fixed transform between:

```
base_link  <->  camera_link
```

This is called **extrinsic calibration**.

Without this transform, the robot cannot understand where the marker is in its own coordinate system.

---

## 3. Converting Marker Pose to Robot Frame

If we know:

- `T_base_camera`
- `T_camera_marker`

Then:

```
T_base_marker = T_base_camera × T_camera_marker
```

Now the robot knows the marker position in its base frame.

This is the key step.

---

## 4. Defining a Goal Position Near the Marker

Usually we do not want the robot to collide with the marker.

We define a small offset, for example:

- Stop 30 cm in front of marker

In marker frame:

```
T_marker_goal = translation(0, 0, 0.30)
```

Then:

```
T_base_goal = T_base_marker × T_marker_goal
```

This gives the robot a safe target pose.

---

## 5. Using This With Different Robot Types

### A) Mobile Robot (Navigation)

Use:

```
x, y = T_base_marker position
yaw  = rotation toward marker
```

Send as goal to:

- Nav2 (ROS2 navigation stack)

The robot drives toward the marker.

---

### B) Robotic Arm (Manipulator)

Use:

```
T_base_goal
```

as target pose in MoveIt.

The arm plans a motion and reaches the marker.

---

### C) Humanoid / Legged Robot

Use marker pose to:

- Walk toward marker
- Align orientation
- Stop at defined offset

---

## 6. Recommended ROS2 TF Setup

Best practice in ROS2 is to publish transforms using `tf2`.

Suggested TF tree:

```
map
 └── base_link
      └── camera_link
           └── aruco_marker_0
```

Publish:

- Static transform: `base_link -> camera_link`
- Dynamic transform: `camera_link -> aruco_marker_X`

Then ROS2 automatically allows you to query:

```
base_link -> aruco_marker_0
```

Example:

```bash
ros2 run tf2_ros tf2_echo base_link aruco_marker_0
```

---

## 7. Important Notes

- Calibration RMSE should be < 0.5 px
- Marker length must be measured accurately
- Camera resolution must match calibration
- Extrinsic transform (base to camera) must be correct
- Frame direction mistakes are the most common source of error

---

## 8. Conceptual Summary

1. Detect marker in camera frame
2. Convert to robot base frame
3. Add safety offset
4. Send target pose to controller
5. Robot reaches marker

This completes the perception → transformation → control pipeline.

---

