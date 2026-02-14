#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
aruco_pose.py  (Python 3.8+)

Real-time ArUco marker detection and camera pose estimation.

Loads: camera_calib.npz
Outputs: camera position relative to marker center (meters)

Supports:
  - OpenCV camera (--source cam)
  - ROS2 image topic (--source ros)

Dependencies:
  pip install numpy opencv-contrib-python

Optional (ROS2 support):
  sudo apt install ros-jazzy-cv-bridge
  source /opt/ros/jazzy/setup.bash

Usage examples:

  # OpenCV camera
  python3 aruco_pose.py --source cam --calib_file camera_calib.npz \
      --aruco_dict 4x4_50 --marker_length 0.05 --target_id 0 --draw_axis

  # ROS2 topic
  python3 aruco_pose.py --source ros --topic /image_raw \
      --calib_file camera_calib.npz --aruco_dict 4x4_50 --marker_length 0.05

Controls:
  q → quit

Author: Arshad Mehmood
"""

import argparse
import os
import numpy as np
import cv2


_ARUCO_DICT_MAP = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
}


def ros2_available() -> bool:
    try:
        import rclpy  # noqa
        from sensor_msgs.msg import Image  # noqa
        from cv_bridge import CvBridge  # noqa
        return True
    except Exception:
        return False


def invert_pose(rvec, tvec):
    """marker-in-camera -> camera-in-marker"""
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec.reshape(3, 1)
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv.reshape(3, 1), t_inv.reshape(3, 1)


def marker_obj_points(marker_length_m: float) -> np.ndarray:
    """4 corners in marker frame with origin at marker center."""
    h = marker_length_m / 2.0
    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float32)


def make_detector(aruco_dict_name: str):
    key = aruco_dict_name.lower()
    if key not in _ARUCO_DICT_MAP:
        raise ValueError(f"Unknown --aruco_dict '{aruco_dict_name}'. Try e.g. 4x4_50, 5x5_100, 6x6_250.")
    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_MAP[key])
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def load_calib(calib_file: str):
    data = np.load(calib_file)
    return data["K"], data["dist"]


def process_frame(frame_bgr, detector, K, dist, objp, args):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    vis = frame_bgr.copy()

    if ids is None or len(ids) == 0:
        cv2.putText(vis, "No ArUco detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return vis, []

    ids_flat = ids.flatten()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    outputs = []
    line = 0

    for i, mid in enumerate(ids_flat):
        if args.target_id >= 0 and int(mid) != args.target_id:
            continue

        imgp = corners[i].reshape(-1, 2).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue

        if args.draw_axis:
            cv2.drawFrameAxes(vis, K, dist, rvec, tvec, args.marker_length * 0.6)

        cam_rvec, cam_tvec = invert_pose(rvec, tvec)
        x, y, z = cam_tvec.ravel()

        txt = f"ID {int(mid)} CamPos[m]: x={x:+.3f} y={y:+.3f} z={z:+.3f}"
        cv2.putText(vis, txt, (10, 30 + 25 * line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        outputs.append((int(mid), float(x), float(y), float(z)))
        line += 1
        if line >= 8:
            break

    return vis, outputs


def run_cam(args):
    print("Source: cam")
    K, dist = load_calib(args.calib_file)
    detector = make_detector(args.aruco_dict)
    objp = marker_obj_points(args.marker_length)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Controls: q = quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera read failed.")
            break

        vis, outputs = process_frame(frame, detector, K, dist, objp, args)
        if args.print_console:
            for mid, x, y, z in outputs:
                print(f"ID {mid} CamPos[m]: x={x:+.3f} y={y:+.3f} z={z:+.3f}")

        cv2.imshow("Aruco Pose (cam)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_ros(args):
    # imports inside so script runs without ROS installed
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    K, dist = load_calib(args.calib_file)
    detector = make_detector(args.aruco_dict)
    objp = marker_obj_points(args.marker_length)

    class RosArucoNode(Node):
        def __init__(self):
            super().__init__("ros_aruco_pose")
            self.bridge = CvBridge()
            self.quit_now = False
            self.sub = self.create_subscription(Image, args.topic, self.cb, 10)
            self.get_logger().info(f"Subscribed to: {args.topic}")
            self.get_logger().info("Controls: q = quit")

        def cb(self, msg: Image):
            if self.quit_now:
                return
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                self.get_logger().warn(f"cv_bridge conversion failed: {e}")
                return

            vis, outputs = process_frame(frame, detector, K, dist, objp, args)

            if args.print_console:
                for mid, x, y, z in outputs:
                    self.get_logger().info(f"ID {mid} CamPos[m]: x={x:+.3f} y={y:+.3f} z={z:+.3f}")

            cv2.imshow("Aruco Pose (ROS)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.quit_now = True

    rclpy.init()
    node = RosArucoNode()

    while rclpy.ok() and not node.quit_now:
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cam", "ros"], required=True)

    ap.add_argument("--calib_file", default="camera_calib.npz")
    ap.add_argument("--aruco_dict", default="4x4_50")
    ap.add_argument("--marker_length", type=float, required=True, help="marker side length in meters, e.g. 0.05")
    ap.add_argument("--target_id", type=int, default=-1, help="-1 = all markers")
    ap.add_argument("--draw_axis", action="store_true")
    ap.add_argument("--print_console", action="store_true")

    # camera source
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)

    # ROS source
    ap.add_argument("--topic", default="/camera/image_raw")
    ap.add_argument("--strict_ros", action="store_true",
                    help="If set and ROS2 is missing, exit instead of falling back to camera.")

    args = ap.parse_args()

    if (args.width == 0) != (args.height == 0):
        raise ValueError("Set both --width and --height, or neither.")

    if not os.path.exists(args.calib_file):
        raise FileNotFoundError(f"Calibration file not found: {args.calib_file}")

    if args.source == "ros":
        if ros2_available():
            print("Source: ros")
            print("Loaded calib:", os.path.abspath(args.calib_file))
            run_ros(args)
            return

        msg = "ROS2 not available (rclpy/cv_bridge/sensor_msgs not found)."
        if args.strict_ros:
            raise RuntimeError(msg + " Install ROS2 + cv_bridge, or run with --source cam.")
        print(msg + " Falling back to OpenCV camera source (--source cam).")
        run_cam(args)
        return

    # source == cam
    print("Source: cam")
    print("Loaded calib:", os.path.abspath(args.calib_file))
    run_cam(args)


if __name__ == "__main__":
    main()
