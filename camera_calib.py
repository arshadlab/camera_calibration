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
camera_calib.py  (Python 3.8+)

Chessboard-based camera calibration tool.

Supports:
  - OpenCV camera (--source cam)
  - ROS2 image topic (--source ros)
  - Image files (--source images)

Saves calibration to: camera_calib.npz

Dependencies:
  pip install numpy opencv-contrib-python

Optional (ROS2 support):
  sudo apt install ros-jazzy-cv-bridge
  source /opt/ros/jazzy/setup.bash

Usage examples:

  # OpenCV camera (manual save - default)
  python3 camera_calib.py --source cam --cols 9 --rows 6 --square 0.024

  # ROS2 topic (manual save)
  python3 camera_calib.py --source ros --topic /image_raw --cols 9 --rows 6 --square 0.024

  # OpenCV camera (auto-save mode)
  python3 camera_calib.py --source cam --cols 9 --rows 6 --square 0.024 --auto

  # From images
  python3 camera_calib.py --source images --pattern "calib_images/<date>/*.jpg" --cols 9 --rows 6 --square 0.024

Controls:
  c → save frame (default manual mode)
  q → stop capture and calibrate

Recommended: RMSE < 0.5 px
Author: Arshad Mehmood
"""

import argparse
import glob
import os
import time
import numpy as np
import cv2


# -------------------- Common calibration helpers --------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def make_objp(chessboard_size, square_size):
    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def compute_reprojection_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error2 = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        total_error2 += err * err
        total_points += len(proj)
    return float(np.sqrt(total_error2 / total_points)) if total_points else float("nan")


def detect_corners(img, chessboard_size, criteria):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if not found:
        return False, None, gray.shape[::-1]
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners2, gray.shape[::-1]


def calibrate_from_paths(image_paths, chessboard_size, square_size, show=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    objp = make_objp(chessboard_size, square_size)

    objpoints = []
    imgpoints = []
    img_size = None
    used = 0

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        found, corners2, size = detect_corners(img, chessboard_size, criteria)
        img_size = size
        if not found:
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        used += 1

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, chessboard_size, corners2, True)
            cv2.putText(vis, os.path.basename(p), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration images", vis)
            cv2.waitKey(30)

    if show:
        cv2.destroyAllWindows()

    if used < 8:
        raise RuntimeError(
            f"Not enough valid chessboard detections: {used}. Need at least ~8 (better 15-30)."
        )

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    rmse = compute_reprojection_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist)

    return {
        "used": used,
        "total": len(image_paths),
        "img_size": img_size,
        "rms": float(ret),
        "rmse_px": float(rmse),
        "K": K,
        "dist": dist,
    }


def show_summary(result, save_path):
    print("\n=== Calibration Result ===")
    print(f"Images used: {result['used']} / {result['total']}")
    print("Calibration RMS:", result["rms"])
    print("Reprojection RMSE (pixels):", result["rmse_px"])
    print("\nCamera Matrix (K):\n", result["K"])
    print("\nDistortion Coefficients:\n", result["dist"].ravel())
    print("\nSaved calibration to:", os.path.abspath(save_path))

    # simple on-screen summary
    img = np.zeros((520, 980, 3), dtype=np.uint8)
    y, ls = 45, 34
    cv2.putText(img, "Calibration Completed", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    y += ls * 2
    cv2.putText(img, f"Images used: {result['used']} / {result['total']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += ls
    cv2.putText(img, f"RMSE(px): {result['rmse_px']:.4f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += ls * 2
    cv2.putText(img, "Press any key to exit", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.imshow("Calibration Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------- Frame saving logic (shared) --------------------

class SaverState:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.saved_paths = []
        self.saved_count = 0
        self.last_save_t = 0.0

    def try_save(self, frame, prefix="calib"):
        fname = os.path.join(self.session_dir, f"{prefix}_{self.saved_count:03d}_{int(time.time()*1000)}.jpg")
        if cv2.imwrite(fname, frame):
            self.saved_paths.append(fname)
            self.saved_count += 1
            return fname
        return None


def process_frame_for_capture(frame, chessboard_size, criteria, args, saver: SaverState, window_title: str):
    found, corners2, _ = detect_corners(frame, chessboard_size, criteria)

    vis = frame.copy()
    if found:
        cv2.drawChessboardCorners(vis, chessboard_size, corners2, True)

    now = time.time()

    # Auto-save
    if args.auto and found and (now - saver.last_save_t) >= args.cooldown:
        saved = saver.try_save(frame)
        if saved:
            saver.last_save_t = now
            print("Saved:", saved)

    status = f"Saved: {saver.saved_count} | Chessboard: {'OK' if found else 'NO'} | q=quit+calib"
    if not args.auto:
        status += " | c=save"
    cv2.putText(vis, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imshow(window_title, vis)
    key = cv2.waitKey(1) & 0xFF

    # Manual save
    if (not args.auto) and key == ord('c'):
        if not found:
            print("Not saved: chessboard not detected.")
        else:
            saved = saver.try_save(frame)
            if saved:
                print("Saved:", saved)

    # Quit
    if key == ord('q') or key == 27:
        return True  # quit

    # Optional auto-stop
    if args.min_samples > 0 and saver.saved_count >= args.min_samples and args.auto:
        print("Reached min_samples. Stopping capture and calibrating...")
        return True

    return False


# -------------------- Sources: images / camera / ROS2 --------------------

def run_images(args, chessboard_size):
    paths = sorted(glob.glob(args.pattern, recursive=True))
    if not paths:
        raise RuntimeError(f"No images found for pattern: {args.pattern}")

    print("Source: images")
    print(f"Found {len(paths)} images. Calibrating...")
    result = calibrate_from_paths(paths, chessboard_size, args.square, show=args.show)
    np.savez(args.save_calib, K=result["K"], dist=result["dist"], img_size=result["img_size"])
    show_summary(result, args.save_calib)


def run_camera(args, chessboard_size):
    session_dir = os.path.abspath(os.path.join(args.out, time.strftime("%Y%m%d_%H%M%S")))
    ensure_dir(session_dir)
    saver = SaverState(session_dir)

    print("Source: cam")
    print("Saving into:", session_dir)
    print(f"Auto capture: save when chessboard detected (cooldown={args.cooldown:.2f}s)" if args.auto
          else "Manual capture: press 'c' to save (only if chessboard detected).")
    print("Press 'q' to stop and calibrate.\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    quit_now = False
    while not quit_now:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera read failed.")
            break

        quit_now = process_frame_for_capture(
            frame, chessboard_size, criteria, args, saver, window_title="Capture (cam calib)"
        )

    cap.release()
    cv2.destroyAllWindows()

    if len(saver.saved_paths) == 0:
        raise RuntimeError("No images saved. Ensure chessboard is visible and detected.")

    print(f"\nCapture ended. Saved {len(saver.saved_paths)} images.")
    print("Running calibration on captured images...")

    result = calibrate_from_paths(saver.saved_paths, chessboard_size, args.square, show=args.show)
    np.savez(args.save_calib, K=result["K"], dist=result["dist"], img_size=result["img_size"])
    show_summary(result, args.save_calib)
    print("Captured images folder:", session_dir)


def ros2_available():
    try:
        import rclpy  # noqa
        from sensor_msgs.msg import Image  # noqa
        from cv_bridge import CvBridge  # noqa
        return True
    except Exception:
        return False


def run_ros(args, chessboard_size):
    # imports inside so script runs without ROS installed
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    class RosNode(Node):
        def __init__(self):
            super().__init__("ros_chess_calib")
            self.bridge = CvBridge()
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            self.quit_now = False

            self.session_dir = os.path.abspath(os.path.join(args.out, time.strftime("%Y%m%d_%H%M%S")))
            ensure_dir(self.session_dir)
            self.saver = SaverState(self.session_dir)

            self.sub = self.create_subscription(Image, args.topic, self.cb, 10)

            self.get_logger().info(f"Source: ros")
            self.get_logger().info(f"Subscribed to: {args.topic}")
            self.get_logger().info(f"Saving into: {self.session_dir}")
            if args.auto:
                self.get_logger().info(f"Auto capture: save when chessboard detected (cooldown={args.cooldown:.2f}s)")
            else:
                self.get_logger().info("Manual capture: press 'c' to save (only if chessboard detected).")
            self.get_logger().info("Press 'q' to stop and calibrate.")

        def cb(self, msg: Image):
            if self.quit_now:
                return
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                self.get_logger().warn(f"cv_bridge conversion failed: {e}")
                return

            self.quit_now = process_frame_for_capture(
                frame, chessboard_size, self.criteria, args, self.saver, window_title="Capture (ROS calib)"
            )

    rclpy.init()
    node = RosNode()

    while rclpy.ok() and not node.quit_now:
        rclpy.spin_once(node, timeout_sec=0.05)

    saved_paths = list(node.saver.saved_paths)
    session_dir = node.session_dir

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

    if len(saved_paths) == 0:
        raise RuntimeError("No images saved. Ensure chessboard is visible and detected.")

    print(f"\nCapture ended. Saved {len(saved_paths)} images.")
    print("Running calibration on captured images...")

    result = calibrate_from_paths(saved_paths, chessboard_size, args.square, show=args.show)
    np.savez(args.save_calib, K=result["K"], dist=result["dist"], img_size=result["img_size"])
    show_summary(result, args.save_calib)
    print("Captured images folder:", session_dir)


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cam", "images", "ros"], required=True,
                    help="Input source: cam (OpenCV), images (glob), ros (ROS2 topic).")

    # Chessboard params
    ap.add_argument("--cols", type=int, default=9, help="inner corners columns")
    ap.add_argument("--rows", type=int, default=6, help="inner corners rows")
    ap.add_argument("--square", type=float, default=0.024, help="square size in meters")

    # Capture behavior
    ap.add_argument("--auto", action="store_true", help="auto-save when chessboard detected (else manual save with 'c')")
    ap.add_argument("--cooldown", type=float, default=0.8, help="auto-save cooldown seconds")
    ap.add_argument("--min_samples", type=int, default=0, help="auto-stop after N saves in auto mode")

    # Output
    ap.add_argument("--out", default="calib_images", help="base output folder for captured images")
    ap.add_argument("--save_calib", default="camera_calib.npz", help="output npz calibration file")
    ap.add_argument("--show", action="store_true", help="show corners during calibration")

    # Camera source args
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=0, help="force width (optional)")
    ap.add_argument("--height", type=int, default=0, help="force height (optional)")

    # Images source args
    ap.add_argument("--pattern", default="calib_images/**/*.jpg", help="glob pattern for images source")

    # ROS source args
    ap.add_argument("--topic", default="/camera/image_raw", help="ROS2 Image topic")
    ap.add_argument("--strict_ros", action="store_true",
                    help="If set and ROS2 is missing, exit instead of falling back to camera.")

    args = ap.parse_args()

    if (args.width == 0) != (args.height == 0):
        raise ValueError("Set both --width and --height, or neither.")

    chessboard_size = (args.cols, args.rows)

    if args.source == "images":
        run_images(args, chessboard_size)
        return

    if args.source == "ros":
        if ros2_available():
            run_ros(args, chessboard_size)
            return
        msg = "ROS2 not available (rclpy/cv_bridge/sensor_msgs not found)."
        if args.strict_ros:
            raise RuntimeError(msg + " Install ROS2 + cv_bridge, or run with --source cam.")
        print(msg + " Falling back to OpenCV camera source (--source cam).")
        run_camera(args, chessboard_size)
        return

    # source == cam
    run_camera(args, chessboard_size)


if __name__ == "__main__":
    main()
