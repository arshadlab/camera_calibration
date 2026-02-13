import argparse
import os
import time
import glob
import cv2
import numpy as np


# ----------------------- Utilities -----------------------

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
        raise RuntimeError(f"Not enough valid chessboard detections: {used}. Need at least ~8 (better 15-30).")

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


def show_calibration_summary(result, save_calib_path):
    print("\n=== Calibration Result ===")
    print(f"Images used: {result['used']} / {result['total']}")
    print("Calibration RMS:", result["rms"])
    print("Reprojection RMSE (pixels):", result["rmse_px"])
    print("\nCamera Matrix (K):\n", result["K"])
    print("\nDistortion Coefficients:\n", result["dist"].ravel())
    print("\nSaved calibration to:", os.path.abspath(save_calib_path))

    # On-screen summary (press any key)
    summary_img = np.zeros((520, 980, 3), dtype=np.uint8)

    y = 45
    ls = 34

    cv2.putText(summary_img, "Calibration Completed", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    y += ls * 2

    cv2.putText(summary_img, f"Images used: {result['used']} / {result['total']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += ls
    cv2.putText(summary_img, f"RMS: {result['rms']:.4f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += ls
    cv2.putText(summary_img, f"RMSE(px): {result['rmse_px']:.4f}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += ls * 2

    cv2.putText(summary_img, "Camera Matrix (K):", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y += ls

    for row in result["K"]:
        txt = "  " + "  ".join([f"{v:8.3f}" for v in row])
        cv2.putText(summary_img, txt, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += ls

    y += ls
    cv2.putText(summary_img, "Distortion:", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y += ls

    dist_flat = result["dist"].ravel()
    txt = "  " + "  ".join([f"{v:.5f}" for v in dist_flat])
    cv2.putText(summary_img, txt, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += ls * 2
    cv2.putText(summary_img, "Press any key to exit", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

    cv2.imshow("Calibration Result", summary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------- Modes: Calibration -----------------------

def run_camera_calib_mode(args, chessboard_size):
    """
    Camera calibration capture:
    - Auto-save when chessboard detected (default), or manual save with --manual + 'c'
    - Press 'q' to stop and calibrate on captured session images
    """
    session_dir = os.path.abspath(os.path.join(args.out, time.strftime("%Y%m%d_%H%M%S")))
    ensure_dir(session_dir)

    print("Mode: camera_calib")
    print("Saving images into:", session_dir)
    if args.manual:
        print("Manual capture: press 'c' to save (only if chessboard detected).")
    else:
        print(f"Auto capture: saving automatically when chessboard detected (cooldown={args.cooldown:.2f}s).")
    print("Press 'q' to stop and run calibration.\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    # Force fixed resolution if provided
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    saved_paths = []
    saved_count = 0
    last_save_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera read failed.")
            break

        found, corners2, _ = detect_corners(frame, chessboard_size, criteria)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, chessboard_size, corners2, True)

        now = time.time()

        # Auto-save
        if (not args.manual) and found and (now - last_save_t) >= args.cooldown:
            fname = os.path.join(session_dir, f"calib_{saved_count:03d}_{int(now*1000)}.jpg")
            if cv2.imwrite(fname, frame):
                saved_paths.append(fname)
                saved_count += 1
                last_save_t = now
                print("Saved:", fname)

        status = f"Saved: {saved_count} | Chessboard: {'OK' if found else 'NO'}"
        status += " | q=quit+calib"
        if args.manual:
            status += " | c=save"
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Capture (calib)", vis)

        key = cv2.waitKey(1) & 0xFF

        # Manual save
        if args.manual and key == ord('c'):
            if not found:
                print("Not saved: chessboard not detected.")
            else:
                fname = os.path.join(session_dir, f"calib_{saved_count:03d}_{int(time.time()*1000)}.jpg")
                if cv2.imwrite(fname, frame):
                    saved_paths.append(fname)
                    saved_count += 1
                    print("Saved:", fname)

        if key == ord('q') or key == 27:
            break

        if args.min_samples > 0 and saved_count >= args.min_samples and (not args.manual):
            print("Reached min_samples. Stopping capture and calibrating...")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(saved_paths) == 0:
        raise RuntimeError("No images saved. Ensure chessboard is visible and detected.")

    print(f"\nCapture ended. Saved {len(saved_paths)} images.")
    print("Running calibration on captured images...")

    result = calibrate_from_paths(saved_paths, chessboard_size, args.square, show=args.show)
    np.savez(args.save_calib, K=result["K"], dist=result["dist"], img_size=result["img_size"])

    show_calibration_summary(result, args.save_calib)
    print("Captured images folder:", session_dir)


def run_images_calib_mode(args, chessboard_size):
    print("Mode: images_calib")
    paths = sorted(glob.glob(args.pattern, recursive=True))
    if not paths:
        raise RuntimeError(f"No images found for pattern: {args.pattern}")
    print(f"Found {len(paths)} images. Calibrating...")

    result = calibrate_from_paths(paths, chessboard_size, args.square, show=args.show)
    np.savez(args.save_calib, K=result["K"], dist=result["dist"], img_size=result["img_size"])

    show_calibration_summary(result, args.save_calib)


# ----------------------- Mode: ArUco pose -----------------------

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


def invert_pose(rvec, tvec):
    """
    Given pose of marker in camera frame (rvec,tvec),
    return pose of camera in marker frame.
    """
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t_inv = -R_inv @ tvec.reshape(3, 1)
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv.reshape(3, 1), t_inv.reshape(3, 1)


def run_aruco_mode(args):
    print("Mode: aruco")
    print("Loading calibration from:", os.path.abspath(args.calib_file))

    data = np.load(args.calib_file)
    K = data["K"]
    dist = data["dist"]

    dict_key = args.aruco_dict.lower()
    if dict_key not in _ARUCO_DICT_MAP:
        raise ValueError(f"Unknown --aruco_dict '{args.aruco_dict}'. Try e.g. 4x4_50, 5x5_100, 6x6_250 ...")

    aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_MAP[dict_key])
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    # Force fixed resolution if provided
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Controls: q = quit")
    if args.target_id >= 0:
        print(f"Tracking target marker ID: {args.target_id}")
    else:
        print("Tracking: all detected markers")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        vis = frame.copy()

        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            rvecs = []
            tvecs = []

            # Define 3D marker corners in marker coordinate frame
            half_len = args.marker_length / 2.0
            obj_points = np.array([
                [-half_len,  half_len, 0],
                [ half_len,  half_len, 0],
                [ half_len, -half_len, 0],
                [-half_len, -half_len, 0],
            ], dtype=np.float32)

            for corner in corners:
                img_points = corner.reshape(-1, 2).astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    img_points,
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)

            line_y = 30
            lines = 0

            for i, mid in enumerate(ids_flat):
                if args.target_id >= 0 and int(mid) != args.target_id:
                    continue

                rvec = rvecs[i].reshape(3, 1)
                tvec = tvecs[i].reshape(3, 1)

                # Draw marker axis (marker frame)
                if args.draw_axis:
                    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, args.marker_length * 0.6)

                # Camera pose in marker frame
                cam_rvec, cam_tvec = invert_pose(rvec, tvec)
                x, y, z = cam_tvec.ravel()  # meters if marker_length is meters

                txt = f"ID {int(mid)} CamPos[m]: x={x:+.3f} y={y:+.3f} z={z:+.3f}"
                cv2.putText(vis, txt, (10, line_y + 25 * lines),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                lines += 1
                if lines >= 6:
                    break  # avoid too much text

        else:
            cv2.putText(vis, "No ArUco detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Aruco Pose", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["camera_calib", "images_calib", "aruco"], required=True)

    # Chessboard calib params
    ap.add_argument("--cols", type=int, default=9, help="chessboard inner corners columns")
    ap.add_argument("--rows", type=int, default=6, help="chessboard inner corners rows")
    ap.add_argument("--square", type=float, default=0.024, help="square size in meters for calibration")

    # Camera / IO
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=0, help="force camera width (0 = don't force)")
    ap.add_argument("--height", type=int, default=0, help="force camera height (0 = don't force)")
    ap.add_argument("--out", default="calib_images", help="base output directory for camera_calib mode")
    ap.add_argument("--pattern", default="calib_images/**/*.jpg", help="glob pattern for images_calib mode (use quotes)")
    ap.add_argument("--show", action="store_true", help="show detected corners during calibration")
    ap.add_argument("--save_calib", default="camera_calib.npz", help="where to save calibration output")

    # Capture control
    ap.add_argument("--cooldown", type=float, default=0.8, help="auto-save cooldown seconds (camera_calib)")
    ap.add_argument("--min_samples", type=int, default=0, help="optional auto stop after N saves (camera_calib)")
    ap.add_argument("--manual", action="store_true",
                    help="manual save (press 'c') instead of auto-save (camera_calib)")

    # ArUco params
    ap.add_argument("--calib_file", default="camera_calib.npz", help="npz file with K/dist (aruco mode)")
    ap.add_argument("--aruco_dict", default="4x4_50", help="e.g. 4x4_50, 5x5_100, 6x6_250 ...")
    ap.add_argument("--marker_length", type=float, default=0.05, help="ArUco marker side length in meters")
    ap.add_argument("--target_id", type=int, default=-1, help="track only this marker id (-1 = all)")
    ap.add_argument("--draw_axis", action="store_true", help="draw marker axis (aruco mode)")

    args = ap.parse_args()

    # Normalize width/height forcing
    if (args.width == 0) != (args.height == 0):
        raise ValueError("Set both --width and --height, or neither.")

    chessboard_size = (args.cols, args.rows)

    if args.mode == "camera_calib":
        run_camera_calib_mode(args, chessboard_size)
    elif args.mode == "images_calib":
        run_images_calib_mode(args, chessboard_size)
    else:
        run_aruco_mode(args)


if __name__ == "__main__":
    main()
