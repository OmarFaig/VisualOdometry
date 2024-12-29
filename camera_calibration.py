import numpy as np
import cv2
import os
import glob

def generate_synthetic_checkerboard_images(num_images=15, size=(9, 9), square_size=50, image_size=(720, 1280)):
    """
    Generate multiple synthetic checkerboard images from different perspectives.

    Args:
        num_images (int): Number of images to generate.
        size (tuple): Number of inner corners per row and column (9x9).
        square_size (int): Size of each square in pixels.
        image_size (tuple): Size of the image (height, width).

    Returns:
        list: List of synthetic checkerboard images.
    """
    rows, cols = size
    board_width = cols * square_size
    board_height = rows * square_size
    checkerboard = np.zeros((board_height, board_width), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                cv2.rectangle(checkerboard,
                              (j * square_size, i * square_size),
                              ((j + 1) * square_size, (i + 1) * square_size),
                              255, -1)

    images = []
    for i in range(num_images):
        rotation_angle = np.random.uniform(-20, 20)  # Random rotation angle in degrees
        translation_x = np.random.uniform(-100, 100)  # Random translation in pixels
        translation_y = np.random.uniform(-100, 100)  # Random translation in pixels

        rotation_matrix = cv2.getRotationMatrix2D((board_width // 2, board_height // 2), rotation_angle, 1)
        translated_board = cv2.warpAffine(checkerboard, rotation_matrix, (board_width, board_height))
        transformed = cv2.warpAffine(translated_board, np.float32([[1, 0, translation_x], [0, 1, translation_y]]), 
                                      (image_size[1], image_size[0]))

        padded_board = cv2.copyMakeBorder(
            transformed,
            max(0, (image_size[0] - board_height) // 2),
            max(0, (image_size[0] - board_height) // 2),
            max(0, (image_size[1] - board_width) // 2),
            max(0, (image_size[1] - board_width) // 2),
            cv2.BORDER_CONSTANT,
            value=0
        )
        images.append(padded_board)

    return images

def calibrate_camera(use_synthetic=False):
    CHECKERBOARD = (9, 9)  # Number of inner corners per row and column
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    if use_synthetic:
        synthetic_images = generate_synthetic_checkerboard_images(num_images=15, size=CHECKERBOARD)
        if not os.path.exists('synthetic_images'):
            os.makedirs('synthetic_images')

        for i, img in enumerate(synthetic_images):
            filename = f"synthetic_images/synthetic_checkerboard_{i}.jpg"
            cv2.imwrite(filename, img)
        print("Generated synthetic checkerboard images for calibration.")
        images = [f"synthetic_images/synthetic_checkerboard_{i}.jpg" for i in range(len(synthetic_images))]

    else:
        print("Using physical checkerboard for calibration.")
        cap = cv2.VideoCapture(0)
        if not os.path.exists('calibration_images'):
            os.makedirs('calibration_images')

        print("Press 'c' to capture an image, 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret:
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
                cv2.imshow('Calibration', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    filename = f'calibration_images/calibration_{len(objpoints)}.jpg'
                    cv2.imwrite(filename, frame)
                    objpoints.append(objp)
                    imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
                    print(f"Captured image {len(objpoints)}")
                elif key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        images = glob.glob('calibration_images/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))

    if len(objpoints) < 5:
        print("Not enough valid images for calibration.")
        return

    print("Calculating camera parameters...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    np.savez('camera_calibration.npz', camera_matrix=mtx, dist_coeffs=dist)
    print("Calibration saved to 'camera_calibration.npz'")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Reprojection error: {mean_error / len(objpoints)}")

if __name__ == "__main__":
    use_synthetic = input("Use synthetic checkerboard? (y/n): ").strip().lower() == 'y'
    calibrate_camera(use_synthetic=use_synthetic)
