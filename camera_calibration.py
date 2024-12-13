import numpy as np
import cv2
import glob
import os

def calibrate_camera():
    # Checkerboard dimensions
    CHECKERBOARD = (6, 9)  # Adjust these numbers based on your checkerboard
    
    # Stop the iteration when specified accuracy is reached
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Create directory for calibration images
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')
    
    img_count = 0
    saved_count = 0
    
    print("Press 'c' to capture an image, 'q' to quit and calculate parameters")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_count += 1
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        # If found, add object points, image points
        if ret:
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
            
            # Show info on frame
            cv2.putText(frame, f"Found checkerboard! Press 'c' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and ret:
            # Save the image
            img_name = f'calibration_images/calibration_{saved_count}.jpg'
            cv2.imwrite(img_name, frame)
            saved_count += 1
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            print(f"Saved {saved_count} images")
            
            if saved_count >= 15:  # We need at least 15 good images
                print("Sufficient images captured. Press 'q' to calculate parameters.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if saved_count < 5:
        print("Not enough images captured for calibration!")
        return
    
    print("Calculating camera parameters...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                      gray.shape[::-1], None, None)
    
    print("\nCamera matrix:")
    print(mtx)
    print("\nDistortion coefficients:")
    print(dist)
    
    # Save the camera calibration results
    np.savez('camera_calibration.npz', 
             camera_matrix=mtx, 
             dist_coeffs=dist)
    
    print("\nCalibration saved to 'camera_calibration.npz'")
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    print(f"\nTotal reprojection error: {mean_error/len(objpoints)}")

if __name__ == "__main__":
    calibrate_camera() 