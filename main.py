import os
import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

class MonocularSlam:
    def __init__(self, camera_matrix):
        """
        Initialize Monocular SLAM system
        Args:
            camera_matrix: 3x3 intrinsic camera matrix
        """
        self.K = camera_matrix
        
        # Initialize feature detector (SIFT works better than ORB for most cases)
        self.feature_detector = cv2.SIFT_create(nfeatures=2000)
        
        # Initialize feature matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Parameters
        self.min_matches = 100
        self.min_parallax = 2.0  # degrees
        self.max_reprojection_error = 4.0
        self.min_distance = 0.05
        
        # Initialize storage for map and trajectory
        self.keyframes = []
        self.poses = []
        self.points_3d = []
        self.current_pose = np.eye(4)
        
        # Add scale estimation parameters
        self.min_triangulation_angle = 2.0  # degrees
        self.max_point_distance = 30.0  # meters
        self.scale_factor = 1.0
        self.last_keyframe_points = None
        self.scale_window = []  # Keep track of recent scales
        
    def process_frame(self, frame):
        """Process a new frame and update the map"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # If this is the first frame
        if not self.keyframes:
            self.keyframes.append((gray, self.current_pose))
            self.poses.append(self.current_pose)
            return True
        
        # Match with last keyframe
        success = self._process_new_frame(gray)
        return success
    
    def _process_new_frame(self, current_frame):
        """Match current frame with last keyframe and update pose"""
        last_frame = self.keyframes[-1][0]
        
        # Detect and match features
        pts1, pts2 = self._match_frames(last_frame, current_frame)
        
        if pts1 is None or len(pts1) < self.min_matches:
            return False
            
        # Normalize points
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K, None)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K, None)
        
        # Estimate essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0., 0.),
                                      method=cv2.RANSAC, prob=0.999, threshold=0.001)
        
        if E is None:
            return False
        
        # Apply mask to points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1_norm[mask.ravel() == 1], 
                                       pts2_norm[mask.ravel() == 1])
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        
        # Triangulate points and estimate scale
        points_3d = self._triangulate_points(pts1, pts2, self.keyframes[-1][1], T)
        
        if len(points_3d) > 0:
            if self.last_keyframe_points is not None:
                # Estimate scale from point cloud distances
                scale = self._estimate_scale(points_3d, self.last_keyframe_points)
                if scale > 0:
                    self.scale_factor *= scale
                    T[:3, 3] *= scale
            
            self.last_keyframe_points = points_3d
        
        # Update current pose
        self.current_pose = self.current_pose @ T
        
        # Add keyframe if we've moved enough
        if self._should_add_keyframe(self.keyframes[-1][1], self.current_pose):
            self.keyframes.append((current_frame, self.current_pose.copy()))
            self.poses.append(self.current_pose.copy())
            if len(points_3d) > 0:
                self.points_3d.extend(points_3d.tolist())
        
        return True
    
    def _match_frames(self, frame1, frame2):
        """Match features between two frames"""
        # Detect keypoints and compute descriptors
        kp1, des1 = self.feature_detector.detectAndCompute(frame1, None)
        kp2, des2 = self.feature_detector.detectAndCompute(frame2, None)
        
        if des1 is None or des2 is None:
            return None, None
            
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) < self.min_matches:
            return None, None
            
        # Extract matched point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2
    
    def _triangulate_points(self, pts1, pts2, pose1, pose2):
        """Triangulate 3D points with filtering"""
        try:
            # Create projection matrices
            P1 = np.hstack((pose1[:3, :3], pose1[:3, 3].reshape(3,1)))
            P2 = np.hstack((pose2[:3, :3], pose2[:3, 3].reshape(3,1)))
            
            # Triangulate points
            points_4d = cv2.triangulatePoints(self.K @ P1, self.K @ P2, pts1.T, pts2.T)
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Initial filtering: remove points at infinity or behind cameras
            valid_points = np.ones(len(points_3d), dtype=bool)
            valid_points &= ~np.isinf(points_3d).any(axis=1)
            valid_points &= ~np.isnan(points_3d).any(axis=1)
            valid_points &= points_3d[:, 2] > 0  # Points in front of both cameras
            
            if not valid_points.any():
                return np.array([])
            
            points_3d = points_3d[valid_points]
            pts1 = pts1[valid_points]
            pts2 = pts2[valid_points]
            
            # Calculate triangulation angles
            rays1 = pts1 - np.array([self.K[0,2], self.K[1,2]])
            rays2 = pts2 - np.array([self.K[0,2], self.K[1,2]])
            rays1 = rays1 / (np.linalg.norm(rays1, axis=1)[:, np.newaxis] + 1e-6)
            rays2 = rays2 / (np.linalg.norm(rays2, axis=1)[:, np.newaxis] + 1e-6)
            
            # Compute angles safely
            cos_angles = np.clip(np.sum(rays1 * rays2, axis=1), -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_angles))
            
            # Final filtering
            good_points = np.ones(len(points_3d), dtype=bool)
            good_points &= angles > self.min_triangulation_angle
            good_points &= np.linalg.norm(points_3d, axis=1) < self.max_point_distance
            
            return points_3d[good_points]
            
        except Exception as e:
            print(f"Triangulation failed: {str(e)}")
            return np.array([])
    
    def _estimate_scale(self, points1, points2):
        """Estimate scale between two point clouds"""
        if len(points1) < 5 or len(points2) < 5:
            return 1.0
        
        try:
            # Calculate centroid distances in both point clouds
            centroid1 = np.mean(points1, axis=0)
            centroid2 = np.mean(points2, axis=0)
            
            # Calculate distances from each point to its centroid
            dist1 = np.linalg.norm(points1 - centroid1, axis=1)
            dist2 = np.linalg.norm(points2 - centroid2, axis=1)
            
            # Use the minimum number of points available in both clouds
            min_points = min(len(dist1), len(dist2))
            dist1 = dist1[:min_points]
            dist2 = dist2[:min_points]
            
            # Get median ratio of distances
            ratios = dist1 / (dist2 + 1e-6)
            scale = np.median(ratios)
            
            # Sanity check on scale
            if scale < 0.1 or scale > 10:
                return 1.0
            
            return scale
        
        except Exception as e:
            print(f"Scale estimation failed: {str(e)}")
            return 1.0
    
    def _should_add_keyframe(self, last_pose, current_pose):
        """Determine if we should add a new keyframe"""
        translation = np.linalg.norm(current_pose[:3, 3] - last_pose[:3, 3])
        rotation = np.arccos((np.trace(current_pose[:3, :3] @ last_pose[:3, :3].T) - 1) / 2)
        return translation > self.min_distance or np.degrees(rotation) > 15
    
    def get_camera_trajectory(self):
        """Return camera trajectory as numpy array"""
        return np.array([pose[:3, 3] for pose in self.poses])
    
    def get_point_cloud(self):
        """Return 3D points as numpy array"""
        return np.array(self.points_3d)

def create_maps(points_3d, trajectory):
    """Create and save 2D and 3D maps"""
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    # Save 3D point cloud
    o3d.io.write_point_cloud("3D_map.pcd", pcd)
    
    # Create 2D trajectory plot
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-')
    plt.scatter(trajectory[0, 0], trajectory[0, 2], c='green', label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', label='End')
    plt.title('Robot Trajectory (Top View)')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig('2D_trajectory.png')
    plt.close()
    
    return pcd

def process_dataset(data_dir, camera_matrix):
    """Process images from a dataset directory"""
    # Load images from dataset
    image_dir = os.path.join(data_dir, "rgb")  # Adjust path according to your dataset structure
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Initialize SLAM system
    slam = MonocularSlam(camera_matrix)
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing frames"):
        # Read image
        frame = cv2.imread(os.path.join(image_dir, img_file))
        
        if frame is None:
            print(f"Failed to read image: {img_file}")
            continue
        
        # Process frame
        success = slam.process_frame(frame)
        
        # Display current frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return slam

def process_live_video(camera_matrix):
    """Process frames from live video"""
    # Initialize SLAM system
    slam = MonocularSlam(camera_matrix)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            success = slam.process_frame(frame)
            
            # Display current frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
    
    return slam

def _load_calib(filepath):
    """
    Loads the calibration of the camera
    Returns: K (camera matrix), P (projection matrix)
    """
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P

def main():
    # Choose processing mode
    use_dataset = True  # Set to False for live video
    
    try:
        if use_dataset:
            # Specify your dataset directory
            data_dir = ""
            K, P = _load_calib(os.path.join(data_dir, "calib.txt"))
            slam = process_dataset(data_dir, K)
        else:
            # For live video, use default camera matrix
            camera_matrix = np.array([[718.856, 0, 607.1928],
                                    [0, 718.856, 185.2157],
                                    [0, 0, 1]])
            slam = process_live_video(camera_matrix)
        
        # Clean up windows
        cv2.destroyAllWindows()
        
        # Get final trajectory and points
        trajectory = slam.get_camera_trajectory()
        points_3d = slam.get_point_cloud()
        
        if len(points_3d) > 0:
            # Create and save maps
            pcd = create_maps(points_3d, trajectory)
            
            # Visualize final point cloud
            o3d.visualization.draw_geometries([pcd])
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
