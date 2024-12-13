import os
import urllib.request
import zipfile
import numpy as np

def download_dataset():
    """Download a sample TUM RGB-D dataset"""
    # Create data directory
    data_dir = "dataset"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset
    dataset_url = "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
    zip_path = os.path.join(data_dir, "dataset.tgz")
    
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    
    # Extract dataset
    print("Extracting dataset...")
    os.system(f"tar -xzf {zip_path} -C {data_dir}")
    
    # Create images directory and copy RGB images
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create camera calibration file
    calib_file = os.path.join(data_dir, "calib.txt")
    # TUM FR1 camera intrinsics
    K = np.array([[517.3, 0, 318.6],
                  [0, 516.5, 255.3],
                  [0, 0, 1]])
    
    # Save calibration
    np.savetxt(calib_file, K.flatten(), fmt='%.6f')
    
    print("Dataset prepared successfully!")
    return data_dir

if __name__ == "__main__":
    data_dir = download_dataset()
    print(f"Dataset downloaded to: {data_dir}") 