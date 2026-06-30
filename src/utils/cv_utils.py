import cv2
import numpy as np


def load_image_with_cv2(image_path: str) -> np.ndarray | None:
    try:
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"WARNING: Failed to load image: {image_path} | Error: Image is None")
            return None
            
        return image

    except Exception as e:
        print(f"WARNING: Exception occurred while loading image: {image_path} | Error: {e}")
        return None


def save_image_with_cv2(image_array: np.ndarray, save_path: str) -> bool:
    try:
        success = cv2.imwrite(
            save_path, 
            image_array, 
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        
        if not success:
            print(f"WARNING: Failed to save image (cv2.imwrite returned False): {save_path}")
            
        return success
        
    except Exception as e:
        print(f"WARNING: Exception occurred while saving image: {save_path} | Error: {e}")
        return False