import av
import numpy as np
from PIL import Image


def load_image_with_av(image_path: str) -> np.ndarray | None:
    try:
        container = av.open(image_path)
        for frame in container.decode(video=0):
            rgb_array = frame.to_ndarray(format="rgb24")
            container.close()
            bgr_array = rgb_array[:, :, ::-1].copy()
            return bgr_array
        container.close()
        return None

    except Exception as e:
        print(f"WARNING: Failed to load image: {image_path} | Error: {e}")
        return None


def save_image_with_av(image_array: np.ndarray, save_path: str) -> bool:
    try:
        rgb_array = image_array[:, :, ::-1].copy()
        image = Image.fromarray(rgb_array.astype(np.uint8))
        image.save(save_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"WARNING: Failed to save image: {save_path} | Error: {e}")
        return False
