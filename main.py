from src.ffmpeg_pipeline import ffmpeg_pipeline
from src.cv2_pipeline import cv2_pipeline
from src.utils.read_yaml import read_yaml

if __name__ == "__main__":
    video_path = "input/694a8baf524205756b53b515_final_camera.mp4"
    output_dir = "output"
    config = dict(read_yaml("config.yml"))
    ffmpeg_pipeline(video_path, output_dir, config)
    cv2_pipeline(video_path, output_dir, config)
