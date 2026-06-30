import os

from src.ffmpeg_pipeline import ffmpeg_pipeline
from src.cv2_pipeline import cv2_pipeline
from src.utils.read_yaml import read_yaml
from src.utils.litellm import run_model

if __name__ == "__main__":
    video_path = "input/India002_final.mp4"
    output_dir = "output"
    
    # run_model(
    #     [
    #         {"type": "text", "text": "What is the capital of France?"},
    #         {"type": "text", "text": "What is the largest mammal?"},
    #     ]
    # )

    config = dict(read_yaml("yolo.config.yaml"))
    # print(f"{config}")
    ffmpeg_pipeline(video_path, output_dir, config)
    cv2_pipeline(video_path, output_dir, config)
