from src.ffmpeg_extraction import ffmpeg_extraction
from src.cv2_extraction import cv2_extraction
from src.utils.run_yolo import run_yolo_detection

if __name__ == "__main__":
    video_path = "input/69b1bc5b452ff6cf1e6ed0a6_final_screen.mp4"
    output_dir = "output"
    ffmpeg_extraction(video_path, output_dir)
    run_yolo_detection(video_path, output_dir, extraction_model="ffmpeg")
    cv2_extraction(video_path, output_dir)
    run_yolo_detection(video_path, output_dir, extraction_model="cv2")
