import os
import cv2
import json
import torch
from ultralytics import YOLO

from src.utils.get_interview_id import get_interview_id
from src.utils.cv_utils import load_image_with_cv2, save_image_with_cv2  # Updated import
from src.utils.yolo_batch import yolo_batch
from src.review_frames import review_frames


# -----------------------------------------------------------------------------
# EXTRACTION & PROCESSING (CV2 PIPELINE)
# -----------------------------------------------------------------------------
class cv2_pipeline:
    def __init__(self, video_path: str, output_dir: str, config: dict) -> None:
        """
        Extract frames from video using OpenCV at specified intervals, then run YOLO
        inference on the extracted frames to detect multiple people and absences.

        Arguments:
            video_path: Path to the input video file.
            output_dir: Directory where extracted frames and results will be saved.
            config: Configuration dictionary containing model paths and thresholds.

        Returns:
            None. Results are saved to disk and printed to console.
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at: {video_path}")

        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.extraction_model = "cv2"
        print(f"Extracting frames from video: {self.video_path} to {self.output_dir} using {self.extraction_model}...")

        # ── output paths ──────────────────────────────────────────────────────────
        self.interview_id = get_interview_id(self.video_path)
        self.frames_dir = os.path.join(self.output_dir, f"{self.interview_id}/{self.extraction_model}/frames")
        self.multiple_people_dir = os.path.join(self.output_dir, f"{self.interview_id}/{self.extraction_model}/multiple_people")
        self.raw_json_path = os.path.join(self.output_dir, f"{self.interview_id}/{self.extraction_model}/yolo_llm_raw.json")
        
        self.video_metadata()

        self.run()

    def batch_setup(self) -> None:
        # ── device setup ──────────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for YOLO inference")

        # ── load YOLO model ───────────────────────────────────────────────────────
        if not os.path.exists(self.config["YOLO_MODEL_PATH"]):
            raise FileNotFoundError(f"YOLO model not found: {self.config['YOLO_MODEL_PATH']}")

        self.model = YOLO(self.config["YOLO_MODEL_PATH"])
        self.model.to(self.device)
        print(f"Loaded YOLO model from: {self.config['YOLO_MODEL_PATH']}")

    def video_metadata(self) -> None:
        # ── video metadata ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS ({self.fps}) for video: {self.video_path}")

        self.duration = int(self.total_frames / self.fps)
        print(f"Video: {os.path.basename(self.video_path)} | FPS: {self.fps:.2f} | Duration: {self.duration}s | Frames: {self.total_frames}")
        
        cap.release()

    def frame_extraction(self) -> None:
        # ── process command ───────────────────────────────────────────────────────
        print(f"INFO: Starting frame extraction using {self.extraction_model}...")
        cap = cv2.VideoCapture(self.video_path)
        
        for sec in range(0, self.duration, self.config["FRAME_SAMPLE_INTERVAL"]):
            frame_number = int(sec * self.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                print(f"\nWARNING: Could not read frame at second {sec} — video may have ended early")
                break
                
            frame_save_path = os.path.join(self.frames_dir, f"{self.interview_id}_{sec:04d}.jpg")
            cv2.imwrite(frame_save_path, frame)

        cap.release()
        print(f"Frame extraction completed successfully for video: {self.video_path} using {self.extraction_model}")

    def run(self) -> None:
        # ── create paths ─────────────────────────────────────────────
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.multiple_people_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.raw_json_path), exist_ok=True)

        self.frame_extraction()
        self.batch_setup()

        # ── prepare frame list ────────────────────────────────────────────────────
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        if not frame_files:
            raise RuntimeError(f"No frame images found in: {self.frames_dir}")

        extracted_frames_count = len(frame_files)
        print(f"Found {extracted_frames_count} extracted frames")

        batch = yolo_batch(self.interview_id, self.frames_dir, self.multiple_people_dir, self.config, save_image_with_cv2)

        # -------------------------------------------------------------------------
        # PROCESS FRAMES
        # -------------------------------------------------------------------------
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(self.frames_dir, frame_file)
            print(f"Processing frame {idx + 1}/{extracted_frames_count}", end="\r")

            # Updated: load_image_with_cv2
            frame = load_image_with_cv2(frame_path)
            if frame is None:
                print(f"WARNING: Could not read frame: {frame_path} — skipping")
                continue

            sec_val = idx * self.config["FRAME_SAMPLE_INTERVAL"]
            frame_number = int(sec_val * self.fps)
            batch.push_frame(frame, sec_val, frame_number)

            if len(batch.frames_batch) >= batch.batch_size or idx == extracted_frames_count - 1:
                batch.process()
                batch.clear_batch()

        print()  
        
        review_raw = review_frames(batch.multiple_people_detections_under_review)
        
        if not review_raw:
            print("WARNING: review_frames returned None. Skipping review process.")
            response = []
        else:
            try:
                review_json = json.loads(review_raw)
                response = review_json.get("results", [])
            except json.JSONDecodeError:
                print("ERROR: review_frames returned invalid JSON.")
                response = []

        for idx, result in enumerate(response):
            if result.get("multiple_people_detected") is True:
                frame_path = batch.multiple_people_detections_under_review[idx - 1].get("frame_path")
                time_sec = batch.multiple_people_detections_under_review[idx - 1].get("time_seconds")
                frame_number = batch.multiple_people_detections_under_review[idx - 1].get("frame_number")
                
                if frame_path:
                    frame = load_image_with_cv2(frame_path)

                    if frame is not None:
                        save_image_with_cv2(frame, os.path.join(self.multiple_people_dir, f"{self.interview_id}_{frame_number}_sec_{time_sec}_multiple_reviewed.jpg"))
        
        # # Check if detections need review
        # try:
        #     response = json.loads(review_frames(batch.multiple_people_detections_under_review))["results"]
        #     print(response)

        #     for idx, result in enumerate(response, start=1):
        #         if result["multiple_people_detected"] == True:
        #             frame_path = batch.multiple_people_detections_under_review[idx - 1]["frame_path"]
        #             time_sec = batch.multiple_people_detections_under_review[idx - 1]["time_seconds"]
        #             frame_number = batch.multiple_people_detections_under_review[idx - 1]["frame_number"]
        #             frame = load_image_with_cv2(frame_path)

        #             if frame is not None:
        #                 save_image_with_cv2(frame, os.path.join(self.multiple_people_dir, f"{self.interview_id}_{frame_number}_sec_{time_sec}_multiple_reviewed.jpg"))
        # except json.JSONDecodeError:
        #     print("ERROR: Failed to decode JSON response")
        #     return
        # # -------------------------------------------------------------------------
        # # FINAL LOGGING (Commented to match original format)
        # # -------------------------------------------------------------------------
        print(f"INFO: YOLO complete | multiple_people={len(batch.multiple_people_detected)} | absence={len(batch.absence_detected)}")

        # # -------------------------------------------------------------------------
        # # SAVE JSON OUTPUT
        # # -------------------------------------------------------------------------
        raw_output = {
            "interview_id": self.interview_id,
            "video_path": self.video_path,
            "fps": self.fps,
            "duration_seconds": self.duration,
            "multiple_people_detections": batch.multiple_people_detected,
            "absence_detections": batch.absence_detected,
        }

        with open(self.raw_json_path, "w") as f:
            json.dump(raw_output, f, indent=2)

        print(f"INFO: YOLO JSON written: {self.raw_json_path}")