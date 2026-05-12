import os
import cv2
import json
import torch
from ultralytics import YOLO

from src.utils.get_interview_id import get_interview_id


# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------
def cv2_pipeline(video_path: str, output_dir: str, config: dict) -> None:
    """
    CV2 Pipeline: Extract frames from video using OpenCV at specified intervals, then run YOLO
    inference on the extracted frames to detect multiple people and absences.

    Arguments:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames and results will be saved.
        config: Configuration dictionary containing model paths and thresholds.

    Returns:
        None. Results are saved to disk and printed to console.
    """
    extraction_model = "cv2"
    print(f"INFO: Starting {extraction_model} pipeline for video: {video_path}")

    # ── output paths ─────────────────────────────────────────────────────────
    interview_id = get_interview_id(video_path)

    frames_dir = os.path.join(output_dir, f"{interview_id}/{extraction_model}/frames")
    multiple_people_dir = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/multiple_people"
    )
    raw_json_path = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/yolo_llm_raw.json"
    )

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(multiple_people_dir, exist_ok=True)
    os.makedirs(os.path.dirname(raw_json_path), exist_ok=True)

    # ── open video & metadata ────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration = int(total_frames / fps)
    print(
        f"INFO: Video: {os.path.basename(video_path)} | FPS: {fps:.2f} | Duration: {duration}s | Frames: {total_frames}"
    )

    # ── device & model setup ─────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: YOLO using device: {device}")

    if not os.path.exists(config["YOLO_MODEL_PATH"]):
        cap.release()
        raise FileNotFoundError(f"YOLO model not found at: {config['YOLO_MODEL_PATH']}")

    model = YOLO(config["YOLO_MODEL_PATH"])
    model.to(device)
    print(f"INFO: YOLO model loaded: {config['YOLO_MODEL_PATH']}")

    # ── state variables ──────────────────────────────────────────────────────
    batch_size = (
        config["BATCH_SIZE_GPU"] if device == "cuda" else config["BATCH_SIZE_CPU"]
    )
    frames_batch = []
    seconds_batch = []
    frame_numbers_batch = []

    multiple_people_detections = []
    absence_detections = []

    # ── batch processing function ────────────────────────────────────────────
    def _process_batch():
        if not frames_batch:
            return

        results = model(
            frames_batch, conf=config["YOLO_CONFIDENCE"], device=device, verbose=False
        )

        for i, r in enumerate(results):
            confidences = [
                float(box.conf[0])
                for box in r.boxes
                if int(box.cls[0]) == config["PERSON_CLASS_ID"]
            ]

            num_people = len(confidences)
            sec_val = seconds_batch[i]
            frame_num = frame_numbers_batch[i]

            if num_people > config["MULTIPLE_PEOPLE_THRESHOLD"]:
                frame_name = f"{interview_id}_{frame_num}_sec_{sec_val}_multiple.jpg"
                frame_path = os.path.join(multiple_people_dir, frame_name)

                if cv2.imwrite(frame_path, frames_batch[i]):
                    multiple_people_detections.append(
                        {
                            "violation_type": "multiple_people",
                            "time_seconds": sec_val,
                            "time_minutes": round(sec_val / 60, 2),
                            "frame_number": frame_num,
                            "num_people": num_people,
                            "confidence": confidences,
                            "frame_name": frame_name,
                            "frame_path": frame_path,
                        }
                    )
                else:
                    print(
                        f"WARNING: Failed to save multiple people frame: {frame_path}"
                    )

            elif num_people <= config["ABSENCE_THRESHOLD"]:
                absence_detections.append(
                    {
                        "violation_type": "absence",
                        "time_seconds": sec_val,
                        "time_minutes": round(sec_val / 60, 2),
                        "frame_number": frame_num,
                        "num_people": 0,
                        "confidence": [],
                    }
                )

    # ── process video (single pass) ──────────────────────────────────────────
    print("INFO: Starting frame extraction and YOLO inference...")

    for sec in range(0, duration, config["FRAME_SAMPLE_INTERVAL"]):
        frame_number = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            print(
                f"\nWARNING: Could not read frame at second {sec} — video may have ended early"
            )
            break

        print(f"Processing second: {sec}/{duration}", end="\r")

        # Save standard extracted frame
        frame_save_path = os.path.join(frames_dir, f"{interview_id}_{sec:04d}.jpg")
        cv2.imwrite(frame_save_path, frame)

        # Queue for YOLO
        frames_batch.append(frame)
        seconds_batch.append(sec)
        frame_numbers_batch.append(frame_number)

        # Trigger batch inference
        if len(frames_batch) >= batch_size:
            _process_batch()
            frames_batch = []
            seconds_batch = []
            frame_numbers_batch = []

    # Process any leftover frames
    if frames_batch:
        _process_batch()

    print()  # Clear line after carriage return
    cap.release()

    print(
        f"INFO: YOLO complete — multiple_people: {len(multiple_people_detections)} frame(s) | absence: {len(absence_detections)} frame(s)"
    )

    # ── write raw JSON ────────────────────────────────────────────────────────
    raw_output = {
        "interview_id": interview_id,
        "video_path": video_path,
        "fps": fps,
        "duration_seconds": duration,
        "multiple_people_detections": multiple_people_detections,
        "absence_detections": absence_detections,
    }

    with open(raw_json_path, "w") as f:
        json.dump(raw_output, f, indent=2)

    print(f"INFO: Raw YOLO JSON written: {raw_json_path}")
