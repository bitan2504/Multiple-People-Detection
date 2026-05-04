import json
import logging
import os

import cv2
import torch
from ultralytics import YOLO
from src.utils.profile import profile

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

YOLO_MODEL_PATH = "yolov8l.pt"
YOLO_CONFIDENCE = 0.6
PERSON_CLASS_ID = 0

FRAME_SAMPLE_INTERVAL = 1  # sample one frame every N seconds (1 = every second)

BATCH_SIZE_GPU = 16  # number of frames per YOLO batch on GPU
BATCH_SIZE_CPU = 4  # number of frames per YOLO batch on CPU

MULTIPLE_PEOPLE_THRESHOLD = 1  # flag frame if num_people > this value (default: 1)
ABSENCE_THRESHOLD = 0  # flag frame if num_people <= this value (default: 0)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


@profile
def run_yolo_detection(video_path: str, output_dir: str, extraction_model: str) -> str:
    """
    Runs YOLO on every FRAME_SAMPLE_INTERVAL seconds of the video.
    Saves flagged frames and writes the raw detections JSON.

    Args:
        video_path    : path to the .mp4 interview video
        output_dir : root output directory for this interview
        extraction_model : the extraction model to use

    Returns:
        raw_json_path : path to the written yolo_llm_raw_<id>.json
    """
    print(
        f"Running YOLO detection on video: {video_path} with extraction model: {extraction_model}"
    )
    interview_id = os.path.basename(video_path).split("/")[-1].split("_")[0]
    frames_dir = os.path.join(output_dir, f"{interview_id}/{extraction_model}/frames")

    # ── output paths ─────────────────────────────────────────────────────────
    multiple_people_dir = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/multiple_people"
    )
    raw_json_path = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/yolo_llm_raw.json"
    )

    os.makedirs(multiple_people_dir, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"  # force CPU for more consistent benchmarking
    log.info("YOLO using device: %s", device)

    # ── load model ────────────────────────────────────────────────────────────
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}")

    model = YOLO(YOLO_MODEL_PATH)
    model.to(device)
    log.info("YOLO model loaded: %s", YOLO_MODEL_PATH)

    # ── open video ────────────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration = int(total_frames / fps)
    log.info(
        "Video: %s | FPS: %.2f | Duration: %ds | Frames: %d",
        os.path.basename(video_path),
        fps,
        duration,
        total_frames,
    )

    # ── batch inference ───────────────────────────────────────────────────────
    batch_size = BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU

    frames_batch = []
    seconds_batch = []
    frame_numbers_batch = []

    multiple_people_detections = []
    absence_detections = []

    def _process_batch():
        results = model(
            frames_batch, conf=YOLO_CONFIDENCE, device=device, verbose=False
        )
        for i, r in enumerate(results):
            confidences = [
                float(box.conf[0])
                for box in r.boxes
                if int(box.cls[0]) == PERSON_CLASS_ID
            ]
            num_people = len(confidences)
            sec_val = seconds_batch[i]
            frame_num = frame_numbers_batch[i]

            if num_people > MULTIPLE_PEOPLE_THRESHOLD:
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
                    log.warning("Failed to save frame: %s", frame_path)

            elif num_people <= ABSENCE_THRESHOLD:
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

    for sec in range(0, duration, FRAME_SAMPLE_INTERVAL):
        frame_number = int(sec * fps)
        frame_path = os.path.join(frames_dir, f"{interview_id}_{sec:04d}.jpg")
        frame = cv2.imread(frame_path)

        print(f"Processing second: {sec}/{duration}", end="\r")

        if frame is None:
            log.warning("Could not read frame at second %d — skipping", sec)
            continue

        frames_batch.append(frame)
        seconds_batch.append(sec)
        frame_numbers_batch.append(frame_number)

        if len(frames_batch) == batch_size or sec >= duration - FRAME_SAMPLE_INTERVAL:
            _process_batch()
            frames_batch = []
            seconds_batch = []
            frame_numbers_batch = []

    cap.release()

    log.info(
        "YOLO complete — multiple_people: %d frame(s) | absence: %d frame(s)",
        len(multiple_people_detections),
        len(absence_detections),
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

    log.info("Raw YOLO JSON written: %s", raw_json_path)

    return raw_json_path
