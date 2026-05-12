import os
import json
import ffmpeg
import torch
import av
import numpy as np
from PIL import Image
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# IMAGE UTILITIES
# -----------------------------------------------------------------------------
def load_image_with_av(image_path: str) -> np.ndarray | None:
    try:
        container = av.open(image_path)
        for frame in container.decode(video=0):
            # decode to RGB first (PyAV native)
            rgb_array = frame.to_ndarray(format="rgb24")
            container.close()
            # convert RGB -> BGR so YOLO gets the same color format
            bgr_array = rgb_array[:, :, ::-1].copy()
            return bgr_array
        container.close()
        return None
    except Exception as e:
        print(f"WARNING: Failed to load image: {image_path} | Error: {e}")
        return None


def save_image_with_av(image_array: np.ndarray, save_path: str) -> bool:
    try:
        # image_array is BGR (after our load conversion), flip back to RGB for PIL
        rgb_array = image_array[:, :, ::-1].copy()
        image = Image.fromarray(rgb_array.astype(np.uint8))
        image.save(save_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"WARNING: Failed to save image: {save_path} | Error: {e}")
        return False


# -----------------------------------------------------------------------------
# EXTRACTION & PROCESSING
# -----------------------------------------------------------------------------
def ffmpeg_pipeline(video_path: str, output_dir: str, config: dict) -> None:
    """
    Extract frames from video using ffmpeg at specified intervals, then run YOLO
    inference on the extracted frames to detect multiple people and absences.

    Arguments:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames and results will be saved.
        config: Configuration dictionary containing model paths and thresholds.\
        
    Returns:
        None. Results are saved to disk and printed to console.
    """
    extraction_model = "ffmpeg"
    print(
        f"Extracting frames from video: {video_path} to {output_dir} using {extraction_model}..."
    )

    # ── output paths ──────────────────────────────────────────────────────────
    interview_id = os.path.basename(video_path).split("/")[-1].split("_")[0]
    frames_dir = os.path.join(output_dir, f"{interview_id}/{extraction_model}/frames")
    multiple_people_dir = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/multiple_people"
    )
    raw_json_path = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/yolo_llm_raw.json"
    )

    # ── create and validate paths ─────────────────────────────────────────────
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(multiple_people_dir, exist_ok=True)
    os.makedirs(os.path.dirname(raw_json_path), exist_ok=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    # ── video metadata ────────────────────────────────────────────────────────
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

    num, den = video_info["r_frame_rate"].split("/")
    fps = int(round(float(num) / float(den)))
    frame_interval = int(fps)
    video_total_frames = int(video_info["nb_frames"])

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration = int(video_total_frames / fps)
    print(
        f"Video: {os.path.basename(video_path)} | FPS: {fps:.2f} | Duration: {duration}s | Frames: {video_total_frames}"
    )

    # ── process command ───────────────────────────────────────────────────────
    try:
        (
            ffmpeg.input(video_path)
            .output(
                os.path.join(frames_dir, f"{interview_id}_%04d.jpg"),
                vf=f"select='not(mod(n,{frame_interval}))'",
                fps_mode="passthrough",
                start_number="0",
                **{
                    "qscale:v": 1,
                },
                qmin=1,
                qmax=1,
            )
            .global_args("-loglevel", "error")
            .run(overwrite_output=True)
        )

    except ffmpeg.Error as e:
        print("ffmpeg error: %s", e.stderr.decode())
        raise RuntimeError(f"ffmpeg failed to extract frames from video: {video_path}")

    print(
        f"Frame extraction completed successfully for video: {video_path} using {extraction_model}"
    )

    # ── device setup ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for YOLO inference")

    # ── load YOLO model ───────────────────────────────────────────────────────
    if not os.path.exists(config["YOLO_MODEL_PATH"]):
        raise FileNotFoundError(f"YOLO model not found: {config['YOLO_MODEL_PATH']}")

    model = YOLO(config["YOLO_MODEL_PATH"])
    model.to(device)
    print(f"Loaded YOLO model from: {config['YOLO_MODEL_PATH']}")

    # ── prepare frame list ────────────────────────────────────────────────────
    frame_files = sorted(
        [
            f
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if not frame_files:
        raise RuntimeError(f"No frame images found in: {frames_dir}")

    extracted_frames_count = len(frame_files)
    print(f"Found {extracted_frames_count} extracted frames")

    # ── batch settings ────────────────────────────────────────────────────────
    batch_size = (
        config["BATCH_SIZE_GPU"] if device == "cuda" else config["BATCH_SIZE_CPU"]
    )
    frames_batch = []
    seconds_batch = []
    frame_numbers_batch = []
    multiple_people_detections = []
    absence_detections = []

    def _process_batch():
        results = model(
            frames_batch,
            conf=config["YOLO_CONFIDENCE"],
            device=device,
            verbose=False,
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

            # -----------------------------------------------------------------
            # MULTIPLE PEOPLE DETECTION
            # -----------------------------------------------------------------
            if num_people > config["MULTIPLE_PEOPLE_THRESHOLD"]:
                frame_name = f"{interview_id}_{frame_num}_sec_{sec_val}_multiple.jpg"
                frame_path = os.path.join(multiple_people_dir, frame_name)

                saved = save_image_with_av(frames_batch[i], frame_path)

                if saved:
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
                    print(f"WARNING: Failed to save frame: {frame_path}")

            # -----------------------------------------------------------------
            # ABSENCE DETECTION
            # -----------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # PROCESS FRAMES
    # -------------------------------------------------------------------------
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)

        print(f"Processing frame {idx + 1}/{extracted_frames_count}", end="\r")

        # load via PyAV — returns BGR array (RGB decoded then flipped in loader)
        frame = load_image_with_av(frame_path)

        if frame is None:
            print(f"WARNING: Could not read frame: {frame_path} — skipping")
            continue

        sec_val = idx * config["FRAME_SAMPLE_INTERVAL"]
        frame_number = int(sec_val * fps)

        frames_batch.append(frame)
        seconds_batch.append(sec_val)
        frame_numbers_batch.append(frame_number)

        if len(frames_batch) >= batch_size or idx == extracted_frames_count - 1:
            _process_batch()
            frames_batch = []
            seconds_batch = []
            frame_numbers_batch = []

    print()  # Clear the loading line

    # -------------------------------------------------------------------------
    # FINAL LOGGING
    # -------------------------------------------------------------------------
    print(
        f"INFO: YOLO complete | multiple_people={len(multiple_people_detections)} | absence={len(absence_detections)}"
    )

    # -------------------------------------------------------------------------
    # SAVE JSON OUTPUT
    # -------------------------------------------------------------------------
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

    print(f"INFO: YOLO JSON written: {raw_json_path}")
