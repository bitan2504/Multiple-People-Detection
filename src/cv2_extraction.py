import os
import cv2
from src.utils.profile import profile


@profile
def cv2_extraction(video_path: str, output_dir: str) -> None:
    """Extract frames from a video using ffmpeg at 1 frame per second."""
    extraction_model = "cv2"
    print(
        f"Extracting frames from video: {video_path} to {output_dir} using {extraction_model}..."
    )

    # ── output paths ─────────────────────────────────────────────────────────
    interview_id = os.path.basename(video_path).split("/")[-1].split("_")[0]
    frames_output_dir = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/frames"
    )

    os.makedirs(frames_output_dir, exist_ok=True)

    # ── open video ────────────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # ── video metadata ────────────────────────────────────────────────────────
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration = int(total_frames / fps)
    print(
        f"Video: {os.path.basename(video_path)} | FPS: {fps:.2f} | Duration: {duration}s | Frames: {total_frames}",
        end="\n",
    )

    # ── process command ────────────────────────────────────────────────────────
    for sec in range(0, duration, 1):
        frame_number = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            print(f"Could not read frame at second {sec} — video may have ended early")
            break  # Exit the loop safely instead of continuing to fail

        cv2.imwrite(
            os.path.join(frames_output_dir, f"{interview_id}_{sec:04d}.jpg"), frame
        )

    cap.release()
    print(
        f"Frame extraction completed successfully for video: {video_path} using {extraction_model}",
    )
