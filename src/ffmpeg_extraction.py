import ffmpeg
import os
from src.utils.profile import profile


@profile
def ffmpeg_extraction(video_path: str, output_dir: str) -> None:
    """Extract frames from a video using ffmpeg at 1 frame per second."""
    extraction_model = "ffmpeg"
    print(
        f"Extracting frames from video: {video_path} to {output_dir} using {extraction_model}..."
    )

    # ── output paths ─────────────────────────────────────────────────────────
    interview_id = os.path.basename(video_path).split("/")[-1].split("_")[0]
    frames_output_dir = os.path.join(
        output_dir, f"{interview_id}/{extraction_model}/frames"
    )

    os.makedirs(frames_output_dir, exist_ok=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    # ── video metadata ────────────────────────────────────────────────────────
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")

    num, den = video_info["r_frame_rate"].split("/")
    fps = int(round(float(num) / float(den)))
    frame_interval = int(fps)
    total_frames = int(video_info["nb_frames"])

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration = int(total_frames / fps)
    print(
        f"Video: {os.path.basename(video_path)} | FPS: {fps:.2f} | Duration: {duration}s | Frames: {total_frames}",
        end="\n",
    )

    # ── process command ────────────────────────────────────────────────────────
    try:
        (
            ffmpeg.input(video_path)
            .output(
                os.path.join(frames_output_dir, f"{interview_id}_%04d.jpg"),
                vf=f"select='not(mod(n,{frame_interval}))',scale=out_color_matrix=bt601:flags=neighbor",
                fps_mode="passthrough",
                start_number="0",
                **{
                    "qscale:v": 2,
                    "pix_fmt": "yuvj444p",
                },
                qmin=2,
                qmax=2,
            )
            .global_args("-loglevel", "error")
            .run(overwrite_output=True)
        )

    except ffmpeg.Error as e:
        print("ffmpeg error: %s", e.stderr.decode())
        raise RuntimeError(f"ffmpeg failed to extract frames from video: {video_path}")

    print(
        f"Frame extraction completed successfully for video: {video_path} using {extraction_model}",
    )
