import os


def get_interview_id(video_path: str) -> str:
    """
    Extracts the interview ID from the given video path.

    Args:
        video_path (str): The path to the video file.
    Returns:
        str: The extracted interview ID.
    """
    return os.path.basename(video_path).split("/")[-1].split("_")[0]
