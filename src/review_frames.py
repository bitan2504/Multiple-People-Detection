import base64
from src.utils.litellm import run_model

# ── PROMPT UPDATED FOR STABILITY ─────────────────────────────────────────────
INITIAL_PROMPT = """
You are a high-precision visual security assistant. Your task is to analyze video frames and detect the presence of multiple people.

Rules:
1. "multiple_people_detected" must be true ONLY if 2 or more distinct human figures are visible.
2. If 0 or 1 person is visible, it must be false.
3. You MUST return a valid JSON object containing a "results" array.
4. Do not include any conversational text, explanations, or Markdown code blocks. Output raw JSON only.
5. "people_detected" should be determined based on clear visual evidence of human figures, such as heads, bodies, or limbs. Do not rely solely on text metadata or confidence scores.
6. You should ignore human figures in photos, posters, or reflections. Focus only on actual people present in the scene.
7. There shall be a picture of an AI assistant in the frame. Do not count the AI assistant as a person. Only count human figures.
8. If the frame is blurry or unclear, make your best effort to determine the presence of people, but prioritize accuracy over guessing. If uncertain, default to "multiple_people_detected": false.
9. "frame_path" should be same as the input frame path for reference.

User Prompt:
Analyze the attached frames in order. For each frame, determine if multiple people are present.

Output Format:
{
  "results": [
    {"multiple_people_detected": true},
    {"multiple_people_detected": false},
  ]
}
"""


def encode_image_to_base64(image_path: str) -> str | None:
    """Reads a local image file and converts it to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"WARNING: Failed to encode image {image_path} - {e}")
        return None


def review_frames(multiple_people_detections_under_review: list) -> dict | None:
    if not multiple_people_detections_under_review:
        print("INFO: No multiple people detections under review.")
        return None

    content = [{"type": "text", "text": INITIAL_PROMPT}]

    print("\nINFO: Preparing Multiple People Detections for LLM Review...")

    for idx, detection in enumerate(multiple_people_detections_under_review, start=1):
        time_sec = detection.get("time_seconds", "Unknown")
        frame_path = detection.get("frame_path")

        # Append Text Metadata
        content.append(
            {
                "type": "text",
                "text": f"Frame number: {time_sec}",
            }
        )

        # Base64 Encode and Append Image
        if frame_path:
            base64_image = encode_image_to_base64(frame_path)
            if base64_image:
                # LiteLLM/OpenAI standard format for base64 images
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    # Execute the model evaluation
    try:
        response = run_model(content)
        return response
    except Exception as e:
        print(f"ERROR: LLM evaluation failed - {e}")
        return None
