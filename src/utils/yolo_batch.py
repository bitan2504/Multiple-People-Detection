import torch
import os
from ultralytics import YOLO


class yolo_batch:
    def __init__(self, interview_id: str, multiple_people_dir: str, config: dict, save_image_function: callable):
        self.interview_id = interview_id
        self.multiple_people_dir = multiple_people_dir
        self.config = config
        self.save_image_function = save_image_function

        # ── device setup ──────────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for YOLO inference")

        # ── load YOLO model ───────────────────────────────────────────────────────
        if not os.path.exists(self.config["YOLO_MODEL_PATH"]):
            raise FileNotFoundError(f"YOLO model not found: {self.config['YOLO_MODEL_PATH']}")

        self.model = YOLO(self.config["YOLO_MODEL_PATH"])
        self.model.to(self.device)
        print(f"Loaded YOLO model from: {self.config['YOLO_MODEL_PATH']}")

        # ── batch settings ────────────────────────────────────────────────────────
        self.batch_size = self.config["BATCH_SIZE_GPU"] if self.device == "cuda" else self.config["BATCH_SIZE_CPU"]
        self.frames_batch = []
        self.seconds_batch = []
        self.frame_numbers_batch = []
        self.multiple_people_detected = []
        self.multiple_people_under_review = []
        self.absence_detected = []

    def push_frame(self, frame, sec_val, frame_num):
        self.frames_batch.append(frame)
        self.seconds_batch.append(sec_val)
        self.frame_numbers_batch.append(frame_num)

    def clear_batch(self):
        self.frames_batch = []
        self.seconds_batch = []
        self.frame_numbers_batch = []

    def process(self):
        results = self.model(
            self.frames_batch,
            conf=self.config["YOLO_CONFIDENCE"],
            device=self.device,
            verbose=False,
        )

        for i, r in enumerate(results):
            confidences = [float(box.conf[0]) for box in r.boxes if int(box.cls[0]) == self.config["PERSON_CLASS_ID"]]

            num_people = len(confidences)
            sec_val = self.seconds_batch[i]
            frame_num = self.frame_numbers_batch[i]

            # -----------------------------------------------------------------
            # MULTIPLE PEOPLE DETECTION
            # -----------------------------------------------------------------
            if num_people > self.config["MULTIPLE_PEOPLE_THRESHOLD"]:
                frame_name = f"{self.interview_id}_{frame_num}_sec_{sec_val}_multiple.jpg"
                frame_path = os.path.join(self.multiple_people_dir, frame_name)

                saved = self.save_image_function(self.frames_batch[i], frame_path)

                if saved:
                    self.multiple_people_detected.append(
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
            elif num_people <= self.config["ABSENCE_THRESHOLD"]:
                self.absence_detected.append(
                    {
                        "violation_type": "absence",
                        "time_seconds": sec_val,
                        "time_minutes": round(sec_val / 60, 2),
                        "frame_number": frame_num,
                        "num_people": 0,
                        "confidence": [],
                    }
                )
