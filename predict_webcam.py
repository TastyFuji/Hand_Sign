"""
predict_webcam.py  --  Real-time ASL Hand Sign prediction via Webcam
Uses MediaPipe for hand detection + trained CNN for letter classification
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ===== CONFIG =====
MODEL_PATH = "artifacts/asl_cnn.h5"
CAMERA_INDEX = 0        # built-in webcam = 0, DroidCam/Iriun = 1, 2, 3
SMOOTH_N = 7            # majority-vote window to reduce label flickering
CONF_THRESHOLD = 0.5    # below this => "Unknown"

# ASL mapping (Sign MNIST: 24 classes, no J/Z because they require motion)
IDX_TO_CHAR = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
}


def label_to_char(label: int) -> str:
    return IDX_TO_CHAR.get(label, "?")


def get_hand_angle(landmarks, fw, fh):
    """Degrees the hand is rotated from vertical (0° = fingers point up)."""
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    dx = (mid_mcp.x - wrist.x) * fw
    dy = (mid_mcp.y - wrist.y) * fh
    return np.degrees(np.arctan2(dx, -dy))


def make_square_box(x1, y1, x2, y2, img_w, img_h, margin=20):
    """Expand bounding box to a square with extra margin for rotation."""
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1) // 2 + margin
    x1 = max(cx - side, 0)
    y1 = max(cy - side, 0)
    x2 = min(cx + side, img_w)
    y2 = min(cy + side, img_h)
    return x1, y1, x2, y2


def rotate_roi(roi_bgr, angle_deg):
    """Rotate ROI around its center to normalize hand orientation."""
    rh, rw = roi_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((rw / 2, rh / 2), angle_deg, 1.0)
    return cv2.warpAffine(roi_bgr, M, (rw, rh),
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR ROI to 28x28 grayscale tensor for CNN."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    img = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    return x.reshape(1, 28, 28, 1)


def majority_vote(queue) -> int | None:
    """Return the most frequent label in the queue."""
    if not queue:
        return None
    vals, counts = np.unique(np.array(queue), return_counts=True)
    return int(vals[np.argmax(counts)])


def open_camera(index: int):
    """Try multiple backends to find a working camera (handles phone webcam green-screen)."""
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Auto", cv2.CAP_ANY),
    ]
    for name, backend in backends:
        print(f"  Trying {name} backend...")
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Test read
        for _ in range(5):
            ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  Camera opened with {name} backend")
            return cap
        cap.release()

    return None


def main():
    # --- Load model ---
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    # --- Camera ---
    print(f"Opening camera (index={CAMERA_INDEX})...")
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        raise RuntimeError(
            f"Cannot open camera index {CAMERA_INDEX}.\n"
            "Try changing CAMERA_INDEX to 1, 2, or 3 in the script."
        )

    # --- MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    pred_queue: deque = deque(maxlen=SMOOTH_N)

    print("Running... Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label_text = "No hand"
        conf_text = ""

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            lm = hand_lm.landmark
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]

            # Bounding box → square with padding (needed for clean rotation)
            pad = 20
            x1 = int(max(min(xs) * w - pad, 0))
            y1 = int(max(min(ys) * h - pad, 0))
            x2 = int(min(max(xs) * w + pad, w))
            y2 = int(min(max(ys) * h + pad, h))
            x1, y1, x2, y2 = make_square_box(x1, y1, x2, y2, w, h)

            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                angle = get_hand_angle(lm, w, h)
                roi = rotate_roi(roi, angle)
                x = preprocess_roi(roi)
                probs = model.predict(x, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(np.max(probs))

                if conf < CONF_THRESHOLD:
                    pred_queue.append(-1)
                else:
                    pred_queue.append(pred)

                voted = majority_vote(pred_queue)

                if voted is None or voted == -1:
                    label_text = f"Unknown ({conf:.2f})"
                else:
                    label_text = f"{label_to_char(voted)} ({conf:.2f})"

                conf_text = f"raw={label_to_char(pred)} conf={conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Show 28x28 ROI preview (top-right, after rotation)
            if roi.size > 0:
                small = x.reshape(28, 28)
                small = (small * 255).astype(np.uint8)
                small = cv2.resize(small, (140, 140), interpolation=cv2.INTER_NEAREST)
                small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                frame[10:150, w - 150:w - 10] = small

        # HUD
        cv2.putText(frame, label_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "Press Q to quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow("ASL Hand Sign - Webcam", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
