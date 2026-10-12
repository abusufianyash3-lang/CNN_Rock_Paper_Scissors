import cv2
from pathlib import Path

# Collect real webcam images for your own CNN dataset.
# Hold one gesture inside the center square, then press:
# r = rock, p = paper, s = scissors, q = quit.

ROOT = Path("dataset")
CLASSES = {"r": "rock", "p": "paper", "s": "scissors"}
for name in CLASSES.values():
    (ROOT / name).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Close other apps using the camera and try again.")

counts = {name: len(list((ROOT / name).glob("*.jpg"))) for name in CLASSES.values()}
print("Keep your hand inside the square. Press r/p/s to save an image; q to quit.")
print("Collect at least 100-200 images per class, varying hand position and lighting.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    size = min(280, h - 20, w - 20)
    x1, y1 = (w - size) // 2, (h - size) // 2
    x2, y2 = x1 + size, y1 + size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "R=Rock  P=Paper  S=Scissors  Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, "Counts: " + " | ".join(f"{k}: {counts[v]}" for k, v in CLASSES.items()),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Collect CNN training images", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key in [ord(k) for k in CLASSES]:
        label = CLASSES[chr(key)]
        crop = frame[y1:y2, x1:x2]
        path = ROOT / label / f"{label}_{counts[label]:05d}.jpg"
        if cv2.imwrite(str(path), crop):
            counts[label] += 1
            print(f"Saved {path}")

cap.release()
cv2.destroyAllWindows()
