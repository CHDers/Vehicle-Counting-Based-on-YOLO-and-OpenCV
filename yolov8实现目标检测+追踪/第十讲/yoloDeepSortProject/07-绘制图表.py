import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")

VIDEO_PATH = "./video/test_person.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("result4.mp4", cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))

analytics = solutions.Analytics(
    type="line",
    writer=out,
    im0_shape=(w, h),
    view_img=True,
)
total_counts = 0
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame_count += 1
        results = model.track(frame, persist=True, verbose=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            for box in boxes:
                total_counts += 1

        analytics.update_line(frame_count, total_counts)

        total_counts = 0
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()