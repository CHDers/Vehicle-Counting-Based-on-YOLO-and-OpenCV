# https://www.bilibili.com/video/BV1AeWoenEVT
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
VIDEO_PATH = "./video/test_person.mp4"
RESULT_PATH = "result3.mp4"
capture = cv2.VideoCapture(VIDEO_PATH)
assert capture.isOpened(), "Error reading video file"

w, h, fps = (int(capture.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(0, int(h/2)), (int(w), int(h/2))]
poly_points = [(710, 200), (1110, 200), (810, 400), (410, 400)]
video_writer = cv2.VideoWriter(
    RESULT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#  声明对象计数函数

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=poly_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2
)
while capture.isOpened():
    success, im0 = capture.read()
    if not success:
        print("视频读取完成")
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

capture.release()
video_writer.release()
cv2.destroyAllWindows()
