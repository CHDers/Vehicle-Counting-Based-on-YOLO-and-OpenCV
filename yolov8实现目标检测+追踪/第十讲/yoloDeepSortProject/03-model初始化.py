# https://www.bilibili.com/video/BV1AeWoenEVT
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# model 6大功能 ： 1. Train() 训练  2. Val() 验证测试 3. predict 预测 4. track 追踪 5. export() 导出 6. Benchmark 基准测试

# 可以接受的来源包括： 图像，url，屏幕截图，视频，流媒体, 文件夹，带通配符的路径...
results = model("./images/000177.jpg")
results2 = model("./video/test_person.mp4",stream=True)

