# https://www.bilibili.com/video/BV1AeWoenEVT
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# 1. 数据来源source  ： 图像路径、视频、目录、URL、设备ID
# 2. conf 最小置信度： 0.25
# 3. iou 交并比 0.7
# 4. max_det 300
# 5. classes : [2,3,7] 只返回指定类别的检测结果
# 6. show : 窗口显示结果
# 7. save 保存结果文件
# 8. save_frames ： 逐帧分析视频 ，但是系统会报warning
# 9. save_txt :将检测结果保存成文本文件
# 10. save_conf : 可以在文本文件中保存置信度
# 11. show_labels : show 为True的时候，是否显示标签
# 12. show_conf : 是否显示置信度
# 13. show_boxes : 是否显示框
# 14. line_width: 设置框框的宽度
OBJ_CLASS_IDS = [0, 39, 53]
results = model.predict("./images/000177.jpg", show=True, line_width=1)
# results = model.predict("./video/test_traffic.mp4")
# model.track()
cv2.waitKey(30000)
