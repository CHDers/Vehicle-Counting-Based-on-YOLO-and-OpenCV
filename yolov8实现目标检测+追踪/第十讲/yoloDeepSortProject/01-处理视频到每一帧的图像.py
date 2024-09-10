
import cv2
import numpy as np

VIDEO_PATH = "./video/test_person.mp4"
RESULT_PATH = "result1.mp4"

#指定多边形的顶点坐标
polygonPoints = np.array([[710,200],[1110,200],[810,400],[410,400]],dtype=np.int32)


if __name__ == '__main__':
   capture = cv2.VideoCapture(VIDEO_PATH)
   if not capture.isOpened():
       print("Error opening video file.")
       exit()
   fps = capture.get(cv2.CAP_PROP_FPS) # 获取帧率
   frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) # 获取宽度
   frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) #获取高度

   videoWriter = None

   while True:
       success, frame = capture.read() #读取视频中的一帧
       if not success:
           print("读取帧失败")
           break
       #  绘制线段
       cv2.line(frame,(0,int(frame_height/2)),(int(frame_width),int(frame_height/2)),(0,0,255),3)

       #  绘制多边形
       cv2.polylines(frame, [polygonPoints], True,(0,0,255),3)

       # 绘制一个与frame大小一样的图像
       mask = np.zeros_like(frame)
       cv2.fillPoly(mask,[polygonPoints],(0,0,255))
       # 融合两张图像
       frame = cv2.addWeighted(frame,0.7,mask,0.3,0)

       if videoWriter is None:
           fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
           videoWriter = cv2.VideoWriter(RESULT_PATH,fourcc,fps,(int(frame_width),int(frame_height)))

       videoWriter.write(frame)
       cv2.imshow("frame",frame)
       cv2.waitKey(1)

   capture.release() #释放capture资源
   videoWriter.release()
   cv2.destroyAllWindows() #关闭掉所有的窗口