# from ultralytics import YOLO
#
# model = YOLO(r"yolo11s.pt")
# model.predict(
#     source=r"D:\picture\people\people.mp4",
#     save=False,
#     show=True,
# )

# print(model.task)
# print(model.names)

# # 可以通过代码查看
# from ultralytics.utils import SETTINGS
# print(SETTINGS['datasets_dir'])
# # 通常位置：
# # Windows: C:\Users\<用户名>\AppData\Roaming\Ultralytics\datasets
# # Linux/Mac: ~/.config/Ultralytics/datasets

# 用摄像头
from ultralytics import YOLO

model = YOLO("results/yolo11m_phone_ft960_2/weights/best.pt")

model.predict(
    source=0,      # 摄像头
    show=True,     # 显示窗口
    save=False,     # 不保存
    conf=0.7
)