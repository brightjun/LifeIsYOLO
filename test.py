import os
from ultralytics import YOLO
import torch
import j

ROOTDIR = "/home/ubuntu/LifeIsYOLO"

model = YOLO(
    "yolov8n.pt"
)  # COCO dataset으로 pretrained된 model을 불러온다. 크기에 따라서 5개의 모델 존재(n, s, m, l, x)

# Use the model
results = model.predict(
    ROOTDIR + "/img.jpeg"
)  # ultralytics 패키지 내에 test용으로 이미 존재해 있는 버스 이미지를 이용해 object detection 수행
result_json = results[0].tojson()
imgSize = results[0].orig_shape

result_json = {"imgSize": imgSize, "Labels": result_json}
print(result_json)
