#coding:utf-8
from ultralytics import YOLO

# 加载训练模型
#model = YOLO("ultralytics/cfg/models/v8/yolov8_SEAttention.yaml").load('yolov8n.pt')
#model = YOLO("ultralytics/cfg/models/v8/yolov8_ECAAttention.yaml").load('yolov8n.pt')
model = YOLO("ultralytics/cfg/models/v8/yolov8_SPTS.yaml").load('yolov8n.pt')
#model = YOLO("ultralytics/cfg/models/v8/yolov8_PHCFE.yaml").load('yolov8n.pt')
#model = YOLO("ultralytics/cfg/models/v8/yolov8_CFFAttention.yaml").load('yolov8n.pt')

# Use the model
# if __name__ == '__main__':
#     # Use the model
     #results = model.train(data='datasets/PoseData/data.yaml', epochs=250, batch=4)  # 训练模型



