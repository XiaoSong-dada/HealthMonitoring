from ultralytics import YOLO

# Load a model
model = YOLO(r'./ultralytics/cfg/models/v8/yolov8-pose.yaml')


# Train the model
results = model.train(data='ultralytics/cfg/datasets/coco8-pose.yaml', epochs=100, imgsz=640)
