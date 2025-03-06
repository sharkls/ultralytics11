from ultralytics import YOLOFusion

# Load a model
model = YOLOFusion("yolo11n-DEYOLO.yaml", task="detect") # build a new model from YAML
# model = YOLOFusion("yolo11n-FMDEA.yaml", task="detect") # build a new model from YAML

# model = YOLOFusion("./ckpt/yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLOFusion("yolo11n.yaml").load("./ckpt/yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="multimodal.yaml", batch=2, epochs=2, imgsz=640, device=0)