from ultralytics.models.yolo.model import YOLOFusion

# Load a model
model = YOLOFusion("/ultralytics/ultralytics/cfg/models/v8/yolov8-DEYOLO.yaml").load("yolov8n.pt")

# Train the model
train_results = model.train(    
    data="multimodal.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    batch=8,
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)