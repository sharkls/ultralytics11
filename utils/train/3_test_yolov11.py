# from ultralytics import YOLOFusion

# # Load a model
# model = YOLOFusion("yolo11n-DEYOLO.yaml", task="detect").load('/ultralytics/runs/detect/train8/weights/last.pt') # build a new model from YAML
# # model = YOLOFusion("./ckpt/yolo11n.pt")  # load a pretrained model (recommended for training)
# # model = YOLOFusion("yolo11n.yaml").load("./ckpt/yolo11n.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.test(data="multimodal.yaml", batch=2, imgsz=640, device=0)

from ultralytics import YOLOMultimodal, YOLO

model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal").load('/ultralytics/runs/multimodal/train66/weights/best.pt')  # load an official model
metrics = model.val(data="multimodal-Myself-v3-test.yaml", batch=16, imgsz=640, device=0, show=True)  # no arguments needed, dataset and settings remembered

# model = YOLO("yolo11s.yaml", task="detect").load('/ultralytics/runs/detect/train8/weights/best.pt')  # load an official model
# metrics = model.val(data="multimodal-Myself-v2-test.yaml", batch=16, imgsz=640, device=0, show=True) 



# Validate the model
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category


