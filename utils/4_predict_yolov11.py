# from ultralytics import YOLOFusion

# # Load a model
# model = YOLOFusion("yolo11n-DEYOLO.yaml", task="detect").load('/ultralytics/runs/detect/train8/weights/last.pt')  # pretrained YOLO11n model

# # Run batched inference on a list of images
# results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


from ultralytics import YOLOMultimodal

# Load a model
model = YOLOMultimodal("yolo11n-DEYOLO.yaml", task="multimodal").load('/ultralytics/runs/multimodal/train25/weights/last.pt')  # pretrained YOLO11n model

# Perform object detection on RGB and IR image
model.predict([["/ultralytics/data/LLVIP/images/visible/test/190001.jpg", "/ultralytics/data/LLVIP/images/infrared/test/190001.jpg"], # corresponding image pair
              ["/ultralytics/data/LLVIP/images/visible/test/190002.jpg", "/ultralytics/data/LLVIP/images/infrared/test/190002.jpg"]], 
              save=True, imgsz=640, conf=0.5)