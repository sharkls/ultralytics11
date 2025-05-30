from ultralytics import YOLOMultimodal, YOLO
import argparse
# Load a model
# model = YOLOFusion("./ckpt/yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLOFusion("yolo11n.yaml").load("./ckpt/yolo11n.pt")  # build from YAML and transfer weights

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", type=str, default="detect", help="task name: multimodal or detect")
#     parser.add_argument("--config", type=str, default="yolo11s.yaml", help="config file: yolo11s.yaml or yolo11n-DEYOLO.yaml or yolo11n-FMDEA.yaml")
#     parser.add_argument("--data", type=str, default="LLVIP.yaml", help="data path: LLVIP.yaml or multimodal.yaml")
#     parser.add_argument("--batch", type=int, default=2, help="batch size")
#     parser.add_argument("--epochs", type=int, default=2, help="epochs")
#     parser.add_argument("--imgsz", type=int, default=640, help="image size")
#     parser.add_argument("--device", type=str, default="0", help="device")
#     args = parser.parse_args()

#     # 模型训练
#     if args.task == "detect":
#         model = YOLO(args.config, args.task) # build a new model from YAML
#     elif args.task == "multimodal":
#         model = YOLOMultimodal(args.config, args.task) # build a new model from YAML
#     results = model.train(data=args.data, batch=args.batch, epochs=args.epochs, imgsz=args.imgsz, device=args.device)


# # # 训练单模态数据
# model = YOLO("yolo11s.yaml", task="detect") # build a new model from YAML
# results = model.train(data="multimodal_ir.yaml", batch=2, epochs=2, imgsz=640, device=0)

# 训练多模态（DEYOLO）
# model = YOLOMultimodal("yolo11n-DEYOLO.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal.yaml", batch=2, epochs=2, imgsz=640, device=0)

# # 训练多模态（FMDEA）
# model = YOLOMultimodal("yolo11n-FMDEA.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal.yaml", batch=2, epochs=2, imgsz=640, device=0)

# # # 训练多模态（EnhancedFMDEA）- 2025.4.21
# model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal_test.yaml", batch=2, epochs=2, imgsz=640, device=0)

# # 训练多模态（EFDEAYOLO）-2025.4.14
# model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal.yaml", batch=64, epochs=300, imgsz=640, device=0)



# 使用100帧LLVIP_RAW数据训练多模态（EFDEAYOLO）-2025.5.16
# model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal_llvip_raw_test.yaml", batch=16, epochs=300, imgsz=640, device=0)

# # # 使用自定义数据训练单模态（YOLO）-2025.5.21
# model = YOLO("yolo11s.yaml", task="detect") # build a new model from YAML
# results = model.train(data="multimodal-Myself-v2.yaml", batch=32, epochs=300, imgsz=640, device=0)

# # # 使用自定义数据集训练多模态（EFDEAYOLO）-2025.5.21
# model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal-Myself.yaml", batch=32, epochs=300, imgsz=640, device=0)

# # 使用自定义数据集训练多模态（EFDEAYOLO）-2025.5.21
# model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal-0521.yaml", batch=2, epochs=4, imgsz=640, device=0)


# # # # 使用自定义数据集Myself-v2训练多模态（EFDEAYOLO）-2025.5.21
# model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal") # build a new model from YAML
# results = model.train(data="multimodal-Myself-v2.yaml", batch=32, epochs=300, imgsz=640, device=0)

# # 使用自定义数据集Myself-v3训练多模态（EFDEAYOLO）-2025.5.26
model = YOLOMultimodal("yolo11s-EFDEA.yaml", task="multimodal") # build a new model from YAML
results = model.train(data="multimodal-Myself-clean-0529.yaml", batch=32, epochs=300, imgsz=640, device=0)

# # # 使用自定义数据训练单模态（YOLO）-2025.5.27
# model = YOLO("yolo11s.yaml", task="detect") # build a new model from YAML
# results = model.train(data="multimodal-Myself-v3.yaml", batch=32, epochs=300, imgsz=640, device=0)


# 断点续训
# 2. 断点续训 - 从最后一个检查点继续训练
# 假设上次训练到第86个epoch中断
# model = YOLOMultimodal('runs/multimodal/multimodal0317/last.pt')  # 加载最后保存的权重
# results = model.train(
#     data="multimodal.yaml",
#     batch=2,
#     epochs=4,
#     imgsz=640,
#     device=0,
#     resume=True  # 关键参数：启用续训
# )