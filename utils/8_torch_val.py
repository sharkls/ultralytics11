import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.augment import LetterBox
import json
import os
from ultralytics import YOLOMultimodal

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Multimodal PyTorch Model Validation')
    
    # 模型相关参数
    parser.add_argument('--pt-path', type=str, default='runs/multimodal/train6/weights/last.pt',
                      help='PyTorch模型路径')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640],
                      help='输入图像尺寸 [height, width]')
    
    # 数据集相关参数
    parser.add_argument('--data-dir', type=str, default='./data/LLVIP',
                      help='数据集根目录')
    parser.add_argument('--split', type=str, default='test',
                      help='数据集划分 (train/val/test)')
    
    # 检测相关参数
    parser.add_argument('--conf-thres', type=float, default=0.5,
                      help='检测置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.001,
                      help='NMS IOU阈值')
    parser.add_argument('--nc', type=int, default=1,
                      help='目标类别数量')
    
    # 批处理参数
    parser.add_argument('--batch-size', type=int, default=1,
                      help='批处理大小')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                      help='运行设备 cuda/cpu')
    parser.add_argument('--save-dir', type=str, default='runs/torch_val',
                      help='结果保存目录')
    parser.add_argument('--visualize', type=bool, default=True,
                      help='是否可视化检测结果')
    
    return parser.parse_args()

def load_extrinsics(extrinsics_path):
    """加载外参矩阵"""
    try:
        homography = np.loadtxt(extrinsics_path)
        if homography.shape != (3, 3):
            raise ValueError(f"单应性矩阵形状错误: {homography.shape}, 应为 (3, 3)")
        return homography
    except Exception as e:
        print(f"加载外参矩阵失败: {e}")
        return None

def preprocess_batch(rgb_paths, ir_paths, extrinsics_list, imgsz=(640, 640), stride=32, device='cuda'):
    """批量图像预处理函数"""
    batch_size = len(rgb_paths)
    rgb_inputs = []
    ir_inputs = []
    updated_Hs = []
    scale_factors = []
    letterbox_infos = []  # 存储letterbox处理信息
    
    for i in range(batch_size):
        # 读取图像
        rgb_img = cv2.imread(rgb_paths[i])
        ir_img = cv2.imread(ir_paths[i])
        if rgb_img is None or ir_img is None:
            raise FileNotFoundError(f"无法读取图像: {rgb_paths[i]} 或 {ir_paths[i]}")
        
        # 保存原始图像尺寸用于后续缩放
        img_h, img_w = rgb_img.shape[:2]
        scale_factor = np.array([img_w/imgsz[1], img_h/imgsz[0], 
                               img_w/imgsz[1], img_h/imgsz[0]])
        scale_factors.append(scale_factor)
        
        # 创建letterbox对象
        letterbox = LetterBox(imgsz, auto=False, stride=stride)
        
        # 计算RGB图像的letterbox参数
        rgb_h, rgb_w = rgb_img.shape[:2]
        r_rgb = min(imgsz[0]/rgb_h, imgsz[1]/rgb_w)
        new_unpad_rgb = int(round(rgb_w * r_rgb)), int(round(rgb_h * r_rgb))
        dw_rgb, dh_rgb = imgsz[1] - new_unpad_rgb[0], imgsz[0] - new_unpad_rgb[1]
        dw_rgb /= 2
        dh_rgb /= 2

        # 计算IR图像的letterbox参数
        ir_h, ir_w = ir_img.shape[:2]
        r_ir = min(imgsz[0]/ir_h, imgsz[1]/ir_w)
        new_unpad_ir = int(round(ir_w * r_ir)), int(round(ir_h * r_ir))
        dw_ir, dh_ir = imgsz[1] - new_unpad_ir[0], imgsz[0] - new_unpad_ir[1]
        dw_ir /= 2
        dh_ir /= 2

        # 构建变换矩阵
        S_rgb = np.eye(3)
        S_rgb[0, 0] = r_rgb
        S_rgb[1, 1] = r_rgb

        S_ir = np.eye(3)
        S_ir[0, 0] = r_ir
        S_ir[1, 1] = r_ir

        T_rgb = np.eye(3)
        T_rgb[0, 2] = dw_rgb
        T_rgb[1, 2] = dh_rgb

        T_ir = np.eye(3)
        T_ir[0, 2] = dw_ir
        T_ir[1, 2] = dh_ir

        # 更新单应性矩阵
        updated_H = T_rgb @ S_rgb @ extrinsics_list[i] @ np.linalg.inv(S_ir) @ np.linalg.inv(T_ir)
        updated_H = np.expand_dims(updated_H, 0)  # [3, 3] -> [1, 3, 3]
        updated_Hs.append(updated_H)
        
        # 应用letterbox变换
        rgb_img = letterbox(image=rgb_img)
        ir_img = letterbox(image=ir_img)
        
        # 保存letterbox处理信息
        letterbox_info = {
            'dw': dw_rgb,
            'dh': dh_rgb,
            'ratio': r_rgb
        }
        letterbox_infos.append(letterbox_info)
        
        # 图像格式转换
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        rgb_img = rgb_img / 255.0
        ir_img = ir_img / 255.0
        
        # HWC to CHW
        rgb_img = rgb_img.transpose(2, 0, 1)
        ir_img = ir_img.transpose(2, 0, 1)
        
        # 转换为tensor
        rgb_img = torch.from_numpy(rgb_img).float()
        ir_img = torch.from_numpy(ir_img).float()
        
        # 添加batch维度
        rgb_img = rgb_img.unsqueeze(0)  # (1, C, H, W)
        ir_img = ir_img.unsqueeze(0)    # (1, C, H, W)
        
        rgb_inputs.append(rgb_img)
        ir_inputs.append(ir_img)
    
    # 堆叠batch
    rgb_inputs = torch.cat(rgb_inputs, dim=0)  # (batch_size, C, H, W)
    ir_inputs = torch.cat(ir_inputs, dim=0)    # (batch_size, C, H, W)
    updated_Hs = torch.from_numpy(np.concatenate(updated_Hs, axis=0)).float()  # (batch_size, 3, 3)
    
    # 将数据移到指定设备
    rgb_inputs = rgb_inputs.to(device)
    ir_inputs = ir_inputs.to(device)
    updated_Hs = updated_Hs.to(device)
    
    return rgb_inputs, ir_inputs, updated_Hs, scale_factors, letterbox_infos

def process_output(output, conf_thres=0.25, iou_thres=0.45, scale_factor=None, nc=1, letterbox_info=None):
    """处理YOLO模型输出，应用NMS"""
    # print(f"Input output shape: {output.shape}")
    
    # 确保输出是2D张量
    if len(output.shape) > 2:
        output = output.squeeze(0)
    # print(f"After squeeze shape: {output.shape}")
    
    # 转置为[8400, 4+nc]格式
    output = output.T  # [4+nc, 8400] -> [8400, 4+nc]
    # print(f"After transpose shape: {output.shape}")
    
    # 获取类别分数
    cls_scores = output[:, 4:4+nc]
    cls_ids = torch.argmax(cls_scores, dim=1)
    scores = torch.max(cls_scores, dim=1)[0]
    
    # 将xywh转换为xyxy格式
    boxes = torch.zeros_like(output[:, :4])
    boxes[:, 0] = output[:, 0] - output[:, 2] / 2  # x1
    boxes[:, 1] = output[:, 1] - output[:, 3] / 2  # y1
    boxes[:, 2] = output[:, 0] + output[:, 2] / 2  # x2
    boxes[:, 3] = output[:, 1] + output[:, 3] / 2  # y2
    
    # 如果提供了letterbox信息，将检测框坐标转换回原始图像尺寸
    if letterbox_info is not None:
        # 1. 先减去填充（反填充）
        boxes[:, 0] = boxes[:, 0] - letterbox_info['dw']  # x1
        boxes[:, 1] = boxes[:, 1] - letterbox_info['dh']  # y1
        boxes[:, 2] = boxes[:, 2] - letterbox_info['dw']  # x2
        boxes[:, 3] = boxes[:, 3] - letterbox_info['dh']  # y2
        
        # 2. 再除以缩放比例（反缩放）
        boxes = boxes / letterbox_info['ratio']
    
    # 应用置信度阈值
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    
    # 对每个类别分别应用NMS
    results = []
    for cls_id in range(nc):
        # 使用布尔掩码进行索引
        cls_mask = (cls_ids == cls_id)
        if not torch.any(cls_mask):
            continue
            
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # 应用NMS
        indices = nms(cls_boxes, cls_scores, iou_thres)
        
        # 组合结果 [x1, y1, x2, y2, conf, class]
        cls_results = torch.zeros((len(indices), 6), device=boxes.device)
        cls_results[:, :4] = cls_boxes[indices]
        cls_results[:, 4] = cls_scores[indices]
        cls_results[:, 5] = cls_id
        
        results.append(cls_results)
    
    if len(results) == 0:
        return torch.zeros((0, 6), device=boxes.device)
    
    final_results = torch.cat(results)
    return final_results

def nms(boxes, scores, iou_threshold):
    """非极大值抑制(NMS)实现"""
    # 按置信度分数降序排序
    order = scores.argsort(descending=True)
    keep = []
    
    while order.numel() > 0:
        # 保留当前最高分的框
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
            
        # 计算当前框与其他框的IOU
        xx1 = torch.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = torch.maximum(torch.zeros_like(xx2), xx2 - xx1)
        h = torch.maximum(torch.zeros_like(yy2), yy2 - yy1)
        intersection = w * h
        
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-16)
        
        # 保留IOU小于阈值的框
        inds = torch.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return torch.tensor(keep)

def load_annotations(annotation_path, rgb_dir):
    """加载标注文件"""
    annotations = {}
    for label_file in annotation_path.glob('*.txt'):
        img_name = label_file.stem
        # 获取原始图像尺寸
        rgb_path = rgb_dir / f"{img_name}.jpg"
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            print(f"警告：无法读取图像 {rgb_path}")
            continue
        img_h, img_w = rgb_img.shape[:2]
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            boxes = []
            for line in lines:
                # YOLO格式: class x_center y_center width height (归一化坐标)
                cls, x, y, w, h = map(float, line.strip().split())
                # 转换为像素坐标 [x1, y1, x2, y2, class]
                x1 = (x - w/2) * img_w
                y1 = (y - h/2) * img_h
                x2 = (x + w/2) * img_w
                y2 = (y + h/2) * img_h
                boxes.append([x1, y1, x2, y2, int(cls)])
            annotations[img_name] = torch.tensor(boxes)
    return annotations

def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    # 获取相交区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算相交区域面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union = box1_area + box2_area - intersection
    
    # 计算IOU
    iou = intersection / (union + 1e-16)
    
    return iou

def calculate_metrics(cls_detections, cls_gt_boxes, iou_thres):
    """计算评估指标"""
    # 初始化
    total_gt = len(cls_gt_boxes)
    total_det = len(cls_detections)
    
    if total_gt == 0:
        return torch.zeros(total_det), torch.zeros(total_det), torch.tensor(0, dtype=torch.int32)
    
    # 计算IOU矩阵
    iou_matrix = torch.zeros((total_det, total_gt))
    for i, det in enumerate(cls_detections):
        for j, gt in enumerate(cls_gt_boxes):
            iou_matrix[i, j] = calculate_iou(det, gt)
    
    # 初始化TP和FP
    tps = torch.zeros(total_det)
    fps = torch.zeros(total_det)
    gt_matched = torch.zeros(total_gt)
    
    # 按置信度排序
    det_scores = cls_detections[:, 4]
    sort_idx = torch.argsort(-det_scores)
    
    # 分配检测框
    for i in sort_idx:
        max_iou = torch.max(iou_matrix[i])
        if max_iou > iou_thres:
            gt_idx = torch.argmax(iou_matrix[i])
            if not gt_matched[gt_idx]:
                tps[i] = 1
                gt_matched[gt_idx] = 1
            else:
                fps[i] = 1
        else:
            fps[i] = 1
    
    # 计算FN
    fn = total_gt - torch.sum(gt_matched)
    fn = fn.to(torch.int32)  # 确保fn是int32类型
    
    return tps, fps, fn

def evaluate_torch(args):
    """评估PyTorch模型"""
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"正在加载PyTorch模型 {args.pt_path} ...")
    model = YOLOMultimodal(args.pt_path)
    model.eval()
    model.to(args.device)
    
    # 加载数据集
    data_dir = Path(args.data_dir)
    extrinsics_dir = data_dir / 'extrinsics' / args.split
    rgb_dir = data_dir / 'images' / 'visible' / args.split
    ir_dir = data_dir / 'images' / 'infrared' / args.split
    annotation_dir = data_dir / 'labels' / 'visible' / args.split
    
    # 加载标注
    annotations = load_annotations(annotation_dir, rgb_dir)
    
    # 初始化评估指标
    stats = []
    seen = 0
    
    # 准备批处理数据
    img_names = list(annotations.keys())
    num_batches = (len(img_names) + args.batch_size - 1) // args.batch_size
    
    # 遍历测试集
    print(f"正在评估模型...")
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, len(img_names))
            batch_names = img_names[start_idx:end_idx]
            
            # 准备批处理输入数据
            rgb_paths = []
            ir_paths = []
            extrinsics_list = []
            for img_name in batch_names:
                rgb_paths.append(rgb_dir / f"{img_name}.jpg")
                ir_paths.append(ir_dir / f"{img_name}.jpg")
                extrinsics_path = extrinsics_dir / f"{img_name}.txt"
                extrinsics = load_extrinsics(extrinsics_path)
                if extrinsics is None:
                    continue
                extrinsics_list.append(extrinsics)
            
            if len(extrinsics_list) == 0:
                continue
                
            # 预处理批处理数据
            try:
                rgb_inputs, ir_inputs, updated_Hs, scale_factors, letterbox_infos = preprocess_batch(
                    rgb_paths,
                    ir_paths,
                    extrinsics_list,
                    args.imgsz,
                    stride=32,
                    device=args.device
                )
                
                # 将数据移到指定设备
                rgb_inputs = rgb_inputs.to(args.device)
                ir_inputs = ir_inputs.to(args.device)
                updated_Hs = updated_Hs.to(args.device)
                
            except Exception as e:
                print(f"批处理图像预处理失败: {e}")
                continue
                
            try:
                # 修改推理方式，参照7_export.py
                # 确保输入数据格式正确
                rgb_tensor = rgb_inputs.to(args.device)  # [batch_size, C, H, W]
                ir_tensor = ir_inputs.to(args.device)    # [batch_size, C, H, W]
                updated_Hs = updated_Hs.to(args.device)  # [batch_size, 3, 3]
                
                # # 打印调试信息
                # print(f"RGB tensor shape: {rgb_tensor.shape}")
                # print(f"IR tensor shape: {ir_tensor.shape}")
                # print(f"Homography shape: {updated_Hs.shape}")
                
                # 执行推理
                outputs = model.forward_multimodal(rgb_tensor, ir_tensor, updated_Hs)
                # print(f"Raw outputs type: {type(outputs)}")
                # if isinstance(outputs, tuple):
                #     print(f"Tuple length: {len(outputs)}")
                #     print(f"First element shape: {outputs[0].shape}")
                
                # 处理输出
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                # print(f"Processed outputs shape: {outputs.shape}")
                
                # 处理每个样本的检测结果
                for i, img_name in enumerate(batch_names):
                    if i >= len(scale_factors):  # 确保索引不越界
                        continue
                        
                    # 获取当前样本的输出
                    current_output = outputs[i:i+1]  # 保持batch维度
                    # print(f"Current output shape: {current_output.shape}")
                    
                    detections = process_output(current_output, args.conf_thres, args.iou_thres, 
                                             scale_factors[i], args.nc, letterbox_infos[i])
                    
                    # 获取真实标注
                    gt_boxes = annotations[img_name]
                    
                    # 更新评估指标
                    seen += 1
                    if len(gt_boxes) > 0:
                        # 对每个类别分别计算TP, FP, FN
                        for cls_id in range(args.nc):
                            # 获取当前类别的检测结果和标注
                            cls_detections = detections[detections[:, 5] == cls_id]
                            cls_gt_boxes = gt_boxes[gt_boxes[:, 4] == cls_id]
                            
                            if len(cls_gt_boxes) == 0 and len(cls_detections) == 0:
                                continue
                            
                            # 计算TP, FP, FN
                            tps, fps, fn = calculate_metrics(cls_detections, cls_gt_boxes, args.iou_thres)
                            
                            # 保存统计信息
                            if len(cls_detections) > 0:
                                # 确保所有数组长度一致
                                confs = cls_detections[:, 4]  # 置信度
                                cls_ids = torch.full((len(confs),), cls_id, dtype=torch.int32)  # 类别ID
                                
                                stats.append([tps, fps, fn, confs, cls_ids])
                    
                    # 可视化结果
                    if args.visualize:
                        rgb_img = cv2.imread(str(rgb_paths[i]))
                        
                        # 绘制原始标签框（红色）
                        gt_boxes = annotations[img_name]
                        for gt_box in gt_boxes:
                            x1, y1, x2, y2, cls_id = gt_box
                            cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(rgb_img, f'GT cls{int(cls_id)}', (int(x1), int(y1)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # 绘制检测结果框（绿色）
                        for det in detections:
                            x1, y1, x2, y2, conf, cls_id = det
                            cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(rgb_img, f'Det cls{int(cls_id)} {conf:.2f}', (int(x1), int(y1)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 添加图例
                        cv2.putText(rgb_img, 'Red: Ground Truth', (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(rgb_img, 'Green: Detection', (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        cv2.imwrite(str(save_dir / f"{img_name}_det.jpg"), rgb_img)
                        
            except Exception as e:
                print(f"PyTorch推理失败: {e}")
                import traceback
                print(f"错误堆栈: {traceback.format_exc()}")
                continue
    
    # 计算AP
    if len(stats) > 0:
        # 解压统计信息
        tps_list, fps_list, fns_list, confs_list, classes_list = zip(*stats)
        
        # 连接所有数组并确保它们在同一个设备上
        tps = torch.cat(tps_list).cpu()  # 移到CPU
        fps = torch.cat(fps_list).cpu()  # 移到CPU
        confs = torch.cat(confs_list).cpu()  # 移到CPU
        classes = torch.cat(classes_list).cpu()  # 移到CPU
        
        # 打印最终数组信息
        print("\n最终数组信息:")
        print(f"TPs: {tps.shape}, {tps.dtype}")
        print(f"FPs: {fps.shape}, {fps.dtype}")
        print(f"Confs: {confs.shape}, {confs.dtype}")
        print(f"Classes: {classes.shape}, {classes.dtype}")
        
        # 计算AP
        try:
            # 将fns_list转换为tensor并确保长度匹配
            fns = torch.tensor(fns_list, dtype=torch.int32).cpu()  # 确保在CPU上
            # 如果fns是二维tensor，取其第一列
            if len(fns.shape) > 1:
                fns = fns[:, 0]  # 只取第一列
            # 如果fns长度不匹配，则重复最后一个值直到长度匹配
            if len(fns) < len(tps):
                fns = torch.cat([fns, fns[-1].repeat(len(tps) - len(fns))])
            
            # 确保所有数组长度一致
            min_len = min(len(tps), len(fps), len(confs), len(classes), len(fns))
            tps = tps[:min_len]
            fps = fps[:min_len]
            confs = confs[:min_len]
            classes = classes[:min_len]
            fns = fns[:min_len]
            
            # 打印处理后的数组信息
            print("\n处理后的数组信息:")
            print(f"TPs: {tps.shape}, {tps.dtype}")
            print(f"FPs: {fps.shape}, {fps.dtype}")
            print(f"Confs: {confs.shape}, {confs.dtype}")
            print(f"Classes: {classes.shape}, {classes.dtype}")
            print(f"FNs: {fns.shape}, {fns.dtype}")
            print(f"fns values: {fns}")
            
            # 按置信度排序
            sort_idx = torch.argsort(-confs)
            tps = tps[sort_idx]
            fps = fps[sort_idx]
            confs = confs[sort_idx]
            classes = classes[sort_idx]
            
            # 计算累积TP和FP
            tp_cumsum = torch.cumsum(tps, dim=0)
            fp_cumsum = torch.cumsum(fps, dim=0)
            
            # 计算召回率和精确率
            recall = tp_cumsum / (tp_cumsum[-1] + fns[-1] + 1e-16)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            
            # 计算AP
            ap = torch.zeros(args.nc, device='cpu')  # 确保在CPU上
            for cls_id in range(args.nc):
                cls_mask = classes == cls_id
                if not torch.any(cls_mask):
                    continue
                    
                cls_recall = recall[cls_mask]
                cls_precision = precision[cls_mask]
                
                # 计算AP
                mrec = torch.cat((torch.tensor([0.], device='cpu'), 
                                cls_recall, 
                                torch.tensor([1.], device='cpu')))
                mpre = torch.cat((torch.tensor([0.], device='cpu'), 
                                cls_precision, 
                                torch.tensor([0.], device='cpu')))
                
                # 计算曲线下面积
                for i in range(mpre.size(0) - 1, 0, -1):
                    mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
                i = torch.where(mrec[1:] != mrec[:-1])[0]
                ap[cls_id] = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            
            # 计算平均精确率和召回率
            p = torch.mean(precision)
            r = torch.mean(recall)
            
            # 打印结果
            print(f"\n评估结果:")
            for cls_id in range(args.nc):
                print(f"类别 {cls_id} AP@0.5: {ap[cls_id]:.4f}")
            print(f"mAP@0.5: {torch.mean(ap):.4f}")
            print(f"Precision: {p:.4f}")
            print(f"Recall: {r:.4f}")
            
            # 保存结果
            results = {
                'mAP@0.5': float(torch.mean(ap)),
                'Precision': float(p),
                'Recall': float(r),
                'conf_thres': args.conf_thres,
                'iou_thres': args.iou_thres,
                'class_AP': {f'class_{i}': float(ap[i]) for i in range(args.nc)}
            }
            
            with open(save_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=4)
                
        except Exception as e:
            print(f"计算AP时出错: {e}")
            print("调试信息:")
            print(f"tps shape: {tps.shape}, dtype: {tps.dtype}, device: {tps.device}")
            print(f"fps shape: {fps.shape}, dtype: {fps.dtype}, device: {fps.device}")
            print(f"confs shape: {confs.shape}, dtype: {confs.dtype}, device: {confs.device}")
            print(f"classes shape: {classes.shape}, dtype: {classes.dtype}, device: {classes.device}")
            print(f"classes values: {classes}")
            print(f"fns shape: {fns.shape}, dtype: {fns.dtype}, device: {fns.device}")
            print(f"fns values: {fns}")
    else:
        print("没有有效的评估结果")

def main():
    args = parse_args()
    evaluate_torch(args)

if __name__ == '__main__':
    main()