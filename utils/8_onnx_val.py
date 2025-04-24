import torch
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.augment import LetterBox
from ultralytics.utils.metrics import ap_per_class
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Multimodal ONNX Model Validation')
    
    # 模型相关参数
    parser.add_argument('--onnx-path', type=str, default='runs/multimodal/train6/weights/last.onnx',
                      help='ONNX模型路径')
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
    parser.add_argument('--save-dir', type=str, default='runs/onnx_val',
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

def preprocess_batch(rgb_paths, ir_paths, extrinsics_list, imgsz=(640, 640), stride=32):
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
        
        # 添加batch维度
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)  # (1, C, H, W)
        ir_img = np.expand_dims(ir_img, 0).astype(np.float32)    # (1, C, H, W)
        
        rgb_inputs.append(rgb_img)
        ir_inputs.append(ir_img)
    
    # 堆叠batch，保持每个样本的独立batch维度
    rgb_inputs = np.concatenate(rgb_inputs, axis=0)  # (batch_size, C, H, W)
    ir_inputs = np.concatenate(ir_inputs, axis=0)    # (batch_size, C, H, W)
    updated_Hs = np.concatenate(updated_Hs, axis=0).astype(np.float32)  # (batch_size, 3, 3)
    
    return rgb_inputs, ir_inputs, updated_Hs, scale_factors, letterbox_infos

def process_output(output, conf_thres=0.25, iou_thres=0.45, scale_factor=None, nc=1, letterbox_info=None):
    """处理YOLO模型输出，应用NMS
    
    Args:
        output: 模型输出，形状为[bs, 4+nc, 8400]，其中：
            - bs: batch size
            - 4: 位置信息(x, y, w, h)
            - nc: 类别数量
            - 8400: 预测框数量
        conf_thres: 置信度阈值
        iou_thres: NMS IOU阈值
        scale_factor: 缩放因子，用于将检测框坐标转换回原始图像尺寸
        nc: 类别数量
        letterbox_info: LetterBox处理信息，包含：
            - dw: 水平填充
            - dh: 垂直填充
            - ratio: 缩放比例
    """
    # 转置为[8400, 4+nc]格式
    output = output.squeeze(0).T  # [4+nc, 8400] -> [8400, 4+nc]
    
    # 获取类别分数
    cls_scores = output[:, 4:4+nc]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = np.max(cls_scores, axis=1)
    
    # 将xywh转换为xyxy格式
    boxes = np.zeros_like(output[:, :4])
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
        cls_mask = cls_ids == cls_id
        if not np.any(cls_mask):
            continue
            
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # 应用NMS
        indices = nms(cls_boxes, cls_scores, iou_thres)
        
        # 组合结果 [x1, y1, x2, y2, conf, class]
        cls_results = np.zeros((len(indices), 6))
        cls_results[:, :4] = cls_boxes[indices]
        cls_results[:, 4] = cls_scores[indices]
        cls_results[:, 5] = cls_id
        
        results.append(cls_results)
    
    if len(results) == 0:
        return np.zeros((0, 6))
    
    final_results = np.vstack(results)
    return final_results

def nms(boxes, scores, iou_threshold):
    """非极大值抑制(NMS)实现"""
    # 按置信度分数降序排序
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        # 保留当前最高分的框
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # 计算当前框与其他框的IOU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area1 + area2 - intersection
        
        iou = intersection / union
        
        # 保留IOU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return np.array(keep)

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
            annotations[img_name] = np.array(boxes)
    return annotations

def calculate_iou(box1, box2):
    """计算两个边界框的IOU
    
    Args:
        box1: 第一个边界框 [x1, y1, x2, y2, ...]
        box2: 第二个边界框 [x1, y1, x2, y2, ...]
    
    Returns:
        float: IOU值
    """
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
    iou = intersection / (union + 1e-16)  # 添加小量避免除零
    
    return iou

def calculate_metrics(cls_detections, cls_gt_boxes, iou_thres):
    # 初始化
    total_gt = len(cls_gt_boxes)
    total_det = len(cls_detections)
    
    if total_gt == 0:
        return np.zeros(total_det), np.zeros(total_det), 0
    
    # 计算IOU矩阵
    iou_matrix = np.zeros((total_det, total_gt))
    for i, det in enumerate(cls_detections):
        for j, gt in enumerate(cls_gt_boxes):
            iou_matrix[i, j] = calculate_iou(det, gt)
    
    # 初始化TP和FP
    tps = np.zeros(total_det)
    fps = np.zeros(total_det)
    gt_matched = np.zeros(total_gt)
    
    # 按置信度排序
    det_scores = cls_detections[:, 4]
    sort_idx = np.argsort(-det_scores)
    
    # 分配检测框
    for i in sort_idx:
        max_iou = np.max(iou_matrix[i])
        if max_iou > iou_thres:
            gt_idx = np.argmax(iou_matrix[i])
            if not gt_matched[gt_idx]:
                tps[i] = 1
                gt_matched[gt_idx] = 1
            else:
                fps[i] = 1
        else:
            fps[i] = 1
    
    # 计算FN
    fn = total_gt - np.sum(gt_matched)
    
    return tps, fps, fn

def evaluate_onnx(args):
    """评估ONNX模型"""
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载ONNX模型
    print(f"正在加载ONNX模型 {args.onnx_path} ...")
    
    # 设置ONNX运行时选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_cpu_mem_arena = False  # 禁用内存区域
    sess_options.enable_mem_pattern = False    # 禁用内存模式
    sess_options.enable_mem_reuse = False      # 禁用内存重用
    sess_options.intra_op_num_threads = 1      # 使用单线程
    sess_options.inter_op_num_threads = 1      # 使用单线程
    
    # 创建ONNX会话
    session = ort.InferenceSession(
        args.onnx_path,
        providers=['CUDAExecutionProvider','CPUExecutionProvider'],
        sess_options=sess_options
    )
    
    # 获取模型输入信息
    input_info = session.get_inputs()
    input_names = [input.name for input in input_info]
    input_shapes = [input.shape for input in input_info]
    
    # 打印模型输入信息
    print("\n模型输入信息:")
    for name, shape in zip(input_names, input_shapes):
        print(f"  {name}: {shape}")
    
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
                stride=32  # 使用固定步长
            )
        except Exception as e:
            print(f"批处理图像预处理失败: {e}")
            continue
            
        # ONNX推理
        onnx_inputs = {
            input_names[0]: rgb_inputs,
            input_names[1]: ir_inputs,
            input_names[2]: updated_Hs
        }
        
        try:
            # 运行推理
            output_names = [output.name for output in session.get_outputs()]
            onnx_outputs = session.run(output_names, onnx_inputs)
            
            # 处理每个样本的检测结果
            for i, img_name in enumerate(batch_names):
                if i >= len(scale_factors):  # 确保索引不越界
                    continue
                    
                # 获取当前样本的输出
                current_output = onnx_outputs[0][i:i+1]  # 保持batch维度
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
                            cls_ids = np.full(len(confs), cls_id, dtype=np.int32)  # 类别ID
                            
                            # 确保fn是一个标量值
                            fn = np.array([fn], dtype=np.int32)
                            
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
            print(f"ONNX推理失败: {e}")
            print("输入数据信息:")
            for name, value in onnx_inputs.items():
                print(f"{name}: 形状={value.shape}, 类型={value.dtype}, 范围=[{value.min()}, {value.max()}]")
            continue
    
    # 计算AP
    if len(stats) > 0:
        # 解压统计信息
        tps_list, fps_list, fns_list, confs_list, classes_list = zip(*stats)
        
        # 连接所有数组
        tps = np.concatenate(tps_list)
        fps = np.concatenate(fps_list)
        confs = np.concatenate(confs_list)
        classes = np.concatenate(classes_list)
        
        # 打印最终数组信息
        print("\n最终数组信息:")
        print(f"TPs: {tps.shape}, {tps.dtype}")
        print(f"FPs: {fps.shape}, {fps.dtype}")
        print(f"Confs: {confs.shape}, {confs.dtype}")
        print(f"Classes: {classes.shape}, {classes.dtype}")
        
        # 计算AP
        try:
            # 将fns_list转换为数组并确保长度匹配
            fns = np.array(fns_list, dtype=np.int32)
            # 如果fns是二维数组，取其第一列
            if len(fns.shape) > 1:
                fns = fns[:, 0]  # 只取第一列
            # 如果fns长度不匹配，则重复最后一个值直到长度匹配
            if len(fns) < len(tps):
                fns = np.pad(fns, (0, len(tps) - len(fns)), mode='edge')
            
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
            
            # 按置信度排序
            sort_idx = np.argsort(-confs)
            tps = tps[sort_idx]
            fps = fps[sort_idx]
            confs = confs[sort_idx]
            classes = classes[sort_idx]
            
            # 计算累积TP和FP
            tp_cumsum = np.cumsum(tps)
            fp_cumsum = np.cumsum(fps)
            
            # 计算召回率和精确率
            recall = tp_cumsum / (tp_cumsum[-1] + fns[-1] + 1e-16)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            
            # 计算AP
            ap = np.zeros(args.nc)
            for cls_id in range(args.nc):
                cls_mask = classes == cls_id
                if not np.any(cls_mask):
                    continue
                    
                cls_recall = recall[cls_mask]
                cls_precision = precision[cls_mask]
                
                # 计算AP
                mrec = np.concatenate(([0.], cls_recall, [1.]))
                mpre = np.concatenate(([0.], cls_precision, [0.]))
                
                # 计算曲线下面积
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                i = np.where(mrec[1:] != mrec[:-1])[0]
                ap[cls_id] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            
            # 计算平均精确率和召回率
            p = np.mean(precision)
            r = np.mean(recall)
            
            # 打印结果
            print(f"\n评估结果:")
            for cls_id in range(args.nc):
                print(f"类别 {cls_id} AP@0.5: {ap[cls_id]:.4f}")
            print(f"mAP@0.5: {np.mean(ap):.4f}")
            print(f"Precision: {p:.4f}")
            print(f"Recall: {r:.4f}")
            
            # 保存结果
            results = {
                'mAP@0.5': float(np.mean(ap)),
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
            print(f"tps shape: {tps.shape}, dtype: {tps.dtype}")
            print(f"fps shape: {fps.shape}, dtype: {fps.dtype}")
            print(f"confs shape: {confs.shape}, dtype: {confs.dtype}")
            print(f"classes shape: {classes.shape}, dtype: {classes.dtype}")
            print(f"classes values: {classes}")
            print(f"fns shape: {fns.shape}, dtype: {fns.dtype}")
            print(f"fns values: {fns}")
    else:
        print("没有有效的评估结果")

def validate_onnx(onnx_path, inputs, updated_homography_np, torch_output, error_threshold, args):
    """验证ONNX模型输出"""
    rgb_input, ir_input = inputs
    providers = ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        return False
        
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    
    # 打印输入输出信息
    print("\nONNX模型输入输出信息:")
    for input in session.get_inputs():
        print(f"输入: {input.name}, 形状: {input.shape}, 类型: {input.type}")
    for output in session.get_outputs():
        print(f"输出: {output.name}, 形状: {output.shape}, 类型: {output.type}")
    
    # 确保输入数据形状正确
    if rgb_input.shape[0] != ir_input.shape[0] or rgb_input.shape[0] != updated_homography_np.shape[0]:
        print(f"警告：输入数据的批处理大小不匹配")
        print(f"RGB输入形状: {rgb_input.shape}")
        print(f"IR输入形状: {ir_input.shape}")
        print(f"外参矩阵形状: {updated_homography_np.shape}")
        # 调整批处理大小
        batch_size = min(rgb_input.shape[0], ir_input.shape[0], updated_homography_np.shape[0])
        rgb_input = rgb_input[:batch_size]
        ir_input = ir_input[:batch_size]
        updated_homography_np = updated_homography_np[:batch_size]
        print(f"已调整批处理大小为: {batch_size}")

    # 确保数据类型正确
    rgb_input = rgb_input.astype(np.float32)
    ir_input = ir_input.astype(np.float32)
    updated_homography_np = updated_homography_np.astype(np.float32)

    # 打印处理后的输入信息
    print("\n处理后的输入数据信息:")
    print(f"RGB输入: 形状={rgb_input.shape}, 类型={rgb_input.dtype}")
    print(f"IR输入: 形状={ir_input.shape}, 类型={ir_input.dtype}")
    print(f"外参矩阵: 形状={updated_homography_np.shape}, 类型={updated_homography_np.dtype}")

    onnx_inputs = {
        input_names[0]: rgb_input,
        input_names[1]: ir_input,
        input_names[2]: updated_homography_np
    }
    
    try:
        onnx_outputs = session.run(output_names, onnx_inputs)
    except Exception as e:
        print(f"ONNX推理失败: {e}")
        print("输入数据信息:")
        for name, value in onnx_inputs.items():
            print(f"{name}: 形状={value.shape}, 类型={value.dtype}, 范围=[{value.min()}, {value.max()}]")
        return False
    
    # 计算误差
    torch_output = torch_output.cpu().numpy()
    onnx_output = onnx_outputs[0]
    
    # 分别计算位置信息和置信度的误差
    # 位置信息误差 (x,y,w,h)
    pos_diff = np.abs(torch_output[:, :4, :] - onnx_output[:, :4, :])
    max_pos_diff = np.max(pos_diff)
    mean_pos_diff = np.mean(pos_diff)
    # 避免除以0
    torch_pos = torch_output[:, :4, :]
    onnx_pos = onnx_output[:, :4, :]
    relative_pos_diff = np.mean(np.abs((torch_pos - onnx_pos) / (np.abs(torch_pos) + 1e-10)))
    
    # 置信度误差
    conf_diff = np.abs(torch_output[:, 4, :] - onnx_output[:, 4, :])
    max_conf_diff = np.max(conf_diff)
    mean_conf_diff = np.mean(conf_diff)
    # 避免除以0
    torch_conf = torch_output[:, 4, :]
    onnx_conf = onnx_output[:, 4, :]
    relative_conf_diff = np.mean(np.abs((torch_conf - onnx_conf) / (np.abs(torch_conf) + 1e-10)))
    
    print(f"\n验证结果:")
    print(f"位置信息误差:")
    print(f"  最大绝对误差: {max_pos_diff:.6f}")
    print(f"  平均绝对误差: {mean_pos_diff:.6f}")
    print(f"  平均相对误差: {relative_pos_diff:.6f}")
    print(f"置信度误差:")
    print(f"  最大绝对误差: {max_conf_diff:.6f}")
    print(f"  平均绝对误差: {mean_conf_diff:.6f}")
    print(f"  平均相对误差: {relative_conf_diff:.6f}")
    
    return max(max_pos_diff, max_conf_diff) < error_threshold

def main():
    args = parse_args()
    evaluate_onnx(args)

if __name__ == '__main__':
    main() 