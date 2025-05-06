import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.augment import LetterBox
import json
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Multimodal TensorRT Model Validation')
    
    # 模型相关参数
    parser.add_argument('--engine-path-fp32', type=str, default='runs/multimodal/train6/weights/last.engine',
                      help='FP32 TensorRT engine文件路径')
    parser.add_argument('--engine-path-fp16', type=str, default='runs/multimodal/train6/weights/last.engine',
                      help='FP16 TensorRT engine文件路径')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640],
                      help='输入图像尺寸 [height, width]')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16'],
                      help='推理精度 (fp32/fp16)')
    
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
    parser.add_argument('--save-dir', type=str, default='runs/trt_val',
                      help='结果保存目录')
    parser.add_argument('--visualize', type=bool, default=True,
                      help='是否可视化检测结果')
    
    return parser.parse_args()

class TRTInference:
    def __init__(self, engine_path_fp32, engine_path_fp16, precision='fp32'):
        """初始化TensorRT推理器
        
        Args:
            engine_path_fp32 (str): FP32 TensorRT engine文件路径
            engine_path_fp16 (str): FP16 TensorRT engine文件路径
            precision (str): 推理精度 ('fp32' 或 'fp16')
        """
        # 根据精度选择engine文件
        engine_path = engine_path_fp16 if precision == 'fp16' else engine_path_fp32
        print(f"使用{precision}精度，加载engine文件: {engine_path}")
        
        # 初始化CUDA
        cuda.init()
        
        # 获取CUDA设备信息
        self.device = cuda.Device(0)  # 使用第一个GPU
        self.cuda_context = self.device.make_context()
        
        # 设置精度
        self.precision = precision
        self.dtype = np.float16 if precision == 'fp16' else np.float32
        
        # 确保PyTorch和TensorRT使用相同的CUDA上下文
        import torch
        dummy_tensor = torch.tensor([1.0], dtype=torch.float32).to("cuda:0")
        
        print("\nGPU信息:")
        print(f"设备名称: {self.device.name()}")
        print(f"计算能力: {self.device.compute_capability()}")
        print(f"总内存: {self.device.total_memory() // (1024*1024)} MB")
        print(f"推理精度: {self.precision}")
        
        try:
            # 创建CUDA流
            self.stream = cuda.Stream()
            
            # 加载engine
            self.logger = trt.Logger(trt.Logger.VERBOSE)
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            print(f"\nEngine文件大小: {len(engine_data) / (1024*1024):.2f} MB")
            
            runtime = trt.Runtime(self.logger)
            runtime.max_threads = os.cpu_count()  # 设置最大线程数
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("无法加载TensorRT engine")
            
            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("无法创建TensorRT执行上下文")
            
            # 获取输入输出绑定
            self.num_io = self.engine.num_io_tensors
            self.input_bindings = []
            self.output_bindings = []
            
            print("\n模型绑定信息:")
            # 遍历所有绑定
            for i in range(self.num_io):
                name = self.engine.get_tensor_name(i)
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_bindings.append(i)
                    print(f"输入 {i}:")
                else:
                    self.output_bindings.append(i)
                    print(f"输出 {i}:")
                print(f"  名称: {name}")
                print(f"  数据类型: {dtype}")
                print(f"  形状: {shape}")
            
            # 获取输入输出形状
            self.input_shapes = []
            self.output_shapes = []
            
            for i in self.input_bindings:
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                self.input_shapes.append(shape)
                
            for i in self.output_bindings:
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                self.output_shapes.append(shape)
            
            # 分配GPU内存
            self.input_buffers = []
            self.output_buffers = []
            
            for shape in self.input_shapes:
                size = int(np.prod(shape) * np.dtype(self.dtype).itemsize)
                try:
                    buf = cuda.mem_alloc(size)
                    self.input_buffers.append(buf)
                except Exception as e:
                    raise RuntimeError(f"无法为输入分配GPU内存: {str(e)}")
                
            for shape in self.output_shapes:
                size = int(np.prod(shape) * np.dtype(self.dtype).itemsize)
                try:
                    buf = cuda.mem_alloc(size)
                    self.output_buffers.append(buf)
                except Exception as e:
                    raise RuntimeError(f"无法为输出分配GPU内存: {str(e)}")
            
            # 设置图像尺寸和步长
            self.imgsz = (self.input_shapes[0][2], self.input_shapes[0][3])  # (H, W)
            self.stride = 32  # 与7_export.py保持一致
            
            print("\n推理器初始化完成")
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            # 清理资源
            self.cleanup()
            raise
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'input_buffers'):
                for buf in self.input_buffers:
                    del buf
            if hasattr(self, 'output_buffers'):
                for buf in self.output_buffers:
                    del buf
            if hasattr(self, 'stream'):
                del self.stream
            if hasattr(self, 'context'):
                del self.context
            if hasattr(self, 'engine'):
                del self.engine
            # 清理CUDA上下文
            if hasattr(self, 'cuda_context'):
                self.cuda_context.pop()
                del self.cuda_context
        except Exception as e:
            print(f"清理资源时发生错误: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()
    
    def preprocess(self, rgb_img, ir_img, homography):
        """预处理输入图像，同时更新单应性矩阵
        
        Args:
            rgb_img (np.ndarray): RGB图像
            ir_img (np.ndarray): 红外图像
            homography (np.ndarray): 单应性矩阵
            
        Returns:
            tuple: 预处理后的输入数据和更新后的单应性矩阵
        """
        # 创建letterbox对象
        letterbox = LetterBox(self.imgsz, auto=False, stride=self.stride)
        
        # 计算RGB图像的letterbox参数
        rgb_h, rgb_w = rgb_img.shape[:2]
        r_rgb = min(self.imgsz[0]/rgb_h, self.imgsz[1]/rgb_w)
        new_unpad_rgb = int(round(rgb_w * r_rgb)), int(round(rgb_h * r_rgb))
        dw_rgb, dh_rgb = self.imgsz[1] - new_unpad_rgb[0], self.imgsz[0] - new_unpad_rgb[1]
        dw_rgb /= 2
        dh_rgb /= 2

        # 计算IR图像的letterbox参数
        ir_h, ir_w = ir_img.shape[:2]
        r_ir = min(self.imgsz[0]/ir_h, self.imgsz[1]/ir_w)
        new_unpad_ir = int(round(ir_w * r_ir)), int(round(ir_h * r_ir))
        dw_ir, dh_ir = self.imgsz[1] - new_unpad_ir[0], self.imgsz[0] - new_unpad_ir[1]
        dw_ir /= 2
        dh_ir /= 2

        # 构建变换矩阵
        dtype = torch.float32  # 使用float32类型
        device = 'cuda'  # 使用CUDA设备
        
        S_rgb = torch.eye(3, device=device, dtype=dtype)
        S_rgb[0, 0] = r_rgb  # x方向缩放
        S_rgb[1, 1] = r_rgb  # y方向缩放

        S_ir = torch.eye(3, device=device, dtype=dtype)
        S_ir[0, 0] = r_ir   # x方向缩放
        S_ir[1, 1] = r_ir   # y方向缩放

        T_rgb = torch.eye(3, device=device, dtype=dtype)
        T_rgb[0, 2] = dw_rgb  # x方向平移
        T_rgb[1, 2] = dh_rgb  # y方向平移

        T_ir = torch.eye(3, device=device, dtype=dtype)
        T_ir[0, 2] = dw_ir  # x方向平移
        T_ir[1, 2] = dh_ir  # y方向平移

        # 将homography转换为tensor
        H = torch.from_numpy(homography).float().to(device)

        # 更新单应性矩阵：新IR -> 原始IR -> 原始RGB -> 新RGB
        # H_new = T_rgb @ S_rgb @ H @ S_ir^(-1) @ T_ir^(-1)
        updated_H = torch.mm(T_rgb, torch.mm(S_rgb, torch.mm(H, 
                            torch.mm(torch.inverse(S_ir), torch.inverse(T_ir)))))

        # 应用letterbox变换
        rgb_img = letterbox(image=rgb_img)
        ir_img = letterbox(image=ir_img)
        
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
        rgb_img = np.expand_dims(rgb_img, 0).astype(self.dtype)
        ir_img = np.expand_dims(ir_img, 0).astype(self.dtype)
        
        # 更新后的单应性矩阵添加batch维度
        updated_H = updated_H.unsqueeze(0)
        
        # 创建letterbox_info字典
        letterbox_info = {
            'dw': dw_rgb,
            'dh': dh_rgb,
            'ratio': r_rgb
        }
        
        return rgb_img, ir_img, updated_H.cpu().numpy(), letterbox_info
    
    def inference(self, rgb_img, ir_img, homography):
        """执行推理
        
        Args:
            rgb_img (np.ndarray): RGB图像
            ir_img (np.ndarray): 红外图像
            homography (np.ndarray): 单应性矩阵
            
        Returns:
            tuple: (推理结果, letterbox_info)
        """
        try:
            # 预处理
            rgb_input, ir_input, homography, letterbox_info = self.preprocess(rgb_img, ir_img, homography)
            
            # # 检查输入数据
            # print("\n输入数据检查:")
            # print(f"RGB输入形状: {rgb_input.shape}, 范围: [{rgb_input.min():.3f}, {rgb_input.max():.3f}]")
            # print(f"IR输入形状: {ir_input.shape}, 范围: [{ir_input.min():.3f}, {ir_input.max():.3f}]")
            # print(f"单应性矩阵形状: {homography.shape}, 范围: [{homography.min():.3f}, {homography.max():.3f}]")
            
            # 检查输入数据是否为连续内存
            if not rgb_input.flags['C_CONTIGUOUS']:
                rgb_input = np.ascontiguousarray(rgb_input)
            if not ir_input.flags['C_CONTIGUOUS']:
                ir_input = np.ascontiguousarray(ir_input)
            if not homography.flags['C_CONTIGUOUS']:
                homography = np.ascontiguousarray(homography)
            
            # 检查输入数据类型
            rgb_input = rgb_input.astype(self.dtype)
            ir_input = ir_input.astype(self.dtype)
            homography = homography.astype(self.dtype)
            
            # 将数据复制到GPU
            try:
                cuda.memcpy_htod_async(self.input_buffers[0], rgb_input.ravel(), self.stream)
                cuda.memcpy_htod_async(self.input_buffers[1], ir_input.ravel(), self.stream)
                cuda.memcpy_htod_async(self.input_buffers[2], homography.ravel(), self.stream)
                self.stream.synchronize()  # 确保数据复制完成
            except Exception as e:
                raise RuntimeError(f"无法将数据复制到GPU: {str(e)}")
            
            # 执行推理
            bindings = [int(buf) for buf in self.input_buffers + self.output_buffers]
            
            # 检查绑定
            # print("\n绑定检查:")
            # for i, binding in enumerate(bindings):
            #     print(f"绑定 {i}: {binding}")
            #     if binding == 0:
            #         raise RuntimeError(f"无效的绑定地址: 绑定 {i} 为0")
            
            try:
                # 设置输入形状
                for i, shape in enumerate(self.input_shapes):
                    name = self.engine.get_tensor_name(i)
                    self.context.set_input_shape(name, shape)
                
                # 检查执行上下文是否准备好
                if not self.context.all_binding_shapes_specified:
                    raise RuntimeError("未指定所有绑定形状")
                if not self.context.all_shape_inputs_specified:
                    raise RuntimeError("未指定所有形状输入")
                
                # 使用同步执行方式
                success = self.context.execute_v2(bindings=bindings)
                if not success:
                    raise RuntimeError("TensorRT推理执行失败")
                
                # 同步CUDA流
                self.stream.synchronize()
            except Exception as e:
                raise RuntimeError(f"TensorRT推理执行错误: {str(e)}")
            
            # 分配输出内存
            outputs = []
            for shape in self.output_shapes:
                output = np.empty(shape, dtype=self.dtype)
                outputs.append(output)
            
            # 将结果复制回CPU
            try:
                for i, output in enumerate(outputs):
                    cuda.memcpy_dtoh_async(output, self.output_buffers[i], self.stream)
                self.stream.synchronize()  # 确保数据复制完成
            except Exception as e:
                raise RuntimeError(f"无法将结果从GPU复制回CPU: {str(e)}")
            
            # # 检查输出数据
            # print("\n输出数据检查:")
            # print(f"输出形状: {outputs[0].shape}, 范围: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
            
            return outputs[0], letterbox_info  # 返回第一个输出和letterbox_info
            
        except Exception as e:
            print(f"推理过程发生错误: {str(e)}")
            raise

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

def evaluate_trt(args):
    """评估TensorRT模型"""
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载TensorRT模型
    print(f"正在加载TensorRT模型...")
    trt_inference = TRTInference(args.engine_path_fp32, args.engine_path_fp16, args.precision)
    
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
            for i, img_name in enumerate(batch_names):
                if i >= len(extrinsics_list):  # 确保索引不越界
                    continue
                    
                # 读取图像
                rgb_img = cv2.imread(str(rgb_paths[i]))
                ir_img = cv2.imread(str(ir_paths[i]))
                if rgb_img is None or ir_img is None:
                    print(f"无法读取图像: {rgb_paths[i]} 或 {ir_paths[i]}")
                    continue
                
                # 执行推理
                try:
                    output, letterbox_info = trt_inference.inference(rgb_img, ir_img, extrinsics_list[i])
                    detections = process_output(output, args.conf_thres, args.iou_thres, 
                                             None, args.nc, letterbox_info)
                    
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
                        # 绘制原始标签框（红色）
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
                    print(f"推理失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"批处理处理失败: {e}")
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

def main():
    args = parse_args()
    evaluate_trt(args)

if __name__ == '__main__':
    main() 