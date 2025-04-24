import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import torch
from ultralytics.data.augment import LetterBox
from pathlib import Path

def build_engine(onnx_path, engine_path, max_batch_size=1, fp16_mode=False):
    """构建TensorRT engine
    
    Args:
        onnx_path (str): ONNX模型路径
        engine_path (str): 保存engine的路径
        max_batch_size (int): 最大batch size
        fp16_mode (bool): 是否使用FP16精度
    """
    # 检查engine文件是否已存在
    if os.path.exists(engine_path):
        print(f"Engine文件已存在: {engine_path}")
        return None
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX模型
    print(f"正在解析ONNX模型: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # 添加更多优化选项
    config.set_flag(trt.BuilderFlag.TF32)  # 启用TF32精度
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)  # 启用稀疏权重
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)  # 遵守精度约束
    
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 设置优化配置文件
    profile = builder.create_optimization_profile()
    
    # 获取所有输入张量
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape
        
        # 打印输入张量信息
        print(f"Input {i} name: {input_tensor.name}")
        print(f"Input {i} shape: {input_shape}")
        
        # 设置固定的输入形状
        if input_tensor.name == 'images' or input_tensor.name == 'images2':
            # 对于图像输入，使用固定尺寸
            min_shape = (1, 3, 640, 640)
            opt_shape = (max_batch_size, 3, 640, 640)
            max_shape = (max_batch_size, 3, 640, 640)
        else:
            # 对于其他输入（如extrinsics），使用原始形状
            min_shape = tuple(max(1, dim) for dim in input_shape)
            opt_shape = tuple(max(max_batch_size, dim) for dim in input_shape)
            max_shape = tuple(max(max_batch_size, dim) for dim in input_shape)
        
        # 设置动态形状范围
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)
    
    # 构建engine
    print("正在构建TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("构建engine失败！")
        return None
    
    # 保存engine
    print(f"正在保存engine到: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    return True

class TRTInference:
    def __init__(self, engine_path):
        """初始化TensorRT推理器
        
        Args:
            engine_path (str): TensorRT engine文件路径
        """
        # 加载engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出绑定
        self.num_io = self.engine.num_io_tensors
        self.input_bindings = []
        self.output_bindings = []
        
        # 遍历所有绑定
        for i in range(self.num_io):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_bindings.append(i)
            else:
                self.output_bindings.append(i)
        
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
            # 将numpy.int64转换为unsigned long
            size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
            self.input_buffers.append(cuda.mem_alloc(size))
            
        for shape in self.output_shapes:
            # 将numpy.int64转换为unsigned long
            size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
            self.output_buffers.append(cuda.mem_alloc(size))
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 设置图像尺寸和步长
        self.imgsz = (self.input_shapes[0][2], self.input_shapes[0][3])  # (H, W)
        self.stride = 32  # 与7_export.py保持一致
    
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
        rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
        ir_img = np.expand_dims(ir_img, 0).astype(np.float32)
        
        # 更新后的单应性矩阵添加batch维度
        updated_H = updated_H.unsqueeze(0)
        
        return rgb_img, ir_img, updated_H.cpu().numpy()
    
    def inference(self, rgb_img, ir_img, homography):
        """执行推理
        
        Args:
            rgb_img (np.ndarray): RGB图像
            ir_img (np.ndarray): 红外图像
            homography (np.ndarray): 单应性矩阵
            
        Returns:
            np.ndarray: 推理结果
        """
        # 预处理
        rgb_input, ir_input, homography = self.preprocess(rgb_img, ir_img, homography)
        
        # 将数据复制到GPU
        cuda.memcpy_htod_async(self.input_buffers[0], rgb_input.ravel(), self.stream)
        cuda.memcpy_htod_async(self.input_buffers[1], ir_input.ravel(), self.stream)
        cuda.memcpy_htod_async(self.input_buffers[2], homography.ravel(), self.stream)
        
        # 执行推理
        bindings = [int(buf) for buf in self.input_buffers + self.output_buffers]
        self.context.execute_v2(bindings=bindings)
        
        # 分配输出内存
        outputs = []
        for shape in self.output_shapes:
            output = np.empty(shape, dtype=np.float32)
            outputs.append(output)
        
        # 将结果复制回CPU
        for i, output in enumerate(outputs):
            cuda.memcpy_dtoh_async(output, self.output_buffers[i], self.stream)
        
        # 同步流
        self.stream.synchronize()
        
        return outputs[0]  # 返回第一个输出
    
    def __del__(self):
        """清理资源"""
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

def visualize_detection(img, detections, conf_thres=0.25, save_path=None):
    """可视化检测结果
    
    Args:
        img (np.ndarray): 原始图像
        detections (np.ndarray): 检测结果 [x1, y1, x2, y2, conf, cls]
        conf_thres (float): 置信度阈值
        save_path (str): 保存路径
    """
    # 创建图像副本
    vis_img = img.copy()
    
    # 绘制检测框
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < conf_thres:
            continue
            
        # 绘制边界框
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # 添加标签
        label = f'cls{int(cls)} {conf:.2f}'
        cv2.putText(vis_img, label, (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存结果
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img

def verify_engine(engine_path, rgb_path, ir_path, homography_path, conf_thres=0.25, visualize=False, save_dir=None):
    """验证TensorRT engine的有效性
    
    Args:
        engine_path (str): TensorRT engine路径
        rgb_path (str): RGB图像路径
        ir_path (str): 红外图像路径
        homography_path (str): 单应性矩阵路径
        conf_thres (float): 置信度阈值
        visualize (bool): 是否可视化结果
        save_dir (str): 可视化结果保存目录
    """
    # 加载图像和单应性矩阵
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path)
    homography = np.loadtxt(homography_path)
    
    if rgb_img is None or ir_img is None:
        print("无法读取图像")
        return False
    
    try:
        # 创建推理器
        trt_inference = TRTInference(engine_path)
        
        # 执行推理
        output = trt_inference.inference(rgb_img, ir_img, homography)
        
        # 检查输出
        if output is not None and output.shape == trt_inference.output_shapes[0]:
            print("✅ Engine验证通过")
            print(f"输出形状: {output.shape}")
            
            # 可视化结果
            if visualize:
                # 处理输出
                detections = process_output(output, conf_thres=conf_thres)
                
                # 可视化检测结果
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{Path(rgb_path).stem}_det.jpg")
                else:
                    save_path = None
                    
                vis_img = visualize_detection(rgb_img, detections, conf_thres, save_path)
                print(f"可视化结果已保存到: {save_path}")
            
            return True
        else:
            print("❌ Engine验证失败：输出形状不匹配")
            return False
            
    except Exception as e:
        print(f"❌ Engine验证失败：{str(e)}")
        return False

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
    
    # 打印检测结果信息
    print("\n检测结果信息:")
    print(f"检测到的目标数量: {len(final_results)}")
    for i, det in enumerate(final_results):
        x1, y1, x2, y2, conf, cls_id = det
        print(f"目标 {i+1}:")
        print(f"  类别: {int(cls_id)}")
        print(f"  置信度: {conf:.4f}")
        print(f"  边界框: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  宽度: {x2-x1:.1f}")
        print(f"  高度: {y2-y1:.1f}")
    
    return final_results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='将ONNX模型转换为TensorRT engine并验证')
    parser.add_argument('--onnx', type=str, default="runs/multimodal/train6/weights/last.onnx", help='ONNX模型路径')
    parser.add_argument('--engine', type=str, default="runs/multimodal/train6/weights/last.engine", help='保存engine的路径')
    parser.add_argument('--max-batch-size', type=int, default=1, help='最大batch size')
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度')
    parser.add_argument('--verify', default=True, help='验证engine')
    parser.add_argument('--rgb', type=str, default="data/LLVIP/images/visible/test/190001.jpg", help='RGB图像路径')
    parser.add_argument('--ir', type=str, default="data/LLVIP/images/infrared/test/190001.jpg", help='红外图像路径')
    parser.add_argument('--homography', type=str, default="data/LLVIP/extrinsics/test/190001.txt", help='单应性矩阵路径')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--visualize', default=True, help='是否可视化结果')
    parser.add_argument('--save-dir', type=str, default="runs/tensorrt_vis", help='可视化结果保存目录')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.engine), exist_ok=True)
    
    # 检查engine文件是否存在
    if os.path.exists(args.engine):
        print(f"Engine文件已存在: {args.engine}")
    else:
        # 构建engine
        result = build_engine(args.onnx, args.engine, args.max_batch_size, args.fp16)
        if result is None:
            print("Engine构建失败！")
            return
        print("Engine构建成功！")
    
    # 验证engine
    if args.verify:
        print("\n开始验证engine...")
        verify_engine(args.engine, args.rgb, args.ir, args.homography, 
                     conf_thres=args.conf_thres, 
                     visualize=args.visualize,
                     save_dir=args.save_dir)

if __name__ == '__main__':
    main() 