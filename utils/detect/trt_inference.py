import os
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics.utils.ops import non_max_suppression
import torch

class TensorRTInference:
    """TensorRT推理类"""
    
    def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.45, nc=3):
        """
        初始化TensorRT推理引擎
        
        Args:
            engine_path (str): TensorRT engine文件路径
            conf_thres (float): 置信度阈值
            iou_thres (float): IOU阈值
            nc (int): 类别数量
        """
        self.engine_path = engine_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.nc = nc
        
        # 初始化TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None
        
        # 加载engine
        self.load_engine()
        
        # 获取输入输出信息
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        # 获取默认输入尺寸
        self.default_input_shape = self.engine.get_tensor_shape(self.input_name)
        print(f"默认输入尺寸: {self.default_input_shape}")
        
        # 处理动态维度，设置具体的输入尺寸
        input_shape_fixed = list(self.default_input_shape)
        for i, dim in enumerate(input_shape_fixed):
            if dim == -1:
                if i == 0:  # batch size
                    input_shape_fixed[i] = 1
                elif i in [2, 3]:  # height, width
                    input_shape_fixed[i] = 640
                else:
                    input_shape_fixed[i] = 1
        
        # 设置输入尺寸
        self.context.set_input_shape(self.input_name, input_shape_fixed)
        print(f"设置固定输入尺寸: {input_shape_fixed}")
        
        # 获取输出尺寸
        self.output_shape = self.context.get_tensor_shape(self.output_name)
        print(f"输出尺寸: {self.output_shape}")
        
        # 分配GPU内存
        self.allocate_buffers()
        
    def load_engine(self):
        """加载TensorRT engine"""
        try:
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()
            print(f"成功加载engine: {self.engine_path}")
            
        except Exception as e:
            raise RuntimeError(f"加载engine失败: {e}")
    
    def allocate_buffers(self):
        """分配GPU内存缓冲区"""
        try:
            # 获取输入输出尺寸
            input_shape = self.context.get_tensor_shape(self.input_name)
            output_shape = self.context.get_tensor_shape(self.output_name)
            
            # 处理动态维度，使用默认尺寸
            # 对于动态batch size，使用1；对于动态H/W，使用640
            input_shape_fixed = list(input_shape)
            output_shape_fixed = list(output_shape)
            
            for i, dim in enumerate(input_shape_fixed):
                if dim == -1:
                    if i == 0:  # batch size
                        input_shape_fixed[i] = 1
                    elif i in [2, 3]:  # height, width
                        input_shape_fixed[i] = 640
                    else:
                        input_shape_fixed[i] = 1
            
            for i, dim in enumerate(output_shape_fixed):
                if dim == -1:
                    if i == 0:  # batch size
                        output_shape_fixed[i] = 1
                    else:
                        output_shape_fixed[i] = 1
            
            # 分配输入缓冲区
            self.input_size = trt.volume(input_shape_fixed) * trt.float32.itemsize
            self.d_input = cuda.mem_alloc(self.input_size)
            
            # 分配输出缓冲区
            self.output_size = trt.volume(output_shape_fixed) * trt.float32.itemsize
            self.d_output = cuda.mem_alloc(self.output_size)
            
            # 创建输出数组
            self.output = np.empty(output_shape_fixed, dtype=np.float32)
            
            print(f"GPU内存分配完成 - 输入: {self.input_size/1024:.1f}KB, 输出: {self.output_size/1024:.1f}KB")
            print(f"固定后的输入尺寸: {input_shape_fixed}")
            print(f"固定后的输出尺寸: {output_shape_fixed}")
            
        except Exception as e:
            raise RuntimeError(f"GPU内存分配失败: {e}")
    
    def preprocess(self, img, target_size=(640, 640)):
        """
        图像预处理
        
        Args:
            img (np.ndarray): 输入图像 (BGR格式)
            target_size (tuple): 目标尺寸 (H, W)
            
        Returns:
            tuple: (预处理后的图像, 原始尺寸, 缩放参数)
        """
        # 获取原始尺寸
        h0, w0 = img.shape[:2]
        
        # 计算缩放比例
        r = min(target_size[0] / h0, target_size[1] / w0)
        
        # 计算缩放后的尺寸
        h_new = int(h0 * r)
        w_new = int(w0 * r)
        
        # 计算填充
        dh = target_size[0] - h_new
        dw = target_size[1] - w_new
        top = dh // 2
        left = dw // 2
        bottom = dh - top
        right = dw - left
        
        # 转换颜色空间并缩放
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        
        # 添加填充
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # 转换为float并归一化
        img_normalized = img_padded.astype(np.float32) / 255.0
        
        # 转换为NCHW格式
        img_nchw = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_nchw, 0)
        
        # 保存预处理参数
        preprocess_params = {
            'ratio': r,
            'pad': (top, left),
            'original_shape': (h0, w0),
            'target_shape': target_size
        }
        
        return img_batch, preprocess_params
    
    def postprocess(self, output, preprocess_params):
        """
        后处理检测结果
        
        Args:
            output (np.ndarray): 模型输出
            preprocess_params (dict): 预处理参数
            
        Returns:
            np.ndarray: 检测框坐标 (N, 4) 格式为 [x1, y1, x2, y2]
        """
        # 转换为PyTorch张量
        if isinstance(output, np.ndarray):
            output_tensor = torch.from_numpy(output)
        else:
            output_tensor = output
            
        # 确保是2D张量
        if output_tensor.ndim == 2:
            output_tensor = output_tensor.unsqueeze(0)
        
        # 非极大值抑制
        preds = non_max_suppression(
            output_tensor, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres, 
            nc=self.nc
        )
        
        # 提取检测框
        boxes_list = []
        for det in preds:
            if det is not None and len(det) > 0:
                boxes = det[:, :4].cpu().numpy()
                boxes_list.extend(boxes)
        
        if not boxes_list:
            return np.zeros((0, 4))
        
        # 还原坐标到原始图像尺寸
        boxes = np.array(boxes_list)
        boxes = self.rescale_boxes(boxes, preprocess_params)
        
        return boxes
    
    def rescale_boxes(self, boxes, preprocess_params):
        """
        将检测框坐标还原到原始图像尺寸
        
        Args:
            boxes (np.ndarray): 检测框坐标 (N, 4)
            preprocess_params (dict): 预处理参数
            
        Returns:
            np.ndarray: 还原后的检测框坐标
        """
        if boxes.size == 0:
            return boxes
        
        r = preprocess_params['ratio']
        top, left = preprocess_params['pad']
        
        # 还原坐标
        boxes[:, 0] = (boxes[:, 0] - left) / r  # x1
        boxes[:, 1] = (boxes[:, 1] - top) / r   # y1
        boxes[:, 2] = (boxes[:, 2] - left) / r  # x2
        boxes[:, 3] = (boxes[:, 3] - top) / r   # y2
        
        # 确保坐标在图像范围内
        h0, w0 = preprocess_params['original_shape']
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w0)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h0)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w0)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h0)
        
        return boxes
    
    def infer(self, img):
        """
        执行推理
        
        Args:
            img (np.ndarray): 输入图像 (BGR格式)
            
        Returns:
            tuple: (检测框, 预处理参数, 推理时间)
        """
        # 预处理
        input_tensor, preprocess_params = self.preprocess(img)
        
        # 确保输入是连续内存
        input_tensor = np.ascontiguousarray(input_tensor)
        
        # 检查输入尺寸是否匹配
        current_input_shape = input_tensor.shape
        if current_input_shape != self.context.get_tensor_shape(self.input_name):
            self.context.set_input_shape(self.input_name, current_input_shape)
            # 重新分配输出缓冲区
            output_shape = self.context.get_tensor_shape(self.output_name)
            self.output = np.empty(output_shape, dtype=np.float32)
        
        # 记录推理开始时间
        start_time = time.time()
        
        # 创建CUDA流
        stream = cuda.Stream()
        
        try:
            # 设置输入输出地址
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            
            # 复制数据到GPU
            cuda.memcpy_htod_async(self.d_input, input_tensor, stream)
            
            # 执行推理
            self.context.execute_async_v3(stream_handle=stream.handle)
            
            # 复制结果到CPU
            cuda.memcpy_dtoh_async(self.output, self.d_output, stream)
            
            # 同步流
            stream.synchronize()
            
        finally:
            # 清理流
            del stream
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 后处理
        boxes = self.postprocess(self.output, preprocess_params)
        
        return boxes, preprocess_params, inference_time
    
    def draw_detections(self, img, boxes, save_path):
        """
        在图像上绘制检测结果并保存
        
        Args:
            img (np.ndarray): 原始图像
            boxes (np.ndarray): 检测框坐标
            save_path (str): 保存路径
        """
        # 复制图像避免修改原图
        img_draw = img.copy()
        
        # 绘制检测框
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # 绘制矩形框
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 添加标签
                label = f"Target"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(img_draw, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 保存图像
        cv2.imwrite(save_path, img_draw)
        print(f"结果已保存到: {save_path}")
    
    def __del__(self):
        """析构函数，清理GPU内存"""
        try:
            if hasattr(self, 'd_input'):
                del self.d_input
            if hasattr(self, 'd_output'):
                del self.d_output
        except:
            pass

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TensorRT目标检测推理')
    
    # 必需参数
    parser.add_argument('--engine_path', type=str, 
                       default='/ultralytics/ckpt/best0829-det.engine',
                       help='TensorRT engine文件路径')
    parser.add_argument('--input_dir', type=str, default='/ultralytics/data/Data/saved_files_1756457235172/XYZ_color',
                       help='输入图像目录路径')
    parser.add_argument('--output_dir', type=str, default='/share/XYZ_color-trtInferencev2',
                       help='输出结果保存目录')
    
    # 可选参数
    parser.add_argument('--conf_thres', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--iou_thres', type=float, default=0.7,
                       help='IOU阈值 (默认: 0.45)')
    parser.add_argument('--nc', type=int, default=3,
                       help='类别数量 (默认: 3)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp'],
                       help='支持的图像文件扩展名')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 检查engine文件是否存在
    if not os.path.exists(args.engine_path):
        raise FileNotFoundError(f"Engine文件不存在: {args.engine_path}")
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化TensorRT推理引擎
    print("正在初始化TensorRT推理引擎...")
    try:
        trt_engine = TensorRTInference(
            args.engine_path,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            nc=args.nc
        )
        print("TensorRT推理引擎初始化成功！")
    except Exception as e:
        print(f"TensorRT推理引擎初始化失败: {e}")
        return
    
    # 查找所有图像文件
    image_files = []
    for ext in args.extensions:
        image_files.extend(Path(args.input_dir).glob(f"*{ext}"))
        image_files.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"在目录 {args.input_dir} 中未找到支持的图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每张图像
    total_time = 0
    for i, image_path in enumerate(image_files):
        print(f"\n处理图像 {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"无法读取图像: {image_path}")
                continue
            
            print(f"图像尺寸: {img.shape}")
            
            # 执行推理
            boxes, preprocess_params, inference_time = trt_engine.infer(img)
            total_time += inference_time
            
            print(f"推理时间: {inference_time:.2f}ms")
            print(f"检测到 {len(boxes)} 个目标")
            
            # 生成输出文件名
            output_filename = f"result_{image_path.stem}.jpg"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # 绘制检测结果并保存
            trt_engine.draw_detections(img, boxes, output_path)
            
        except Exception as e:
            print(f"处理图像 {image_path.name} 时出错: {e}")
            continue
    
    # 输出统计信息
    print(f"\n=== 处理完成 ===")
    print(f"总处理图像数: {len(image_files)}")
    print(f"总推理时间: {total_time:.2f}ms")
    print(f"平均推理时间: {total_time/len(image_files):.2f}ms")
    print(f"结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 