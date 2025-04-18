import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import torch
from ultralytics.data.augment import LetterBox

def build_engine(onnx_path, engine_path, max_batch_size=1, fp16_mode=False):
    """构建TensorRT engine
    
    Args:
        onnx_path (str): ONNX模型路径
        engine_path (str): 保存engine的路径
        max_batch_size (int): 最大batch size
        fp16_mode (bool): 是否使用FP16精度
    """
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
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 设置优化配置文件
    profile = builder.create_optimization_profile()
    # 获取输入张量
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    # 设置动态形状范围
    profile.set_shape(input_tensor.name, 
                     (1, input_shape[1], input_shape[2], input_shape[3]),  # 最小形状
                     (max_batch_size, input_shape[1], input_shape[2], input_shape[3]),  # 最优形状
                     (max_batch_size, input_shape[1], input_shape[2], input_shape[3]))  # 最大形状
    config.add_optimization_profile(profile)
    
    # 构建engine
    print("正在构建TensorRT engine...")
    engine = builder.build_engine(network, config)
    
    # 保存engine
    print(f"正在保存engine到: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    return engine

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
        self.input_binding = self.engine.get_binding_index('input')  # 假设输入名为'input'
        self.output_binding = self.engine.get_binding_index('output')  # 假设输出名为'output'
        
        # 获取输入输出形状
        self.input_shape = self.engine.get_binding_shape(self.input_binding)
        self.output_shape = self.engine.get_binding_shape(self.output_binding)
        
        # 分配GPU内存
        self.input_buffer = cuda.mem_alloc(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        self.output_buffer = cuda.mem_alloc(np.prod(self.output_shape) * np.dtype(np.float32).itemsize)
        
        # 创建CUDA流
        self.stream = cuda.Stream()
        
        # 设置图像尺寸和步长
        self.imgsz = (self.input_shape[2], self.input_shape[3])  # (H, W)
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
        cuda.memcpy_htod_async(self.input_buffer, rgb_input.ravel(), self.stream)
        cuda.memcpy_htod_async(self.input_buffer, ir_input.ravel(), self.stream)
        cuda.memcpy_htod_async(self.input_buffer, homography.ravel(), self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=[int(self.input_buffer), int(self.output_buffer)], stream_handle=self.stream.handle)
        
        # 分配输出内存
        output = np.empty(self.output_shape, dtype=np.float32)
        
        # 将结果复制回CPU
        cuda.memcpy_dtoh_async(output, self.output_buffer, self.stream)
        
        # 同步流
        self.stream.synchronize()
        
        return output
    
    def __del__(self):
        """清理资源"""
        del self.input_buffer
        del self.output_buffer
        del self.stream
        del self.context
        del self.engine

def verify_engine(engine_path, rgb_path, ir_path, homography_path):
    """验证TensorRT engine的有效性
    
    Args:
        engine_path (str): TensorRT engine路径
        rgb_path (str): RGB图像路径
        ir_path (str): 红外图像路径
        homography_path (str): 单应性矩阵路径
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
        if output is not None and output.shape == trt_inference.output_shape:
            print("✅ Engine验证通过")
            print(f"输出形状: {output.shape}")
            return True
        else:
            print("❌ Engine验证失败：输出形状不匹配")
            return False
            
    except Exception as e:
        print(f"❌ Engine验证失败：{str(e)}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='将ONNX模型转换为TensorRT engine并验证')
    parser.add_argument('--onnx', type=str, default="runs/multimodal/train6/weights/last.onnx", help='ONNX模型路径')
    parser.add_argument('--engine', type=str, default="runs/multimodal/train6/weights/last.engine", help='保存engine的路径')
    parser.add_argument('--max-batch-size', type=int, default=1, help='最大batch size')
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度')
    parser.add_argument('--verify', action='store_true', help='验证engine')
    parser.add_argument('--rgb', type=str, default="data/LLVIP/images/visible/test/190001.jpg", help='RGB图像路径')
    parser.add_argument('--ir', type=str, default="data/LLVIP/images/infrared/test/190001.jpg", help='红外图像路径')
    parser.add_argument('--homography', type=str, default="data/LLVIP/extrinsics/test/190001.txt", help='单应性矩阵路径')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.engine), exist_ok=True)
    
    # 构建engine
    engine = build_engine(args.onnx, args.engine, args.max_batch_size, args.fp16)
    if engine is None:
        print("Engine构建失败！")
        return
    
    print("Engine构建成功！")
    
    # 验证engine
    if args.verify:
        print("\n开始验证engine...")
        verify_engine(args.engine, args.rgb, args.ir, args.homography)

if __name__ == '__main__':
    main() 