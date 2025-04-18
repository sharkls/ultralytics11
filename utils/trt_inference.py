import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import torch
from pathlib import Path

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
    
    def preprocess(self, rgb_img, ir_img, homography):
        """预处理输入图像
        
        Args:
            rgb_img (np.ndarray): RGB图像
            ir_img (np.ndarray): 红外图像
            homography (np.ndarray): 单应性矩阵
            
        Returns:
            tuple: 预处理后的输入数据
        """
        # 图像预处理
        def process_img(img):
            # 调整大小
            img = cv2.resize(img, (self.input_shape[2], self.input_shape[3]))
            # 归一化
            img = img.astype(np.float32) / 255.0
            # HWC to CHW
            img = img.transpose(2, 0, 1)
            # 添加batch维度
            img = np.expand_dims(img, 0)
            return img
        
        # 处理RGB和红外图像
        rgb_input = process_img(rgb_img)
        ir_input = process_img(ir_img)
        
        # 处理单应性矩阵
        homography = np.expand_dims(homography, 0)  # 添加batch维度
        
        return rgb_input, ir_input, homography
    
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TensorRT推理')
    parser.add_argument('--engine', type=str, default="runs/multimodal/train6/weights/last.engine", help='TensorRT engine路径')
    parser.add_argument('--rgb', type=str, default="data/LLVIP/images/visible/test/190001.jpg", help='RGB图像路径')
    parser.add_argument('--ir', type=str, default="data/LLVIP/images/infrared/test/190001.jpg", help='红外图像路径')
    parser.add_argument('--homography', type=str, default="data/LLVIP/extrinsics/test/190001.txt", help='单应性矩阵路径')
    args = parser.parse_args()
    
    # 加载图像和单应性矩阵
    rgb_img = cv2.imread(args.rgb)
    ir_img = cv2.imread(args.ir)
    homography = np.loadtxt(args.homography)
    
    if rgb_img is None or ir_img is None:
        print("无法读取图像")
        return
    
    # 创建推理器
    trt_inference = TRTInference(args.engine)
    
    # 执行推理
    output = trt_inference.inference(rgb_img, ir_img, homography)
    
    # 处理输出结果
    # 这里需要根据你的模型输出格式进行处理
    print("推理完成，输出形状:", output.shape)

if __name__ == '__main__':
    main() 