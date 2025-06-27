import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import subprocess
import onnx
import argparse
from ultralytics.utils.ops import non_max_suppression
import time

"""
YOLO姿态估计多batch推理测试工具

支持的功能：
1. 加载多张图像进行batch推理
2. PyTorch模型batch推理
3. ONNX模型batch推理（支持CPU和GPU）
4. TensorRT engine batch推理
5. 性能对比和结果可视化

使用方法：
# 使用两张图像进行batch推理
python 10_multi_batch_pose_inference.py --model_path ckpt/yolo11m-pose.pt --image_paths data/test1.jpg data/test2.jpg

# 使用GPU进行ONNX batch推理
python 10_multi_batch_pose_inference.py --model_path ckpt/yolo11m-pose.pt --image_paths data/test1.jpg data/test2.jpg --use_gpu_onnx

# 指定batch大小
python 10_multi_batch_pose_inference.py --model_path ckpt/yolo11m-pose.pt --image_paths data/test1.jpg data/test2.jpg --batch_size 4
"""

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO姿态估计多batch推理测试')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, default='ckpt/yolo11m-pose.pt',
                      help='PyTorch模型路径')
    parser.add_argument('--onnx_path', type=str, default='ckpt/yolo11m-pose.onnx',
                      help='ONNX模型保存路径')
    parser.add_argument('--engine_path', type=str, default='ckpt/yolo11m-pose.engine',
                      help='TensorRT engine保存路径')
    
    # 图像相关参数
    parser.add_argument('--image_paths', nargs='+', 
                      default=['/ultralytics/c++/Output/Vis_Object_Regions/4/0_0.jpg', '/ultralytics/c++/Output/Vis_Object_Regions/4/1_0.jpg'],
                      help='输入图像路径列表')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='batch大小')
    parser.add_argument('--max_size', type=int, default=640,
                      help='最大输入尺寸')
    parser.add_argument('--stride', type=int, default=32,
                      help='模型步长')
    
    # 后处理参数
    parser.add_argument('--conf_thres', type=float, default=0.25,
                      help='置信度阈值')
    parser.add_argument('--iou_thres', type=float, default=0.7,
                      help='IOU阈值')
    
    # 输出相关参数
    parser.add_argument('--save_dir', type=str, default='runs/pose_batch/',
                      help='结果保存目录')
    parser.add_argument('--save_torch', type=str, default='result_torch_batch.jpg',
                      help='PyTorch结果保存文件名')
    parser.add_argument('--save_onnx', type=str, default='result_onnx_batch.jpg',
                      help='ONNX结果保存文件名')
    parser.add_argument('--save_engine', type=str, default='result_engine_batch.jpg',
                      help='TensorRT结果保存文件名')
    
    # TensorRT相关参数
    parser.add_argument('--fp16', action='store_true',
                      help='是否使用FP16精度')
    parser.add_argument('--min_batch', type=int, default=1,
                      help='最小批次大小')
    parser.add_argument('--max_batch', type=int, default=8,
                      help='最大批次大小')
    
    # ONNX Runtime相关参数
    parser.add_argument('--use_gpu_onnx', action='store_true',
                      help='是否使用GPU版本的onnxruntime进行推理')
    parser.add_argument('--onnx_provider', type=str, default='auto',
                      choices=['auto', 'cpu', 'gpu', 'cuda'],
                      help='ONNX Runtime推理提供者')
    
    # 性能测试参数
    parser.add_argument('--warmup', type=int, default=10,
                      help='预热次数')
    parser.add_argument('--iterations', type=int, default=100,
                      help='测试迭代次数')
    
    return parser.parse_args()

def calculate_target_size(orig_shape, max_size=640, stride=32):
    """
    根据原始图像尺寸计算目标尺寸
    
    Args:
        orig_shape (tuple): 原始图像尺寸 (h, w)
        max_size (int): 最大尺寸
        stride (int): 模型步长
    
    Returns:
        tuple: 目标尺寸 (h, w)
    """
    h, w = orig_shape
    # 计算缩放比例
    r = min(max_size / h, max_size / w)
    # 计算缩放后的尺寸
    h_new = int(h * r)
    w_new = int(w * r)
    # 确保尺寸是stride的整数倍
    h_new = (h_new // stride) * stride
    w_new = (w_new // stride) * stride
    return h_new, w_new

def preprocess_batch(images, max_size=640, stride=32):
    """
    批量预处理函数
    
    Args:
        images (list): 输入图像列表，BGR格式
        max_size (int): 最大尺寸
        stride (int): 模型步长
    
    Returns:
        tuple: (预处理后的batch图像, 原始图像尺寸列表, 预处理参数列表)
    """
    print(f"\n=== 批量预处理详细信息 ===")
    print(f"输入图像数量: {len(images)}")
    print(f"最大尺寸: {max_size}")
    print(f"步长: {stride}")
    
    batch_inputs = []
    orig_shapes = []
    preprocess_params_list = []
    
    # 首先计算所有图像的目标尺寸，取最大值确保一致性
    target_h, target_w = 0, 0
    for i, img in enumerate(images):
        if not isinstance(img, np.ndarray):
            raise TypeError("输入图像必须是numpy数组")
        if len(img.shape) != 3:
            raise ValueError("输入图像必须是3通道图像")
        
        h0, w0 = img.shape[:2]
        h, w = calculate_target_size((h0, w0), max_size, stride)
        target_h = max(target_h, h)
        target_w = max(target_w, w)
        print(f"图像 {i}: 原始尺寸 {h0}x{w0} -> 目标尺寸 {h}x{w}")
    
    print(f"统一目标尺寸: {target_h}x{target_w}")
    
    for i, img in enumerate(images):
        print(f"\n--- 处理图像 {i} ---")
        # 获取原始尺寸
        h0, w0 = img.shape[:2]
        orig_shapes.append((h0, w0))
        
        print(f"  原始尺寸: {h0}x{w0}")
        
        # 计算缩放比例
        r = min(target_h / h0, target_w / w0)
        print(f"  缩放比例: {r:.6f}")
        
        # 计算填充
        dh, dw = target_h - int(h0 * r), target_w - int(w0 * r)
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left
        
        print(f"  填充参数: top={top}, bottom={bottom}, left={left}, right={right}")
        
        # 转换颜色空间并缩放
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        
        print(f"  缩放后尺寸: {img.shape}")
        
        # 添加填充
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        print(f"  填充后尺寸: {img.shape}")
        
        # 转换为float并归一化
        img = img.astype(np.float32) / 255.0
        
        print(f"  归一化后数据范围: [{img.min():.6f}, {img.max():.6f}]")
        
        # 转换为NCHW格式
        img = img.transpose(2, 0, 1)
        print(f"  NCHW格式后形状: {img.shape}")
        
        batch_inputs.append(img)
        
        # 保存预处理参数用于后处理
        preprocess_params = {
            'ratio': r,
            'pad': (top, left),
            'stride': stride,
            'scaled_shape': (target_h, target_w)
        }
        preprocess_params_list.append(preprocess_params)
        
        print(f"  预处理参数: {preprocess_params}")
    
    # 堆叠为batch
    batch_input = np.stack(batch_inputs, axis=0)
    
    print(f"\n=== 批量预处理完成 ===")
    print(f"最终batch输入形状: {batch_input.shape}")
    print(f"最终batch输入数据类型: {batch_input.dtype}")
    print(f"最终batch输入数据范围: [{batch_input.min():.6f}, {batch_input.max():.6f}]")
    
    return batch_input, orig_shapes, preprocess_params_list

def export_onnx(model_path, onnx_path, max_size=640, stride=32):
    """
    ONNX导出函数，支持动态输入
    
    Args:
        model_path (str): 模型路径
        onnx_path (str): 输出ONNX路径
        max_size (int): 最大尺寸
        stride (int): 模型步长
    """
    model = YOLO(model_path)
    
    # 使用动态尺寸
    model.export(
        format='onnx',
        imgsz=max_size,
        simplify=True,
        dynamic=True,
        opset=16,
        half=False,
        int8=False,
        device='cpu',
        optimize=False,
        name=onnx_path
    )
    print(f"ONNX模型已导出到: {onnx_path}")

def check_onnxruntime_providers():
    """
    检查onnxruntime可用的推理提供者
    
    Returns:
        dict: 可用提供者信息
    """
    providers_info = {
        'available_providers': ort.get_available_providers(),
        'cuda_available': False,
        'cpu_available': False
    }
    
    print(f"ONNX Runtime版本: {ort.__version__}")
    print(f"可用提供者: {providers_info['available_providers']}")
    
    if 'CUDAExecutionProvider' in providers_info['available_providers']:
        providers_info['cuda_available'] = True
        print("✓ CUDA提供者可用")
    else:
        print("✗ CUDA提供者不可用")
    
    if 'CPUExecutionProvider' in providers_info['available_providers']:
        providers_info['cpu_available'] = True
        print("✓ CPU提供者可用")
    else:
        print("✗ CPU提供者不可用")
    
    return providers_info

def check_cuda_available():
    """
    检查CUDA是否可用
    
    Returns:
        tuple: (是否可用, 错误信息)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "PyTorch未检测到CUDA设备"
        
        import pycuda.driver as cuda
        import pycuda.autoinit
        cuda.init()
        
        if cuda.Device.count() == 0:
            return False, "未检测到CUDA设备"
            
        try:
            device = cuda.Device(0)
            props = device.get_attributes()
            memory = cuda.mem_get_info()
            print(f"CUDA设备信息:")
            print(f"- 设备名称: {device.name()}")
            print(f"- 总显存: {memory[1] / 1024**2:.1f}MB")
            print(f"- 可用显存: {memory[0] / 1024**2:.1f}MB")
            
            test_array = cuda.mem_alloc(1024)
            del test_array
            
            return True, None
        except Exception as e:
            return False, f"CUDA设备信息获取失败: {e}"
            
    except ImportError as e:
        return False, f"导入CUDA相关库失败: {e}"
    except Exception as e:
        return False, f"CUDA初始化失败: {e}"

def onnx2engine(onnx_path, engine_path, fp16=True, max_size=640, stride=32, batch_size=2):
    """
    TensorRT engine转换函数，支持动态输入和batch
    
    Args:
        onnx_path (str): ONNX模型路径
        engine_path (str): 输出engine路径
        fp16 (bool): 是否使用FP16
        max_size (int): 最大尺寸
        stride (int): 模型步长
        batch_size (int): batch大小
    """
    cuda_available, error_msg = check_cuda_available()
    if not cuda_available:
        print(f"警告：{error_msg}，将跳过TensorRT转换")
        return False

    h, w = calculate_target_size((max_size, max_size), max_size, stride)

    min_batch = 1
    max_batch = batch_size
    min_h = stride
    max_h = h
    min_w = stride
    max_w = w

    try:
        import pycuda.driver as cuda
        free_mem, total_mem = cuda.mem_get_info()
        required_mem = max_batch * 3 * max_h * max_w * 4
        if free_mem < required_mem:
            print(f"警告：可用显存不足，需要至少{required_mem/1024**2:.1f}MB，当前可用{free_mem/1024**2:.1f}MB")
            max_batch = min(max_batch, int(free_mem / (3 * max_h * max_w * 4)))
            print(f"已自动调整最大批次大小为: {max_batch}")
    except Exception as e:
        print(f"警告：无法检查显存: {e}")

    fp16_flag = "--fp16" if fp16 else ""
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path} {fp16_flag} " \
          f"--memPoolSize=workspace:4096 " \
          f"--shapes=images:{max_batch}x3x{max_h}x{max_w} " \
          f"--minShapes=images:{min_batch}x3x{min_h}x{min_w} " \
          f"--maxShapes=images:{max_batch}x3x{max_h}x{max_w} " \
          f"--optShapes=images:{batch_size}x3x{h}x{w}"
    
    print(f"正在转换为TensorRT engine: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"engine已保存到: {engine_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"TensorRT转换失败: {e}")
        return False

def run_onnx_batch(onnx_path, batch_input, use_gpu=False, provider='auto'):
    """
    运行ONNX模型batch推理
    
    Args:
        onnx_path (str): ONNX模型路径
        batch_input (np.ndarray): 预处理后的batch输入图像
        use_gpu (bool): 是否使用GPU推理
        provider (str): 推理提供者
    
    Returns:
        list: 模型输出
    """
    if provider == 'auto':
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    elif provider == 'cpu':
        providers = ['CPUExecutionProvider']
    elif provider in ['gpu', 'cuda']:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        available_providers = session.get_providers()
        print(f"ONNX Runtime可用提供者: {available_providers}")
        print(f"当前使用的提供者: {session.get_provider_options()}")
        
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batch_input})
        print(f"ONNX模型输出形状: {[out.shape for out in outputs]}")
        return outputs
        
    except Exception as e:
        print(f"ONNX推理失败: {e}")
        if use_gpu and provider == 'auto':
            print("GPU推理失败，尝试使用CPU推理...")
            return run_onnx_batch(onnx_path, batch_input, use_gpu=False, provider='cpu')
        else:
            raise

def run_engine_batch(engine_path, batch_input):
    """
    运行TensorRT模型batch推理
    
    Args:
        engine_path (str): TensorRT engine路径
        batch_input (np.ndarray): 预处理后的batch输入图像
    
    Returns:
        np.ndarray: 模型输出
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        cuda.init()
        
        print(f"\n=== TensorRT Engine推理详细信息 ===")
        print(f"Engine路径: {engine_path}")
        print(f"输入数据形状: {batch_input.shape}")
        print(f"输入数据类型: {batch_input.dtype}")
        print(f"输入数据范围: [{batch_input.min():.6f}, {batch_input.max():.6f}]")
        print(f"输入数据均值: {batch_input.mean():.6f}")
        print(f"输入数据标准差: {batch_input.std():.6f}")
        
        # 打印输入数据的前几个元素
        print(f"\n输入数据前20个元素:")
        flat_input = batch_input.flatten()
        for i in range(min(20, len(flat_input))):
            print(f"  input[{i}] = {flat_input[i]:.6f}")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        
        input_shape = batch_input.shape
        context.set_input_shape(input_name, input_shape)
        
        output_shape = context.get_tensor_shape(output_name)
        print(f"\nTensorRT模型输入形状: {input_shape}")
        print(f"TensorRT模型输出形状: {output_shape}")
        print(f"输出数据大小: {np.prod(output_shape)}")

        batch_input = np.ascontiguousarray(batch_input)
        d_input = cuda.mem_alloc(batch_input.nbytes)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        try:
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))

            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_input, batch_input, stream)
            
            print(f"\n开始TensorRT推理...")
            start_time = time.time()
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
            end_time = time.time()
            print(f"TensorRT推理完成，耗时: {(end_time - start_time) * 1000:.2f}ms")
            
            print(f"\n=== TensorRT输出详细信息 ===")
            print(f"输出数据形状: {output.shape}")
            print(f"输出数据类型: {output.dtype}")
            print(f"输出数据范围: [{output.min():.6f}, {output.max():.6f}]")
            print(f"输出数据均值: {output.mean():.6f}")
            print(f"输出数据标准差: {output.std():.6f}")
            
            # 分析输出数据的详细结构
            if len(output.shape) == 3:  # [batch, feature_dim, num_anchors]
                batch_size, feature_dim, num_anchors = output.shape
                print(f"\n输出数据结构分析:")
                print(f"  Batch大小: {batch_size}")
                print(f"  特征维度: {feature_dim}")
                print(f"  Anchor数量: {num_anchors}")
                
                # 分析每个batch的输出
                for b in range(batch_size):
                    print(f"\n  Batch {b} 输出分析:")
                    batch_output = output[b]  # [feature_dim, num_anchors]
                    
                    # 分析边界框坐标 (前4个特征)
                    bbox_output = batch_output[:4]  # [4, num_anchors]
                    print(f"    边界框坐标范围:")
                    print(f"      x: [{bbox_output[0].min():.6f}, {bbox_output[0].max():.6f}]")
                    print(f"      y: [{bbox_output[1].min():.6f}, {bbox_output[1].max():.6f}]")
                    print(f"      w: [{bbox_output[2].min():.6f}, {bbox_output[2].max():.6f}]")
                    print(f"      h: [{bbox_output[3].min():.6f}, {bbox_output[3].max():.6f}]")
                    
                    # 分析置信度 (第5个特征)
                    conf_output = batch_output[4]  # [num_anchors]
                    print(f"    置信度范围: [{conf_output.min():.6f}, {conf_output.max():.6f}]")
                    print(f"    置信度均值: {conf_output.mean():.6f}")
                    
                    # 分析前几个anchor的详细输出
                    print(f"    前5个Anchor的详细输出:")
                    for i in range(min(5, num_anchors)):
                        anchor_output = batch_output[:, i]  # [feature_dim]
                        print(f"      Anchor {i}:")
                        print(f"        BBox: ({anchor_output[0]:.6f}, {anchor_output[1]:.6f}, {anchor_output[2]:.6f}, {anchor_output[3]:.6f})")
                        print(f"        Conf: {anchor_output[4]:.6f}")
                        
                        # 分析关键点 (如果有)
                        if feature_dim > 5:
                            kpt_start = 5
                            kpt_dim = feature_dim - kpt_start
                            print(f"        Keypoints: {kpt_dim} values")
                            if kpt_dim >= 6:  # 至少2个关键点
                                print(f"          Kpt0: ({anchor_output[kpt_start]:.6f}, {anchor_output[kpt_start+1]:.6f}, {anchor_output[kpt_start+2]:.6f})")
                                print(f"          Kpt1: ({anchor_output[kpt_start+3]:.6f}, {anchor_output[kpt_start+4]:.6f}, {anchor_output[kpt_start+5]:.6f})")
            
            # 打印输出数据的前50个元素
            print(f"\n输出数据前50个元素:")
            flat_output = output.flatten()
            for i in range(min(50, len(flat_output))):
                print(f"  output[{i}] = {flat_output[i]:.6f}")
            
            print(f"=== TensorRT Engine推理详细信息结束 ===\n")
            
            return output
        finally:
            del d_input
            del d_output
            del stream
            
    except Exception as e:
        print(f"TensorRT推理失败: {e}")
        raise

def keypoints_rescale_batch(kpts_list, orig_shapes, preprocess_params_list):
    """
    批量关键点坐标还原函数
    
    Args:
        kpts_list (list): 关键点坐标列表
        orig_shapes (list): 原始图像尺寸列表
        preprocess_params_list (list): 预处理参数列表
    
    Returns:
        list: 还原后的关键点坐标列表
    """
    print(f"\n=== 关键点坐标还原详细信息 ===")
    print(f"输入关键点列表长度: {len(kpts_list)}")
    print(f"原始图像尺寸列表长度: {len(orig_shapes)}")
    print(f"预处理参数列表长度: {len(preprocess_params_list)}")
    
    rescaled_kpts = []
    
    for i, (kpts, orig_shape, preprocess_params) in enumerate(zip(kpts_list, orig_shapes, preprocess_params_list)):
        print(f"\n--- 处理Batch {i} 关键点 ---")
        h0, w0 = orig_shape
        r = preprocess_params['ratio']
        top, left = preprocess_params['pad']
        
        print(f"  原始图像尺寸: {h0}x{w0}")
        print(f"  缩放比例: {r:.6f}")
        print(f"  填充参数: top={top}, left={left}")
        print(f"  输入关键点形状: {kpts.shape}")
        
        kpts = np.array(kpts)
        
        if kpts.ndim == 3:  # (N, 17, 2/3)
            print(f"  关键点维度: 3D (N, 17, 2/3)")
            print(f"  关键点数量: {kpts.shape[0]}")
            print(f"  每个关键点特征数: {kpts.shape[2]}")
            
            # 显示前几个关键点的原始值
            if len(kpts) > 0:
                print(f"  前3个关键点的原始值:")
                for j in range(min(3, len(kpts))):
                    print(f"    关键点组 {j}: {kpts[j][:3]}")  # 显示前3个关键点
            
            kpts[..., 0] = (kpts[..., 0] - left) / r
            kpts[..., 1] = (kpts[..., 1] - top) / r
            
            # 显示还原后的值
            if len(kpts) > 0:
                print(f"  前3个关键点的还原后值:")
                for j in range(min(3, len(kpts))):
                    print(f"    关键点组 {j}: {kpts[j][:3]}")
                    
        elif kpts.ndim == 2:  # (N, 51/34)
            print(f"  关键点维度: 2D (N, 51/34)")
            print(f"  关键点数量: {kpts.shape[0]}")
            print(f"  特征维度: {kpts.shape[1]}")
            
            for j in range(0, kpts.shape[1], 3):
                kpts[:, j] = (kpts[:, j] - left) / r
                kpts[:, j+1] = (kpts[:, j+1] - top) / r
        else:
            print(f"  警告: 不支持的关键点维度: {kpts.ndim}")
            raise ValueError(f"不支持的关键点维度: {kpts.ndim}")
        
        print(f"  还原后关键点形状: {kpts.shape}")
        rescaled_kpts.append(kpts)
    
    print(f"=== 关键点坐标还原完成 ===")
    return rescaled_kpts

def boxes_rescale_batch(boxes_list, orig_shapes, preprocess_params_list):
    """
    批量检测框坐标还原函数
    
    Args:
        boxes_list (list): 检测框坐标列表
        orig_shapes (list): 原始图像尺寸列表
        preprocess_params_list (list): 预处理参数列表
    
    Returns:
        list: 还原后的检测框坐标列表
    """
    print(f"\n=== 检测框坐标还原详细信息 ===")
    print(f"输入检测框列表长度: {len(boxes_list)}")
    print(f"原始图像尺寸列表长度: {len(orig_shapes)}")
    print(f"预处理参数列表长度: {len(preprocess_params_list)}")
    
    rescaled_boxes = []
    
    for i, (boxes, orig_shape, preprocess_params) in enumerate(zip(boxes_list, orig_shapes, preprocess_params_list)):
        print(f"\n--- 处理Batch {i} 检测框 ---")
        if boxes.size == 0:
            print(f"  无检测框，跳过")
            rescaled_boxes.append(boxes)
            continue
            
        boxes = np.array(boxes)
        r = preprocess_params['ratio']
        top, left = preprocess_params['pad']
        
        print(f"  原始图像尺寸: {orig_shape}")
        print(f"  缩放比例: {r:.6f}")
        print(f"  填充参数: top={top}, left={left}")
        print(f"  输入检测框形状: {boxes.shape}")
        
        # 显示前几个检测框的原始值
        if len(boxes) > 0:
            print(f"  前3个检测框的原始值:")
            for j in range(min(3, len(boxes))):
                print(f"    检测框 {j}: {boxes[j]}")
        
        boxes[:, 0] = (boxes[:, 0] - left) / r  # x1
        boxes[:, 1] = (boxes[:, 1] - top) / r   # y1
        boxes[:, 2] = (boxes[:, 2] - left) / r  # x2
        boxes[:, 3] = (boxes[:, 3] - top) / r   # y2
        
        # 显示还原后的值
        if len(boxes) > 0:
            print(f"  前3个检测框的还原后值:")
            for j in range(min(3, len(boxes))):
                print(f"    检测框 {j}: {boxes[j]}")
        
        print(f"  还原后检测框形状: {boxes.shape}")
        rescaled_boxes.append(boxes)
    
    print(f"=== 检测框坐标还原完成 ===")
    return rescaled_boxes

def plot_pose_batch(images, keypoints_list, save_path, boxes_list=None):
    """
    批量绘制姿态估计结果
    
    Args:
        images (list): 原始图像列表
        keypoints_list (list): 关键点坐标列表
        save_path (str): 保存路径
        boxes_list (list, optional): 检测框坐标列表
    """
    # 计算拼接后的图像尺寸
    total_width = sum(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)
    
    # 创建拼接图像
    combined_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    
    x_offset = 0
    for i, (img, keypoints) in enumerate(zip(images, keypoints_list)):
        h, w = img.shape[:2]
        
        # 复制图像到拼接位置
        combined_image[:h, x_offset:x_offset+w] = img
        
        # 绘制关键点
        keypoints = np.array(keypoints)
        if keypoints.ndim > 2:
            keypoints = keypoints.squeeze()
        
        if keypoints.ndim == 3 and keypoints.shape[-1] == 3:
            for kpt in keypoints:
                for x, y, conf in kpt:
                    if conf > 0.3:
                        cv2.circle(combined_image, (int(x + x_offset), int(y)), 3, (0,255,0), -1)
        elif keypoints.ndim == 2 and keypoints.shape[1] % 3 == 0:
            for kpt in keypoints:
                for j in range(0, keypoints.shape[1], 3):
                    x, y, conf = kpt[j], kpt[j+1], kpt[j+2]
                    if conf > 0.3:
                        cv2.circle(combined_image, (int(x + x_offset), int(y)), 3, (0,255,0), -1)
        
        # 绘制检测框
        if boxes_list is not None and i < len(boxes_list):
            boxes = boxes_list[i]
            if boxes.size > 0:
                boxes = np.array(boxes)
                if boxes.ndim == 1:
                    boxes = boxes[None, :]
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(combined_image, (x1 + x_offset, y1), (x2 + x_offset, y2), (0,0,255), 2)
        
        x_offset += w
    
    cv2.imwrite(save_path, combined_image)
    print(f"批量结果已保存到: {save_path}")

def postprocess_pose_output_batch(raw_output, conf_thres=0.25, iou_thres=0.45, nc=1):
    """
    批量后处理姿态估计输出
    
    Args:
        raw_output (np.ndarray): 模型原始输出
        conf_thres (float): 置信度阈值
        iou_thres (float): IOU阈值
        nc (int): 类别数
    
    Returns:
        tuple: (关键点坐标列表, 检测框坐标列表)
    """
    print(f"\n=== 后处理详细信息 ===")
    print(f"输入原始输出形状: {raw_output.shape}")
    print(f"置信度阈值: {conf_thres}")
    print(f"IOU阈值: {iou_thres}")
    print(f"类别数: {nc}")
    
    if isinstance(raw_output, np.ndarray):
        raw_output = torch.from_numpy(raw_output)
    
    print(f"转换为PyTorch张量形状: {raw_output.shape}")
    print(f"原始输出数据范围: [{raw_output.min().item():.6f}, {raw_output.max().item():.6f}]")
    print(f"原始输出数据均值: {raw_output.mean().item():.6f}")
    
    kpts_list = []
    boxes_list = []
    
    # 处理每个batch的输出
    for i in range(raw_output.shape[0]):
        print(f"\n--- 处理Batch {i} ---")
        batch_output = raw_output[i:i+1]  # 保持batch维度
        print(f"Batch {i} 输出形状: {batch_output.shape}")
        
        # 分析原始输出数据
        if len(batch_output.shape) == 3:  # [1, feature_dim, num_anchors]
            _, feature_dim, num_anchors = batch_output.shape
            print(f"  Batch {i} 数据结构: feature_dim={feature_dim}, num_anchors={num_anchors}")
            
            # 分析置信度分布
            if feature_dim > 4:
                conf_data = batch_output[0, 4, :]  # [num_anchors]
                print(f"  置信度统计:")
                print(f"    范围: [{conf_data.min().item():.6f}, {conf_data.max().item():.6f}]")
                print(f"    均值: {conf_data.mean().item():.6f}")
                print(f"    标准差: {conf_data.std().item():.6f}")
                
                # 统计超过阈值的anchor数量
                above_thresh = (conf_data > conf_thres).sum().item()
                print(f"    超过阈值 {conf_thres} 的anchor数量: {above_thresh}/{num_anchors}")
                
                # 分析前几个anchor的置信度
                print(f"    前10个anchor的置信度:")
                for j in range(min(10, num_anchors)):
                    print(f"      Anchor {j}: {conf_data[j].item():.6f}")
        
        preds = non_max_suppression(batch_output, conf_thres=conf_thres, iou_thres=iou_thres, nc=nc)
        print(f"  NMS后检测数量: {len(preds)}")
        
        batch_kpts = []
        batch_boxes = []
        
        for det_idx, det in enumerate(preds):
            if det is not None and len(det) > 0:
                print(f"  Detection {det_idx}: {len(det)} 个目标")
                kpt_start = 6
                kpt_dim = 51  # 17*3
                for row_idx, row in enumerate(det):
                    print(f"    目标 {row_idx}:")
                    print(f"      边界框: {row[:4].cpu().numpy()}")
                    print(f"      置信度: {row[4].item():.6f}")
                    print(f"      类别: {row[5].item():.0f}")
                    
                    kpts = row[kpt_start:kpt_start+kpt_dim].reshape(-1, 3).cpu().numpy()
                    print(f"      关键点形状: {kpts.shape}")
                    if len(kpts) > 0:
                        print(f"      前3个关键点: {kpts[:3]}")
                    
                    batch_kpts.append(kpts)
                    box = row[:4].cpu().numpy()
                    batch_boxes.append(box)
            else:
                print(f"  Detection {det_idx}: 无检测结果")
        
        if len(batch_kpts) == 0:
            print(f"  Batch {i}: 无有效检测，使用默认值")
            batch_kpts = [np.zeros((17, 3))]
            batch_boxes = [np.zeros(4)]
        
        print(f"  Batch {i} 最终结果: {len(batch_kpts)} 个目标")
        kpts_list.append(np.array(batch_kpts))
        boxes_list.append(np.array(batch_boxes))
    
    print(f"\n=== 后处理完成 ===")
    print(f"总batch数: {len(kpts_list)}")
    for i, (kpts, boxes) in enumerate(zip(kpts_list, boxes_list)):
        print(f"  Batch {i}: {len(kpts)} 个关键点组, {len(boxes)} 个边界框")
    
    return kpts_list, boxes_list

def benchmark_inference(inference_func, inputs, warmup=10, iterations=100, name="推理"):
    """
    性能基准测试
    
    Args:
        inference_func (callable): 推理函数
        inputs: 输入数据
        warmup (int): 预热次数
        iterations (int): 测试迭代次数
        name (str): 测试名称
    
    Returns:
        dict: 性能统计信息
    """
    print(f"\n开始{name}性能测试...")
    
    # 预热
    print(f"预热 {warmup} 次...")
    for _ in range(warmup):
        inference_func(*inputs)
    
    # 性能测试
    print(f"测试 {iterations} 次...")
    times = []
    for _ in range(iterations):
        start_time = time.time()
        inference_func(*inputs)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 统计信息
    times = np.array(times)
    stats = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
        'fps': 1000.0 / np.mean(times)
    }
    
    print(f"{name}性能统计:")
    print(f"  平均时间: {stats['mean']:.2f}ms")
    print(f"  标准差: {stats['std']:.2f}ms")
    print(f"  最小时间: {stats['min']:.2f}ms")
    print(f"  最大时间: {stats['max']:.2f}ms")
    print(f"  中位数: {stats['p50']:.2f}ms")
    print(f"  95%分位数: {stats['p95']:.2f}ms")
    print(f"  99%分位数: {stats['p99']:.2f}ms")
    print(f"  平均FPS: {stats['fps']:.2f}")
    
    return stats

if __name__ == "__main__":
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查ONNX Runtime提供者
    print("=" * 50)
    print("检查ONNX Runtime提供者")
    print("=" * 50)
    onnx_providers = check_onnxruntime_providers()
    
    # 检查CUDA可用性
    print("\n" + "=" * 50)
    print("检查CUDA可用性")
    print("=" * 50)
    cuda_available, error_msg = check_cuda_available()
    if not cuda_available:
        print(f"警告：{error_msg}")
        print("将只执行ONNX推理，跳过TensorRT转换和推理")
    
    # 根据参数调整推理策略
    if args.use_gpu_onnx and not onnx_providers['cuda_available']:
        print("警告：请求使用GPU ONNX推理，但CUDA提供者不可用，将使用CPU推理")
        args.use_gpu_onnx = False
        args.onnx_provider = 'cpu'
    
    if not onnx_providers['cpu_available']:
        print("错误：CPU提供者不可用，无法进行推理")
        exit(1)
    
    # 读取图像
    print("\n" + "=" * 50)
    print("读取图像")
    print("=" * 50)
    images = []
    for img_path in args.image_paths:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        images.append(img)
        print(f"图像 {img_path}: {img.shape}")
    
    # 确保batch大小不超过图像数量
    args.batch_size = min(args.batch_size, len(images))
    print(f"使用batch大小: {args.batch_size}")
    
    # 如果图像数量不足，重复使用图像
    while len(images) < args.batch_size:
        images.extend(images[:args.batch_size - len(images)])
    
    # 只使用需要的图像数量
    images = images[:args.batch_size]
    
    # 获取PyTorch模型结果
    print("\n" + "=" * 50)
    print("PyTorch模型推理")
    print("=" * 50)
    model = YOLO(args.model_path)
    
    # PyTorch batch推理 - 修复：直接使用模型进行推理
    torch_results = []
    for img in images:
        result = model(img, verbose=False)  # 直接传入numpy数组
        torch_results.append(result[0])
    
    # 保存PyTorch结果
    torch_img = torch_results[0].plot()
    torch_save_path = os.path.join(args.save_dir, args.save_torch)
    cv2.imwrite(torch_save_path, torch_img)
    print(f"PyTorch结果已保存: {torch_save_path}")
    
    # 导出ONNX和engine
    if not os.path.exists(args.onnx_path):
        export_onnx(args.model_path, args.onnx_path, args.max_size, args.stride)
    else:
        print(f"ONNX文件已存在: {args.onnx_path}，跳过导出。")

    engine_converted = False
    if cuda_available and not os.path.exists(args.engine_path):
        engine_converted = onnx2engine(args.onnx_path, args.engine_path, 
                                     fp16=args.fp16, max_size=args.max_size, 
                                     stride=args.stride, batch_size=args.batch_size)
    else:
        if os.path.exists(args.engine_path):
            print(f"engine文件已存在: {args.engine_path}，跳过转换。")
            engine_converted = True
        else:
            print("跳过TensorRT转换，因为CUDA不可用")

    # 预处理图像
    print("\n" + "=" * 50)
    print("预处理图像")
    print("=" * 50)
    batch_input, orig_shapes, preprocess_params_list = preprocess_batch(images, args.max_size, args.stride)
    print(f"Batch输入形状: {batch_input.shape}")

    # ONNX推理
    print("\n" + "=" * 50)
    print("ONNX模型推理")
    print("=" * 50)
    onnx_outputs = run_onnx_batch(args.onnx_path, batch_input, args.use_gpu_onnx, args.onnx_provider)
    onnx_kpts_list, onnx_boxes_list = postprocess_pose_output_batch(onnx_outputs[0], 
                                                                  conf_thres=args.conf_thres, 
                                                                  iou_thres=args.iou_thres, 
                                                                  nc=1)
    onnx_kpts_list = keypoints_rescale_batch(onnx_kpts_list, orig_shapes, preprocess_params_list)
    onnx_boxes_list = boxes_rescale_batch(onnx_boxes_list, orig_shapes, preprocess_params_list)
    onnx_save_path = os.path.join(args.save_dir, args.save_onnx)
    plot_pose_batch(images.copy(), onnx_kpts_list, onnx_save_path, boxes_list=onnx_boxes_list)

    # TensorRT推理
    if engine_converted and cuda_available:
        print("\n" + "=" * 50)
        print("TensorRT模型推理")
        print("=" * 50)
        try:
            engine_outputs = run_engine_batch(args.engine_path, batch_input.astype(np.float32))
            print(f"engine_outputs.shape: {engine_outputs.shape}")
            engine_kpts_list, engine_boxes_list = postprocess_pose_output_batch(engine_outputs, 
                                                                             conf_thres=args.conf_thres, 
                                                                             iou_thres=args.iou_thres, 
                                                                             nc=1)
            engine_kpts_list = keypoints_rescale_batch(engine_kpts_list, orig_shapes, preprocess_params_list)
            engine_boxes_list = boxes_rescale_batch(engine_boxes_list, orig_shapes, preprocess_params_list)
            engine_save_path = os.path.join(args.save_dir, args.save_engine)
            plot_pose_batch(images.copy(), engine_kpts_list, engine_save_path, boxes_list=engine_boxes_list)
        except Exception as e:
            print(f"TensorRT推理失败: {e}")
    else:
        print("跳过TensorRT推理，因为engine转换失败或CUDA不可用")
    
    # 性能基准测试
    print("\n" + "=" * 50)
    print("性能基准测试")
    print("=" * 50)
    
    # ONNX性能测试
    def onnx_inference():
        return run_onnx_batch(args.onnx_path, batch_input, args.use_gpu_onnx, args.onnx_provider)
    
    onnx_stats = benchmark_inference(onnx_inference, (), 
                                   warmup=args.warmup, iterations=args.iterations, 
                                   name="ONNX")
    
    # TensorRT性能测试
    if engine_converted and cuda_available:
        def engine_inference():
            return run_engine_batch(args.engine_path, batch_input.astype(np.float32))
        
        engine_stats = benchmark_inference(engine_inference, (), 
                                         warmup=args.warmup, iterations=args.iterations, 
                                         name="TensorRT")
        
        # 性能对比
        print("\n" + "=" * 50)
        print("性能对比")
        print("=" * 50)
        speedup = onnx_stats['mean'] / engine_stats['mean']
        print(f"TensorRT相对于ONNX的加速比: {speedup:.2f}x")
        print(f"ONNX平均FPS: {onnx_stats['fps']:.2f}")
        print(f"TensorRT平均FPS: {engine_stats['fps']:.2f}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50) 