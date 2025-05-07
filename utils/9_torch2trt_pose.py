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

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO姿态估计模型转换和推理')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, default='ckpt/yolo11m-pose.pt',
                      help='PyTorch模型路径')
    parser.add_argument('--onnx_path', type=str, default='ckpt/yolo11m-pose.onnx',
                      help='ONNX模型保存路径')
    parser.add_argument('--engine_path', type=str, default='ckpt/yolo11m-pose.engine',
                      help='TensorRT engine保存路径')
    
    # 推理相关参数
    parser.add_argument('--image_path', type=str, default='data/coco8-pose/images/val/000000000113.jpg',
                      help='输入图像路径')
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
    parser.add_argument('--save_dir', type=str, default='runs/pose/',
                      help='结果保存目录')
    parser.add_argument('--save_torch', type=str, default='result_torch.jpg',
                      help='PyTorch结果保存文件名')
    parser.add_argument('--save_onnx', type=str, default='result_onnx.jpg',
                      help='ONNX结果保存文件名')
    parser.add_argument('--save_engine', type=str, default='result_engine.jpg',
                      help='TensorRT结果保存文件名')
    
    # TensorRT相关参数
    parser.add_argument('--fp16', action='store_true',
                      help='是否使用FP16精度')
    parser.add_argument('--min_batch', type=int, default=1,
                      help='最小批次大小')
    parser.add_argument('--max_batch', type=int, default=4,
                      help='最大批次大小')
    
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

def preprocess(img, max_size=640, stride=32):
    """
    优化后的预处理函数，动态计算目标尺寸
    
    Args:
        img (np.ndarray): 输入图像，BGR格式
        max_size (int): 最大尺寸
        stride (int): 模型步长
    
    Returns:
        tuple: (预处理后的图像, 原始图像尺寸, 缩放和填充参数)
    """
    # 参数验证
    if not isinstance(img, np.ndarray):
        raise TypeError("输入图像必须是numpy数组")
    if len(img.shape) != 3:
        raise ValueError("输入图像必须是3通道图像")
    if not isinstance(max_size, int) or max_size <= 0:
        raise ValueError("max_size必须是正整数")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride必须是正整数")

    # 获取原始尺寸
    h0, w0 = img.shape[:2]
    
    # 计算目标尺寸
    h, w = calculate_target_size((h0, w0), max_size, stride)
    
    # 计算缩放比例
    r = min(h / h0, w / w0)
    
    # 计算填充
    dh, dw = h - int(h0 * r), w - int(w0 * r)
    top, left = dh // 2, dw // 2
    bottom, right = dh - top, dw - left
    
    # 转换颜色空间并缩放
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    
    # 添加填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 转换为float并归一化
    img = img.astype(np.float32) / 255.0
    
    # 转换为NCHW格式
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    
    # 保存预处理参数用于后处理
    preprocess_params = {
        'ratio': r,
        'pad': (top, left),
        'stride': stride,
        'scaled_shape': (h, w)
    }
    
    return img, (h0, w0), preprocess_params

def export_onnx(model_path, onnx_path, max_size=640, stride=32):
    """
    优化后的ONNX导出函数，支持动态输入
    
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
        imgsz=max_size,  # 使用最大尺寸
        simplify=True,
        dynamic=True,  # 启用动态尺寸
        opset=16,
        half=False,
        int8=False,
        device='cpu',
        optimize=False,
        name=onnx_path
    )
    print(f"ONNX模型已导出到: {onnx_path}")

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
        
        # 初始化CUDA环境
        import pycuda.driver as cuda
        import pycuda.autoinit  # 自动初始化CUDA环境
        cuda.init()
        
        # 检查CUDA设备
        if cuda.Device.count() == 0:
            return False, "未检测到CUDA设备"
            
        # 获取GPU信息
        try:
            device = cuda.Device(0)
            props = device.get_attributes()
            memory = cuda.mem_get_info()
            print(f"CUDA设备信息:")
            print(f"- 设备名称: {device.name()}")
            print(f"- 总显存: {memory[1] / 1024**2:.1f}MB")
            print(f"- 可用显存: {memory[0] / 1024**2:.1f}MB")
            
            # 测试CUDA功能
            test_array = cuda.mem_alloc(1024)
            del test_array  # 使用Python的垃圾回收来释放内存
            
            return True, None
        except Exception as e:
            return False, f"CUDA设备信息获取失败: {e}"
            
    except ImportError as e:
        return False, f"导入CUDA相关库失败: {e}"
    except Exception as e:
        return False, f"CUDA初始化失败: {e}"

def onnx2engine(onnx_path, engine_path, fp16=True, max_size=640, stride=32):
    """
    优化后的TensorRT engine转换函数，支持动态输入
    
    Args:
        onnx_path (str): ONNX模型路径
        engine_path (str): 输出engine路径
        fp16 (bool): 是否使用FP16
        max_size (int): 最大尺寸
        stride (int): 模型步长
    """
    # 检查CUDA是否可用
    cuda_available, error_msg = check_cuda_available()
    if not cuda_available:
        print(f"警告：{error_msg}，将跳过TensorRT转换")
        return False

    # 计算目标尺寸
    h, w = calculate_target_size((max_size, max_size), max_size, stride)

    # 设置动态尺寸范围
    min_batch = 1
    max_batch = 4
    min_h = stride
    max_h = h
    min_w = stride
    max_w = w

    # 检查显存是否足够
    try:
        import pycuda.driver as cuda
        free_mem, total_mem = cuda.mem_get_info()
        required_mem = max_batch * 3 * max_h * max_w * 4  # 估计所需显存
        if free_mem < required_mem:
            print(f"警告：可用显存不足，需要至少{required_mem/1024**2:.1f}MB，当前可用{free_mem/1024**2:.1f}MB")
            # 调整批次大小
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
          f"--optShapes=images:1x3x{h}x{w}"
    
    print(f"正在转换为TensorRT engine: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"engine已保存到: {engine_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"TensorRT转换失败: {e}")
        return False

def keypoints_rescale(kpts, orig_shape, preprocess_params):
    """
    优化后的关键点坐标还原函数
    
    Args:
        kpts (np.ndarray): 关键点坐标
        orig_shape (tuple): 原始图像尺寸 (h0, w0)
        preprocess_params (dict): 预处理参数
    
    Returns:
        np.ndarray: 还原后的关键点坐标
    """
    h0, w0 = orig_shape
    r = preprocess_params['ratio']
    top, left = preprocess_params['pad']
    
    # 确保输入是numpy数组
    kpts = np.array(kpts)
    
    # 处理不同维度的输入
    if kpts.ndim == 3:  # (N, 17, 2/3)
        kpts[..., 0] = (kpts[..., 0] - left) / r
        kpts[..., 1] = (kpts[..., 1] - top) / r
    elif kpts.ndim == 2:  # (N, 51/34)
        for i in range(0, kpts.shape[1], 3):
            kpts[:, i] = (kpts[:, i] - left) / r
            kpts[:, i+1] = (kpts[:, i+1] - top) / r
    else:
        raise ValueError(f"不支持的关键点维度: {kpts.ndim}")
    
    return kpts

def run_onnx(onnx_path, img):
    """
    运行ONNX模型推理
    
    Args:
        onnx_path (str): ONNX模型路径
        img (np.ndarray): 预处理后的输入图像
    
    Returns:
        list: 模型输出
    """
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    print(f"ONNX模型输出形状: {[out.shape for out in outputs]}")
    return outputs

def run_engine(engine_path, img):
    """
    运行TensorRT模型推理，支持动态输入
    
    Args:
        engine_path (str): TensorRT engine路径
        img (np.ndarray): 预处理后的输入图像
    
    Returns:
        np.ndarray: 模型输出
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # 自动初始化CUDA环境
        
        # 确保CUDA环境已初始化
        cuda.init()
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # 获取输入输出名和shape（TensorRT 10.x）
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        
        # 设置动态输入尺寸
        input_shape = img.shape
        context.set_input_shape(input_name, input_shape)
        
        # 获取输出尺寸
        output_shape = context.get_tensor_shape(output_name)
        print(f"TensorRT模型输入形状: {input_shape}")
        print(f"TensorRT模型输出形状: {output_shape}")

        # 保证输入是连续内存
        img = np.ascontiguousarray(img)
        d_input = cuda.mem_alloc(img.nbytes)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        try:
            # TensorRT 10.x 推荐 set_tensor_address
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))

            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_input, img, stream)
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
            
            return output
        finally:
            # 清理CUDA资源
            del d_input
            del d_output
            del stream
            
    except Exception as e:
        print(f"TensorRT推理失败: {e}")
        raise

def plot_pose(image, keypoints, save_path, boxes=None):
    """
    绘制姿态估计结果
    
    Args:
        image (np.ndarray): 原始图像
        keypoints (np.ndarray): 关键点坐标
        save_path (str): 保存路径
        boxes (np.ndarray, optional): 检测框坐标
    """
    keypoints = np.array(keypoints)
    if keypoints.ndim > 2:
        keypoints = keypoints.squeeze()
    if boxes is not None:
        boxes = np.array(boxes)
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
    if keypoints.ndim == 3 and keypoints.shape[-1] == 3:
        for kpt in keypoints:
            for x, y, conf in kpt:
                if conf > 0.3:
                    cv2.circle(image, (int(x), int(y)), 3, (0,255,0), -1)
    elif keypoints.ndim == 2 and keypoints.shape[1] % 3 == 0:
        for kpt in keypoints:
            for i in range(0, keypoints.shape[1], 3):
                x, y, conf = kpt[i], kpt[i+1], kpt[i+2]
                if conf > 0.3:
                    cv2.circle(image, (int(x), int(y)), 3, (0,255,0), -1)
    elif keypoints.ndim == 1 and keypoints.shape[0] % 3 == 0:
        for i in range(0, keypoints.shape[0], 3):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf > 0.3:
                cv2.circle(image, (int(x), int(y)), 3, (0,255,0), -1)
    else:
        print(f'未知关键点shape: {keypoints.shape}')
    cv2.imwrite(save_path, image)
    print(f"结果已保存到: {save_path}")

def compare_outputs(torch_kpts, engine_kpts):
    """
    比较PyTorch和TensorRT模型的输出
    
    Args:
        torch_kpts (np.ndarray): PyTorch模型输出的关键点
        engine_kpts (np.ndarray): TensorRT模型输出的关键点
    """
    torch_kpts = np.array(torch_kpts)
    engine_kpts = np.array(engine_kpts)
    # 只对比人数最少的部分
    n = min(torch_kpts.shape[0], engine_kpts.shape[0])
    torch_kpts = torch_kpts[:n]
    engine_kpts = engine_kpts[:n]
    # 只对比关键点数最少的部分
    k = min(torch_kpts.shape[1], engine_kpts.shape[1])
    torch_kpts = torch_kpts[:, :k]
    engine_kpts = engine_kpts[:, :k]
    # 位置误差（xy）
    torch_xy = torch_kpts[..., :2]
    engine_xy = engine_kpts[..., :2]
    diff_xy = np.abs(torch_xy - engine_xy)
    print(f'关键点位置最大绝对误差: {diff_xy.max():.4f}')
    print(f'关键点位置平均绝对误差: {diff_xy.mean():.4f}')
    # 置信度误差（conf）
    if torch_kpts.shape[-1] > 2 and engine_kpts.shape[-1] > 2:
        torch_conf = torch_kpts[..., 2]
        engine_conf = engine_kpts[..., 2]
        diff_conf = np.abs(torch_conf - engine_conf)
        print(f'关键点置信度最大绝对误差: {diff_conf.max():.4f}')
        print(f'关键点置信度平均绝对误差: {diff_conf.mean():.4f}')
    else:
        print('无置信度信息，仅对比位置xy。')

def postprocess_pose_output(raw_output, conf_thres=0.25, iou_thres=0.45, nc=1):
    """
    后处理姿态估计输出
    
    Args:
        raw_output (np.ndarray): 模型原始输出
        conf_thres (float): 置信度阈值
        iou_thres (float): IOU阈值
        nc (int): 类别数
    
    Returns:
        tuple: (关键点坐标, 检测框坐标)
    """
    if isinstance(raw_output, np.ndarray):
        raw_output = torch.from_numpy(raw_output)
    if raw_output.ndim == 2:
        raw_output = raw_output.unsqueeze(0)
    preds = non_max_suppression(raw_output, conf_thres=conf_thres, iou_thres=iou_thres, nc=nc)
    kpts_list = []
    boxes_list = []
    for det in preds:
        if det is not None and len(det) > 0:
            kpt_start = 6
            kpt_dim = 51  # 17*3
            for row in det:
                kpts = row[kpt_start:kpt_start+kpt_dim].reshape(-1, 3).cpu().numpy()
                kpts_list.append(kpts)
                box = row[:4].cpu().numpy()
                boxes_list.append(box)
    if len(kpts_list) == 0:
        return np.zeros((1, 17, 3)), np.zeros((1, 4))
    return np.array(kpts_list), np.array(boxes_list)

def boxes_rescale(boxes, orig_shape, preprocess_params):
    """
    还原检测框坐标
    
    Args:
        boxes (np.ndarray): 检测框坐标 (N, 4) 格式为 [x1, y1, x2, y2]
        orig_shape (tuple): 原始图像尺寸 (h0, w0)
        preprocess_params (dict): 预处理参数
    
    Returns:
        np.ndarray: 还原后的检测框坐标
    """
    if boxes.size == 0:
        return boxes
        
    boxes = np.array(boxes)
    r = preprocess_params['ratio']
    top, left = preprocess_params['pad']
    
    # 处理检测框坐标
    boxes[:, 0] = (boxes[:, 0] - left) / r  # x1
    boxes[:, 1] = (boxes[:, 1] - top) / r   # y1
    boxes[:, 2] = (boxes[:, 2] - left) / r  # x2
    boxes[:, 3] = (boxes[:, 3] - top) / r   # y2
    
    return boxes

if __name__ == "__main__":
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查CUDA可用性
    cuda_available, error_msg = check_cuda_available()
    if not cuda_available:
        print(f"警告：{error_msg}")
        print("将只执行ONNX推理，跳过TensorRT转换和推理")
    
    # 读取图像
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {args.image_path}")
    print("原始图像尺寸: ", image.shape)
    
    # 获取PyTorch模型结果
    model = YOLO(args.model_path)
    torch_results = model(image)
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
                                     stride=args.stride)
    else:
        if os.path.exists(args.engine_path):
            print(f"engine文件已存在: {args.engine_path}，跳过转换。")
            engine_converted = True
        else:
            print("跳过TensorRT转换，因为CUDA不可用")

    # 预处理图像
    img_input, orig_shape, preprocess_params = preprocess(image, args.max_size, args.stride)
    print("预处理后图像尺寸: ", img_input.shape)

    # ONNX推理
    onnx_outputs = run_onnx(args.onnx_path, img_input)
    onnx_kpts, onnx_boxes = postprocess_pose_output(onnx_outputs[0], 
                                                  conf_thres=args.conf_thres, 
                                                  iou_thres=args.iou_thres, 
                                                  nc=1)
    onnx_kpts = keypoints_rescale(onnx_kpts, orig_shape, preprocess_params)
    onnx_boxes = boxes_rescale(onnx_boxes, orig_shape, preprocess_params)
    onnx_save_path = os.path.join(args.save_dir, args.save_onnx)
    plot_pose(image.copy(), onnx_kpts, onnx_save_path, boxes=onnx_boxes)

    # TensorRT推理
    if engine_converted and cuda_available:
        try:
            engine_outputs = run_engine(args.engine_path, img_input.astype(np.float32))
            engine_kpts, engine_boxes = postprocess_pose_output(engine_outputs, 
                                                             conf_thres=args.conf_thres, 
                                                             iou_thres=args.iou_thres, 
                                                             nc=1)
            engine_kpts = keypoints_rescale(engine_kpts, orig_shape, preprocess_params)
            engine_boxes = boxes_rescale(engine_boxes, orig_shape, preprocess_params)
            print(f"engine_kpts.shape: {engine_kpts.shape}, engine_boxes.shape: {engine_boxes.shape}")
            engine_save_path = os.path.join(args.save_dir, args.save_engine)
            plot_pose(image.copy(), engine_kpts, engine_save_path, boxes=engine_boxes)

            # 对比输出
            if torch_results is not None:
                torch_kpts = torch_results[0].keypoints.data.cpu().numpy()
                compare_outputs(torch_kpts, engine_kpts)
        except Exception as e:
            print(f"TensorRT推理失败: {e}")
    else:
        print("跳过TensorRT推理，因为engine转换失败或CUDA不可用") 