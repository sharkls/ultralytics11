import os
import torch
import cv2
import numpy as np
import onnxruntime as ort
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX分类模型转TensorRT和推理')
    parser.add_argument('--onnx_path', type=str, default='/ultralytics/ckpt/best0703cls.onnx', help='ONNX模型路径')
    parser.add_argument('--engine_path', type=str, default='/ultralytics/ckpt/best0703cls.engine', help='TensorRT engine保存路径')
    parser.add_argument('--image_paths', nargs='+', default=['/ultralytics/c++/Output/Vis_Object_Regions/1/0_0.jpg','/ultralytics/c++/Output/Vis_Object_Regions/1/1_0.jpg'], help='输入图像路径列表')
    parser.add_argument('--max_size', type=int, default=224, help='输入尺寸')
    parser.add_argument('--save_dir', type=str, default='runs/classify/', help='结果保存目录')
    parser.add_argument('--save_onnx', type=str, default='result_onnx.jpg', help='ONNX结果保存文件名')
    parser.add_argument('--save_engine', type=str, default='result_engine.jpg', help='TensorRT结果保存文件名')
    parser.add_argument('--fp16', action='store_true', help='是否使用FP16精度')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Injures', 'Person', 'Solider'], help='类别名称列表')
    parser.add_argument('--min_batch', type=int, default=1, help='最小batch大小')
    parser.add_argument('--max_batch', type=int, default=8, help='最大batch大小')
    parser.add_argument('--opt_batch', type=int, default=4, help='最优batch大小')
    return parser.parse_args()

def preprocess(img, size=224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh))
    pad_h, pad_w = size - nh, size - nw
    img = cv2.copyMakeBorder(img, pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2, cv2.BORDER_CONSTANT, value=(114,114,114))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3,1,1)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape(3,1,1)
    img = (img.transpose(2, 0, 1) - mean) / std
    img = img[None].astype(np.float32)
    return img

def check_cuda_available():
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "PyTorch未检测到CUDA设备"
        import pycuda.driver as cuda
        import pycuda.autoinit
        cuda.init()
        if cuda.Device.count() == 0:
            return False, "未检测到CUDA设备"
        return True, None
    except Exception as e:
        return False, str(e)

def onnx2engine(onnx_path, engine_path, fp16=True, size=224, min_batch=1, max_batch=8, opt_batch=4):
    cuda_available, error_msg = check_cuda_available()
    if not cuda_available:
        print(f"警告：{error_msg}，将跳过TensorRT转换")
        return False
    fp16_flag = "--fp16" if fp16 else ""
    # 修改trtexec命令，确保maxShapes和shapes一致
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path} {fp16_flag} --shapes=images:{max_batch}x3x{size}x{size} --minShapes=images:{min_batch}x3x{size}x{size} --maxShapes=images:{max_batch}x3x{size}x{size} --optShapes=images:{opt_batch}x3x{size}x{size}"
    print(f"正在转换为TensorRT engine: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"engine已保存到: {engine_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"TensorRT转换失败: {e}")
        return False

def run_onnx(onnx_path, img):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    return outputs[0]  # logits

def run_engine(engine_path, img, batch_size=None):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    # 设置动态batch大小
    if batch_size is not None:
        context.set_input_shape(input_name, (batch_size, img.shape[1], img.shape[2], img.shape[3]))
    
    img = np.ascontiguousarray(img)
    d_input = cuda.mem_alloc(img.nbytes)
    output = np.empty(context.get_tensor_shape(output_name), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, img, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    return output

def postprocess_classify_output(logits, class_names, topk=5, is_prob=False):
    if is_prob:
        # 如果输入已经是概率值，直接使用
        probs = logits
    else:
        # 如果输入是logits，需要做softmax
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).cpu().numpy()
    topk_idx = probs.argsort(axis=-1)[:, -topk:][:, ::-1]
    topk_probs = np.take_along_axis(probs, topk_idx, axis=-1)
    return topk_idx, topk_probs

def visualize(img_path, pred_idx, pred_probs, class_names, save_path):
    img = cv2.imread(img_path)
    label = f"{class_names[pred_idx[0]]} ({pred_probs[0]:.2f})"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(save_path, img)
    print(f"结果已保存到: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查ONNX文件是否存在
    if not os.path.exists(args.onnx_path):
        raise FileNotFoundError(f"ONNX文件不存在: {args.onnx_path}")
    
    # 根据图像路径列表确定batch大小
    batch_size = len(args.image_paths)
    print(f"检测到{batch_size}张图像，batch大小设置为: {batch_size}")
    
    # 读取所有图片并预处理
    img_batch = []
    for img_path in args.image_paths:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img_input = preprocess(image, size=args.max_size)
        img_batch.append(img_input)
    
    # 将列表转换为numpy数组
    img_batch = np.concatenate(img_batch, axis=0)
    print(f"创建了batch大小为{batch_size}的输入数据，形状: {img_batch.shape}")
    
    # 获取类别名称
    class_names = args.class_names
    
    # 1. TensorRT engine转换
    engine_converted = False
    if not os.path.exists(args.engine_path):
        engine_converted = onnx2engine(args.onnx_path, args.engine_path, fp16=args.fp16, size=args.max_size, 
                                     min_batch=args.min_batch, max_batch=args.max_batch, opt_batch=args.opt_batch)
    else:
        print(f"engine文件已存在: {args.engine_path}，跳过转换。")
        engine_converted = True
    
    # 2. ONNX推理
    print("\n=== ONNX推理 ===")
    onnx_logits = run_onnx(args.onnx_path, img_batch)
    onnx_topk_idx, onnx_topk_probs = postprocess_classify_output(onnx_logits, class_names, topk=5, is_prob=True)
    topk = min(5, len(class_names))
    print("ONNX预测Top{} (batch={}):".format(topk, batch_size))
    for b in range(min(3, batch_size)):  # 只显示前3个batch的结果
        print(f"  图像{b+1}:")
        for i in range(topk):
            print(f"    {class_names[onnx_topk_idx[b][i]]}: {onnx_topk_probs[b][i]:.4f}")
    visualize(args.image_paths[0], [onnx_topk_idx[0][0]], [onnx_topk_probs[0][0]], class_names, os.path.join(args.save_dir, args.save_onnx))
    
    # 3. TensorRT推理
    if engine_converted:
        try:
            print("\n=== TensorRT推理 ===")
            engine_logits = run_engine(args.engine_path, img_batch.astype(np.float32), batch_size=batch_size)
            engine_topk_idx, engine_topk_probs = postprocess_classify_output(engine_logits, class_names, topk=5, is_prob=True)
            topk = min(5, len(class_names))
            print("TensorRT预测Top{} (batch={}):".format(topk, batch_size))
            for b in range(min(3, batch_size)):  # 只显示前3个batch的结果
                print(f"  图像{b+1}:")
                for i in range(topk):
                    print(f"    {class_names[engine_topk_idx[b][i]]}: {engine_topk_probs[b][i]:.4f}")
            visualize(args.image_paths[0], [engine_topk_idx[0][0]], [engine_topk_probs[0][0]], class_names, os.path.join(args.save_dir, args.save_engine))
        except Exception as e:
            print(f"TensorRT推理失败: {e}")
    else:
        print("跳过TensorRT推理，因为engine转换失败或CUDA不可用") 