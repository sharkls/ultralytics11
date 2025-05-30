import os
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_path, engine_path, min_shape=(1, 3, 32, 32), opt_shape=(1, 3, 640, 640), max_shape=(4, 3, 1280, 1280), fp16_mode=False):
    """构建TensorRT engine，支持动态尺寸（b, 3, w, h）"""
    if os.path.exists(engine_path):
        print(f"删除已存在的engine文件: {engine_path}")
        os.remove(engine_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    print(f"正在解析ONNX模型: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("启用FP16模式")
        else:
            print("警告：平台不支持FP16，将使用FP32")

    profile = builder.create_optimization_profile()
    print("\n模型输入信息:")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape
        input_dtype = input_tensor.dtype
        print(f"输入 {i}: 名称: {input_tensor.name}, 形状: {input_shape}, 数据类型: {input_dtype}")

        # 支持动态shape (b, 3, w, h)
        if len(input_shape) == 4:
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            print(f"  动态shape: min={min_shape}, opt={opt_shape}, max={max_shape}")
        else:
            raise ValueError(f"未知输入shape: {input_shape}")
    config.add_optimization_profile(profile)

    print("\n网络信息:")
    print(f"层数: {network.num_layers}")
    print(f"输入数量: {network.num_inputs}")
    print(f"输出数量: {network.num_outputs}")

    print("\n正在构建TensorRT engine...")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("构建engine失败")
    except Exception as e:
        print(f"构建engine时发生错误: {str(e)}")
        return None

    print(f"正在保存engine到: {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("Engine构建完成！")
    return True

def check_cuda_available():
    try:
        # import torch
        # if not torch.cuda.is_available():
        #     return False, "PyTorch未检测到CUDA设备"
        import pycuda.driver as cuda
        import pycuda.autoinit
        cuda.init()
        if cuda.Device.count() == 0:
            return False, "未检测到CUDA设备"
        return True, None
    except Exception as e:
        return False, f"CUDA不可用: {e}"

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX转TensorRT engine（姿态模型专用，支持动态尺寸）')
    # parser.add_argument('--onnx', type=str, default='ckpt/yolo11m-pose.onnx', help='ONNX模型路径')
    # parser.add_argument('--engine', type=str, default='ckpt/yolo11m-pose32.engine', help='TensorRT engine保存路径')
    parser.add_argument('--onnx', type=str, default='/mnt/ddsproject-example/Output/Configs/Model/yolo11m-pose.onnx', help='ONNX模型路径')
    parser.add_argument('--engine', type=str, default='/mnt/ddsproject-example/Output/Configs/Model/yolo11m-pose32.engine', help='TensorRT engine保存路径')
    parser.add_argument('--min-shape', type=int, nargs=4, default=[1,3,32,32], help='最小输入shape (b,c,w,h)')
    parser.add_argument('--opt-shape', type=int, nargs=4, default=[1,3,640,640], help='最优输入shape (b,c,w,h)')
    parser.add_argument('--max-shape', type=int, nargs=4, default=[1,3,1280,1280], help='最大输入shape (b,c,w,h)')
    parser.add_argument('--fp16', default=False, action='store_true', help='使用FP16精度')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.engine), exist_ok=True)
    cuda_available, error_msg = check_cuda_available()
    if not cuda_available:
        print(f"警告：{error_msg}，跳过TensorRT转换")
        return
    if not os.path.exists(args.onnx):
        print(f"ONNX文件不存在: {args.onnx}")
        return
    result = build_engine(args.onnx, args.engine, tuple(args.min_shape), tuple(args.opt_shape), tuple(args.max_shape), args.fp16)
    if result is None:
        print("Engine构建失败！")
    else:
        print("Engine构建成功！")

if __name__ == "__main__":
    main() 