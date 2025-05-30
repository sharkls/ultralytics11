import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import argparse


def build_engine(onnx_path, engine_path, max_batch_size=1, fp16_mode=False):
    """构建TensorRT engine
    Args:
        onnx_path (str): ONNX模型路径
        engine_path (str): 保存engine的路径
        max_batch_size (int): 最大batch size
        fp16_mode (bool): 是否使用FP16精度
    """
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

        # 针对动态shape，手动指定
        if len(input_shape) == 4:
            # 典型的图像输入
            min_shape = (1, 3, 640, 640)
            opt_shape = (1, 3, 640, 640)
            max_shape = (1, 3, 640, 640)
        elif len(input_shape) == 3:
            # 例如单应性矩阵 (1, 3, 3)
            min_shape = (1, 3, 3)
            opt_shape = (1, 3, 3)
            max_shape = (1, 3, 3)
        else:
            raise ValueError(f"未知输入shape: {input_shape}")

        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        print(f"  固定shape: min={min_shape}, opt={opt_shape}, max={max_shape}")
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


def main():
    parser = argparse.ArgumentParser(description='将ONNX模型转换为TensorRT engine')
    parser.add_argument('--onnx', type=str, default="runs/multimodal/train64/weights/best.onnx", help='ONNX模型路径')
    parser.add_argument('--engine', type=str, default="runs/multimodal/train64/weights/best32.engine", help='保存engine的路径')
    parser.add_argument('--max-batch-size', type=int, default=1, help='最大batch size')
    parser.add_argument('--fp16', default=False, action='store_true', help='使用FP16精度')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.engine), exist_ok=True)
    result = build_engine(args.onnx, args.engine, args.max_batch_size, args.fp16)
    if result is None:
        print("Engine构建失败！")
    else:
        print("Engine构建成功！")

if __name__ == '__main__':
    main() 