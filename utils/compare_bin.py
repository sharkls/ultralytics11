import numpy as np

# 读取bin文件
def load_bin(path, dtype=np.float32):
    return np.fromfile(path, dtype=dtype)

# 路径
# engine输入数据
# bin1 = 'c++/Output/preprocess_yolov11pose.bin'
# bin2 = 'deploy/Output/Lib/preprocess_trt_infer.bin'

# engine输出结果
bin1 = 'c++/Output/output_yolov11pose.bin'
bin2 = 'deploy/Output/Lib/output_trt_infer.bin'
# 加载
arr1 = load_bin(bin1)
arr2 = load_bin(bin2)

print(f"arr1 shape: {arr1.shape}, arr2 shape: {arr2.shape}")

# 形状是否一致
if arr1.shape != arr2.shape:
    print("Shape 不一致！")
else:
    # 计算绝对误差和相对误差
    abs_diff = np.abs(arr1 - arr2)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)
    print(f"最大绝对误差: {max_abs}")
    print(f"平均绝对误差: {mean_abs}")

    # 打印前10个不同的值
    idx = np.where(abs_diff > 1e-6)[0]
    print(f"不同元素数量: {len(idx)}")
    if len(idx) > 0:
        for i in idx[:10]:
            print(f"idx {i}: yolov11pose={arr1[i]}, trt_infer={arr2[i]}, diff={abs_diff[i]}")