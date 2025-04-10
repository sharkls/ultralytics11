import torch
import numpy as np

# 矩阵归一化函数
def normalize_matrix(H):
    return H / H[2,2] if abs(H[2,2]) > 1e-10 else H

def update_homography_correct(H_original, M_global, orig_corner, new_corner):
    """
    正确的单应性矩阵更新方法
    :param H_original: 原始单应性矩阵 (3x3)
    :param M_global: 全局变换矩阵 (3x3)
    :param orig_corner: 原始子图左上角在全局坐标系中的坐标 (x, y, 1)
    :param new_corner: 变换后子图左上角在裁剪后坐标系中的坐标 (x, y, 1)
    :return: 更新后的单应性矩阵 (3x3)
    """
    # 1. 构造坐标系转换矩阵
    T_orig = torch.eye(3, dtype=H_original.dtype)
    T_orig[0:2, 2] = orig_corner[:2]  # 原始局部->全局
    
    T_new_inv = torch.eye(3, dtype=H_original.dtype)
    T_new_inv[0:2, 2] = -new_corner[:2]  # 全局->新局部
    
    # 2. 组合完整变换链
    H_new = T_new_inv @ M_global @ T_orig @ H_original
    return H_new / H_new[2,2]  # 归一化


# 1. 初始设置保持不变
patch_pos = torch.tensor([0., 134., 471., 494.])
A_local_orig = torch.tensor([320., 226., 1.0])
A_local_orig = torch.tensor([220., 126., 1.0])
scale = 0.8
H_original = torch.tensor([
    [7.0177e-01, 1.8738e-02, 3.3375e+01],
    [7.1160e-03, 9.9684e-01, 9.6879e+00],
    [4.8825e-05, 6.5607e-05, 1.0083e+00]
])

# 计算原始B点
B_local_orig = H_original @ A_local_orig
B_local_orig = B_local_orig / B_local_orig[2]  # 确保归一化
print(f"方法1 - 原始A点在子图上的坐标: ({A_local_orig[0]:.2f}, {A_local_orig[1]:.2f})")
print(f"方法1 - 原始B点在子图上的坐标: ({B_local_orig[0]:.2f}, {B_local_orig[1]:.2f})")

# 2. 构建变换矩阵
center_orig = torch.tensor([0., 0., 1.0])
T1 = torch.tensor([[1, 0, -640], [0, 1, -640], [0, 0, 1]], dtype=torch.float32)
S = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=torch.float32)
T2 = torch.tensor([[1, 0, 100], [0, 1, 50], [0, 0, 1]], dtype=torch.float32)

# 计算裁剪参数
M_before_crop = T2 @ S @ T1
new_center = M_before_crop @ center_orig
new_center = new_center / new_center[2]
print("new_center[2]: ", new_center[2])
crop_left = new_center[0] - 320
crop_top = new_center[1] - 320
T_crop = torch.tensor([[1, 0, -crop_left], [0, 1, -crop_top], [0, 0, 1]], dtype=torch.float32)
T3 = torch.tensor([[1, 0, 320], [0, 1, 320], [0, 0, 1]], dtype=torch.float32)

# 完整变换矩阵
M = T3 @ T_crop @ T2 @ S @ T1
# print("M矩阵:")


# 3. 计算变换后的位置
patch_corner_orig = torch.tensor([patch_pos[0], patch_pos[1], 1.0])
patch_corner_new = M @ patch_corner_orig
patch_corner_new = patch_corner_new / patch_corner_new[2]  # 确保归一化
print("patch_corner_new: ", patch_corner_new)

# 计算A点的新位置
A_global = torch.tensor([[1, 0, patch_pos[0]], [0, 1, patch_pos[1]], [0, 0, 1]], dtype=torch.float32) @ A_local_orig
A_final = M @ A_global
A_final = A_final / A_final[2]  # 确保归一化
print("A_final: ", A_final)
A_new_local = A_final - patch_corner_new

print(f"\nA点的新坐标(子图坐标系): ({A_new_local[0]:.2f}, {A_new_local[1]:.2f})")

# 4. 使用原始H矩阵计算B点的新坐标 - 方法1：分步更新H矩阵
# 构造从原始局部坐标系到全局坐标系的变换矩阵
T_orig = torch.eye(3, dtype=H_original.dtype)
T_orig[0:2, 2] = patch_pos[:2]  # 原始局部->全局

# 构造从全局坐标系到新局部坐标系的变换矩阵
T_new_inv = torch.eye(3, dtype=H_original.dtype)
T_new_inv[0:2, 2] = -patch_corner_new[:2]  # 全局->新局部

# 分步计算H_new，每步都归一化
H_temp1 = T_orig @ H_original
H_temp1 = normalize_matrix(H_temp1)
print("H_temp1[2,2]: ", H_temp1[2,2])

H_temp2 = M @ H_temp1
H_temp2 = normalize_matrix(H_temp2)
print("H_temp2[2,2]: ", H_temp2[2,2])

H_new = T_new_inv @ H_temp2
H_new = normalize_matrix(H_new)
print("H_new[2,2]: ", H_new[2,2])

# 使用新的H矩阵计算B点
B_new = H_new @ A_new_local
print("B_new原始值:", B_new)
print("B_new[2]:", B_new[2])

print(f"方法2 - 使用复杂H矩阵计算的B点坐标: ({B_new[0]:.2f}, {B_new[1]:.2f})")

# 5. 方法3a：简化H矩阵更新 - 原始方法
H_simple = H_original.clone()
# print("H_original: ", H_original)

H_simple[:2, :] = H_original[:2, :] * scale  # 对前两行整体缩放
# print("H_simple:")
# print(H_simple)

# 将A点从新坐标系转换到原始坐标系的比例
A_scaled = A_new_local.clone()
A_scaled[0] = A_new_local[0] / scale
A_scaled[1] = A_new_local[1] / scale
A_scaled[2] = 1.0

# 使用简化H矩阵计算B点
B_simple = H_simple @ A_scaled
B_simple = B_simple / B_simple[2]  # 归一化
print(f"方法3a - 使用简化H矩阵计算的B点坐标(原方法): ({B_simple[0]:.2f}, {B_simple[1]:.2f})")

# 5b. 正确方法：使用完整变换链
H_new_correct = update_homography_correct(
    H_original=H_original,
    M_global=M,
    orig_corner=patch_corner_orig[:2],
    new_corner=patch_corner_new[:2]
)

B_combined = H_new_correct @ torch.cat([A_new_local[:2], torch.tensor([1.0])])
B_combined = B_combined.clone() / B_combined[2]
print(f"方法5b - 正确组合矩阵计算的B点坐标: ({B_combined[0]:.2f}, {B_combined[1]:.2f})")

# 6. 直接使用变换矩阵计算B点的新坐标
B_global = torch.tensor([[1, 0, patch_pos[0]], [0, 1, patch_pos[1]], [0, 0, 1]], dtype=torch.float32) @ torch.tensor([B_local_orig[0], B_local_orig[1], 1.0], dtype=torch.float32)
B_final = M @ B_global
B_final = B_final / B_final[2]  # 确保归一化
print("B_final: ", B_final)
B_direct = B_final - patch_corner_new
print(f"方法4 - 直接变换得到的B点坐标: ({B_direct[0]:.2f}, {B_direct[1]:.2f})")

# 7. 计算各方法之间的差异
diff_complex_direct = torch.norm(B_new[:2] - B_direct[:2])
diff_simple_direct = torch.norm(B_simple[:2] - B_direct[:2])
diff_combined_direct = torch.norm(B_combined[:2] - B_direct[:2])
diff_simple_combined = torch.norm(B_simple[:2] - B_combined[:2])

print(f"\n复杂H矩阵与直接变换的差异: {diff_complex_direct:.6f}")
print(f"简化H矩阵(原方法)与直接变换的差异: {diff_simple_direct:.6f}")
print(f"组合矩阵(修正方法)与直接变换的差异: {diff_combined_direct:.6f}")
print(f"简化H矩阵与组合矩阵的差异: {diff_simple_combined:.6f}")