import torch
import numpy as np

# 矩阵归一化函数
def normalize_matrix(H):
    return H / H[2,2] if abs(H[2,2]) > 1e-10 else H

# 1. 初始设置保持不变
patch_pos = torch.tensor([0., 134., 471., 494.])        # 全局坐标系下的子图角点坐标
# A_local_orig = torch.tensor([320., 226., 1.0])
A_local_orig = torch.tensor([340., 270., 1.0])          # 子图坐标系下的A点坐标
scale = 1.1                                               # 放缩因子
translation = [200, 200]                                # 平移因子
H_original = torch.tensor([                             # 原始H矩阵
    [7.0177e-01, 1.8738e-02, 3.3375e+01],
    [7.1160e-03, 9.9684e-01, 9.6879e+00],
    [4.8825e-05, 6.5607e-05, 1.0083e+00]
])

# 计算原始B点
B_local_orig = H_original @ A_local_orig
print("B_local_orig: ", B_local_orig, A_local_orig)
B_local_orig = B_local_orig / B_local_orig[2]           # B 点在子图坐标系下的坐标
print(f"方法1 - 原始A点在子图上的坐标: ({A_local_orig[0]:.2f}, {A_local_orig[1]:.2f})")
print(f"方法1 - 原始B点在子图上的坐标: ({B_local_orig[0]:.2f}, {B_local_orig[1]:.2f})")

# 2. 构建变换矩阵
T1 = torch.tensor([[1, 0, -640], [0, 1, -640], [0, 0, 1]], dtype=torch.float32)
S = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=torch.float32)
T2 = torch.tensor([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]], dtype=torch.float32) # < 0向右下平移
T_crop = torch.tensor([[1, 0, 320], [0, 1, 320], [0, 0, 1]], dtype=torch.float32)


# 计算裁剪参数
M_before_crop = T_crop @ T2 @ S @ T1

center_orig = torch.tensor([0., 0., 1.0])
new_center = M_before_crop @ center_orig
print("中心点：", new_center)
# crop_left = new_center[0] + 320     # 裁剪之前的角点坐标
# crop_top = new_center[1] + 320
# T_crop = torch.tensor([[1, 0, -crop_left], [0, 1, -crop_top], [0, 0, 1]], dtype=torch.float32)

# 完整变换矩阵
M = T2 @ S @ T1

# 3. 计算角点变换后的位置
patch_corner_orig = torch.tensor([patch_pos[0], patch_pos[1], 1.0])
patch_corner_new = M_before_crop @ patch_corner_orig
patch_corner_new_update = patch_corner_new / patch_corner_new[2]  # 确保归一化
print("新角点坐标：", patch_corner_new_update)


# # 约束角点位置
crop_x, crop_y = False, False
delta_x, delta_y = 0, 0
if patch_corner_new_update[0] < 0:
    delta_x = patch_corner_new_update[0]
    crop_x = True
if patch_corner_new_update[1]< 0:
    delta_y = patch_corner_new_update[1]
    crop_y = True
crop = torch.tensor([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]], dtype=torch.float32)
crop_inv = torch.tensor([[1, 0, -delta_x], [0, 1, -delta_y], [0, 0, 1]], dtype=torch.float32)

print("patch_corner_new: ", patch_corner_new_update)

# 计算A点的新位置
A_global = torch.tensor([[1, 0, patch_pos[0]], [0, 1, patch_pos[1]], [0, 0, 1]], dtype=torch.float32) @ A_local_orig
print("A_global: ", A_global)
A_final = M_before_crop @ A_global
print("New_A_global: ", A_final)
A_final = A_final / A_final[2]  # 确保归一化
A_final = A_final - torch.tensor([patch_corner_new_update[0], patch_corner_new_update[1], 0],dtype=torch.float32) + torch.tensor([delta_x, delta_y, 0],dtype=torch.float32)   # 新A点裁剪之后的子图坐标
# A_final = 
    
A_new_local = A_final
print(f"\nA点的新坐标(子图坐标系): ({A_new_local[0]:.2f}, {A_new_local[1]:.2f})")
print(A_new_local)


# 根据新A点和变换矩阵计算B点坐标
A_new_beforecrop_local = A_new_local - torch.tensor([delta_x, delta_y, 0],dtype=torch.float32)      # 计算新A点裁剪之前的子图坐标
print(f"A_new_beforecrop_local: ({A_new_beforecrop_local[0]:.2f}, {A_new_beforecrop_local[1]:.2f})")
S_inv = torch.tensor([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]], dtype=torch.float32)
# 反缩放后的坐标
# B_new_afterSinv_local = S_inv @ A_new_beforecrop_local      # [340, 270]
# print(B_new_afterSinv_local, A_new_beforecrop_local)
# B_new_afterSinv_local = H_original @ S_inv @ A_new_beforecrop_local # (243.66, 271.57)
# print(B_new_afterSinv_local)
# B_new_afterSinv_local = B_new_afterSinv_local / B_new_afterSinv_local[2]
# print(f"B_new_afterSinv_local: ({B_new_afterSinv_local[0]:.2f}, {B_new_afterSinv_local[1]:.2f})")
# B_new_beforecrop_local = S @ H_original @ S_inv @ A_new_beforecrop_local
# print(f"B_new_beforecrop_local: ({B_new_beforecrop_local[0]:.2f}, {B_new_beforecrop_local[1]:.2f})")
B_new_aftercrop_local = crop @ S @ H_original @ S_inv @ A_new_beforecrop_local                          # 进行反放缩和映射
B_new_local = B_new_aftercrop_local / B_new_aftercrop_local[2]
print("B_new_aftercrop_local: ", B_new_aftercrop_local)
print(f"B_new_aftercrop_local: ({B_new_aftercrop_local[0]:.2f}, {B_new_aftercrop_local[1]:.2f})")

# 计算B点的新坐标
B_global = torch.tensor([[1, 0, patch_pos[0]], [0, 1, patch_pos[1]], [0, 0, 1]], dtype=torch.float32) @ torch.tensor([B_local_orig[0], B_local_orig[1], 1.0], dtype=torch.float32)
print(f"\nB_global: ({B_global[0]:.2f}, {B_global[1]:.2f})")
B_new_global = M_before_crop @ B_global
B_new_global = B_new_global / B_new_global[2]
B_new_global = B_new_global - patch_corner_new_update + torch.tensor([delta_x, delta_y, 0],dtype=torch.float32)   # 新B点裁剪之后的子图坐标
# print("B_new_global: ", B_new_global)
print(f"B_new_global: ({B_new_global[0]:.2f}, {B_new_global[1]:.2f})")






# 4. 方法4a：简化H矩阵更新 - 原始方法
H_simple = H_original.clone()
H_simple[:2, :] = H_original[:2, :] * scale  # 对前两行整体缩放

# 将A点从新坐标系转换到原始坐标系的比例
A_scaled = A_new_local.clone()
A_scaled[0] = A_new_local[0] / scale
A_scaled[1] = A_new_local[1] / scale
A_scaled[2] = 1.0

# 使用简化H矩阵计算B点
B_simple = H_simple @ A_scaled
B_simple = B_simple / B_simple[2]  # 归一化
print(f"方法4a - 使用简化H矩阵计算的B点坐标(原方法): ({B_simple[0]:.2f}, {B_simple[1]:.2f})")

# 4b. 使用原始H矩阵计算B点的新坐标 - 方法1：分步更新H矩阵
# 构造从原始局部坐标系到全局坐标系的变换矩阵
T_orig = torch.eye(3, dtype=H_original.dtype)
T_orig[0:2, 2] = patch_pos[:2]  # 原始局部->全局

# 构造从全局坐标系到新局部坐标系的变换矩阵
T_new_inv = torch.eye(3, dtype=H_original.dtype)
T_new_inv[0:2, 2] = -patch_corner_new[:2]  # 全局->新局部

# 分步计算H_new，每步都归一化
H_temp1 = T_orig @ H_original
H_temp1 = normalize_matrix(H_temp1)

H_temp2 = M @ H_temp1
H_temp2 = normalize_matrix(H_temp2)

H_new = T_new_inv @ H_temp2
H_new = normalize_matrix(H_new)


# 使用新的H矩阵计算B点
B_new = H_new @ A_new_local
print("B_new原始值:", B_new)
print("B_new[2]:", B_new[2])
print(f"方法4b - 使用复杂H矩阵计算的B点坐标: ({B_new[0]:.2f}, {B_new[1]:.2f})")



# 4c. 使用原始H矩阵计算B点的新坐标 - 方法1：分步更新H矩阵
# 构造从原始局部坐标系到全局坐标系的变换矩阵
T_orig = torch.eye(3, dtype=H_original.dtype)
T_orig[0:2, 2] = patch_pos[:2]  # 原始局部->全局

# 构造从全局坐标系到新局部坐标系的变换矩阵
T_new_inv = torch.eye(3, dtype=H_original.dtype)
T_new_inv[0:2, 2] = -patch_corner_new[:2]  # 全局->新局部

T_crop = torch.eye(3, dtype=H_original.dtype)
if crop_x:
    T_crop[0, 2] = -crop_left
if crop_y:
    T_crop[1, 2] = -crop_top


H_new = T_new_inv @ M @ T_orig @ H_original # A_local -> B_local -> B_global -> B_global_update -> B_local_update -> B_local

# # 分步计算H_new，每步都归一化
# H_temp1 = T_orig @ H_original
# H_temp1 = normalize_matrix(H_temp1)

# H_temp2 = M @ H_temp1
# H_temp2 = normalize_matrix(H_temp2)

# H_new = T_new_inv @ H_temp2
# H_new = normalize_matrix(H_new)


# 使用新的H矩阵计算B点
B_new = H_new @ A_new_local
print("B_new原始值:", B_new)
print("B_new[2]:", B_new[2])
print(f"方法4b - 使用复杂H矩阵计算的B点坐标: ({B_new[0]:.2f}, {B_new[1]:.2f})")



# 5. 直接使用变换矩阵计算B点的新坐标
B_global = torch.tensor([[1, 0, patch_pos[0]], [0, 1, patch_pos[1]], [0, 0, 1]], dtype=torch.float32) @ torch.tensor([B_local_orig[0], B_local_orig[1], 1.0], dtype=torch.float32)
B_final = M @ B_global
B_final = B_final / B_final[2]  # 确保归一化
print("B_final: ", B_final)
B_direct = B_final - patch_corner_new
print(f"方法5 - 直接变换得到的B点坐标: ({B_direct[0]:.2f}, {B_direct[1]:.2f})")


# 6. 计算各方法之间的差异
diff_complex_direct = torch.norm(B_new[:2] - B_direct[:2])
diff_simple_direct = torch.norm(B_simple[:2] - B_direct[:2])
# diff_combined_direct = torch.norm(B_combined[:2] - B_direct[:2])
# diff_simple_combined = torch.norm(B_simple[:2] - B_combined[:2])

print(f"\n复杂H矩阵与直接变换的差异: {diff_complex_direct:.6f}")
print(f"简化H矩阵(原方法)与直接变换的差异: {diff_simple_direct:.6f}")
# print(f"组合矩阵(修正方法)与直接变换的差异: {diff_combined_direct:.6f}")
# print(f"简化H矩阵与组合矩阵的差异: {diff_simple_combined:.6f}")