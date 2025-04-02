import torch

def update_homography_correct(H, dx, dy):
    # 构造T1矩阵（平移dx, dy）
    T1 = torch.eye(3)
    T1[0, 2] = dx
    T1[1, 2] = dy

    # 构造T2_inv矩阵（平移-dx, -dy）
    T2_inv = torch.eye(3)
    T2_inv[0, 2] = -dx
    T2_inv[1, 2] = -dy

    # 计算新的单应性矩阵：H_new = T2_inv @ H @ T1
    H_new = T2_inv @ H @ T1
    return H_new

# 原始单应性矩阵H
H = torch.tensor([
    [7.1002e-01, 2.9826e-02, 8.3776e+01],
    [7.1160e-03, 9.9684e-01, 8.4853e+00],
    [4.8825e-05, 6.5607e-05, 1.0000e+00]
], dtype=torch.float32)

# 验证原始映射
A = torch.tensor([200., 100., 1.0])
B = H @ A
B = B / B[2]
print(f"原始映射: A({A[0]:.1f}, {A[1]:.1f}) -> B({B[0]:.1f}, {B[1]:.1f})")

# 平移量
dx, dy = 100, 50

# 使用正确方法更新矩阵
H_new = update_homography_correct(H, dx, dy)

# 验证新的映射
A_new = torch.tensor([100., 50., 1.0])  # 平移后的A点
B_new = H_new @ A_new
B_new = B_new / B_new[2]

print(f"\n平移后:")
print(f"A_new({A_new[0]:.1f}, {A_new[1]:.1f}) -> B_new({B_new[0]:.1f}, {B_new[1]:.1f})")
print(f"期望的B点坐标: ({B[0]-dx:.1f}, {B[1]-dy:.1f})")

# 计算误差
expected_B = torch.tensor([B[0]-dx, B[1]-dy])
actual_B = B_new[:2]
error = torch.norm(expected_B - actual_B)
print(f"映射误差: {error:.6f}")