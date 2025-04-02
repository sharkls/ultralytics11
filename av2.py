import torch

# Original Homography H
H = torch.tensor([
    [7.1002e-01, 2.9826e-02, 8.3776e+01],
    [7.1160e-03, 9.9684e-01, 8.4853e+00],
    [4.8825e-05, 6.5607e-05, 1.0000e+00]
], dtype=torch.float32)

# --- Transformation Parameters ---
s = 1.2
tx_trans, ty_trans = 15, 20
tx_crop, ty_crop = 20, 10

# --- Transformation Matrices ---
S = torch.tensor([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=torch.float32)
T_trans = torch.tensor([[1, 0, tx_trans], [0, 1, ty_trans], [0, 0, 1]], dtype=torch.float32)
T_crop = torch.tensor([[1, 0, -tx_crop], [0, 1, -ty_crop], [0, 0, 1]], dtype=torch.float32)

# --- Combined Transformation ---
M = T_crop @ T_trans @ S
M_inv = torch.inverse(M)

# --- Update Homography ---
H_new = M @ H @ M_inv

# --- Verification ---
# 1. Choose an original point A
A_orig = torch.tensor([200., 100., 1.0])

# 2. Calculate original B using H
B_orig = H @ A_orig
B_orig = B_orig / B_orig[2] # Normalize

# 3. Calculate transformed A using M
A_final = M @ A_orig
A_final = A_final / A_final[2] # Normalize

# 4. Calculate expected transformed B using M
B_final_expected = M @ B_orig
B_final_expected = B_final_expected / B_final_expected[2] # Normalize

# 5. Calculate actual transformed B using H_new
B_final_actual = H_new @ A_final
B_final_actual = B_final_actual / B_final_actual[2] # Normalize

# --- Print Results ---
print(f"Original A: ({A_orig[0]:.2f}, {A_orig[1]:.2f})")
print(f"Original B (H @ A_orig): ({B_orig[0]:.2f}, {B_orig[1]:.2f})")
print("-" * 20)
print(f"Transformed A (M @ A_orig): ({A_final[0]:.2f}, {A_final[1]:.2f})")
print(f"Expected Transformed B (M @ B_orig): ({B_final_expected[0]:.2f}, {B_final_expected[1]:.2f})")
print(f"Actual Transformed B (H_new @ A_final): ({B_final_actual[0]:.2f}, {B_final_actual[1]:.2f})")
print("-" * 20)

# Calculate error
error = torch.norm(B_final_expected[:2] - B_final_actual[:2])
print(f"Mapping Error after update: {error:.6f}")

print("\nUpdated Homography Matrix (H_new):")
print(H_new)