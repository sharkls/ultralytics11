import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics.data.augment import LetterBox

def load_extrinsics(extrinsics_path, device):
    """加载外参矩阵"""
    try:
        homography = np.loadtxt(extrinsics_path)
        homography = torch.from_numpy(homography).float().to(device)
        if homography.shape != (3, 3):
            raise ValueError(f"单应性矩阵形状错误: {homography.shape}, 应为 (3, 3)")
        return homography
    except Exception as e:
        print(f"加载外参矩阵失败: {e}")
        return None

def preprocess_image_and_homography(rgb_path, ir_path, extrinsics, imgsz=(640, 640), stride=32, device='cuda'):
    """图像预处理函数，同时更新单应性矩阵"""
    # 读取图像
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path)
    if rgb_img is None or ir_img is None:
        raise FileNotFoundError(f"无法读取图像: {rgb_path} 或 {ir_path}")
    
    # 创建letterbox对象
    letterbox = LetterBox(imgsz, auto=False, stride=stride)
    
    # 计算RGB图像的letterbox参数
    rgb_h, rgb_w = rgb_img.shape[:2]
    r_rgb = min(imgsz[0]/rgb_h, imgsz[1]/rgb_w)
    new_unpad_rgb = int(round(rgb_w * r_rgb)), int(round(rgb_h * r_rgb))
    dw_rgb, dh_rgb = imgsz[1] - new_unpad_rgb[0], imgsz[0] - new_unpad_rgb[1]
    dw_rgb /= 2
    dh_rgb /= 2

    # 计算IR图像的letterbox参数
    ir_h, ir_w = ir_img.shape[:2]
    r_ir = min(imgsz[0]/ir_h, imgsz[1]/ir_w)
    new_unpad_ir = int(round(ir_w * r_ir)), int(round(ir_h * r_ir))
    dw_ir, dh_ir = imgsz[1] - new_unpad_ir[0], imgsz[0] - new_unpad_ir[1]
    dw_ir /= 2
    dh_ir /= 2

    # 构建变换矩阵
    dtype = extrinsics.dtype
    S_rgb = torch.eye(3, device=device, dtype=dtype)
    S_rgb[0, 0] = r_rgb  # x方向缩放
    S_rgb[1, 1] = r_rgb  # y方向缩放

    S_ir = torch.eye(3, device=device, dtype=dtype)
    S_ir[0, 0] = r_ir   # x方向缩放
    S_ir[1, 1] = r_ir   # y方向缩放

    T_rgb = torch.eye(3, device=device, dtype=dtype)
    T_rgb[0, 2] = dw_rgb  # x方向平移
    T_rgb[1, 2] = dh_rgb  # y方向平移

    T_ir = torch.eye(3, device=device, dtype=dtype)
    T_ir[0, 2] = dw_ir  # x方向平移
    T_ir[1, 2] = dh_ir  # y方向平移

    # 更新单应性矩阵：新IR -> 原始IR -> 原始RGB -> 新RGB
    # H_new = T_rgb @ S_rgb @ H @ S_ir^(-1) @ T_ir^(-1)
    updated_H = torch.mm(T_rgb, torch.mm(S_rgb, torch.mm(extrinsics, 
                        torch.mm(torch.inverse(S_ir), torch.inverse(T_ir)))))

    # 应用letterbox变换
    rgb_img = letterbox(image=rgb_img)
    ir_img = letterbox(image=ir_img)
    
    # 图像格式转换
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)
    
    # 归一化
    rgb_img = rgb_img / 255.0
    ir_img = ir_img / 255.0
    
    # HWC to CHW
    rgb_img = rgb_img.transpose(2, 0, 1)
    ir_img = ir_img.transpose(2, 0, 1)
    
    # 添加batch维度
    rgb_img = np.expand_dims(rgb_img, 0).astype(np.float32)
    ir_img = np.expand_dims(ir_img, 0).astype(np.float32)
    
    # 保存原始尺寸
    original_sizes = torch.tensor([[[rgb_h, rgb_w], [ir_h, ir_w]]], device=device, dtype=torch.float32)
    
    return rgb_img, ir_img, updated_H.unsqueeze(0), original_sizes

def visualize_registration(rgb_img, ir_img, H, save_path='registration_vis.jpg', save_prefix='vis', is_normalized=False):
    """
    可视化配准结果，显示RGB图像、变换后的红外图像及其融合效果。

    Args:
        rgb_img (np.ndarray | torch.Tensor): RGB图像 (HWC 或 CHW)
        ir_img (np.ndarray | torch.Tensor): 红外图像 (HWC 或 CHW)
        H (np.ndarray | torch.Tensor): 单应性矩阵 (3x3)
        save_path (str): 保存可视化结果的路径
        save_prefix (str): 图像标题前缀
        is_normalized (bool): 图像是否已经归一化(0-1)
    """
    try:
        # --- 数据格式转换 ---
        def process_img_input(img):
            """Helper function to convert tensor/numpy to HWC uint8 numpy"""
            if isinstance(img, torch.Tensor):
                if img.dim() == 4: img = img.squeeze(0) # Remove batch
                img = img.detach().cpu().numpy()
                if img.shape[0] == 3: img = img.transpose(1, 2, 0) # CHW -> HWC
            elif isinstance(img, np.ndarray):
                if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0) # Remove batch
                if img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1, 2, 0) # CHW -> HWC
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # Ensure HWC
            if img.ndim != 3 or img.shape[2] not in [1, 3]:
                 raise ValueError(f"Image shape {img.shape} not convertible to HWC")

            return img

        rgb_img_np = process_img_input(rgb_img)
        ir_img_np = process_img_input(ir_img)

        if H is None:
            print("警告：未提供单应性矩阵，使用单位矩阵进行可视化。")
            H = np.eye(3, dtype=np.float32)
        elif isinstance(H, torch.Tensor):
             if H.dim() == 3: H = H.squeeze(0) # Remove batch
             H = H.detach().cpu().numpy()

        if not isinstance(H, np.ndarray) or H.shape != (3, 3):
            print(f"错误：无效的单应性矩阵形状: {H.shape}, 应为 (3, 3)")
            return
        H = H.astype(np.float32)

        # --- 像素值范围处理 ---
        if is_normalized:
            rgb_img_np = (rgb_img_np * 255).clip(0, 255)
            ir_img_np = (ir_img_np * 255).clip(0, 255)

        rgb_img_np = rgb_img_np.astype(np.uint8)
        ir_img_np = ir_img_np.astype(np.uint8)

        # --- 确保是3通道BGR ---
        def ensure_bgr(img):
            if len(img.shape) == 2 or img.shape[2] == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3: # Assume input might be RGB from numpy
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV
            return img # Should already be BGR if 3 channels

        rgb_img_bgr = ensure_bgr(rgb_img_np)
        ir_img_bgr = ensure_bgr(ir_img_np)

        # --- 图像变换和融合 ---
        target_size = (rgb_img_bgr.shape[1], rgb_img_bgr.shape[0]) # (width, height)
        warped_ir = cv2.warpPerspective(ir_img_bgr, H, target_size)

        alpha = 0.5
        fused_img = cv2.addWeighted(rgb_img_bgr, alpha, warped_ir, 1 - alpha, 0)

        # --- 创建对比图 ---
        comparison = np.hstack([rgb_img_bgr, warped_ir, fused_img])

        # --- 添加标题 ---
        h, w = comparison.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w / 1920, h / 1080) * 0.8
        title_color = (0, 255, 0) # Green
        cv2.putText(comparison, f'{save_prefix} RGB', (10, 30), font, font_scale, title_color, 1)
        cv2.putText(comparison, f'{save_prefix} Warped IR', (w // 3 + 10, 30), font, font_scale, title_color, 1)
        cv2.putText(comparison, f'{save_prefix} Fused', (2 * w // 3 + 10, 30), font, font_scale, title_color, 1)

        # --- 添加分隔线 ---
        line_color = (255, 0, 0) # Blue
        cv2.line(comparison, (w // 3, 0), (w // 3, h), line_color, 1)
        cv2.line(comparison, (2 * w // 3, 0), (2 * w // 3, h), line_color, 1)

        # --- 保存图像 ---
        cv2.imwrite(save_path, comparison)
        print(f"可视化结果已保存到: {save_path}")

    except Exception as e:
        print(f"可视化时发生错误: {e}")
        import traceback
        traceback.print_exc()
