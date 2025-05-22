import torch
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from ultralytics import YOLOMultimodal
from ultralytics.data.augment import LetterBox
from pathlib import Path
import onnx

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Multimodal Model Export and Validation')
    
    # 模型相关参数
    parser.add_argument('--weights', type=str, default='/ultralytics/runs/multimodal/train64/weights/best.pt',
                      help='训练好的模型权重路径')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640],
                      help='输入图像尺寸 [height, width]')
    
    # 导出相关参数
    parser.add_argument('--export-path', type=str, default='/ultralytics/runs/export/EFDE-YOLO-Myslef-v2',
                      help='ONNX模型导出路径')
    parser.add_argument('--opset', type=int, default=16,
                      help='ONNX opset版本')
    parser.add_argument('--dynamic', type=bool, default=True,
                      help='是否使用动态输入尺寸')
    parser.add_argument('--simplify', type=bool, default=True,
                      help='是否简化ONNX模型')
    
    # 验证相关参数
    parser.add_argument('--extrinsics', type=str, default='/ultralytics/data/Myself-v2/extrinsics/test/000020.txt',
                      help='RGB图像路径')
    parser.add_argument('--rgb-path', type=str, default='/ultralytics/data/Myself-v2/images/visible/test/000020.jpg',
                      help='RGB图像路径')
    parser.add_argument('--ir-path', type=str, default='/ultralytics/data/Myself-v2/images/infrared/test/000020.jpg',
                      help='红外图像路径')
    parser.add_argument('--pos-error-threshold', type=float, default=10.0,
                      help='位置误差阈值（像素）')
    parser.add_argument('--conf-error-threshold', type=float, default=0.01,
                      help='置信度误差阈值')
    parser.add_argument('--device', type=str, default='cuda',
                      help='运行设备 cuda/cpu')
    
    # 检测相关参数
    parser.add_argument('--conf-thres', type=float, default=0.5,
                      help='检测置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                      help='NMS IOU阈值')
    
    # 其他参数
    parser.add_argument('--visualize', type=bool, default=True, # 默认显示
                          help='是否在预处理前后可视化配准效果')
    
    return parser.parse_args()

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

def export_model(model, args):
    """导出模型为ONNX格式"""
    print(f"正在导出模型到 {args.export_path}.onnx ...")
    success = model.export(
        format="onnx",
        dynamic=args.dynamic,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        name=args.export_path
    )
    return success

def run_inference(model, inputs, device):
    """运行PyTorch模型推理"""
    rgb_tensor, ir_tensor = inputs
    rgb_tensor = rgb_tensor.to(device)
    ir_tensor = ir_tensor.to(device)
    
    homography = torch.eye(3).unsqueeze(0).to(device)
    original_sizes = torch.tensor([[[640, 640], [640, 640]]]).to(device)
    
    with torch.no_grad():
        return model(rgb_tensor, ir_tensor, homography)

def validate_onnx(onnx_path, inputs, updated_homography_np, torch_output, pos_error_threshold, conf_error_threshold, args):
    """验证ONNX模型输出"""
    rgb_input, ir_input = inputs
    providers = ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        return False
        
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    
    if len(input_names) != 3:
        print(f"警告：ONNX模型期望 {len(input_names)} 个输入, 但代码准备了3个。请检查模型导出。")

    onnx_inputs = {
        input_names[0]: rgb_input,
        input_names[1]: ir_input,
        input_names[2]: updated_homography_np
    }
    
    print("rgb_input shape:",onnx_inputs[input_names[0]].shape)
    print("ir_input shape:",onnx_inputs[input_names[1]].shape)
    print("updated_homography_np shape:",onnx_inputs[input_names[2]].shape)
    
    onnx_outputs = session.run(output_names, onnx_inputs)
    
    # 计算误差
    torch_output = torch_output.cpu().numpy()
    onnx_output = onnx_outputs[0]
    
    # 检查输出维度
    print(f"PyTorch输出形状: {torch_output.shape}")
    print(f"ONNX输出形状: {onnx_output.shape}")
    
    # 计算原始输出的误差
    print("\n原始输出误差:")
    raw_diff = np.abs(torch_output - onnx_output)
    max_raw_diff = np.max(raw_diff)
    mean_raw_diff = np.mean(raw_diff)
    relative_raw_diff = np.mean(np.abs((torch_output - onnx_output) / (np.abs(torch_output) + 1e-10)))
    print(f"  最大绝对误差: {max_raw_diff:.6f}")
    print(f"  平均绝对误差: {mean_raw_diff:.6f}")
    print(f"  平均相对误差: {relative_raw_diff:.6f}")
    
    # 分别计算位置信息和置信度的误差
    # 位置信息误差 (x,y,w,h)
    pos_diff = np.abs(torch_output[:, :4, :] - onnx_output[:, :4, :])
    max_pos_diff = np.max(pos_diff)
    mean_pos_diff = np.mean(pos_diff)
    # 避免除以0
    torch_pos = torch_output[:, :4, :]
    onnx_pos = onnx_output[:, :4, :]
    relative_pos_diff = np.mean(np.abs((torch_pos - onnx_pos) / (np.abs(torch_pos) + 1e-10)))
    
    # 置信度误差
    # 对于single-cls模型，置信度可能在最后一个维度
    if torch_output.shape[1] == 5:  # 单类别模型
        torch_conf = torch_output[:, 4, :]
        onnx_conf = onnx_output[:, 4, :]
    else:  # 多类别模型
        torch_conf = torch_output[:, 4:, :].max(axis=1)  # 取所有类别中的最大置信度
        onnx_conf = onnx_output[:, 4:, :].max(axis=1)
    
    conf_diff = np.abs(torch_conf - onnx_conf)
    max_conf_diff = np.max(conf_diff)
    mean_conf_diff = np.mean(conf_diff)
    relative_conf_diff = np.mean(np.abs((torch_conf - onnx_conf) / (np.abs(torch_conf) + 1e-10)))
    
    # 打印数值范围统计
    print("\n数值范围统计:")
    print("位置信息 (x,y,w,h):")
    print(f"  PyTorch - 最小值: {torch_pos.min():.6f}, 最大值: {torch_pos.max():.6f}, 平均值: {torch_pos.mean():.6f}")
    print(f"  ONNX    - 最小值: {onnx_pos.min():.6f}, 最大值: {onnx_pos.max():.6f}, 平均值: {onnx_pos.mean():.6f}")
    print(f"  标准差  - PyTorch: {torch_pos.std():.6f}, ONNX: {onnx_pos.std():.6f}")
    
    print("\n置信度:")
    print(f"  PyTorch - 最小值: {torch_conf.min():.6f}, 最大值: {torch_conf.max():.6f}, 平均值: {torch_conf.mean():.6f}")
    print(f"  ONNX    - 最小值: {onnx_conf.min():.6f}, 最大值: {onnx_conf.max():.6f}, 平均值: {onnx_conf.mean():.6f}")
    print(f"  标准差  - PyTorch: {torch_conf.std():.6f}, ONNX: {onnx_conf.std():.6f}")
    
    print(f"\n转换后误差:")
    print(f"位置信息误差:")
    print(f"  最大绝对误差: {max_pos_diff:.6f}")
    print(f"  平均绝对误差: {mean_pos_diff:.6f}")
    print(f"  平均相对误差: {relative_pos_diff:.6f}")
    print(f"置信度误差:")
    print(f"  最大绝对误差: {max_conf_diff:.6f}")
    print(f"  平均绝对误差: {mean_conf_diff:.6f}")
    print(f"  平均相对误差: {relative_conf_diff:.6f}")
    
    # 分别检查位置和置信度误差是否超过阈值
    pos_check = max_pos_diff < pos_error_threshold
    conf_check = max_conf_diff < conf_error_threshold
    
    if pos_check and conf_check:
        print("\n✅ 验证通过：")
        print(f"  位置误差 ({max_pos_diff:.6f}) < 阈值 ({pos_error_threshold})")
        print(f"  置信度误差 ({max_conf_diff:.6f}) < 阈值 ({conf_error_threshold})")
        return True
    else:
        print("\n❌ 验证失败：")
        if not pos_check:
            print(f"  位置误差 ({max_pos_diff:.6f}) >= 阈值 ({pos_error_threshold})")
        if not conf_check:
            print(f"  置信度误差 ({max_conf_diff:.6f}) >= 阈值 ({conf_error_threshold})")
        return False

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
                # !! 处理 NumPy CHW -> HWC !!
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
        # !! 使用处理后的 ir_img_bgr !!
        warped_ir = cv2.warpPerspective(ir_img_bgr, H, target_size)

        alpha = 0.5
        # !! 使用处理后的 rgb_img_bgr !!
        fused_img = cv2.addWeighted(rgb_img_bgr, alpha, warped_ir, 1 - alpha, 0)

        # --- 创建对比图 ---
        # !! 使用处理后的 rgb_img_bgr !!
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

def nms(boxes, scores, iou_threshold):
    """非极大值抑制(NMS)实现
    
    Args:
        boxes (np.ndarray): 边界框坐标 [N, 4]，格式为[x1, y1, x2, y2]
        scores (np.ndarray): 置信度分数 [N]
        iou_threshold (float): IOU阈值
        
    Returns:
        np.ndarray: 保留的检测框索引
    """
    # 按置信度分数降序排序
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        # 保留当前最高分的框
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # 计算当前框与其他框的IOU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area1 + area2 - intersection
        
        iou = intersection / union
        
        # 保留IOU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return np.array(keep)

def visualize_detections(rgb_img, ir_img, torch_output, onnx_output, save_path, conf_thres=0.5, iou_thres=0.45):
    """
    可视化PyTorch和ONNX模型的检测结果
    
    Args:
        rgb_img (np.ndarray): RGB图像 (HWC)
        ir_img (np.ndarray): 红外图像 (HWC)
        torch_output (np.ndarray): PyTorch模型输出
        onnx_output (np.ndarray): ONNX模型输出
        save_path (str): 保存路径
        conf_thres (float): 置信度阈值
        iou_thres (float): NMS IOU阈值
    """
    try:
        # 处理PyTorch输出
        torch_pred = torch_output[0].T  # 转置为 [N, C]
        torch_conf = torch_pred[:, 4:].max(axis=1)  # 获取最大置信度
        torch_boxes = torch_pred[:, :4]  # 获取边界框
        torch_mask = torch_conf > conf_thres
        torch_boxes = torch_boxes[torch_mask]
        torch_conf = torch_conf[torch_mask]
        
        # 处理ONNX输出
        onnx_pred = onnx_output[0].T  # 转置为 [N, C]
        onnx_conf = onnx_pred[:, 4:].max(axis=1)  # 获取最大置信度
        onnx_boxes = onnx_pred[:, :4]  # 获取边界框
        onnx_mask = onnx_conf > conf_thres
        onnx_boxes = onnx_boxes[onnx_mask]
        onnx_conf = onnx_conf[onnx_mask]
        
        # 应用NMS
        if len(torch_boxes) > 0:
            torch_keep = nms(torch_boxes, torch_conf, iou_thres)
            torch_boxes = torch_boxes[torch_keep]
            torch_conf = torch_conf[torch_keep]
            
        if len(onnx_boxes) > 0:
            onnx_keep = nms(onnx_boxes, onnx_conf, iou_thres)
            onnx_boxes = onnx_boxes[onnx_keep]
            onnx_conf = onnx_conf[onnx_keep]
        
        # 创建可视化图像
        h, w = rgb_img.shape[:2]
        vis_img = np.zeros((h, w*3, 3), dtype=np.uint8)
        
        # 复制原始图像
        vis_img[:, :w] = rgb_img
        vis_img[:, w:w*2] = rgb_img.copy()
        vis_img[:, w*2:] = rgb_img.copy()
        
        # 绘制PyTorch检测结果
        for box, conf in zip(torch_boxes, torch_conf):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img[:, :w], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img[:, :w], f'{conf:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制ONNX检测结果
        for box, conf in zip(onnx_boxes, onnx_conf):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img[:, w:w*2], (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_img[:, w:w*2], f'{conf:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        title_color = (255, 255, 255)
        cv2.putText(vis_img, 'Original', (10, 30), font, font_scale, title_color, 2)
        cv2.putText(vis_img, 'PyTorch', (w + 10, 30), font, font_scale, title_color, 2)
        cv2.putText(vis_img, 'ONNX', (w*2 + 10, 30), font, font_scale, title_color, 2)
        
        # 添加分隔线
        cv2.line(vis_img, (w, 0), (w, h), (255, 0, 0), 2)
        cv2.line(vis_img, (w*2, 0), (w*2, h), (255, 0, 0), 2)
        
        # 保存结果
        cv2.imwrite(save_path, vis_img)
        print(f"检测结果可视化已保存到: {save_path}")
        
    except Exception as e:
        print(f"可视化检测结果时发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = parse_args()
    output_dir = Path(args.export_path).parent
    output_dir.mkdir(parents=True, exist_ok=True) # 确保目录存在
    
    # 确保输出目录存在
    Path(args.export_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型 {args.weights} ...")
    model = YOLOMultimodal(args.weights)
    model.eval()
    model.to(args.device)  # 将模型移动到指定设备
    
    # 加载外参矩阵
    print("正在加载外参矩阵...")
    extrinsics = load_extrinsics(args.extrinsics, args.device)
    if extrinsics is None:
        return

    # --- 可视化：预处理前 ---
    if args.visualize:
        print("正在可视化原始图像配准效果...")
        try:
            raw_rgb = cv2.imread(args.rgb_path)
            raw_ir = cv2.imread(args.ir_path)
            if raw_rgb is not None and raw_ir is not None:
                    # 注意：这里使用原始 extrinsics
                visualize_registration(
                    raw_rgb,
                    raw_ir,
                    extrinsics, # 原始H
                    save_path=str(output_dir / f'{Path(args.export_path).stem}_vis_raw.jpg'),
                    save_prefix='Raw',
                    is_normalized=False # 原始图像未归一化
                )
            else:
                print("警告：无法读取原始图像进行可视化。")
        except Exception as e:
            print(f"原始图像可视化失败: {e}")

    # 准备输入数据 (预处理)
    print("正在处理输入图像和更新单应性矩阵...")
    try:
        # !! 接收 original_sizes !!
        rgb_input_np, ir_input_np, updated_homography, original_sizes = preprocess_image_and_homography(
            args.rgb_path,
            args.ir_path,
            extrinsics, # 传入原始H
            args.imgsz,
            model.stride,
            args.device
        )
        # rgb_input_np 和 ir_input_np 是 (1, C, H, W) 格式, float32, 0-1 范围
        # updated_homography 是 (1, 3, 3) 格式, torch.Tensor
        # original_sizes 是 (1, 2, 2) 格式, torch.Tensor
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return
    
     # --- 可视化：预处理后 ---
    if args.visualize:
        print("正在可视化预处理后图像配准效果...")
        try:
            visualize_registration(
                rgb_input_np, # 传入 (1, C, H, W) numpy 数组
                ir_input_np,  # 传入 (1, C, H, W) numpy 数组
                updated_homography, # 传入 (1, 3, 3) Tensor
                save_path=str(output_dir / f'{Path(args.export_path).stem}_vis_processed.jpg'),
                save_prefix='Processed',
                is_normalized=True # 预处理后图像已归一化
            )
        except Exception as e:
            print(f"处理后图像可视化失败: {e}")

    # PyTorch模型推理
    print("运行PyTorch模型推理...")
    rgb_tensor = torch.from_numpy(rgb_input_np).to(args.device)
    ir_tensor = torch.from_numpy(ir_input_np).to(args.device)

    with torch.no_grad():
        outputs = model.forward_multimodal(rgb_tensor, ir_tensor, updated_homography)
        # 如果是元组，取第一个元素（通常是检测结果）
        torch_output = outputs[0] if isinstance(outputs, tuple) else outputs
    
    # 导出模型
    if export_model(model, args):
        print("模型导出成功")
    else:
        print("模型导出失败")
        return
    
    # 验证ONNX模型
    print("正在验证ONNX模型...")
    # 根据weights路径设置onnx路径
    pt_name = Path(args.weights).stem  # 获取PT文件名（不含扩展名）
    onnx_path = str(Path(args.weights).parent / f'{pt_name}.onnx')  # 使用相同的文件名，但扩展名改为.onnx
    print(f"ONNX模型路径: {onnx_path}")
    
    # 读取原始RGB图像用于可视化
    rgb_img = cv2.imread(args.rgb_path)
    if rgb_img is None:
        print(f"无法读取RGB图像: {args.rgb_path}")
        return
    
    # 验证ONNX模型并获取输出
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_names = [input.name for input in session.get_inputs()]
    onnx_inputs = {
        input_names[0]: rgb_input_np,
        input_names[1]: ir_input_np,
        input_names[2]: updated_homography.cpu().numpy()
    }
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_output = onnx_outputs[0]
    
    # 可视化检测结果
    det_save_path = str(output_dir / f'{Path(args.export_path).stem}_detections.jpg')
    visualize_detections(
        rgb_img,
        None,  # 这里不需要IR图像
        torch_output.cpu().numpy(),
        onnx_output,
        det_save_path,
        args.conf_thres,
        args.iou_thres
    )
    
    if validate_onnx(onnx_path, (rgb_input_np, ir_input_np), updated_homography.cpu().numpy(), torch_output, args.pos_error_threshold, args.conf_error_threshold, args):
        print("\n✅ 验证通过：PyTorch和ONNX模型输出一致")
    else:
        print("\n❌ 验证失败：PyTorch和ONNX模型输出存在显著差异")

    # 检查ONNX模型是否支持动态形状
    model = onnx.load(onnx_path)
    print("模型是否支持动态形状:", model.graph.input[0].type.tensor_type.shape.dim[0].dim_param != "")

if __name__ == '__main__':
    main()