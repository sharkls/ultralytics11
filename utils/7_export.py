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
    parser.add_argument('--weights', type=str, default='runs/multimodal/train6/weights/last.pt',
                      help='训练好的模型权重路径')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640],
                      help='输入图像尺寸 [height, width]')
    
    # 导出相关参数
    parser.add_argument('--export-path', type=str, default='./runs/export/EFDE-YOLO',
                      help='ONNX模型导出路径')
    parser.add_argument('--opset', type=int, default=16,
                      help='ONNX opset版本')
    parser.add_argument('--dynamic', type=bool, default=True,
                      help='是否使用动态输入尺寸')
    parser.add_argument('--simplify', type=bool, default=True,
                      help='是否简化ONNX模型')
    
    # 验证相关参数
    parser.add_argument('--extrinsics', type=str, default='./data/LLVIP/extrinsics/test/190001.txt',
                      help='RGB图像路径')
    parser.add_argument('--rgb-path', type=str, default='./data/LLVIP/images/visible/test/190001.jpg',
                      help='RGB图像路径')
    parser.add_argument('--ir-path', type=str, default='./data/LLVIP/images/infrared/test/190001.jpg',
                      help='红外图像路径')
    parser.add_argument('--error-threshold', type=float, default=1e-4,
                      help='验证误差阈值')
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

def validate_onnx(onnx_path, inputs, updated_homography_np, torch_output, error_threshold, args):
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
    conf_diff = np.abs(torch_output[:, 4, :] - onnx_output[:, 4, :])
    max_conf_diff = np.max(conf_diff)
    mean_conf_diff = np.mean(conf_diff)
    # 避免除以0
    torch_conf = torch_output[:, 4, :]
    onnx_conf = onnx_output[:, 4, :]
    relative_conf_diff = np.mean(np.abs((torch_conf - onnx_conf) / (np.abs(torch_conf) + 1e-10)))
    
    print(f"\n验证结果:")
    print(f"位置信息误差:")
    print(f"  最大绝对误差: {max_pos_diff:.6f}")
    print(f"  平均绝对误差: {mean_pos_diff:.6f}")
    print(f"  平均相对误差: {relative_pos_diff:.6f}")
    print(f"置信度误差:")
    print(f"  最大绝对误差: {max_conf_diff:.6f}")
    print(f"  平均绝对误差: {mean_conf_diff:.6f}")
    print(f"  平均相对误差: {relative_conf_diff:.6f}")
    
    # 处理检测结果
    def process_output(output, conf_thres=0.5, iou_thres=0.45):
        """处理YOLO模型输出，应用NMS
        
        Args:
            output (np.ndarray): YOLO模型输出 (1,5,8400)，其中5表示[x,y,w,h,conf]
            conf_thres (float): 置信度阈值
            iou_thres (float): NMS IOU阈值
            
        Returns:
            np.ndarray: 处理后的检测结果 [M, 6]，格式为[x1, y1, x2, y2, conf, class]
        """
        # 移除batch维度并转置为[8400, 5]格式
        output = output.squeeze(0).T
        
        # 将xywh转换为xyxy格式
        boxes = np.zeros_like(output[:, :4])
        boxes[:, 0] = output[:, 0] - output[:, 2] / 2  # x1
        boxes[:, 1] = output[:, 1] - output[:, 3] / 2  # y1
        boxes[:, 2] = output[:, 0] + output[:, 2] / 2  # x2
        boxes[:, 3] = output[:, 1] + output[:, 3] / 2  # y2
        
        # 获取置信度
        scores = output[:, 4]
        
        # 应用置信度阈值
        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        
        # 如果没有检测到目标，返回空数组
        if len(boxes) == 0:
            return np.zeros((0, 6))
        
        # 应用NMS
        indices = nms(boxes, scores, iou_thres)
        
        # 组合结果 [x1, y1, x2, y2, conf, class]
        results = np.zeros((len(indices), 6))
        results[:, :4] = boxes[indices]
        results[:, 4] = scores[indices]
        results[:, 5] = 0  # 类别ID（单类别模型，类别为0）
        
        return results
    
    # 处理PyTorch和ONNX输出
    torch_results = process_output(torch_output, args.conf_thres, args.iou_thres)
    onnx_results = process_output(onnx_output, args.conf_thres, args.iou_thres)
    
    # 可视化结果
    def visualize_detections(image, detections, save_path, imgsz=(640, 640)):
        """可视化检测结果"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # 计算缩放比例
        r = min(imgsz[0] / h, imgsz[1] / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = imgsz[1] - new_unpad[0], imgsz[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # 将检测框坐标转换回原始图像尺寸
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # 反向应用letterbox变换
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r
            
            # 确保坐标在图像范围内
            x1 = max(0, min(int(x1), w))
            y1 = max(0, min(int(y1), h))
            x2 = max(0, min(int(x2), w))
            y2 = max(0, min(int(y2), h))
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制置信度
            cv2.putText(img, f'{conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(save_path, img)

    # 读取原始图像
    rgb_img = cv2.imread(args.rgb_path)
    if rgb_img is not None:
        # 可视化PyTorch结果
        visualize_detections(rgb_img, torch_results, 
                           str(Path(args.export_path).parent / 'torch_detections.jpg'), args.imgsz)
        # 可视化ONNX结果
        visualize_detections(rgb_img, onnx_results, 
                           str(Path(args.export_path).parent / 'onnx_detections.jpg'), args.imgsz)
    
    # 返回验证结果
    return max(max_pos_diff, max_conf_diff) < error_threshold

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
    # onnx_path = f"{args.export_path}.onnx"
    onnx_path = "runs/multimodal/train6/weights/last.onnx"
    # !! 移除 original_sizes.cpu().numpy() !!
    if validate_onnx(onnx_path, (rgb_input_np, ir_input_np), updated_homography.cpu().numpy(), torch_output, args.error_threshold, args):
        print("\n✅ 验证通过：PyTorch和ONNX模型输出一致")
    else:
        print("\n❌ 验证失败：PyTorch和ONNX模型输出存在显著差异")

    # 检查ONNX模型是否支持动态形状
    model = onnx.load(onnx_path)
    print("模型是否支持动态形状:", model.graph.input[0].type.tensor_type.shape.dim[0].dim_param != "")

if __name__ == '__main__':
    main()