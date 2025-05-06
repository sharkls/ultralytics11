import torch
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from ultralytics import YOLOMultimodal
from ultralytics.data.augment import LetterBox
from pathlib import Path
import onnx
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Multimodal Model Split Export and Validation')
    
    # 模型相关参数
    parser.add_argument('--weights', type=str, default='runs/multimodal/train6/weights/last.pt',
                      help='训练好的模型权重路径')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640],
                      help='输入图像尺寸 [height, width]')
    
    # 导出相关参数
    parser.add_argument('--export-dir', type=str, default='./runs/export/EFDE-YOLO-split',
                      help='ONNX模型导出目录')
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
    parser.add_argument('--pos-error-threshold', type=float, default=1.0,
                      help='位置误差阈值（像素）')
    parser.add_argument('--conf-error-threshold', type=float, default=1e-4,
                      help='置信度误差阈值')
    parser.add_argument('--device', type=str, default='cuda',
                      help='运行设备 cuda/cpu')
    
    # 检测相关参数
    parser.add_argument('--conf-thres', type=float, default=0.5,
                      help='检测置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                      help='NMS IOU阈值')
    
    # 其他参数
    parser.add_argument('--visualize', type=bool, default=True,
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
    S_rgb[0, 0] = r_rgb
    S_rgb[1, 1] = r_rgb

    S_ir = torch.eye(3, device=device, dtype=dtype)
    S_ir[0, 0] = r_ir
    S_ir[1, 1] = r_ir

    T_rgb = torch.eye(3, device=device, dtype=dtype)
    T_rgb[0, 2] = dw_rgb
    T_rgb[1, 2] = dh_rgb

    T_ir = torch.eye(3, device=device, dtype=dtype)
    T_ir[0, 2] = dw_ir
    T_ir[1, 2] = dh_ir

    # 更新单应性矩阵
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
    
    return rgb_img, ir_img, updated_H.unsqueeze(0)

def export_split_models(model, args):
    """将模型分为三个部分导出为ONNX格式"""
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    modules = list(model.model._modules.values())
    backbone1 = nn.Sequential(*modules[0:11])   # 0~10
    backbone2 = nn.Sequential(*modules[11:22])  # 11~21
    head      = nn.Sequential(*modules[22:38])  # 22~37

    # 导出backbone1
    print("正在导出backbone1 (RGB分支)...")
    dummy_rgb = torch.randn(1, 3, args.imgsz[0], args.imgsz[1]).to(args.device)
    feat1 = backbone1(dummy_rgb)
    torch.onnx.export(
        backbone1, dummy_rgb, str(export_dir / 'backbone1.onnx'),
        opset_version=args.opset,
        input_names=['rgb'], output_names=['feat1'],
        dynamic_axes={'rgb': {0: 'batch'}, 'feat1': {0: 'batch'}} if args.dynamic else None
    )

    # 导出backbone2
    print("正在导出backbone2 (IR分支)...")
    dummy_ir = torch.randn(1, 3, args.imgsz[0], args.imgsz[1]).to(args.device)
    feat2 = backbone2(dummy_ir)
    torch.onnx.export(
        backbone2, dummy_ir, str(export_dir / 'backbone2.onnx'),
        opset_version=args.opset,
        input_names=['ir'], output_names=['feat2'],
        dynamic_axes={'ir': {0: 'batch'}, 'feat2': {0: 'batch'}} if args.dynamic else None
    )

    # 导出head
    print("正在导出head (特征融合和检测头)...")
    dummy_homo = torch.eye(3).unsqueeze(0).to(args.device)
    torch.onnx.export(
        head, (feat1, feat2, dummy_homo), str(export_dir / 'head.onnx'),
        opset_version=args.opset,
        input_names=['feat1', 'feat2', 'homography'], output_names=['output'],
        dynamic_axes={'feat1': {0: 'batch'}, 'feat2': {0: 'batch'}, 'homography': {0: 'batch'}, 'output': {0: 'batch'}} if args.dynamic else None
    )
    return True

def validate_split_models(export_dir, inputs, updated_homography_np, torch_output, pos_error_threshold, conf_error_threshold, args):
    """验证分离后的ONNX模型输出"""
    rgb_input, ir_input = inputs
    providers = ['CPUExecutionProvider']
    
    try:
        # 加载三个ONNX模型
        backbone1_session = ort.InferenceSession(str(Path(export_dir) / 'backbone1.onnx'), providers=providers)
        backbone2_session = ort.InferenceSession(str(Path(export_dir) / 'backbone2.onnx'), providers=providers)
        head_session = ort.InferenceSession(str(Path(export_dir) / 'head.onnx'), providers=providers)
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        return False
    
    # 运行backbone1 (RGB分支)
    backbone1_input = {backbone1_session.get_inputs()[0].name: rgb_input}
    backbone1_output = backbone1_session.run(None, backbone1_input)[0]
    
    # 运行backbone2 (IR分支)
    backbone2_input = {backbone2_session.get_inputs()[0].name: ir_input}
    backbone2_output = backbone2_session.run(None, backbone2_input)[0]
    
    # 运行head (特征融合和检测头)
    head_input = {
        head_session.get_inputs()[0].name: backbone1_output,
        head_session.get_inputs()[1].name: backbone2_output,
        head_session.get_inputs()[2].name: updated_homography_np
    }
    onnx_output = head_session.run(None, head_input)[0]
    
    # 计算误差
    torch_output = torch_output.cpu().numpy()
    
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
    pos_diff = np.abs(torch_output[:, :4, :] - onnx_output[:, :4, :])
    max_pos_diff = np.max(pos_diff)
    mean_pos_diff = np.mean(pos_diff)
    torch_pos = torch_output[:, :4, :]
    onnx_pos = onnx_output[:, :4, :]
    relative_pos_diff = np.mean(np.abs((torch_pos - onnx_pos) / (np.abs(torch_pos) + 1e-10)))
    
    # 置信度误差
    if torch_output.shape[1] == 5:  # 单类别模型
        torch_conf = torch_output[:, 4, :]
        onnx_conf = onnx_output[:, 4, :]
    else:  # 多类别模型
        torch_conf = torch_output[:, 4:, :].max(axis=1)
        onnx_conf = onnx_output[:, 4:, :].max(axis=1)
    
    conf_diff = np.abs(torch_conf - onnx_conf)
    max_conf_diff = np.max(conf_diff)
    mean_conf_diff = np.mean(conf_diff)
    relative_conf_diff = np.mean(np.abs((torch_conf - onnx_conf) / (np.abs(torch_conf) + 1e-10)))
    
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
    """可视化配准结果"""
    try:
        # 数据格式转换
        def process_img_input(img):
            if isinstance(img, torch.Tensor):
                if img.dim() == 4: img = img.squeeze(0)
                img = img.detach().cpu().numpy()
                if img.shape[0] == 3: img = img.transpose(1, 2, 0)
            elif isinstance(img, np.ndarray):
                if img.ndim == 4 and img.shape[0] == 1: img = img.squeeze(0)
                if img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1, 2, 0)
            else:
                raise TypeError(f"不支持的图像类型: {type(img)}")

            if img.ndim != 3 or img.shape[2] not in [1, 3]:
                 raise ValueError(f"图像形状 {img.shape} 无法转换为HWC格式")

            return img

        rgb_img_np = process_img_input(rgb_img)
        ir_img_np = process_img_input(ir_img)

        if H is None:
            print("警告：未提供单应性矩阵，使用单位矩阵进行可视化。")
            H = np.eye(3, dtype=np.float32)
        elif isinstance(H, torch.Tensor):
             if H.dim() == 3: H = H.squeeze(0)
             H = H.detach().cpu().numpy()

        if not isinstance(H, np.ndarray) or H.shape != (3, 3):
            print(f"错误：无效的单应性矩阵形状: {H.shape}, 应为 (3, 3)")
            return
        H = H.astype(np.float32)

        # 像素值范围处理
        if is_normalized:
            rgb_img_np = (rgb_img_np * 255).clip(0, 255)
            ir_img_np = (ir_img_np * 255).clip(0, 255)

        rgb_img_np = rgb_img_np.astype(np.uint8)
        ir_img_np = ir_img_np.astype(np.uint8)

        # 确保是3通道BGR
        def ensure_bgr(img):
            if len(img.shape) == 2 or img.shape[2] == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        rgb_img_bgr = ensure_bgr(rgb_img_np)
        ir_img_bgr = ensure_bgr(ir_img_np)

        # 图像变换和融合
        target_size = (rgb_img_bgr.shape[1], rgb_img_bgr.shape[0])
        warped_ir = cv2.warpPerspective(ir_img_bgr, H, target_size)

        alpha = 0.5
        fused_img = cv2.addWeighted(rgb_img_bgr, alpha, warped_ir, 1 - alpha, 0)

        # 创建对比图
        comparison = np.hstack([rgb_img_bgr, warped_ir, fused_img])

        # 添加标题
        h, w = comparison.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w / 1920, h / 1080) * 0.8
        title_color = (0, 255, 0)
        cv2.putText(comparison, f'{save_prefix} RGB', (10, 30), font, font_scale, title_color, 1)
        cv2.putText(comparison, f'{save_prefix} Warped IR', (w // 3 + 10, 30), font, font_scale, title_color, 1)
        cv2.putText(comparison, f'{save_prefix} Fused', (2 * w // 3 + 10, 30), font, font_scale, title_color, 1)

        # 添加分隔线
        line_color = (255, 0, 0)
        cv2.line(comparison, (w // 3, 0), (w // 3, h), line_color, 1)
        cv2.line(comparison, (2 * w // 3, 0), (2 * w // 3, h), line_color, 1)

        # 保存图像
        cv2.imwrite(save_path, comparison)
        print(f"可视化结果已保存到: {save_path}")

    except Exception as e:
        print(f"可视化时发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = parse_args()
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型 {args.weights} ...")
    model = YOLOMultimodal(args.weights)
    model.eval()
    model.to(args.device)
    
    # 加载外参矩阵
    print("正在加载外参矩阵...")
    extrinsics = load_extrinsics(args.extrinsics, args.device)
    if extrinsics is None:
        return

    # 可视化预处理前
    if args.visualize:
        print("正在可视化原始图像配准效果...")
        try:
            raw_rgb = cv2.imread(args.rgb_path)
            raw_ir = cv2.imread(args.ir_path)
            if raw_rgb is not None and raw_ir is not None:
                visualize_registration(
                    raw_rgb,
                    raw_ir,
                    extrinsics,
                    save_path=str(export_dir / 'vis_raw.jpg'),
                    save_prefix='Raw',
                    is_normalized=False
                )
            else:
                print("警告：无法读取原始图像进行可视化。")
        except Exception as e:
            print(f"原始图像可视化失败: {e}")

    # 准备输入数据
    print("正在处理输入图像和更新单应性矩阵...")
    try:
        rgb_input_np, ir_input_np, updated_homography = preprocess_image_and_homography(
            args.rgb_path,
            args.ir_path,
            extrinsics,
            args.imgsz,
            model.stride,
            args.device
        )
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return
    
    # 可视化预处理后
    if args.visualize:
        print("正在可视化预处理后图像配准效果...")
        try:
            visualize_registration(
                rgb_input_np,
                ir_input_np,
                updated_homography,
                save_path=str(export_dir / 'vis_processed.jpg'),
                save_prefix='Processed',
                is_normalized=True
            )
        except Exception as e:
            print(f"处理后图像可视化失败: {e}")

    # PyTorch模型推理
    print("运行PyTorch模型推理...")
    rgb_tensor = torch.from_numpy(rgb_input_np).to(args.device)
    ir_tensor = torch.from_numpy(ir_input_np).to(args.device)

    with torch.no_grad():
        outputs = model.forward_multimodal(rgb_tensor, ir_tensor, updated_homography)
        torch_output = outputs[0] if isinstance(outputs, tuple) else outputs
    
    # 导出分离的模型
    if export_split_models(model, args):
        print("模型分离导出成功")
    else:
        print("模型分离导出失败")
        return
    
    # 验证分离后的ONNX模型
    print("正在验证分离后的ONNX模型...")
    if validate_split_models(export_dir, (rgb_input_np, ir_input_np), updated_homography.cpu().numpy(), torch_output, args.pos_error_threshold, args.conf_error_threshold, args):
        print("\n✅ 验证通过：PyTorch和分离后的ONNX模型输出一致")
    else:
        print("\n❌ 验证失败：PyTorch和分离后的ONNX模型输出存在显著差异")

    # 检查ONNX模型是否支持动态形状
    for model_name in ['backbone1', 'backbone2', 'head']:
        model = onnx.load(str(export_dir / f'{model_name}.onnx'))
        print(f"{model_name}模型是否支持动态形状:", model.graph.input[0].type.tensor_type.shape.dim[0].dim_param != "")

if __name__ == '__main__':
    main() 