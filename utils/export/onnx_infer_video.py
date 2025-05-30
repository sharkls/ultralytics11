import torch
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
from ultralytics.data.augment import LetterBox
import os

# 复制自8_onnx_val.py

def process_output(output, conf_thres=0.25, iou_thres=0.45, nc=1, letterbox_info=None):
    output = output.squeeze(0).T  # [4+nc, N] -> [N, 4+nc]
    cls_scores = output[:, 4:4+nc]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = np.max(cls_scores, axis=1)
    boxes = np.zeros_like(output[:, :4])
    boxes[:, 0] = output[:, 0] - output[:, 2] / 2  # x1
    boxes[:, 1] = output[:, 1] - output[:, 3] / 2  # y1
    boxes[:, 2] = output[:, 0] + output[:, 2] / 2  # x2
    boxes[:, 3] = output[:, 1] + output[:, 3] / 2  # y2
    if letterbox_info is not None:
        boxes[:, 0] = boxes[:, 0] - letterbox_info['dw']
        boxes[:, 1] = boxes[:, 1] - letterbox_info['dh']
        boxes[:, 2] = boxes[:, 2] - letterbox_info['dw']
        boxes[:, 3] = boxes[:, 3] - letterbox_info['dh']
        boxes = boxes / letterbox_info['ratio']
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    results = []
    for cls_id in range(nc):
        cls_mask = cls_ids == cls_id
        if not np.any(cls_mask):
            continue
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        indices = nms(cls_boxes, cls_scores, iou_thres)
        cls_results = np.zeros((len(indices), 6))
        cls_results[:, :4] = cls_boxes[indices]
        cls_results[:, 4] = cls_scores[indices]
        cls_results[:, 5] = cls_id
        results.append(cls_results)
    if len(results) == 0:
        return np.zeros((0, 6))
    final_results = np.vstack(results)
    # 类间NMS
    global_boxes = final_results[:, :4]
    global_scores = final_results[:, 4]
    keep = nms(global_boxes, global_scores, iou_thres)
    final_results = final_results[keep]
    return final_results

def nms(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
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
        iou = intersection / (union + 1e-16)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX 多模态推理并保存视频')
    parser.add_argument('--onnx-path', type=str, default='runs/multimodal/train76/weights/best0529.onnx', help='ONNX模型路径')
    parser.add_argument('--data-dir', type=str, default='./data/Data/0528-test', help='数据集根目录')
    parser.add_argument('--split', type=str, default='train', help='数据集划分 (train/val/test)')
    parser.add_argument('--output-mp4', type=str, default='runs/infer_video.mp4', help='输出视频路径')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='输入图像尺寸 [h, w]')
    parser.add_argument('--nc', type=int, default=2, help='类别数')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IOU阈值')
    parser.add_argument('--stride', type=int, default=32, help='letterbox步长')
    return parser.parse_args()

def main():
    args = parse_args()
    imgsz = args.imgsz if len(args.imgsz) == 2 else [args.imgsz[0], args.imgsz[0]]
    # 拼接路径
    rgb_dir = Path(args.data_dir) / 'images' / 'visible' / args.split
    ir_dir = Path(args.data_dir) / 'images' / 'infrared' / args.split
    homo_dir = Path(args.data_dir) / 'extrinsics' / args.split
    # 遍历所有jpg，按img_name组装ir和homo路径
    rgb_files = sorted(list(rgb_dir.glob('*.jpg')))
    ir_files = []
    homo_files = []
    valid_rgb_files = []
    for rgb_path in rgb_files:
        img_name = rgb_path.stem
        ir_path = ir_dir / f'{img_name}.jpg'
        homo_path = homo_dir / f'{img_name}.txt'
        if ir_path.exists() and homo_path.exists():
            valid_rgb_files.append(rgb_path)
            ir_files.append(ir_path)
            homo_files.append(homo_path)
        else:
            print(f"跳过: {rgb_path}, {ir_path}, {homo_path}")
    # 加载ONNX模型
    session = ort.InferenceSession(args.onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    input_names = [input.name for input in session.get_inputs()]
    # 视频写入器初始化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    for idx, (rgb_path, ir_path, homo_path) in enumerate(zip(valid_rgb_files, ir_files, homo_files)):
        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        # 预处理
        letterbox = LetterBox(imgsz, auto=False, stride=args.stride)
        rgb_h, rgb_w = rgb_img.shape[:2]
        r_rgb = min(imgsz[0]/rgb_h, imgsz[1]/rgb_w)
        new_unpad_rgb = int(round(rgb_w * r_rgb)), int(round(rgb_h * r_rgb))
        dw_rgb, dh_rgb = imgsz[1] - new_unpad_rgb[0], imgsz[0] - new_unpad_rgb[1]
        dw_rgb /= 2
        dh_rgb /= 2
        S_rgb = np.eye(3)
        S_rgb[0, 0] = r_rgb
        S_rgb[1, 1] = r_rgb
        T_rgb = np.eye(3)
        T_rgb[0, 2] = dw_rgb
        T_rgb[1, 2] = dh_rgb
        # IR
        ir_h, ir_w = ir_img.shape[:2]
        r_ir = min(imgsz[0]/ir_h, imgsz[1]/ir_w)
        new_unpad_ir = int(round(ir_w * r_ir)), int(round(ir_h * r_ir))
        dw_ir, dh_ir = imgsz[1] - new_unpad_ir[0], imgsz[0] - new_unpad_ir[1]
        dw_ir /= 2
        dh_ir /= 2
        S_ir = np.eye(3)
        S_ir[0, 0] = r_ir
        S_ir[1, 1] = r_ir
        T_ir = np.eye(3)
        T_ir[0, 2] = dw_ir
        T_ir[1, 2] = dh_ir
        # homography
        H = np.loadtxt(str(homo_path))
        updated_H = T_rgb @ S_rgb @ H @ np.linalg.inv(S_ir) @ np.linalg.inv(T_ir)
        updated_H = np.expand_dims(updated_H, 0).astype(np.float32)
        # letterbox
        rgb_input = letterbox(image=rgb_img)
        ir_input = letterbox(image=ir_img)
        letterbox_info = {'dw': dw_rgb, 'dh': dh_rgb, 'ratio': r_rgb}
        rgb_input = cv2.cvtColor(rgb_input, cv2.COLOR_BGR2RGB) / 255.0
        ir_input = cv2.cvtColor(ir_input, cv2.COLOR_BGR2RGB) / 255.0
        rgb_input = rgb_input.transpose(2, 0, 1)[None].astype(np.float32)
        ir_input = ir_input.transpose(2, 0, 1)[None].astype(np.float32)
        # onnx推理
        onnx_inputs = {input_names[0]: rgb_input, input_names[1]: ir_input, input_names[2]: updated_H}
        output = session.run(None, onnx_inputs)[0]
        detections = process_output(output, args.conf_thres, args.iou_thres, args.nc, letterbox_info)
        # 可视化
        vis_img = cv2.imread(str(rgb_path))
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            color = (0, 0, 255) if int(cls_id) == 0 else (0, 255, 0)
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_img, f'cls{int(cls_id)} {conf:.2f}', (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # 初始化视频写入器
        if out is None:
            h, w = vis_img.shape[:2]
            out = cv2.VideoWriter(args.output_mp4, fourcc, 20, (w, h))
        out.write(vis_img)
        print(f"已处理: {rgb_path.name}")
    if out is not None:
        out.release()
        print(f"视频已保存到: {args.output_mp4}")
    else:
        print("没有有效帧，未生成视频。")

if __name__ == '__main__':
    main() 