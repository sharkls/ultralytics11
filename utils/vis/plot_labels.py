import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# 颜色和类别名可自定义
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def draw_yolo_label(img, label_path, class_names=None):
    h, w = img.shape[:2]
    if not Path(label_path).exists():
        return img  # 没有标签直接返回原图
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, bw, bh = map(float, parts)
        cls = int(cls)
        # 反归一化
        cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if class_names:
            label = class_names[cls]
        else:
            label = str(cls)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def process_all_images(images_root, labels_root, output_root, subdirs, class_names=None):
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    output_root = Path(output_root)
    for subdir in subdirs:
        img_dir = images_root / subdir
        label_dir = labels_root / subdir
        out_dir = output_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        for img_path in tqdm(img_files, desc=f"处理{subdir}子集"):
            label_path = label_dir / (img_path.stem + '.txt')
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            img_drawn = draw_yolo_label(img, label_path, class_names)
            save_path = out_dir / img_path.name
            cv2.imwrite(str(save_path), img_drawn)


def main():
    parser = argparse.ArgumentParser(description='批量绘制YOLO标签到图片并保存')
    parser.add_argument('--images_root', type=str, default='data/LLVIP_RAW_TEST/images/visible', help='图片根目录')
    parser.add_argument('--labels_root', type=str, default='data/LLVIP_RAW_TEST/labels/visible', help='标签根目录')
    parser.add_argument('--output_root', type=str, default='runs/plot_labels', help='输出保存目录')
    parser.add_argument('--subdirs', type=str, nargs='+', default=['train', 'val'], help='处理的子目录')
    parser.add_argument('--class_names', type=str, default=None, help='类别名txt文件路径（可选）')
    args = parser.parse_args()

    class_names = None
    if args.class_names is not None:
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    process_all_images(args.images_root, args.labels_root, args.output_root, args.subdirs, class_names)
    print('全部处理完成！')

if __name__ == '__main__':
    main()
