import os
import argparse
from glob import glob
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="批量重命名图片，从19000开始递增")
    parser.add_argument('--input_dir', type=str, default="/ultralytics/data/Test_unmatch/Test_track/visible5", help='输入图片文件夹路径')
    parser.add_argument('--output_dir', type=str, default="/ultralytics/data/Test_unmatch/Test_track/visible5_acc", help='重命名后图片保存文件夹路径')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有图片路径并排序（保证顺序一致）
    img_paths = sorted(glob(os.path.join(args.input_dir, '*')))
    if not img_paths:
        print("未找到图片文件！")
        return

    start_idx = 19000
    for i, img_path in enumerate(img_paths):
        ext = os.path.splitext(img_path)[-1]
        new_name = f"{start_idx + i}{ext}"
        new_path = os.path.join(args.output_dir, new_name)
        shutil.copy(img_path, new_path)
        print(f"{img_path} -> {new_path}")

if __name__ == '__main__':
    main()