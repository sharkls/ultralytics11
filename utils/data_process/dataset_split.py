import os
import shutil
import argparse
from glob import glob

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_sorted_files(folder):
    files = sorted(glob(os.path.join(folder, '*')))
    return files

def process_scene(scene_idx, split, output_dirs, counter, args):
    vis_dir = os.path.join(args.images_root, f'visible/visible{scene_idx}')
    ir_dir = os.path.join(args.images_root, f'infrared/infrared{scene_idx}')
    label_dir = os.path.join(args.labels_root, f'visible{scene_idx}')

    vis_files = get_sorted_files(vis_dir)
    ir_files = get_sorted_files(ir_dir)
    label_files = get_sorted_files(label_dir)

    # 取三类文件的交集（按文件名不含扩展名）
    vis_names = {os.path.splitext(os.path.basename(f))[0]: f for f in vis_files}
    ir_names = {os.path.splitext(os.path.basename(f))[0]: f for f in ir_files}
    label_names = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}
    common_keys = sorted(set(vis_names) & set(ir_names) & set(label_names))

    num_files = len(common_keys)
    indices = list(range(num_files))

    if split == 'train':
        selected = indices
    elif split == 'val':
        selected = indices[:num_files//2]
    elif split == 'test':
        selected = indices[num_files//2:]
    else:
        raise ValueError('split must be train/val/test')

    for idx in selected:
        key = common_keys[idx]
        vis_file = vis_names[key]
        ir_file = ir_names[key]
        label_file = label_names[key]
        new_name = f'0528_{counter:06d}'
        vis_ext = os.path.splitext(vis_file)[1]
        ir_ext = os.path.splitext(ir_file)[1]
        label_ext = os.path.splitext(label_file)[1]

        shutil.copy(vis_file, os.path.join(output_dirs['visible'], new_name + vis_ext))
        shutil.copy(ir_file, os.path.join(output_dirs['infrared'], new_name + ir_ext))
        shutil.copy(label_file, os.path.join(output_dirs['labels'], new_name + label_ext))
        counter += 1
    return counter

def main():
    parser = argparse.ArgumentParser(description="自动划分并重命名多场景多模态数据集")
    parser.add_argument('--images_root', type=str, default='/share/Data/0528all/images', help='原始图像根目录')
    parser.add_argument('--labels_root', type=str, default='/share/Data/0528all/labels', help='原始标签根目录')
    parser.add_argument('--output_root', type=str, default='/share/Data/0528all_split_new-v3', help='输出数据集根目录')
    parser.add_argument('--scene_num', type=int, default=7, help='场景总数')
    parser.add_argument('--train_scene', type=int, default=5, help='用于训练的场景数')
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    for img_type in ['infrared', 'visible']:
        for split in splits:
            ensure_dir(os.path.join(args.output_root, f'images/{img_type}/{split}'))
    for split in splits:
        ensure_dir(os.path.join(args.output_root, f'labels/{split}'))

    counter = 1
    # 跳过第一个场景，用第2~5个场景作为训练集
    for scene_idx in range(2, 2 + 4):  # 2,3,4,5
        output_dirs = {
            'visible': os.path.join(args.output_root, 'images/visible/train'),
            'infrared': os.path.join(args.output_root, 'images/infrared/train'),
            'labels': os.path.join(args.output_root, 'labels/train'),
        }
        counter = process_scene(scene_idx, 'train', output_dirs, counter, args)

    # 验证集和测试集部分不变
    for scene_idx in range(args.train_scene + 1, args.scene_num + 1):
        # 验证集
        output_dirs = {
            'visible': os.path.join(args.output_root, 'images/visible/val'),
            'infrared': os.path.join(args.output_root, 'images/infrared/val'),
            'labels': os.path.join(args.output_root, 'labels/val'),
        }
        counter = process_scene(scene_idx, 'val', output_dirs, counter, args)
        # 测试集
        output_dirs = {
            'visible': os.path.join(args.output_root, 'images/visible/test'),
            'infrared': os.path.join(args.output_root, 'images/infrared/test'),
            'labels': os.path.join(args.output_root, 'labels/test'),
        }
        counter = process_scene(scene_idx, 'test', output_dirs, counter, args)

    print('数据集划分与重命名完成！')

if __name__ == '__main__':
    main()
