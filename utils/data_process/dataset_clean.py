import os
import argparse

def sync_subfolders(visible_root, infrared_root, extrinsics_root, labels_visible_root):
    for subdir, _, files in os.walk(visible_root):
        # 只处理有图片的子文件夹
        rel_path = os.path.relpath(subdir, visible_root)
        if rel_path == '.':
            rel_path = ''
        # 获取当前visible子文件夹下所有图片名（不含扩展名）
        visible_imgs = set(os.path.splitext(f)[0] for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        if not visible_imgs:
            continue
        # 构造infrared、extrinsics、labels/visible对应子文件夹路径
        ir_subdir = os.path.join(infrared_root, rel_path)
        ex_subdir = os.path.join(extrinsics_root, rel_path)
        label_vis_subdir = os.path.join(labels_visible_root, rel_path)
        # 同步infrared
        if os.path.exists(ir_subdir):
            ir_files = [f for f in os.listdir(ir_subdir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in ir_files:
                name, _ = os.path.splitext(f)
                if name not in visible_imgs:
                    os.remove(os.path.join(ir_subdir, f))
        # 同步extrinsics
        if os.path.exists(ex_subdir):
            ex_files = [f for f in os.listdir(ex_subdir) if f.lower().endswith('.txt')]
            for f in ex_files:
                name, _ = os.path.splitext(f)
                if name not in visible_imgs:
                    os.remove(os.path.join(ex_subdir, f))
        # 同步labels/visible
        if os.path.exists(label_vis_subdir):
            label_files = [f for f in os.listdir(label_vis_subdir) if f.lower().endswith('.txt')]
            for f in label_files:
                name, _ = os.path.splitext(f)
                if name not in visible_imgs:
                    os.remove(os.path.join(label_vis_subdir, f))
    print('同步完成，infrared、extrinsics、labels/visible已与images/visible保持一致。')

def main():
    parser = argparse.ArgumentParser(description="同步删除infrared、extrinsics、labels/visible中无对应images/visible图片的文件（支持多级子文件夹）")
    parser.add_argument('--images_visible', type=str, default='data/Myself-v3-clean-0529/images/visible', help='可见光图片根目录')
    parser.add_argument('--images_infrared', type=str, default='data/Myself-v3-clean-0529/images/infrared', help='红外图片根目录')
    parser.add_argument('--extrinsics', type=str, default='data/Myself-v3-clean-0529/extrinsics', help='外参txt根目录')
    parser.add_argument('--labels_visible', type=str, default='data/Myself-v3-clean-0529/labels/visible', help='labels/visible根目录')
    args = parser.parse_args()
    sync_subfolders(args.images_visible, args.images_infrared, args.extrinsics, args.labels_visible)

if __name__ == '__main__':
    main()