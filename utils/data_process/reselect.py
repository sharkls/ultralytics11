import os
import argparse


def sync_infrared_with_visible(visible_dir, infrared_dir):
    visible_imgs = set([f for f in os.listdir(visible_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    infrared_imgs = [f for f in os.listdir(infrared_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    removed = 0
    for img in infrared_imgs:
        if img not in visible_imgs:
            os.remove(os.path.join(infrared_dir, img))
            removed += 1
    print(f"已删除{removed}张红外图片，infrared文件夹现有{len(visible_imgs)}张图片，与visible保持一致。")


def main():
    parser = argparse.ArgumentParser(description="根据visible文件夹图片名，删除infrared中无对应图片，保持一一对应")
    parser.add_argument('--visible', type=str, default='/share/Data/0528/tmp-6/visible', help='可见光图片文件夹路径')
    parser.add_argument('--infrared', type=str, default='/share/Data/0528/tmp-6/infrared', help='红外图片文件夹路径')
    args = parser.parse_args()
    sync_infrared_with_visible(args.visible, args.infrared)

if __name__ == '__main__':
    main()
