import os
import argparse
import shutil


def select_max_interval_images(src_visible, src_infrared, dst_visible, dst_infrared, num_select=200):
    # 获取所有图片名，并排序
    visible_imgs = sorted([f for f in os.listdir(src_visible) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total = len(visible_imgs)
    if total < num_select:
        selected_imgs = visible_imgs
    else:
        # 计算等间隔索引
        step = total / num_select
        indices = [int(i * step) for i in range(num_select)]
        selected_imgs = [visible_imgs[i] for i in indices]

    # 创建目标文件夹
    os.makedirs(dst_visible, exist_ok=True)
    os.makedirs(dst_infrared, exist_ok=True)

    for img_name in selected_imgs:
        src_vis_path = os.path.join(src_visible, img_name)
        src_ir_path = os.path.join(src_infrared, img_name)
        dst_vis_path = os.path.join(dst_visible, img_name)
        dst_ir_path = os.path.join(dst_infrared, img_name)
        if not os.path.exists(src_ir_path):
            print(f"警告: 红外图像 {src_ir_path} 不存在，跳过。")
            continue
        shutil.copy2(src_vis_path, dst_vis_path)
        shutil.copy2(src_ir_path, dst_ir_path)
    print(f"已成功挑选并复制{len(selected_imgs)}张图片到 {dst_visible} 和 {dst_infrared}")


def main():
    parser = argparse.ArgumentParser(description="等间隔挑选图片并分别保存到visible和infrared文件夹")
    parser.add_argument('--src_visible', type=str, default='/share/Data/0528/tmp-6/xyz', help='可见光图像文件夹路径')
    parser.add_argument('--src_infrared', type=str, default='/share/Data/0528/tmp-6/M3JUVC', help='红外图像文件夹路径')
    parser.add_argument('--dst_visible', type=str, default='/share/Data/0528/tmp-6/visible', help='保存可见光图像的新文件夹')
    parser.add_argument('--dst_infrared', type=str, default='/share/Data/0528/tmp-6/infrared', help='保存红外图像的新文件夹')
    parser.add_argument('--num', type=int, default=200, help='要挑选的图片数量，默认200')
    args = parser.parse_args()

    select_max_interval_images(
        args.src_visible,
        args.src_infrared,
        args.dst_visible,
        args.dst_infrared,
        args.num
    )

if __name__ == '__main__':
    main()
