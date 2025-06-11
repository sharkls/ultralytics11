import argparse
import cv2
import os

def split_binocular_image(image_path, output_dir, total_width, total_height):
    """
    将双目相机图像分割为左右眼图像
    
    Args:
        image_path (str): 输入图像路径
        output_dir (str): 输出目录路径
        total_width (int): 输入图像总宽度
        total_height (int): 输入图像总高度
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 检查图像尺寸是否正确
    if width != total_width or height != total_height:
        raise ValueError(f"输入图像尺寸不正确，应为{total_width}x{total_height}，实际为{width}x{height}")
    
    # 计算单眼图像宽度
    single_width = total_width // 2
    
    # 分割左右眼图像
    left_eye = img[:, :single_width]
    right_eye = img[:, single_width:]
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分割后的图像
    left_path = os.path.join(output_dir, "left.jpg")
    right_path = os.path.join(output_dir, "right.jpg")
    
    cv2.imwrite(left_path, left_eye)
    cv2.imwrite(right_path, right_eye)
    
    print(f"左眼图像已保存至: {left_path}")
    print(f"右眼图像已保存至: {right_path}")

def main():
    parser = argparse.ArgumentParser(description="双目相机图像分割工具")
    parser.add_argument("--input", type=str, default="/ultralytics/data/Data/BibocularPositioning/27.jpg",
                      help="输入图像路径")
    parser.add_argument("--output_dir", type=str, default="/ultralytics/data/Data/BibocularPositioning/",
                      help="输出目录路径")
    parser.add_argument("--width", type=int, default=2560,
                      help="输入图像总宽度")
    parser.add_argument("--height", type=int, default=960,
                      help="输入图像总高度")
    
    args = parser.parse_args()
    
    try:
        split_binocular_image(args.input, args.output_dir, args.width, args.height)
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
