import cv2
import numpy as np
import argparse

# 加载单通道图片（灰度图）
def show_depthmap_with_pixel_value(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法加载图片: {image_path}")
        return

    window_name = 'Depthmap Viewer'

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            pixel_value = img[y, x]
            # 在窗口标题显示像素值
            cv2.displayOverlay(window_name, f"位置: ({x}, {y}) 像素值: {pixel_value}", 500)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按ESC退出
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV可视化单通道深度图，并支持鼠标查看像素值")
    parser.add_argument('--image_path', type=str, default="/share/545.jpg", help='要加载的单通道图片路径')
    parser.add_argument('--normalize', action='store_true', help='是否将图片归一化到0-255后显示')
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法加载图片: {args.image_path}")
        exit(1)
    if args.normalize:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')
    window_name = 'Depthmap Viewer'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            pixel_value = img[y, x]
            cv2.displayOverlay(window_name, f"位置: ({x}, {y}) 像素值: {pixel_value}", 500)
    cv2.setMouseCallback(window_name, mouse_callback)
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()
