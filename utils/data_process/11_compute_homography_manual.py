import cv2
import numpy as np
import sys
import select
import termios
import tty
import os

def select_points(event, x, y, flags, param):
    """鼠标事件回调函数"""
    if event == cv2.EVENT_LBUTTONDOWN:
        points = param[0]
        image = param[1]
        
        # 记录坐标点
        points.append((x, y))
        # 在图像上画点
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow(param[2], image)

def verify_homography(img_vis, img_therm, H, points_vis, points_therm):
    """验证单应性矩阵的正确性"""
    # 使用单应性矩阵变换可见光图像中的点
    points_vis_homo = np.hstack((points_vis, np.ones((points_vis.shape[0], 1))))
    transformed_points = np.dot(H, points_vis_homo.T).T
    transformed_points = transformed_points / transformed_points[:, 2:]
    transformed_points = transformed_points[:, :2]
    
    # 计算每对点的误差
    errors = np.sqrt(np.sum((transformed_points - points_therm) ** 2, axis=1))
    mean_error = np.mean(errors)
    
    print(f"平均投影误差: {mean_error:.2f}像素")
    for i, err in enumerate(errors):
        print(f"点对 {i+1} 误差: {err:.2f}像素")
    
    # 如果误差过大，重新计算单应性矩阵
    if mean_error > 20:
        print("警告：投影误差过大，重新计算单应性矩阵")
        H_new, _ = cv2.findHomography(points_therm, points_vis, cv2.RANSAC, 3.0)
        return H_new
    return H

def manual_compute_homography(image_vis_path, image_therm_path, save_path):
    """
    通过手动选择匹配点计算单应性矩阵
    
    Args:
        image_vis_path (str): 可见光图像路径
        image_therm_path (str): 红外图像路径
        save_path (str): 结果保存路径
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 读取图像
    img_vis = cv2.imread(image_vis_path)
    img_therm = cv2.imread(image_therm_path)
    
    # 存储选择的点
    points_vis = []
    points_therm = []
    
    def reset_images():
        """重置图像和点"""
        points_vis.clear()
        points_therm.clear()
        return img_vis.copy(), img_therm.copy()
    
    # 创建图像副本用于绘制
    img_vis_copy, img_therm_copy = reset_images()
    
    # 设置窗口
    cv2.namedWindow('Visible Image', cv2.WINDOW_KEEPRATIO)  # 使用WINDOW_KEEPRATIO保持图像比例
    cv2.namedWindow('Thermal Image', cv2.WINDOW_KEEPRATIO)
    
    # 设置窗口大小
    cv2.resizeWindow('Visible Image', 960, 1080)
    cv2.resizeWindow('Thermal Image', 960, 1080)
    
    cv2.setMouseCallback('Visible Image', select_points, [points_vis, img_vis_copy, 'Visible Image'])
    cv2.setMouseCallback('Thermal Image', select_points, [points_therm, img_therm_copy, 'Thermal Image'])
    
    print("请在两张图像上依次点击对应的特征点（至少4对）")
    print("按 'q' 或在终端输入 'q' 完成点选")
    print("按 'r' 或在终端输入 'r' 重新开始")
    
    def is_data():
        """检查终端是否有输入"""
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    
    # 保存终端设置
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # 设置终端为raw模式
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            # 显示图像
            cv2.imshow('Visible Image', img_vis_copy)
            cv2.imshow('Thermal Image', img_therm_copy)
            
            # 检查OpenCV窗口按键
            key_cv = cv2.waitKey(1) & 0xFF
            
            # 检查终端输入
            if is_data():
                key_terminal = sys.stdin.read(1)
                if key_terminal == 'q':
                    break
                elif key_terminal == 'r':
                    img_vis_copy, img_therm_copy = reset_images()
                    cv2.setMouseCallback('Visible Image', select_points, [points_vis, img_vis_copy, 'Visible Image'])
                    cv2.setMouseCallback('Thermal Image', select_points, [points_therm, img_therm_copy, 'Thermal Image'])
            
            # 检查OpenCV窗口按键
            if key_cv == ord('q'):
                break
            elif key_cv == ord('r'):
                img_vis_copy, img_therm_copy = reset_images()
                cv2.setMouseCallback('Visible Image', select_points, [points_vis, img_vis_copy, 'Visible Image'])
                cv2.setMouseCallback('Thermal Image', select_points, [points_therm, img_therm_copy, 'Thermal Image'])
    
    finally:
        # 恢复终端设置
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    cv2.destroyAllWindows()
    
    # 确保点数相等且至少有4对点
    if len(points_vis) != len(points_therm) or len(points_vis) < 4:
        print("错误：两图像中的点数不相等或点数少于4对")
        return None
    
    # 转换为numpy数组
    points_vis = np.float32(points_vis)
    points_therm = np.float32(points_therm)
    
    # 计算单应性矩阵 - 从红外到可见光的变换
    H, mask = cv2.findHomography(points_therm, points_vis, cv2.RANSAC, 3.0)
    
    if H is not None:
        print("\n计算得到的单应性矩阵：")
        print(H)
        
        # 验证单应性矩阵
        H = verify_homography(img_therm, img_vis, H, points_therm, points_vis)  # 注意参数顺序改变
        
        # 测试变换结果
        test_points = np.float32([[0,0], [img_vis.shape[1],0], 
                                [0,img_vis.shape[0]], [img_vis.shape[1],img_vis.shape[0]]])
        test_transformed = cv2.perspectiveTransform(test_points.reshape(-1,1,2), H)
        print("图像四角变换结果：")
        print(test_transformed.reshape(-1,2))
        
        return H
    return None

def warp_thermal_to_visible(image_vis_path, image_therm_path, H, save_path):
    """
    将红外图像通过单应性矩阵投影到可见光图像上
    
    Args:
        image_vis_path (str): 可见光图像路径
        image_therm_path (str): 红外图像路径
        H (np.ndarray): 单应性矩阵
        save_path (str): 结果保存路径
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 读取图像
    img_vis = cv2.imread(image_vis_path)
    img_therm = cv2.imread(image_therm_path)
    
    # 打印变换前后的图像尺寸
    print(f"可见光图像尺寸: {img_vis.shape}")
    print(f"红外图像原始尺寸: {img_therm.shape}")
    print(f"变换后的目标尺寸: {(img_vis.shape[1], img_vis.shape[0])}")
    
    # 使用单应性矩阵进行投影变换 - 不需要反向映射标志
    warped_therm = cv2.warpPerspective(
        img_therm, 
        H, 
        (img_vis.shape[1], img_vis.shape[0])
    )
    
    # 创建融合显示
    # 方法1：简单叠加
    alpha = 0.5
    fused_simple = cv2.addWeighted(img_vis, alpha, warped_therm, 1-alpha, 0)
    
    # 方法2：并排显示
    combined = np.hstack((img_vis, warped_therm))
    
    # 显示结果
    cv2.namedWindow('Original Visible', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Warped Thermal', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Fused Result', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('Side by Side', cv2.WINDOW_KEEPRATIO)
    
    cv2.resizeWindow('Original Visible', 960, 1080)
    cv2.resizeWindow('Warped Thermal', 960, 1080)
    cv2.resizeWindow('Fused Result', 960, 1080)
    cv2.resizeWindow('Side by Side', 1920, 1080)
    
    cv2.imshow('Original Visible', img_vis)
    cv2.imshow('Warped Thermal', warped_therm)
    cv2.imshow('Fused Result', fused_simple)
    cv2.imshow('Side by Side', combined)
    
    # 修改保存结果的路径
    warped_path = os.path.join(save_path, 'warped_thermal.jpg')
    fused_path = os.path.join(save_path, 'fused_result.jpg')
    side_path = os.path.join(save_path, 'side_by_side.jpg')
    
    cv2.imwrite(warped_path, warped_therm)
    cv2.imwrite(fused_path, fused_simple)
    cv2.imwrite(side_path, combined)
    
    print("结果已保存：")
    print(f"- {warped_path}: 变换后的红外图像")
    print(f"- {fused_path}: 融合结果")
    print(f"- {side_path}: 并排对比图")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置路径
    # image_vis = '/ultralytics/data/LLVIP/images/visible/train/010001.jpg'
    # image_therm = '/ultralytics/data/LLVIP/images/infrared/train/010001.jpg'
    # image_vis = 'data/Test/images/visible/train/frame009.jpg'
    # image_therm = 'data/Test/images/infrared/train/frame009.jpg'
    # image_vis = 'data/homography/visible_010001.jpg'
    # image_therm = 'data/homography/infrared_010001.jpg'

    # 
    image_vis = 'data/LLVIP_RAW_TEST/images/visible/train/020002.jpg'
    image_therm = 'data/LLVIP_RAW_TEST/images/infrared/train/020002.jpg'
    save_path = 'runs/mapping_matrix'
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 检查是否存在已保存的单应性矩阵
    matrix_path = os.path.join(save_path, 'manual_homography_matrix.npy')
    if os.path.exists(matrix_path):
        H = np.load(matrix_path)
    else:
        H = manual_compute_homography(image_vis, image_therm, save_path)
    # H = manual_compute_homography(image_vis, image_therm, save_path)
    
    if H is not None:
        print(f'手动选择点计算的单应性矩阵 H:\n{H}')
        # 保存单应性矩阵
        np.save(matrix_path, H)
        
        # 使用计算出的单应性矩阵进行投影变换
        warp_thermal_to_visible(image_vis, image_therm, H, save_path)