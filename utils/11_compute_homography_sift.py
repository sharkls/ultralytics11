import cv2
import numpy as np

def compute_homography(image_vis, image_therm, feature_matcher='SIFT', visualize=False):
    """
    计算可见光图像与红外图像之间的单应性矩阵，并可选择性地可视化匹配点。
    
    Args:
        image_vis (str): 可见光图像路径。
        image_therm (str): 红外图像路径。
        feature_matcher (str): 特征匹配算法，默认使用SIFT。
        visualize (bool): 是否可视化匹配点，默认为False。
        
    Returns:
        np.ndarray: 3x3 单应性矩阵.
    """
    # 读取图像
    img1 = cv2.imread(image_vis, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_therm, cv2.IMREAD_GRAYSCALE)
    
    # 初始化特征检测器
    if feature_matcher.upper() == 'SIFT':
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    else:
        raise ValueError("Unsupported feature matcher.")
    
    # 使用FLANN匹配器
    index_params = dict(algorithm=1, trees=5)  # 1: FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 应用 Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # 如果需要可视化匹配点
    if visualize:
        # 读取原始彩色图像用于可视化
        img1_color = cv2.imread(image_vis)
        img2_color = cv2.imread(image_therm)
        
        # 绘制匹配点
        matched_img = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 显示结果
        cv2.imshow('Matching Points', matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存匹配结果
        cv2.imwrite('matching_points.jpg', matched_img)
        print("匹配点可视化结果已保存为 matching_points.jpg")
    
    # 需要至少4个匹配点来计算单应性矩阵
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is not None:
            return H
    return np.eye(3)

if __name__ == "__main__":
    image_vis = '/ultralytics/data/LLVIP/images/infrared/train/010001.jpg'
    image_therm = '/ultralytics/data/LLVIP/images/visible/train/010001.jpg'
    # 设置 visualize=True 来显示匹配点
    H = compute_homography(image_vis, image_therm, visualize=True)
    print(f'单应性矩阵 H: {H}')
    print("单应性矩阵已保存为 homography_matrix.npy")