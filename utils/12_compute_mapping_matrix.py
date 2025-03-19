import cv2
import numpy as np
import os

class ImageMatcher:
    def __init__(self, vis_path, therm_path, save_path='runs/mapping_matrix'):
        """
        初始化图像匹配器
        
        Args:
            vis_path (str): 可见光图像路径
            therm_path (str): 红外图像路径
            save_path (str): 结果保存路径
        """
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        
        self.img_vis = cv2.imread(vis_path)
        self.img_therm = cv2.imread(therm_path)
        
        # 确保图像尺寸相同
        if self.img_vis.shape != self.img_therm.shape:
            self.img_therm = cv2.resize(self.img_therm, 
                                      (self.img_vis.shape[1], self.img_vis.shape[0]))
        
        # 转换为灰度图
        self.img_vis_gray = cv2.cvtColor(self.img_vis, cv2.COLOR_BGR2GRAY)
        self.img_therm_gray = cv2.cvtColor(self.img_therm, cv2.COLOR_BGR2GRAY)
        
        # 添加预处理步骤
        self.preprocess_images()
        
        # 存储计算结果
        self.flow = None
        self.homography = None
        self.refined_flow = None

    def preprocess_images(self):
        """图像预处理，增强特征"""
        # 对可见光图像进行直方图均衡化
        self.img_vis_gray = cv2.equalizeHist(self.img_vis_gray)
        
        # 对红外图像进行对比度增强
        alpha = 1.5  # 对比度增强系数
        beta = 10    # 亮度增强系数
        self.img_therm_gray = cv2.convertScaleAbs(self.img_therm_gray, alpha=alpha, beta=beta)
        
        # 应用高斯滤波减少噪声
        self.img_vis_gray = cv2.GaussianBlur(self.img_vis_gray, (3, 3), 0)
        self.img_therm_gray = cv2.GaussianBlur(self.img_therm_gray, (3, 3), 0)

    def compute_homography(self):
        """计算单应性矩阵"""
        # 使用SIFT特征检测和匹配
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.img_vis_gray, None)
        kp2, des2 = sift.detectAndCompute(self.img_therm_gray, None)
        
        # FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 应用比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            self.homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return self.homography
        return None

    def compute_optical_flow(self):
        """计算光流，调整参数以提高精度"""
        self.flow = cv2.calcOpticalFlowFarneback(
            self.img_vis_gray,
            self.img_therm_gray,
            None,
            pyr_scale=0.5,    # 保持0.5
            levels=6,         # 增加金字塔层数
            winsize=21,       # 增加窗口大小
            iterations=5,     # 增加迭代次数
            poly_n=7,         # 增加多项式展开
            poly_sigma=1.5,   # 调整高斯标准差
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN  # 使用高斯加权
        )
        return self.flow

    def refine_local_matching(self, window_size=31):
        """局部优化匹配"""
        if self.flow is None:
            self.compute_optical_flow()
            
        h, w = self.flow.shape[:2]
        self.refined_flow = self.flow.copy()
        half_window = window_size // 2
        
        for y in range(half_window, h - half_window, window_size):
            for x in range(half_window, w - half_window, window_size):
                template = self.img_vis_gray[y-half_window:y+half_window+1, 
                                          x-half_window:x+half_window+1]
                
                search_y = int(y + self.flow[y, x, 1])
                search_x = int(x + self.flow[y, x, 0])
                
                if (search_y >= half_window and search_y < h - half_window and 
                    search_x >= half_window and search_x < w - half_window):
                    
                    search_region = self.img_therm_gray[search_y-half_window:search_y+half_window+1,
                                                      search_x-half_window:search_x+half_window+1]
                    
                    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(result)
                    
                    self.refined_flow[y, x] = [max_loc[0] - half_window, max_loc[1] - half_window]
        
        return self.refined_flow

    def post_process_warped(self, warped):
        """对变换后的图像进行后处理"""
        # 创建mask去除边缘伪影
        mask = np.ones_like(warped, dtype=np.uint8) * 255
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        
        # 应用mask
        warped = cv2.bitwise_and(warped, mask)
        
        return warped

    def warp_image(self, method='flow'):
        """根据指定方法变换图像"""
        if method == 'flow':
            if self.flow is None:
                self.compute_optical_flow()
            
            h, w = self.flow.shape[:2]
            map_x = np.float32(np.indices((w, h))[0].T)
            map_y = np.float32(np.indices((w, h))[1].T)
            
            map_x = map_x + self.flow[..., 0]
            map_y = map_y + self.flow[..., 1]
            
            warped = cv2.remap(self.img_therm, map_x, map_y, 
                             cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
            
        elif method == 'homography':
            if self.homography is None:
                self.compute_homography()
            
            warped = cv2.warpPerspective(
                self.img_therm,
                self.homography,
                (self.img_vis.shape[1], self.img_vis.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
            )
        
        # 应用后处理
        warped = self.post_process_warped(warped)
        return warped

    def weighted_fusion(self, vis, therm):
        """加权融合两张图像"""
        # 计算每个图像的权重
        vis_weight = cv2.Laplacian(cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        therm_weight = cv2.Laplacian(cv2.cvtColor(therm, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        
        # 归一化权重
        vis_weight = np.abs(vis_weight)
        therm_weight = np.abs(therm_weight)
        sum_weight = vis_weight + therm_weight
        vis_weight = vis_weight / (sum_weight + 1e-6)
        therm_weight = therm_weight / (sum_weight + 1e-6)
        
        # 扩展维度以适应彩色图像
        vis_weight = np.expand_dims(vis_weight, axis=2)
        therm_weight = np.expand_dims(therm_weight, axis=2)
        
        # 加权融合
        fused = vis * vis_weight + therm * therm_weight
        return fused.astype(np.uint8)

    def visualize_results(self):
        """可视化结果，重点展示最终的融合效果"""
        # 计算不同方法的结果
        flow_warped = self.warp_image('flow')
        homo_warped = self.warp_image('homography')
        
        # 使用加权融合
        flow_fused = self.weighted_fusion(self.img_vis, flow_warped)
        homo_fused = self.weighted_fusion(self.img_vis, homo_warped)
        
        # 创建对比图
        h, w = self.img_vis.shape[:2]
        comparison = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # 左上：原始可见光图像
        comparison[:h, :w] = self.img_vis
        # 右上：原始红外图像
        comparison[:h, w:] = self.img_therm
        # 左下：光流法结果
        comparison[h:, :w] = flow_warped
        # 右下：最终融合结果
        comparison[h:, w:] = flow_fused
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Visible', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Thermal', (w+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Warped', (10, h+30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Final Fusion', (w+10, h+30), font, 1, (255, 255, 255), 2)
        
        # 显示对比图
        cv2.namedWindow('Results Comparison', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Results Comparison', 1920, 1080)
        cv2.imshow('Results Comparison', comparison)
        
        # 单独显示最终融合结果
        cv2.namedWindow('Final Result', cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Final Result', 960, 1080)
        cv2.imshow('Final Result', flow_fused)
        
        # 保存结果到指定路径
        comparison_path = os.path.join(self.save_path, 'comparison.jpg')
        fusion_path = os.path.join(self.save_path, 'final_fusion.jpg')
        warped_path = os.path.join(self.save_path, 'warped_thermal.jpg')
        flow_path = os.path.join(self.save_path, 'optical_flow.npy')
        homo_path = os.path.join(self.save_path, 'homography_matrix.npy')
        
        cv2.imwrite(comparison_path, comparison)
        cv2.imwrite(fusion_path, flow_fused)
        cv2.imwrite(warped_path, flow_warped)
        
        if self.flow is not None:
            np.save(flow_path, self.flow)
        if self.homography is not None:
            np.save(homo_path, self.homography)
        
        print(f"\n所有结果已保存到目录: {self.save_path}")
        print(f"- comparison.jpg: 对比图")
        print(f"- final_fusion.jpg: 最终融合结果")
        print(f"- warped_thermal.jpg: 变换后的红外图像")
        print(f"- optical_flow.npy: 光流场数据")
        if self.homography is not None:
            print(f"- homography_matrix.npy: 单应性矩阵")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # 图像路径
    vis_path = '/ultralytics/data/LLVIP/images/visible/train/010001.jpg'
    therm_path = '/ultralytics/data/LLVIP/images/infrared/train/010001.jpg'
    save_path = 'runs/mapping_matrix'
    
    # 创建匹配器实例
    matcher = ImageMatcher(vis_path, therm_path, save_path)
    
    # 计算光流并检查结果
    flow = matcher.compute_optical_flow()
    if flow is None:
        print("光流计算失败")
        return
    
    # 计算单应性矩阵并检查结果
    H = matcher.compute_homography()
    if H is None:
        print("单应性矩阵计算失败")
    
    # 进行局部优化
    matcher.refine_local_matching()
    
    # 显示结果
    matcher.visualize_results()

if __name__ == "__main__":
    main() 