"""
Feature matching and spatial transformation modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

class FeatureMatching(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # 特征降维，减少计算量
        self.dim_reduce = Conv(channel, channel//2, 1)
        
        # 特征描述器
        self.descriptor = nn.Sequential(
            Conv(channel//2, channel//2, 3),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
            Conv(channel//2, channel//4, 1)
        )
        
    def extract_keypoints(self, features, top_k=512):
        """提取特征图中的关键点"""
        b, c, h, w = features.shape
        response = torch.norm(features, dim=1)
        response_flat = response.view(b, -1)
        values, indices = response_flat.topk(top_k, dim=-1)
        y = (indices // w).float() / (h-1) * 2 - 1
        x = (indices % w).float() / (w-1) * 2 - 1
        return torch.stack([x, y], dim=-1)

    def compute_correlation(self, desc1, desc2):
        """计算描述子之间的相关性"""
        desc1 = F.normalize(desc1, p=2, dim=1)
        desc2 = F.normalize(desc2, p=2, dim=1)
        correlation = torch.bmm(desc1.transpose(1, 2), desc2)
        return correlation

    def estimate_transform(self, pts1, pts2, weights):
        """估计变换矩阵"""
        batch_size = pts1.size(0)
        H = []
        for b in range(batch_size):
            A = []
            for i in range(pts1.size(1)):
                x1, y1 = pts1[b, i]
                x2, y2 = pts2[b, i]
                w = weights[b, i]
                A.append(w * torch.tensor([
                    [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2],
                    [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2]
                ]))
            A = torch.cat(A, dim=0)
            _, _, Vh = torch.svd(A)
            h = Vh[-1].view(3, 3)
            H.append(h)
        return torch.stack(H)

    def forward(self, rgb_feat, ir_feat):
        # 特征降维
        rgb_reduced = self.dim_reduce(rgb_feat)
        ir_reduced = self.dim_reduce(ir_feat)
        
        # 提取描述子
        rgb_desc = self.descriptor(rgb_reduced)
        ir_desc = self.descriptor(ir_reduced)
        
        # 提取关键点
        rgb_kpts = self.extract_keypoints(rgb_desc)
        ir_kpts = self.extract_keypoints(ir_desc)
        
        # 计算相关性
        correlation = self.compute_correlation(rgb_desc, ir_desc)
        
        # 生成权重
        weights = F.softmax(correlation, dim=-1)
        
        # 估计变换矩阵
        transform_matrix = self.estimate_transform(rgb_kpts, ir_kpts, weights)
        
        return transform_matrix, rgb_desc, ir_desc

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('transform_matrix', torch.eye(3).unsqueeze(0))

    def update_transform(self, matrix):
        self.transform_matrix = matrix

    def forward(self, x):
        b, c, h, w = x.size()
        grid = torch.nn.functional.affine_grid(
            self.transform_matrix[:, :2, :], 
            size=(b, c, h, w),
            align_corners=True
        )
        return torch.nn.functional.grid_sample(
            x, grid, 
            align_corners=True,
            mode='bilinear'
        )