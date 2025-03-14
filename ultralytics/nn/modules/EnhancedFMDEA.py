import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv  # 假设您有这个基础卷积模块
from .DEA import DECA, DEPA  # 假设您有这些基础模块

class EfficientFeatureAlignment(nn.Module):
    """高效特征对齐模块"""
    def __init__(self, channels, refinement_steps=3):
        super().__init__()
        self.channels = channels
        self.refinement_steps = refinement_steps
        
        # 特征相似度评估
        self.similarity = nn.Sequential(
            Conv(channels * 2, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            Conv(channels // 2, 2, 1)  # 预测x,y方向的偏移
        )
        
        # 修改通道注意力机制，移除BatchNorm
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 使用普通卷积层替代Conv类（因为Conv类包含BatchNorm）
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            Conv(2, 1, 7, p=3),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.enhancement = Conv(channels, channels, 3, p=1)

    def forward(self, rgb_feat, ir_feat, homography):
        b, c, h, w = rgb_feat.shape
        current_ir = ir_feat

        # 确保homography与特征图具有相同的数据类型
        current_H = homography.to(dtype=rgb_feat.dtype, device=rgb_feat.device)
        if len(current_H.shape) == 2:
            current_H = current_H.unsqueeze(0).repeat(b, 1, 1)
        # current_H = homography.unsqueeze(0).repeat(b, 1, 1) if len(homography.shape) == 2 else homography
        
        for i in range(self.refinement_steps):
            # 1. 应用当前变换
            grid = self.get_grid(current_H, h, w, ir_feat.device)
            warped_ir = F.grid_sample(
                current_ir,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            
            # 2. 特征相似度评估和偏移预测
            concat_feat = torch.cat([rgb_feat, warped_ir], dim=1)
            offset = self.similarity(concat_feat)
            
            # 3. 更新变换矩阵
            # delta_H = self.offset_to_H(offset, h, w)
            # current_H = torch.bmm(current_H, delta_H)
            # 更新变换矩阵时确保数据类型一致
            delta_H = self.offset_to_H(offset, h, w)
            current_H = torch.bmm(current_H.to(delta_H.dtype), delta_H)
            
            # 4. 注意力增强
            # 通道注意力
            ca = self.channel_attention(warped_ir)
            warped_ir = warped_ir * ca
            
            # 空间注意力
            avg_out = torch.mean(warped_ir, dim=1, keepdim=True)
            max_out, _ = torch.max(warped_ir, dim=1, keepdim=True)
            spatial = torch.cat([avg_out, max_out], dim=1)
            sa = self.spatial_attention(spatial)
            warped_ir = warped_ir * sa
            
            # 5. 特征增强
            current_ir = self.enhancement(warped_ir)
            
            # 6. 残差连接
            if i == self.refinement_steps - 1:
                current_ir = current_ir + ir_feat
        
        return current_ir, current_H

    def get_grid(self, H, height, width, device):
        """生成采样网格"""
        # 获取H的数据类型
        dtype = H.dtype
        
        # 1. 生成网格点并确保数据类型匹配
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing='ij'
        )
        
        # 2. 重塑为齐次坐标 (x,y,1)，保持数据类型一致
        points = torch.stack([x, y, torch.ones_like(x, dtype=dtype)], dim=0)
        points = points.reshape(3, -1)
        
        # 3. 扩展points到batch维度并应用变换
        batch_size = H.size(0)
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
        transformed = torch.bmm(H, points)  # 现在两个输入的数据类型一致
        
        # 4. 归一化齐次坐标
        transformed = transformed / (transformed[:, 2:3, :] + 1e-8)
        
        # 5. 重塑为网格格式
        grid = transformed[:, :2, :].permute(0, 2, 1).reshape(batch_size, height, width, 2)
        
        return grid

    def offset_to_H(self, offset, height, width):
        """将偏移量转换为增量单应性矩阵"""
        b = offset.size(0)
        dx = offset[:, 0].mean() / width
        dy = offset[:, 1].mean() / height
        
        # 使用与offset相同的数据类型和设备
        H = torch.eye(3, device=offset.device, dtype=offset.dtype)
        H = H.unsqueeze(0).repeat(b, 1, 1)
        H[:, 0, 2] = dx
        H[:, 1, 2] = dy
        
        return H

class EnhancedFMDEA(nn.Module):
    """改进的FMDEA模块"""
    def __init__(self, init_homography, channel=[512, 512], 
                 kernel_size=80, p_kernel=None, m_kernel=None, 
                 refinement_steps=3):
        super().__init__()
        
        # 注册初始单应性矩阵
        if not isinstance(init_homography, torch.Tensor):
            init_homography = torch.tensor(init_homography, dtype=torch.float32)
        self.register_buffer('init_H', init_homography)
        
        # 使用高效特征对齐模块
        self.alignment = EfficientFeatureAlignment(channel[1], refinement_steps)
        
        # DECA和DEPA模块
        self.deca = DECA(channel[0], kernel_size, p_kernel)
        self.depa = DEPA(channel[1], m_kernel)
        
        # 最终激活
        self.act = nn.Sigmoid()
        
        # 预热参数
        self.register_buffer('current_step', torch.tensor(0))
        self.warmup_steps = 1000

    def forward(self, x):
        rgb_feat, ir_feat = x[0], x[1]
        
        # 更新训练步数
        if self.training:
            self.current_step += 1
        
        # 1. 特征对齐
        aligned_ir, refined_H = self.alignment(rgb_feat, ir_feat, self.init_H)
        
        # 2. 特征融合
        result_vi, result_ir = self.depa(self.deca([rgb_feat, aligned_ir]))
        fused_feat = self.act(result_vi + result_ir)
        
        return fused_feat