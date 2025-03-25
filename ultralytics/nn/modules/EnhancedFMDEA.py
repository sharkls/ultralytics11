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
    def __init__(self, channel=[512, 512], 
                 kernel_size=80, p_kernel=None, m_kernel=None, 
                 refinement_steps=3):
        super().__init__()
        
        # 参数验证
        assert len(channel) == 2, "channel参数必须包含两个值 [rgb_channels, ir_channels]"
        assert kernel_size > 0, "kernel_size必须为正数"
        assert refinement_steps > 0, "refinement_steps必须为正数"
        
        # 使用高效特征对齐模块
        self.alignment = EfficientFeatureAlignment(channel[1], refinement_steps)
        
        # DECA和DEPA模块
        self.deca = DECA(channel[0], kernel_size, p_kernel)
        self.depa = DEPA(channel[1], m_kernel)
        
       # 使用Parameter替代buffer以支持梯度更新
        self.current_step = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.warmup_steps = 1000
        
        # 使用nn.Sequential优化激活函数的应用
        self.fusion = nn.Sequential(nn.Sigmoid())
    
    def adjust_homography_for_scale(self, homography, original_sizes, feature_size):
        """
        根据特征图尺寸调整单应性矩阵
        
        Args:
            homography (torch.Tensor): 原始单应性矩阵 [B, 3, 3]
            original_sizes (torch.Tensor): 原始图像尺寸 [B, 2, 2]，其中2表示rgb和ir，2表示(H, W)
            feature_size (tuple): 特征图尺寸 (H, W)
        """
        B = homography.shape[0]
        device = homography.device
        dtype = homography.dtype
        
        # 为每个样本创建对应的变换矩阵
        transformed_H = []
        for b in range(B):
            # 获取当前样本的原始尺寸
            rgb_h, rgb_w = original_sizes[b, 0]  # RGB尺寸
            ir_h, ir_w = original_sizes[b, 1]    # IR尺寸
            feat_h, feat_w = feature_size
            
            # 计算RGB和IR图像的缩放比例
            scale_h_rgb = rgb_h / feat_h
            scale_w_rgb = rgb_w / feat_w
            scale_h_ir = ir_h / feat_h
            scale_w_ir = ir_w / feat_w
            
            # 构建RGB图像的缩放矩阵（逆变换）
            S_rgb = torch.eye(3, device=device, dtype=dtype)
            S_rgb[0, 0] = 1/scale_w_rgb  # x方向缩放
            S_rgb[1, 1] = 1/scale_h_rgb  # y方向缩放
            
            # 构建IR图像的缩放矩阵
            S_ir = torch.eye(3, device=device, dtype=dtype)
            S_ir[0, 0] = scale_w_ir    # x方向缩放
            S_ir[1, 1] = scale_h_ir    # y方向缩放
            
            # 计算当前样本的变换矩阵: S_rgb @ H @ S_ir
            current_H = torch.mm(torch.mm(S_rgb, homography[b]), S_ir)
            transformed_H.append(current_H)
        
        # 堆叠所有样本的变换矩阵
        return torch.stack(transformed_H)
    
    def forward_v1(self, inputs):
        """
        前向传播
        
        Args:
            inputs (list): [rgb_features, ir_features, extrinsics]
                - rgb_features (torch.Tensor): RGB特征图 [B, C, H, W]
                - ir_features (torch.Tensor): 红外特征图 [B, C, H, W]
                - extrinsics (torch.Tensor): 外参矩阵 [B, 3, 3]
                
        Returns:
            torch.Tensor: 融合后的特征图 [B, C, H, W]
        """
        # 1. 输入验证和解包
        assert len(inputs) == 3, "输入必须包含RGB特征、红外特征和外参矩阵"
        rgb_feat, ir_feat, extrinsics = inputs

        # 2. 形状检查
        self._check_input_shapes(rgb_feat, ir_feat, extrinsics)

        # 4. 特征对齐
        aligned_ir, refined_H = self.alignment(rgb_feat, ir_feat, extrinsics)
        
        # 5. 特征融合
        deca_features = self.deca([rgb_feat, aligned_ir])
        result_vi, result_ir = self.depa(deca_features)
        
        # 6. 融合结果
        fused_feat = self.fusion(result_vi + result_ir)
        
        return fused_feat

    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs (list): [rgb_feat, ir_feat, extrinsics, original_sizes]
                - rgb_feat (torch.Tensor): RGB特征图 [B, C, H, W]
                - ir_feat (torch.Tensor): 红外特征图 [B, C, H, W]
                - extrinsics (torch.Tensor): 外参矩阵 [B, 3, 3]
                - original_sizes (dict): 原始图像尺寸 [B, num_modalities, 2]
        
        Returns:
            torch.Tensor: 融合后的特征图 [B, C, H, W]
        """
        # 获取输入
        rgb_feat, ir_feat, extrinsics, original_sizes = inputs
        
        # 检查输入
        self._check_input_shapes(rgb_feat, ir_feat, extrinsics, original_sizes)
        
        # 获取当前特征图尺寸
        current_feature_size = rgb_feat.shape[-2:]  # (H, W)
        
        # 调整单应性矩阵以适应当前特征图尺度
        current_H = self.adjust_homography_for_scale(
            extrinsics, 
            original_sizes,
            current_feature_size
        )
        
        # 特征对齐
        aligned_ir, refined_H = self.alignment(rgb_feat, ir_feat, current_H)
        
        # 特征融合
        deca_features = self.deca([rgb_feat, aligned_ir])
        result_vi, result_ir = self.depa(deca_features)
        
        # 融合结果
        fused_feat = self.fusion(result_vi + result_ir)
        
        return fused_feat

    def _check_input_shapes(self, rgb_feat, ir_feat, extrinsics, original_sizes):
        """验证输入张量的形状"""
        B = rgb_feat.size(0)
        
        # 检查特征图维度
        assert rgb_feat.dim() == 4, f"RGB特征维度应为4，当前为{rgb_feat.dim()}"
        assert ir_feat.dim() == 4, f"红外特征维度应为4，当前为{ir_feat.dim()}"
        
        # 检查特征图形状匹配
        b, c, h, w = rgb_feat.shape
        assert ir_feat.shape == (b, c, h, w), \
            f"红外特征形状{ir_feat.shape}与RGB特征形状{rgb_feat.shape}不匹配"
        
        # 检查外参矩阵
        assert extrinsics.dim() == 3, f"外参矩阵维度应为3，当前为{extrinsics.dim()}"
        assert extrinsics.shape == (B, 3, 3), \
            f"外参矩阵形状应为({B}, 3, 3)，当前为{extrinsics.shape}"
        
        # 检查原始尺寸张量
        assert original_sizes.dim() == 3, f"原始尺寸维度应为3，当前为{original_sizes.dim()}"
        assert original_sizes.shape == (B, 2, 2), \
            f"原始尺寸形状应为({B}, 2, 2)，当前为{original_sizes.shape}"