import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", message="grid_sampler_2d_backward_cuda")
from .conv import Conv
from .DEA import DECA, DEPA

class EfficientFeatureAlignment(nn.Module):
    """高效特征对齐模块"""
    def __init__(self, channels, refinement_steps=3):
        super().__init__()
        self.channels = channels
        self.refinement_steps = refinement_steps
        
        # 特征相似度评估
        self.similarity = nn.Sequential(
            Conv(channels * 2, channels // 2, 1),
            nn.ReLU(),
            Conv(channels // 2, 2, 1)  # 预测x,y方向的偏移
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
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
        
        # 特征质量评估
        self.quality_assessment = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, ir_feat, homography):
        b, c, h, w = rgb_feat.shape
        current_ir = ir_feat
        current_H = homography.to(dtype=rgb_feat.dtype, device=rgb_feat.device)
        
        if len(current_H.shape) == 2:
            current_H = current_H.unsqueeze(0).repeat(b, 1, 1)
            
        # 初始变换
        grid = self.get_grid(current_H, h, w, ir_feat.device)
        warped_ir = F.grid_sample(
            current_ir, 
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # 特征质量评估
        rgb_quality = self.quality_assessment(rgb_feat)
        ir_quality = self.quality_assessment(warped_ir)
        
        # 注意力增强
        ca = self.channel_attention(warped_ir)
        warped_ir = warped_ir * ca
        
        avg_out = torch.mean(warped_ir, dim=1, keepdim=True)
        max_out, _ = torch.max(warped_ir, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial)
        enhanced_ir = warped_ir * sa
        
        # 残差连接
        final_ir = enhanced_ir + ir_feat
        
        return final_ir, current_H, rgb_quality, ir_quality

    def get_grid(self, H, height, width, device):
        """生成采样网格"""
        dtype = H.dtype
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing='ij'
        )
        
        points = torch.stack([x, y, torch.ones_like(x, dtype=dtype)], dim=0)
        points = points.reshape(3, -1)
        
        batch_size = H.size(0)
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
        transformed = torch.bmm(H, points)
        
        transformed = transformed / (transformed[:, 2:3, :] + 1e-8)
        grid = transformed[:, :2, :].permute(0, 2, 1).reshape(batch_size, height, width, 2)
        
        return grid

class EFDEA(nn.Module):
    """改进的FMDEA模块"""
    def __init__(self, channel=[512, 512], kernel_size=80, p_kernel=None, m_kernel=None):
        super().__init__()
        
        assert len(channel) == 2, "channel参数必须包含两个值 [rgb_channels, ir_channels]"
        
        # 特征对齐模块
        self.alignment = EfficientFeatureAlignment(channel[1])
        
        # DECA和DEPA模块
        self.deca = DECA(channel[0], kernel_size, p_kernel)
        self.depa = DEPA(channel[1], m_kernel)
        
        # 自适应融合权重
        self.weight_vi = nn.Parameter(torch.ones(1))
        self.weight_ir = nn.Parameter(torch.ones(1))
        
        # 激活函数
        self.fusion = nn.Sequential(nn.Sigmoid())
        
    def adjust_homography_for_scale(self, homography, original_sizes, feature_size):
        """调整单应性矩阵以适应特征图尺度"""
        # 计算缩放因子 (从640到80/40/20)
        scale_factor = original_sizes[0, 0, 0] / feature_size[0]

        # 确保scale_factor在正确的设备上
        scale_factor = scale_factor.to(homography.device)
        
        # 直接调整相关元素
        H_new = homography.clone()
        H_new[..., :2, 2] *= scale_factor  # 调整平移项
        H_new[..., 2, :2] /= scale_factor  # 调整透视项
        
        return H_new

    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs: [rgb_feat, ir_feat, extrinsics, original_sizes]
                - rgb_feat: RGB特征图 [B, C, H, W]
                - ir_feat: 红外特征图 [B, C, H, W]
                - extrinsics: 外参矩阵 [B, 3, 3]
                - original_sizes: 原始图像尺寸 [B, 2, 2]
        """
        rgb_feat, ir_feat, extrinsics, original_sizes = inputs
        
        # 检查输入
        self._check_input_shapes(rgb_feat, ir_feat, extrinsics, original_sizes)
        
        # 获取当前特征图尺寸(80*80、40*40或20*20)
        current_feature_size = rgb_feat.shape[-2:]
        
        # 调整单应性矩阵
        current_H = self.adjust_homography_for_scale(
            extrinsics, 
            original_sizes,
            current_feature_size
        )
        
        # 特征对齐和质量评估
        aligned_ir, refined_H, rgb_quality, ir_quality = self.alignment(rgb_feat, ir_feat, current_H)
        
        # 特征融合
        deca_features = self.deca([rgb_feat, aligned_ir])
        result_vi, result_ir = self.depa(deca_features)
        
        # 使用自适应权重和质量评分进行融合
        fused_feat = self.fusion(
            self.weight_vi * rgb_quality * result_vi + 
            self.weight_ir * ir_quality * result_ir
        )
        
        return fused_feat

    def _check_input_shapes(self, rgb_feat, ir_feat, extrinsics, original_sizes):
        """验证输入张量的形状"""
        B = rgb_feat.size(0)
        
        assert rgb_feat.dim() == 4, f"RGB特征维度应为4，当前为{rgb_feat.dim()}"
        assert ir_feat.dim() == 4, f"红外特征维度应为4，当前为{ir_feat.dim()}"
        
        b, c, h, w = rgb_feat.shape
        assert ir_feat.shape == (b, c, h, w), \
            f"红外特征形状{ir_feat.shape}与RGB特征形状{rgb_feat.shape}不匹配"
        
        assert extrinsics.dim() == 3, f"外参矩阵维度应为3，当前为{extrinsics.dim()}"
        assert extrinsics.shape == (B, 3, 3), \
            f"外参矩阵形状应为({B}, 3, 3)，当前为{extrinsics.shape}"
        
        assert original_sizes.dim() == 3, f"原始尺寸维度应为3，当前为{original_sizes.dim()}"
        assert original_sizes.shape == (B, 2, 2), \
            f"原始尺寸形状应为({B}, 2, 2)，当前为{original_sizes.shape}"