import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv  # 假设您有这个基础卷积模块
from .DEA import DECA, DEPA  # 假设您有这些基础模块

class FeaturePyramidAlignment(nn.Module):
    """多尺度特征对齐模块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 多尺度特征提取
        self.downsample = nn.ModuleList([
            Conv(channels, channels, 3, 2) for _ in range(2)
        ])
        
        # 特征增强
        self.enhance = nn.ModuleList([
            nn.Sequential(
                Conv(channels, channels, 3),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ) for _ in range(3)
        ])
        
        # 上采样融合
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv(channels, channels, 3),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ) for _ in range(2)
        ])

    def forward(self, x):
        # 自顶向下路径
        features = [x]
        for down in self.downsample:
            features.append(down(features[-1]))
        
        # 增强特征
        enhanced = [enhance(feat) for enhance, feat in zip(self.enhance, features)]
        
        # 自底向上路径
        results = [enhanced[-1]]
        for i, up in enumerate(self.upsample):
            feat = up(results[-1]) + enhanced[-(i+2)]
            results.append(feat)
            
        return results[-1]

class AdaptiveFeatureAlignment(nn.Module):
    """改进的自适应特征对齐模块"""
    def __init__(self, channels, refinement_steps=3):
        super().__init__()
        self.channels = channels
        self.refinement_steps = refinement_steps
        
        # 特征金字塔对齐
        self.pyramid_align = FeaturePyramidAlignment(channels)
        
        # 偏移预测器
        self.offset_predictor = nn.Sequential(
            Conv(channels * 2, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            Conv(channels // 2, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(),
            Conv(channels // 4, 2, 1)
        )
        
        # 注意力模块
        self.attention = nn.Sequential(
            Conv(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            Conv(channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.enhancement = nn.Sequential(
            Conv(channels, channels, 3, p=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            Conv(channels, channels, 3, p=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def get_grid(self, H, height, width, device):
        """生成采样网格"""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        )
        points = torch.stack([x, y, torch.ones_like(x)]).reshape(3, -1)
        
        # 批处理变换
        if len(H.shape) == 3:
            points = points.unsqueeze(0).expand(H.size(0), -1, -1)
            transformed = torch.bmm(H, points)
        else:
            transformed = H @ points
            
        transformed = transformed / (transformed[..., 2:3] + 1e-8)
        grid = transformed[..., :2].reshape(-1, height, width, 2)
        
        return grid

    def forward(self, rgb_feat, ir_feat, homography):
        b, c, h, w = rgb_feat.shape
        current_ir = ir_feat
        current_H = homography.unsqueeze(0).repeat(b, 1, 1) if len(homography.shape) == 2 else homography
        
        # 残差连接的起始特征
        identity = current_ir
        
        for i in range(self.refinement_steps):
            # 1. 特征金字塔对齐
            pyramid_feat = self.pyramid_align(current_ir)
            
            # 2. 生成采样网格并变换特征
            grid = self.get_grid(current_H, h, w, ir_feat.device)
            warped_ir = F.grid_sample(
                pyramid_feat,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            
            # 3. 预测特征偏移
            concat_feat = torch.cat([rgb_feat, warped_ir], dim=1)
            offset = self.offset_predictor(concat_feat)
            
            # 4. 更新变换矩阵
            delta_H = self.offset_to_H(offset, h, w)
            current_H = torch.bmm(current_H, delta_H)
            
            # 5. 注意力加权
            attention_weights = self.attention(warped_ir)
            current_ir = warped_ir * attention_weights
            
            # 6. 特征增强
            current_ir = self.enhancement(current_ir)
            
            # 7. 残差连接
            if i == self.refinement_steps - 1:
                current_ir = current_ir + identity
        
        return current_ir, current_H

    def offset_to_H(self, offset, height, width):
        """将偏移量转换为增量单应性矩阵"""
        b = offset.size(0)
        dx = offset[:, 0].mean() / width
        dy = offset[:, 1].mean() / height
        
        H = torch.eye(3, device=offset.device).unsqueeze(0).repeat(b, 1, 1)
        H[:, 0, 2] = dx
        H[:, 1, 2] = dy
        
        return H

class ModifiedFMDEA(nn.Module):
    """改进的FMDEA模块"""
    def __init__(self, init_homography, channel=[512, 512], 
                 kernel_size=80, p_kernel=None, m_kernel=None, 
                 refinement_steps=3):
        super().__init__()
        
        # 注册初始单应性矩阵
        if not isinstance(init_homography, torch.Tensor):
            init_homography = torch.tensor(init_homography, dtype=torch.float32)
        self.register_buffer('init_H', init_homography)
        
        # 特征对齐模块
        self.alignment = AdaptiveFeatureAlignment(channel[1], refinement_steps)
        
        # DECA和DEPA模块
        self.deca = DECA(channel[0], kernel_size, p_kernel)
        self.depa = DEPA(channel[1], m_kernel)
        
        # 最终激活
        self.act = nn.Sigmoid()
        
        # 预热参数
        self.register_buffer('current_step', torch.tensor(0))
        self.warmup_steps = 1000

    def get_current_refinement_steps(self):
        """获取当前的优化步数（用于预热）"""
        if self.training:
            progress = min(self.current_step / self.warmup_steps, 1.0)
            return max(1, int(self.alignment.refinement_steps * progress))
        return self.alignment.refinement_steps

    def forward(self, x):
        rgb_feat, ir_feat = x[0], x[1]
        
        # 更新训练步数
        if self.training:
            self.current_step += 1
        
        # 临时设置优化步数（用于预热）
        original_steps = self.alignment.refinement_steps
        self.alignment.refinement_steps = self.get_current_refinement_steps()
        
        # 1. 特征对齐
        aligned_ir, refined_H = self.alignment(rgb_feat, ir_feat, self.init_H)
        
        # 2. 特征融合
        result_vi, result_ir = self.depa(self.deca([rgb_feat, aligned_ir]))
        fused_feat = self.act(result_vi + result_ir)
        
        # 恢复原始步数
        self.alignment.refinement_steps = original_steps
        
        return fused_feat

# 训练技巧函数
def get_warmup_scheduler(optimizer, warmup_steps=1000):
    """获取预热学习率调度器"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1.0, warmup_steps))
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 使用示例
def main():
    # 初始化参数（只需要单应性矩阵）
    init_homography = torch.eye(3)  # 这里应该是您通过标定得到的实际单应性矩阵
    
    # 创建模型
    model = ModifiedFMDEA(
        init_homography=init_homography,
        channel=[512, 512],
        refinement_steps=3
    )
    
    # 优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_warmup_scheduler(optimizer)
    
    # 使用示例
    rgb_feat = torch.randn(1, 512, 64, 64)
    ir_feat = torch.randn(1, 512, 64, 64)
    
    # 训练模式
    model.train()
    fused_feat = model([rgb_feat, ir_feat])

"""
    为了确保稳定收敛，代码中包含了以下改进：
    1.预热策略：
        逐步增加refinement_steps
        使用学习率预热
        渐进式特征对齐
    2.多尺度处理：
        添加特征金字塔对齐
        多级特征增强
        上下采样融合
    3.稳定性优化：
        残差连接
        注意力机制
        边界约束
    4.训练技巧：
        批归一化
        梯度裁剪（需要在训练循环中实现）
        学习率调度
    使用建议：
        1.开始训练时使用较小的学习率
        2.监控对齐模块的输出，确保变换合理
        3.可以根据实际需求调整预热步数和refinement_steps
        4.如果显存不足，可以减少特征金字塔的层数
"""