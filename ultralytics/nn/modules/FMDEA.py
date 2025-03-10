from .DEA import DECA, DEPA
from .conv import Conv
import torch
import torch.nn as nn
import torch.nn.functional as F

class FMDEA(nn.Module):
    """Enhanced FMDEA module with fixed mapping matrix"""
    def __init__(self, mapping_matrix, channel=[512, 512], kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        """
        Args:
            mapping_matrix: IR到RGB的3x3映射矩阵，可以是numpy数组或torch.Tensor
            channel: 特征通道数
            kernel_size: DECA模块的核大小
            p_kernel: DECA模块的金字塔卷积核参数
            m_kernel: DEPA模块的合并卷积核参数
            reduction: 通道注意力的降维比例
        """
        super().__init__()

        # 关闭确定性算法警告
        import warnings
        warnings.filterwarnings("ignore", message="grid_sampler_2d_backward_cuda")
        

        # 确保映射矩阵是float32类型
        if not isinstance(mapping_matrix, torch.Tensor):
            mapping_matrix = torch.tensor(mapping_matrix, dtype=torch.float32)
        else:
            mapping_matrix = mapping_matrix.to(torch.float32)
        self.register_buffer('mapping_matrix', mapping_matrix)
        
        # 原有的DECA和DEPA模块
        self.deca = DECA(channel[0], kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel[1], m_kernel)
        self.act = nn.Sigmoid()
        
        # 可选：添加轻量级的特征细化模块
        self.refine = Conv(channel[1], channel[1], 3)

    def warp_ir_to_rgb(self, ir_feat, rgb_size):
        """使用固定映射矩阵将IR特征图变换到RGB视角"""
        b, c, h, w = ir_feat.size()
        
        # 确保在计算inverse时使用float32精度
        inv_matrix = torch.inverse(self.mapping_matrix.float())
        
        # 生成网格点（使用float32）
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, rgb_size[0], device=ir_feat.device, dtype=torch.float32),
            torch.linspace(-1, 1, rgb_size[1], device=ir_feat.device, dtype=torch.float32)
        )
        ones = torch.ones_like(grid_x)
        grid = torch.stack([grid_x, grid_y, ones]).reshape(3, -1)
        
        # 计算变换后的网格点
        warped_grid = inv_matrix @ grid
        warped_grid = warped_grid / warped_grid[2:3]
        warped_grid = warped_grid[:2].reshape(2, rgb_size[0], rgb_size[1]).permute(1, 2, 0)
        
        # 扩展到batch维度并转换到输入特征的数据类型
        warped_grid = warped_grid.unsqueeze(0).expand(b, -1, -1, -1).to(ir_feat.dtype)
        
        # 进行采样
        warped_feat = F.grid_sample(
            ir_feat, 
            warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # 特征细化
        refined_feat = self.refine(warped_feat)
        
        return refined_feat

    def forward(self, x):
        rgb_feat, ir_feat = x[0], x[1]
        
        # 1. 使用映射矩阵将IR特征图对齐到RGB视角
        aligned_ir = self.warp_ir_to_rgb(ir_feat, rgb_feat.shape[2:])
        
        # 2. 使用对齐后的特征进行融合
        result_vi, result_ir = self.depa(self.deca([rgb_feat, aligned_ir]))
        
        return self.act(result_vi + result_ir)

# 使用示例
def main():
    # 1. 创建模型
    model = FMDEA(channel=[512, 512])
    
    # 2. 设置预先标定好的映射矩阵
    mapping_matrix = torch.tensor([
        [1.1, 0.1, 10],  # 示例矩阵，需要替换为实际的标定矩阵
        [-0.05, 0.95, -5],
        [0, 0, 1]
    ], dtype=torch.float)
    model.set_mapping_matrix(mapping_matrix)
    
    # 3. 使用模型
    rgb_feat = torch.randn(1, 512, 64, 64)
    ir_feat = torch.randn(1, 512, 64, 64)
    fused_feat = model([rgb_feat, ir_feat])