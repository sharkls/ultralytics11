from .DEA import DECA, DEPA
from .conv import Conv
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAlignment(nn.Module):
    """自适应对齐模块"""
    def __init__(self, channels):
        super().__init__()
        self.offset_predictor = nn.Sequential(
            Conv(channels, channels//2, 3),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(),
            Conv(channels//2, 2, 3)  # 预测x,y方向的偏移
        )
        
        self.attention = nn.Sequential(
            Conv(channels, channels//2, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(),
            Conv(channels//2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, feat):
        # 预测空间偏移
        offset = self.offset_predictor(feat)
        b, c, h, w = feat.shape
        
        # 生成基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device),
            torch.linspace(-1, 1, w, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        
        # 应用预测的偏移
        grid = grid + offset
        grid = grid.permute(0, 2, 3, 1)
        
        # 进行特征变换
        aligned_feat = F.grid_sample(
            feat, 
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # 计算注意力权重
        attention_weights = self.attention(feat)
        
        return aligned_feat * attention_weights

class RobustFMDEA(nn.Module):
    """Enhanced FMDEA module with robust feature alignment"""
    def __init__(self, mapping_matrix, camera_params=None, channel=[512, 512], 
                 kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        """
        Args:
            mapping_matrix: IR到RGB的3x3映射矩阵
            camera_params: 相机内参和畸变参数
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
        
        # 注册映射矩阵
        if not isinstance(mapping_matrix, torch.Tensor):
            mapping_matrix = torch.tensor(mapping_matrix, dtype=torch.float32)
        self.register_buffer('mapping_matrix', mapping_matrix.to(torch.float32))
        
        # 注册相机参数（如果有）
        if camera_params is not None:
            for key, value in camera_params.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32)
                self.register_buffer(f'camera_{key}', value)
        
        # 特征细化模块
        self.refinement = nn.Sequential(
            AdaptiveAlignment(channel[1]),
            Conv(channel[1], channel[1], 3),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU()
        )
        
        # 原有的DECA和DEPA模块
        self.deca = DECA(channel[0], kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel[1], m_kernel)
        self.act = nn.Sigmoid()
        
        # 特征一致性评估
        self.consistency_estimator = nn.Sequential(
            Conv(channel[1] * 2, channel[1], 1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(),
            Conv(channel[1], 1, 1),
            nn.Sigmoid()
        )

    def warp_with_camera_params(self, ir_feat, rgb_size):
        """考虑相机参数的特征变换"""
        b, c, h, w = ir_feat.size()
        
        # 计算完整的变换矩阵
        if hasattr(self, 'camera_rgb_intrinsic') and hasattr(self, 'camera_ir_intrinsic'):
            transform = (self.camera_rgb_intrinsic @ 
                        self.mapping_matrix @ 
                        torch.inverse(self.camera_ir_intrinsic))
        else:
            transform = self.mapping_matrix
            
        # 确保使用float32进行计算
        inv_matrix = torch.inverse(transform.float())
        
        # 生成采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, rgb_size[0], device=ir_feat.device, dtype=torch.float32),
            torch.linspace(-1, 1, rgb_size[1], device=ir_feat.device, dtype=torch.float32),
            indexing='ij'
        )
        ones = torch.ones_like(grid_x)
        grid = torch.stack([grid_x, grid_y, ones]).reshape(3, -1)
        
        # 应用变换
        warped_grid = inv_matrix @ grid
        warped_grid = warped_grid / warped_grid[2:3]
        warped_grid = warped_grid[:2].reshape(2, rgb_size[0], rgb_size[1]).permute(1, 2, 0)
        
        # 扩展到batch维度
        warped_grid = warped_grid.unsqueeze(0).expand(b, -1, -1, -1).to(ir_feat.dtype)
        
        # 进行特征采样
        warped_feat = F.grid_sample(
            ir_feat,
            warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped_feat

    def compute_consistency(self, rgb_feat, aligned_ir):
        """计算特征一致性"""
        concat_feat = torch.cat([rgb_feat, aligned_ir], dim=1)
        consistency_score = self.consistency_estimator(concat_feat)
        return consistency_score

    def forward(self, x):
        rgb_feat, ir_feat = x[0], x[1]
        
        # 1. 基础几何对齐
        base_aligned = self.warp_with_camera_params(ir_feat, rgb_feat.shape[2:])
        
        # 2. 自适应特征细化
        refined_ir = self.refinement(base_aligned)
        
        # 3. 计算特征一致性
        consistency = self.compute_consistency(rgb_feat, refined_ir)
        
        # 4. 特征融合
        result_vi, result_ir = self.depa(self.deca([rgb_feat, refined_ir]))
        fused_feat = self.act(result_vi + result_ir)
        
        if self.training:
            return fused_feat, {
                'consistency': consistency,
                'aligned_ir': refined_ir
            }
        return fused_feat

def consistency_loss(rgb_feat, aligned_ir, consistency_score):
    """特征一致性损失"""
    # 结构相似性损失
    ssim_loss = 1 - F.cosine_similarity(
        rgb_feat.flatten(2), 
        aligned_ir.flatten(2)
    ).mean()
    
    # 加权一致性损失
    weighted_loss = ssim_loss * consistency_score.mean()
    
    return weighted_loss

# 使用示例
def main():
    # 相机参数示例
    camera_params = {
        'rgb_intrinsic': torch.tensor([
            [fx_rgb, 0, cx_rgb],
            [0, fy_rgb, cy_rgb],
            [0, 0, 1]
        ]),
        'ir_intrinsic': torch.tensor([
            [fx_ir, 0, cx_ir],
            [0, fy_ir, cy_ir],
            [0, 0, 1]
        ])
    }
    
    # 创建模型
    mapping_matrix = torch.eye(3)  # 示例：使用单位矩阵
    model = RobustFMDEA(
        mapping_matrix=mapping_matrix,
        camera_params=camera_params,
        channel=[512, 512]
    )
    
    # 使用模型
    rgb_feat = torch.randn(1, 512, 64, 64)
    ir_feat = torch.randn(1, 512, 64, 64)
    
    if model.training:
        fused_feat, aux_outputs = model([rgb_feat, ir_feat])
        # 计算一致性损失
        cons_loss = consistency_loss(
            rgb_feat, 
            aux_outputs['aligned_ir'],
            aux_outputs['consistency']
        )
    else:
        fused_feat = model([rgb_feat, ir_feat])