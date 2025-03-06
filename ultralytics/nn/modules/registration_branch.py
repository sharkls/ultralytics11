"""
Registration branch and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
from .registration import FeatureMatching, SpatialTransformer

class RegistrationBranch(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            Conv(channel, channel//2, 1),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
            Conv(channel//2, channel//4, 3),
            nn.BatchNorm2d(channel//4),
            nn.ReLU()
        )
        
        # 配准质量评估器
        self.quality_estimator = nn.Sequential(
            Conv(channel//2, channel//4, 3),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(),
            Conv(channel//4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, ir_feat, aligned_ir):
        rgb_desc = self.feature_extractor(rgb_feat)
        ir_desc = self.feature_extractor(ir_feat)
        aligned_desc = self.feature_extractor(aligned_ir)
        
        return {
            'rgb_desc': rgb_desc,
            'ir_desc': ir_desc,
            'aligned_desc': aligned_desc,
            'quality_map': self.quality_estimator(torch.cat([rgb_desc, aligned_desc], dim=1))
        }

class RegistrationModule(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.feature_matching = FeatureMatching(channel)
        self.spatial_transformer = SpatialTransformer()
        self.registration_branch = RegistrationBranch(channel)

    def forward(self, rgb_feat, ir_feat):
        # 特征匹配和空间对齐
        transform_matrix, rgb_desc, ir_desc = self.feature_matching(rgb_feat, ir_feat)
        self.spatial_transformer.update_transform(transform_matrix)
        aligned_ir = self.spatial_transformer(ir_feat)
        
        # 配准分支处理
        reg_output = self.registration_branch(rgb_feat, ir_feat, aligned_ir)
        reg_output.update({
            'transform_matrix': transform_matrix,
            'aligned_ir': aligned_ir
        })
        
        return reg_output

class RegistrationLoss(nn.Module):
    def __init__(self, weights={'desc': 1.0, 'smooth': 0.5, 'quality': 0.3}):
        super().__init__()
        self.weights = weights
        
    def descriptor_loss(self, rgb_desc, ir_desc, aligned_desc):
        pos_loss = F.mse_loss(rgb_desc, aligned_desc)
        neg_loss = torch.max(torch.zeros_like(pos_loss),
                           0.5 - F.mse_loss(rgb_desc, ir_desc))
        return pos_loss + neg_loss
    
    def smoothness_loss(self, transform_matrix):
        identity = torch.eye(3, device=transform_matrix.device)
        return F.mse_loss(transform_matrix, identity.expand_as(transform_matrix))
    
    def quality_loss(self, quality_map, aligned_desc, rgb_desc):
        desc_diff = F.mse_loss(aligned_desc, rgb_desc, reduction='none').mean(1, keepdim=True)
        return F.mse_loss(quality_map, (-desc_diff).exp())
    
    def forward(self, reg_output):
        losses = {
            'desc_loss': self.weights['desc'] * self.descriptor_loss(
                reg_output['rgb_desc'],
                reg_output['ir_desc'],
                reg_output['aligned_desc']
            ),
            'smooth_loss': self.weights['smooth'] * self.smoothness_loss(
                reg_output['transform_matrix']
            ),
            'quality_loss': self.weights['quality'] * self.quality_loss(
                reg_output['quality_map'],
                reg_output['aligned_desc'],
                reg_output['rgb_desc']
            )
        }
        return sum(losses.values()), losses