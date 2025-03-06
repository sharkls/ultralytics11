# 在您的主模型中使用配准模块
class YourModel(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.registration = RegistrationModule(channel)
        # 其他模型组件...

    def forward(self, rgb_feat, ir_feat):
        # 获取配准结果
        reg_output = self.registration(rgb_feat, ir_feat)
        aligned_ir = reg_output['aligned_ir']
        
        # 使用对齐后的特征进行后续处理
        # ...

        if self.training:
            return final_output, reg_output
        return final_output

# 训练代码
def train_step(model, rgb_data, ir_data, targets, optimizer, detection_criterion, registration_criterion):
    optimizer.zero_grad()
    
    # 前向传播
    main_output, reg_output = model(rgb_data, ir_data)
    
    # 计算检测损失
    detection_loss = detection_criterion(main_output, targets)
    
    # 计算配准损失
    reg_loss, reg_losses = registration_criterion(reg_output)
    
    # 总损失
    total_loss = detection_loss + reg_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    return {
        'detection_loss': detection_loss.item(),
        'registration_loss': reg_loss.item(),
        **{k: v.item() for k, v in reg_losses.items()}
    }