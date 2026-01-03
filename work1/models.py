"""
神经网络模型定义
"""
import torch
import torch.nn as nn

from config import FORWARD_CONFIG, BACKWARD_CONFIG


class ForwardNet(nn.Module):
    """
    前向网络：结构参数 (3维) → 吸收光谱 (200维)
    
    架构: Input(3) → [Linear→BN→LeakyReLU/Tanh→Dropout] × 5 → Linear(200) → Sigmoid
    """
    def __init__(self, config=None):
        super(ForwardNet, self).__init__()
        
        if config is None:
            config = FORWARD_CONFIG
        
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        hidden_dims = config["hidden_dims"]
        dropout = config["dropout"]
        use_batch_norm = config["use_batch_norm"]
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            # 交替激活函数：奇数层 LeakyReLU(0.2)，偶数层 Tanh
            if i % 2 == 0:
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层：使用 Sigmoid 约束输出到 [0, 1]
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier Uniform 权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class BackwardNet(nn.Module):
    """
    后向网络：吸收光谱 (200维) → 结构参数 (3维)
    
    架构: Input(200) → [Linear→BN→LeakyReLU/Tanh→Dropout] × 5 → Linear(3)
    注意：输出层无激活函数（参数经 Z-Score 标准化，可为任意实数）
    """
    def __init__(self, config=None):
        super(BackwardNet, self).__init__()
        
        if config is None:
            config = BACKWARD_CONFIG
        
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        hidden_dims = config["hidden_dims"]
        dropout = config["dropout"]
        use_batch_norm = config["use_batch_norm"]
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            # 交替激活函数
            if i % 2 == 0:
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层：无激活函数
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier Uniform 权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class TandemNet(nn.Module):
    """
    串联网络：组合后向网络与前向网络
    
    流程: [目标光谱] → [后向网络] → [预测参数] → [前向网络(冻结)] → [重建光谱]
    """
    def __init__(self, backward_net, forward_net, freeze_forward=True):
        super(TandemNet, self).__init__()
        
        self.backward_net = backward_net
        self.forward_net = forward_net
        
        if freeze_forward:
            self.freeze_forward_net()
    
    def forward(self, target_spectrum):
        """
        Args:
            target_spectrum: 目标吸收光谱 (batch_size, 200)
        
        Returns:
            predicted_params: 预测的结构参数 (batch_size, 3)
            reconstructed_spectrum: 重建的吸收光谱 (batch_size, 200)
        """
        predicted_params = self.backward_net(target_spectrum)
        reconstructed_spectrum = self.forward_net(predicted_params)
        return predicted_params, reconstructed_spectrum
    
    def freeze_forward_net(self):
        """冻结前向网络参数"""
        for param in self.forward_net.parameters():
            param.requires_grad = False
        self.forward_net.eval()
    
    def unfreeze_forward_net(self):
        """解冻前向网络参数"""
        for param in self.forward_net.parameters():
            param.requires_grad = True


def count_parameters(model):
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("=" * 50)
    print("前向网络测试")
    forward_net = ForwardNet()
    print(forward_net)
    print(f"可训练参数: {count_parameters(forward_net):,}")
    
    # 测试前向传播
    x = torch.randn(4, 3)
    y = forward_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    print("\n" + "=" * 50)
    print("后向网络测试")
    backward_net = BackwardNet()
    print(backward_net)
    print(f"可训练参数: {count_parameters(backward_net):,}")
    
    # 测试前向传播
    x = torch.randn(4, 200)
    y = backward_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    
    print("\n" + "=" * 50)
    print("串联网络测试")
    tandem_net = TandemNet(backward_net, forward_net, freeze_forward=True)
    print(f"后向网络可训练参数: {count_parameters(backward_net):,}")
    print(f"前向网络可训练参数: {count_parameters(forward_net):,}")
    
    # 测试前向传播
    spectrum = torch.randn(4, 200)
    params, recon = tandem_net(spectrum)
    print(f"输入光谱: {spectrum.shape}")
    print(f"预测参数: {params.shape}")
    print(f"重建光谱: {recon.shape}")

