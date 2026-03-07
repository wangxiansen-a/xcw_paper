"""
神经网络模型定义
正向网络: 基于LSTM（参照论文架构）
后向网络: MLP（串联网络逆向设计）
"""
import torch
import torch.nn as nn

from config import FORWARD_CONFIG, BACKWARD_CONFIG


class LSTMForwardNet(nn.Module):
    """
    基于LSTM的前向网络：结构参数 (4维) → 吸收光谱 (200维)

    参照论文架构:
    - 将4个结构参数视为长度为4的序列，每个时间步输入1个参数
    - 4层LSTM，隐藏层分别为 10, 30, 50, 80 个单元
    - 最后一层线性连接输出光谱

    架构: Input(4,1) → LSTM(1→10) → LSTM(10→30) → LSTM(30→50) → LSTM(50→80) → Linear(80→200) → Sigmoid
    """
    def __init__(self, config=None):
        super(LSTMForwardNet, self).__init__()

        if config is None:
            config = FORWARD_CONFIG

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.hidden_sizes = config["lstm_hidden_sizes"]
        dropout = config.get("dropout", 0.1)

        # 构建多层LSTM（每层不同hidden_size，需逐层构建）
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        prev_size = 1  # 每个时间步输入1个特征
        for hidden_size in self.hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(input_size=prev_size, hidden_size=hidden_size, batch_first=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size

        self.dropout = nn.Dropout(dropout)

        # 输出层: 取最后时间步的输出，线性映射到光谱维度
        self.fc_out = nn.Linear(self.hidden_sizes[-1], self.output_dim)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # 设置遗忘门偏置为1，有助于长期记忆
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        Args:
            x: 结构参数 (batch_size, 4)
        Returns:
            预测光谱 (batch_size, 200)
        """
        # 将参数reshape为序列: (batch, seq_len=4, feature=1)
        out = x.unsqueeze(-1)

        for lstm, ln in zip(self.lstm_layers, self.layer_norms):
            out, _ = lstm(out)
            out = ln(out)
            out = self.dropout(out)

        # 取最后一个时间步的输出
        last_output = out[:, -1, :]  # (batch, hidden_size[-1])

        # 线性映射到光谱维度
        output = self.fc_out(last_output)
        output = self.sigmoid(output)

        return output


class BackwardNet(nn.Module):
    """
    后向网络（MLP）：吸收光谱 (200维) → 结构参数 (4维)

    架构: Input(200) → [Linear→BN→LeakyReLU/Tanh→Dropout] × 5 → Linear(4)
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

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if i % 2 == 0:
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
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
    串联网络：组合后向网络（MLP）与前向网络（LSTM）

    流程: [目标光谱] → [后向网络MLP] → [预测参数] → [前向网络LSTM(冻结)] → [重建光谱]
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
            predicted_params: 预测的结构参数 (batch_size, 4)
            reconstructed_spectrum: 重建的吸收光谱 (batch_size, 200)
        """
        predicted_params = self.backward_net(target_spectrum)
        reconstructed_spectrum = self.forward_net(predicted_params)
        return predicted_params, reconstructed_spectrum

    def freeze_forward_net(self):
        """冻结前向网络参数（保持train模式，因为cuDNN LSTM不支持eval模式下的backward）"""
        for param in self.forward_net.parameters():
            param.requires_grad = False

    def unfreeze_forward_net(self):
        """解冻前向网络参数"""
        for param in self.forward_net.parameters():
            param.requires_grad = True


def count_parameters(model):
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 50)
    print("LSTM前向网络测试")
    forward_net = LSTMForwardNet()
    print(forward_net)
    print(f"可训练参数: {count_parameters(forward_net):,}")

    x = torch.randn(4, 4)
    y = forward_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")

    print("\n" + "=" * 50)
    print("后向网络测试")
    backward_net = BackwardNet()
    print(backward_net)
    print(f"可训练参数: {count_parameters(backward_net):,}")

    x = torch.randn(4, 200)
    y = backward_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")

    print("\n" + "=" * 50)
    print("串联网络测试")
    tandem_net = TandemNet(backward_net, forward_net, freeze_forward=True)
    print(f"后向网络可训练参数: {count_parameters(backward_net):,}")
    print(f"前向网络可训练参数: {count_parameters(forward_net):,}")

    spectrum = torch.randn(4, 200)
    params, recon = tandem_net(spectrum)
    print(f"输入光谱: {spectrum.shape}")
    print(f"预测参数: {params.shape}")
    print(f"重建光谱: {recon.shape}")
