"""
神经网络模型定义
正向网络: 基于 Mamba（选择性状态空间模型）
后向网络: MLP（串联网络逆向设计）

Mamba 架构参考:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FORWARD_CONFIG, BACKWARD_CONFIG


# =============================================================================
# Selective SSM（选择性状态空间模型）
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型

    核心公式:
        h(t) = A_bar * h(t-1) + B_bar * x(t)
        y(t) = C(t) * h(t)

    其中 B, C, dt 是输入依赖的（选择性机制），A 是固定参数
    """
    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)

        # A: 固定参数，使用 log-spaced 初始化（类 HiPPO）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # D: 残差连接参数
        self.D = nn.Parameter(torch.ones(d_inner))

        # 输入依赖的投影: x → dt, B, C
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt 投影: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        # dt 偏置初始化: 使初始 dt 在合理范围内
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_inner)
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        # 从输入计算 dt, B, C（选择性机制的核心）
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt: (B, L, dt_rank) → (B, L, d_inner)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)  # 确保 dt > 0

        # A: 恢复为负数（保证稳定性）
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # 选择性扫描（sequential scan）
        y = self._selective_scan(x, dt, A, B, C)

        # 残差连接: y = y + D * x
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x

        return y

    def _selective_scan(self, x, dt, A, B, C):
        """
        选择性扫描（顺序实现）

        离散化:
            A_bar = exp(dt * A)
            B_bar = dt * B

        递推:
            h(t) = A_bar(t) * h(t-1) + B_bar(t) * x(t)
            y(t) = C(t) * h(t)

        Args:
            x: (B, L, d_inner) - 输入
            dt: (B, L, d_inner) - 离散化步长
            A: (d_inner, d_state) - 状态转移矩阵（固定，负值）
            B: (B, L, d_state) - 输入矩阵（输入依赖）
            C: (B, L, d_state) - 输出矩阵（输入依赖）
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # 离散化: A_bar = exp(dt * A)
        # dt: (B, L, d_inner) → (B, L, d_inner, 1)
        # A: (d_inner, d_state) → (1, 1, d_inner, d_state)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, d_state)
        A_bar = torch.exp(dt_A)

        # B_bar = dt * B
        # dt: (B, L, d_inner) → (B, L, d_inner, 1)
        # B: (B, L, d_state) → (B, L, 1, d_state)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # 顺序扫描
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            # h(t) = A_bar(t) * h(t-1) + B_bar(t) * x(t)
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y(t) = C(t) * h(t), 对 d_state 维求和
            y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y


# =============================================================================
# Mamba Block
# =============================================================================

class MambaBlock(nn.Module):
    """
    标准 Mamba Block

    结构:
        Input
          │
        LayerNorm
          │
        Linear(D → 2*E)  → 分成 x 和 z 两个分支
          │                    │
        Conv1d(E, k=4)       SiLU (门控分支)
          │                    │
        SiLU                   │
          │                    │
        Selective SSM          │
          │                    │
          × ←──────────────────┘ (逐元素相乘)
          │
        Linear(E → D)
          │
          + Residual
          │
        Output
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)

        # 输入投影: D → 2*E（一次投影，拆分为 x 和 z 分支）
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 深度可分离因果卷积（捕获局部上下文，弥补 SSM 因果设置下的短板）
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # 深度可分离
        )

        # 选择性 SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state)

        # 输出投影: E → D
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        residual = x

        x = self.norm(x)

        # 投影并拆分为两个分支
        xz = self.in_proj(x)  # (B, L, 2*E)
        x_branch, z = xz.chunk(2, dim=-1)  # 各 (B, L, E)

        # x 分支: Conv1d → SiLU → SSM
        x_branch = x_branch.transpose(1, 2)  # (B, E, L) for Conv1d
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]  # 因果：截断到原始长度
        x_branch = x_branch.transpose(1, 2)  # (B, L, E)
        x_branch = F.silu(x_branch)
        x_branch = self.ssm(x_branch)

        # z 分支: SiLU 门控
        z = F.silu(z)

        # 逐元素相乘（门控机制，类 GLU）
        output = x_branch * z

        # 输出投影
        output = self.out_proj(output)
        output = self.dropout(output)

        # 残差连接
        return output + residual


# =============================================================================
# Mamba 前向网络
# =============================================================================

class MambaForwardNet(nn.Module):
    """
    基于 Mamba 的前向网络：结构参数 (6维) → 光谱 (500维)

    将 6 个结构参数视为长度为 6 的序列，每个时间步输入 1 个参数，
    嵌入到 d_model 维后通过 N 个 Mamba Block，取最后时间步的输出
    线性映射到光谱维度。

    架构: Input(6,1) → Embedding(1→D) → [MambaBlock]×N → Norm → LastStep → Linear(D→500) → Sigmoid
    """
    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = FORWARD_CONFIG

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        d_model = config["d_model"]
        d_state = config["d_state"]
        d_conv = config["d_conv"]
        expand = config["expand"]
        n_layers = config["n_layers"]
        dropout = config.get("dropout", 0.1)

        # 输入嵌入: 将每个标量参数嵌入到 d_model 维
        self.embedding = nn.Linear(1, d_model)

        # Mamba Block 堆叠
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.output_dim)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Args:
            x: 结构参数 (batch_size, 6)
        Returns:
            预测光谱 (batch_size, 500)
        """
        # (B, 6) → (B, 6, 1) → (B, 6, d_model)
        x = x.unsqueeze(-1)
        x = self.embedding(x)

        # 通过 Mamba Block 堆叠
        for layer in self.layers:
            x = layer(x)

        # 取最后一个时间步
        x = self.norm(x)
        x = x[:, -1, :]  # (B, d_model)

        # 映射到光谱维度
        output = self.head(x)
        output = self.sigmoid(output)

        return output


# =============================================================================
# MLP 后向网络
# =============================================================================

class BackwardNet(nn.Module):
    """
    后向网络（MLP）：光谱 (500维) → 结构参数 (6维)

    架构: Input(500) → [Linear→BN→LeakyReLU/Tanh→Dropout] × 5 → Linear(6)
    """
    def __init__(self, config=None):
        super().__init__()

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# 串联网络
# =============================================================================

class TandemNet(nn.Module):
    """
    串联网络：组合后向网络（MLP）与前向网络（Mamba）

    流程: [目标光谱] → [后向网络MLP] → [预测参数] → [前向网络Mamba(冻结)] → [重建光谱]
    """
    def __init__(self, backward_net, forward_net, freeze_forward=True):
        super().__init__()

        self.backward_net = backward_net
        self.forward_net = forward_net

        if freeze_forward:
            self.freeze_forward_net()

    def forward(self, target_spectrum):
        predicted_params = self.backward_net(target_spectrum)
        reconstructed_spectrum = self.forward_net(predicted_params)
        return predicted_params, reconstructed_spectrum

    def freeze_forward_net(self):
        """冻结前向网络参数（不调用eval，避免影响反向传播）"""
        for param in self.forward_net.parameters():
            param.requires_grad = False

    def unfreeze_forward_net(self):
        for param in self.forward_net.parameters():
            param.requires_grad = True


def count_parameters(model):
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 50)
    print("Mamba前向网络测试")
    forward_net = MambaForwardNet()
    print(forward_net)
    print(f"可训练参数: {count_parameters(forward_net):,}")

    x = torch.randn(4, 6)
    y = forward_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")

    print("\n" + "=" * 50)
    print("后向网络测试")
    backward_net = BackwardNet()
    print(f"可训练参数: {count_parameters(backward_net):,}")

    x = torch.randn(4, 500)
    y = backward_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")

    print("\n" + "=" * 50)
    print("串联网络测试")
    tandem_net = TandemNet(backward_net, forward_net, freeze_forward=True)
    print(f"后向网络可训练参数: {count_parameters(backward_net):,}")
    print(f"前向网络可训练参数: {count_parameters(forward_net):,}")

    spectrum = torch.randn(4, 500)
    params, recon = tandem_net(spectrum)
    print(f"输入光谱: {spectrum.shape}")
    print(f"预测参数: {params.shape}")
    print(f"重建光谱: {recon.shape}")
