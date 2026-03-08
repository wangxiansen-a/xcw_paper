"""
神经网络模型定义
正向网络: 基于双向 Mamba（Bi-Mamba，选择性状态空间模型）
后向网络: 基于双向 Mamba（与正向网络对称）

Mamba 架构参考:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
双向 Mamba 参考:
  Zhu et al., "Vision Mamba: Efficient Visual Representation Learning
              with Bidirectional State Space Model", 2024

改进要点:
  1. 并行关联扫描替代顺序 for 循环，大幅提升 GPU 运算效率
  2. 双向 Mamba Block 替代单向，消除因果性与物理任务的逻辑错位
  3. 全局平均池化替代取最后时间步，避免信息瓶颈
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FORWARD_CONFIG, BACKWARD_CONFIG


# =============================================================================
# 并行关联扫描（Parallel Associative Scan）
# =============================================================================

def parallel_scan(gates, tokens):
    """
    并行关联扫描，用于高效计算一阶线性递推:
        h[t] = gates[t] * h[t-1] + tokens[t],  h[-1] = 0

    使用 Hillis-Steele 风格的递归加倍算法:
      - 计算复杂度: O(L log L)
      - 深度（并行步数）: O(log L)

    相比顺序 for 循环（O(L) 串行步），在 GPU 上可获得显著加速，
    尤其对于后向网络的 500 长序列（9 步 vs 500 步）。

    算法原理:
      关联操作 (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)
      通过递归加倍，每步将每个位置的"可见范围"翻倍。

    Args:
        gates:  (B, L, D, N) - 乘法系数（离散化后的 A_bar）
        tokens: (B, L, D, N) - 加法项（B_bar * x）
    Returns:
        h:      (B, L, D, N) - 扫描结果
    """
    T = gates.shape[1]
    num_steps = int(math.ceil(math.log2(max(T, 2))))

    for i in range(num_steps):
        stride = 2 ** i
        if stride >= T:
            break

        # 构建移位后的版本（对 dim=1 左填充 stride 个 identity 元素）
        # F.pad 从最后一维向前指定: (N_left, N_right, D_left, D_right, L_left, L_right)
        padding = [0, 0, 0, 0, stride, 0]
        shifted_gates = F.pad(gates[:, :-stride], padding, value=1.0)
        shifted_tokens = F.pad(tokens[:, :-stride], padding, value=0.0)

        # 关联操作: (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)
        tokens = gates * shifted_tokens + tokens
        gates = gates * shifted_gates

    return tokens


# =============================================================================
# Selective SSM（选择性状态空间模型）
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型

    核心公式:
        h(t) = A_bar * h(t-1) + B_bar * x(t)
        y(t) = C(t) * h(t)

    其中 B, C, dt 是输入依赖的（选择性机制），A 是固定参数。
    使用并行关联扫描替代顺序 for 循环，大幅提升 GPU 运算效率。
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

        # 离散化
        # dt: (B, L, d_inner) → (B, L, d_inner, 1)
        # A: (d_inner, d_state) → (1, 1, d_inner, d_state)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, d_state)
        A_bar = torch.exp(dt_A)  # 离散化后的 A

        # B_bar = dt * B（欧拉近似）
        # dt: (B, L, d_inner, 1), B: (B, L, 1, d_state)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # tokens = B_bar * x（扫描的加法项）
        # x: (B, L, d_inner) → (B, L, d_inner, 1)
        scan_tokens = B_bar * x.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # 并行关联扫描（替代顺序 for 循环）
        h = parallel_scan(A_bar, scan_tokens)  # (B, L, d_inner, d_state)

        # 输出: y(t) = C(t) * h(t)，对 d_state 维求和
        # C: (B, L, d_state) → (B, L, 1, d_state)
        y = torch.sum(h * C.unsqueeze(2), dim=-1)  # (B, L, d_inner)

        # 残差连接: y = y + D * x
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x

        return y


# =============================================================================
# 双向 Mamba Block（Bidirectional Mamba Block）
# =============================================================================

class BiMambaBlock(nn.Module):
    """
    双向 Mamba Block

    解决标准（单向因果）Mamba 与物理回归任务的逻辑错位:
      - 结构参数（如宽度、高度、折射率）之间没有严格时序依赖
      - 光谱（频率/波长）需要双向信息交互

    前向和后向分支共享输入投影，各自拥有独立的卷积层和 SSM，
    输出通过逐元素相加合并后经门控机制输出。

    结构:
        Input
          │
        LayerNorm
          │
        Linear(D → 2*E)  → 分成 x 和 z 两个分支
          │                    │
          ├── 前向分支          │
          │   Conv1d → SiLU    │
          │   → SSM_fwd        │
          │                    │
          ├── 后向分支          │
          │   Flip → Conv1d    │
          │   → SiLU → SSM_bwd│
          │   → Flip           │
          │                    │
          + (前向 + 后向)      SiLU (门控)
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

        # 共享输入投影: D → 2*E（拆分为 x 分支和 z 门控分支）
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 前向分支: 深度可分离因果卷积 + SSM
        self.conv1d_fwd = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
        )
        self.ssm_fwd = SelectiveSSM(self.d_inner, d_state)

        # 后向分支: 深度可分离因果卷积 + SSM（对翻转序列操作）
        self.conv1d_bwd = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
        )
        self.ssm_bwd = SelectiveSSM(self.d_inner, d_state)

        # 输出投影: E → D（前向+后向通过相加合并，维度不变）
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

        # 投影并拆分为 x 分支和 z 门控分支
        xz = self.in_proj(x)  # (B, L, 2*E)
        x_branch, z = xz.chunk(2, dim=-1)  # 各 (B, L, E)

        # ---- 前向分支: Conv1d → SiLU → SSM ----
        x_fwd = x_branch.transpose(1, 2)  # (B, E, L)
        x_fwd = self.conv1d_fwd(x_fwd)[:, :, :seq_len]  # 因果截断
        x_fwd = x_fwd.transpose(1, 2)  # (B, L, E)
        x_fwd = F.silu(x_fwd)
        x_fwd = self.ssm_fwd(x_fwd)

        # ---- 后向分支: Flip → Conv1d → SiLU → SSM → Flip ----
        x_bwd = x_branch.flip(1)  # 翻转序列（右→左）
        x_bwd = x_bwd.transpose(1, 2)
        x_bwd = self.conv1d_bwd(x_bwd)[:, :, :seq_len]
        x_bwd = x_bwd.transpose(1, 2)
        x_bwd = F.silu(x_bwd)
        x_bwd = self.ssm_bwd(x_bwd)
        x_bwd = x_bwd.flip(1)  # 翻转回原始顺序

        # 合并两个方向（逐元素相加）
        x_merged = x_fwd + x_bwd

        # z 门控（类 GLU 机制）
        z = F.silu(z)
        output = x_merged * z

        # 输出投影 + 残差连接
        output = self.out_proj(output)
        output = self.dropout(output)

        return output + residual


# =============================================================================
# Mamba 前向网络
# =============================================================================

class MambaForwardNet(nn.Module):
    """
    基于双向 Mamba 的前向网络：结构参数 (6维) → 光谱 (500维)

    将 6 个结构参数视为长度为 6 的序列，每个时间步输入 1 个参数，
    嵌入到 d_model 维后通过 N 个双向 Mamba Block。
    使用全局平均池化（替代取最后时间步）提取全序列特征，
    避免信息瓶颈，线性映射到光谱维度。

    架构: Input(6,1) → Embed(1→D) → [BiMambaBlock]×N → Norm → MeanPool → Linear(D→500) → Sigmoid
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

        # 双向 Mamba Block 堆叠
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
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

        # 通过双向 Mamba Block 堆叠
        for layer in self.layers:
            x = layer(x)

        # 全局平均池化（替代 x[:, -1, :]，避免信息瓶颈）
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, d_model)

        # 映射到光谱维度
        output = self.head(x)
        output = self.sigmoid(output)

        return output


# =============================================================================
# Mamba 后向网络
# =============================================================================

class MambaBackwardNet(nn.Module):
    """
    基于双向 Mamba 的后向网络：光谱 (500维) → 结构参数 (6维)

    与前向网络对称：
    将 500 维光谱视为长度为 500 的序列，每个时间步输入 1 个值，
    嵌入到 d_model 维后通过 N 个双向 Mamba Block。
    使用全局平均池化（替代取最后时间步）提取全序列特征，
    避免 500 维信息被压缩到单个 Token 的严重瓶颈。
    输出层无激活函数（参数经 Z-Score 标准化，可为任意实数）。

    架构: Input(500,1) → Embed(1→D) → [BiMambaBlock]×N → Norm → MeanPool → Linear(D→6)
    """
    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = BACKWARD_CONFIG

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        d_model = config["d_model"]
        d_state = config["d_state"]
        d_conv = config["d_conv"]
        expand = config["expand"]
        n_layers = config["n_layers"]
        dropout = config.get("dropout", 0.1)

        # 输入嵌入: 将每个标量嵌入到 d_model 维
        self.embedding = nn.Linear(1, d_model)

        # 双向 Mamba Block 堆叠
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        # 输出层（无 Sigmoid，参数可为任意实数）
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.output_dim)

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
            x: 光谱 (batch_size, 500)
        Returns:
            预测参数 (batch_size, 6)
        """
        # (B, 500) → (B, 500, 1) → (B, 500, d_model)
        x = x.unsqueeze(-1)
        x = self.embedding(x)

        # 通过双向 Mamba Block 堆叠
        for layer in self.layers:
            x = layer(x)

        # 全局平均池化（替代 x[:, -1, :]，避免信息瓶颈）
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, d_model)

        # 映射到参数维度（无激活函数）
        output = self.head(x)

        return output


# =============================================================================
# 串联网络
# =============================================================================

class TandemNet(nn.Module):
    """
    串联网络：组合后向网络（Bi-Mamba）与前向网络（Bi-Mamba）

    流程: [目标光谱] → [后向网络Bi-Mamba] → [预测参数] → [前向网络Bi-Mamba(冻结)] → [重建光谱]

    注: 冻结前向网络后，梯度仍会穿透前向网络回传给 predicted_params，
    从而指导后向网络更新（TBA 标准范式）。
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
    print("Bi-Mamba前向网络测试")
    forward_net = MambaForwardNet()
    print(forward_net)
    print(f"可训练参数: {count_parameters(forward_net):,}")

    x = torch.randn(4, 6)
    y = forward_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")

    print("\n" + "=" * 50)
    print("Bi-Mamba后向网络测试")
    backward_net = MambaBackwardNet()
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
