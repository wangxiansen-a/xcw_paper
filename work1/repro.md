# 光学器件神经网络正向/逆向设计项目复现文档

## 1. 项目概述

### 1.1 研究背景

本项目实现了基于神经网络的光学器件正向设计与逆向设计系统。核心思想是利用深度学习建立**结构参数**与**吸收光谱**之间的映射关系。

### 1.2 核心任务

| 任务类型 | 输入 | 输出 | 维度变换 |
|---------|------|------|---------|
| 正向设计 | 结构参数 | 吸收光谱 | 3 → 200 |
| 逆向设计 | 吸收光谱 | 结构参数 | 200 → 3 |

### 1.3 技术方案

采用**串联网络(Tandem Network)** 架构解决逆向设计中的多对一问题：

```
[目标光谱] → [后向网络] → [预测参数] → [前向网络(冻结)] → [重建光谱]
                 ↑                                              ↓
                 └─────────── 损失反向传播 ←──────────────────┘
```

---

## 2. 环境配置

### 2.1 依赖安装

```bash
# Python >= 3.8
pip install torch>=1.10.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=0.24.0
pip install matplotlib>=3.4.0
pip install tqdm>=4.62.0  # 可选
```

或使用 `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2.2 项目结构

```
work1/
├── config.py           # 配置文件（超参数、路径）
├── data_loader.py      # 数据加载与预处理
├── models.py           # 神经网络模型定义
├── train_forward.py    # 前向网络训练
├── train_tandem.py     # 串联网络训练
├── evaluate.py         # 模型评估与可视化
├── main.py             # 主程序入口
├── utils.py            # 工具函数
├── requirements.txt    # 依赖列表
└── checkpoints/        # 模型保存目录
```

---

## 3. 数据格式与预处理

### 3.1 原始数据格式

数据文件位于 `paper_works/` 目录：

| 文件 | 说明 | 存储格式 |
|------|------|---------|
| `input.csv` | 结构参数 | 3行 × N列，每列一个样本 |
| `output.csv` | 吸收光谱 | 200行 × N列，每列一个样本 |

**关键**：数据按列存储，需转置后使用。

### 3.2 数据加载流程 (data_loader.py)

```python
def load_raw_data():
    # 加载并转置数据
    params = pd.read_csv(PARAMS_FILE, header=None).values[:3, :].T   # (N, 3)
    spectra = pd.read_csv(SPECTRA_FILE, header=None).values[:200, :].T  # (N, 200)
    return params.astype(np.float32), spectra.astype(np.float32)
```

### 3.3 数据标准化

使用两种不同的标准化方法：

#### 3.3.1 结构参数：Z-Score 标准化

```python
class ZScoreNormalizer:
    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std = np.where(self.std == 0, 1.0, self.std)  # 防止除零
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean
```

#### 3.3.2 吸收光谱：Min-Max 标准化

```python
class MinMaxNormalizer:
    def fit(self, data):
        self.min_val = np.min(data, axis=0, keepdims=True)
        self.max_val = np.max(data, axis=0, keepdims=True)
        self.range_val = np.where(self.max_val - self.min_val == 0, 1.0, 
                                   self.max_val - self.min_val)
    
    def transform(self, data):
        return (data - self.min_val) / self.range_val
    
    def inverse_transform(self, data):
        return data * self.range_val + self.min_val
```

### 3.4 数据集划分

```python
# 90% 训练，10% 测试
params_train, params_test, spectra_train, spectra_test = train_test_split(
    params, spectra, 
    test_size=0.1,      # TEST_RATIO
    random_state=42     # RANDOM_SEED
)
```

**关键**：标准化器必须只在训练集上 `fit()`，然后对测试集做 `transform()`。

---

## 4. 神经网络架构 (models.py)

### 4.1 前向网络 (ForwardNet)

**功能**：结构参数 (3维) → 吸收光谱 (200维)

```
Input(3) → [Linear→BN→LeakyReLU→Dropout] × 5 → Linear(200) → Sigmoid
```

**详细配置**：

```python
FORWARD_CONFIG = {
    "input_dim": 3,
    "output_dim": 200,
    "hidden_dims": [128, 256, 512, 512, 256],  # 5层隐藏层
    "dropout": 0.1,
    "use_batch_norm": True,
}
```

**架构细节**：

1. 隐藏层结构（每层）：
   - Linear 全连接层
   - BatchNorm1d（可选，默认启用）
   - 激活函数：奇数层 LeakyReLU(0.2)，偶数层 Tanh（交替使用）
   - Dropout (0.1)

2. 输出层：
   - Linear(256, 200)
   - Sigmoid（因为光谱经 Min-Max 归一化到 [0, 1]）

3. 权重初始化：Xavier Uniform

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

### 4.2 后向网络 (BackwardNet)

**功能**：吸收光谱 (200维) → 结构参数 (3维)

```
Input(200) → [Linear→BN→LeakyReLU/Tanh→Dropout] × 5 → Linear(3)
```

**详细配置**：

```python
BACKWARD_CONFIG = {
    "input_dim": 200,
    "output_dim": 3,
    "hidden_dims": [256, 512, 512, 256, 128],  # 对称设计
    "dropout": 0.1,
    "use_batch_norm": True,
}
```

**关键区别**：输出层无激活函数（参数经 Z-Score 标准化，可为任意实数）

### 4.3 串联网络 (TandemNet)

**功能**：组合后向网络与前向网络

```python
class TandemNet(nn.Module):
    def __init__(self, backward_net, forward_net, freeze_forward=True):
        self.backward_net = backward_net
        self.forward_net = forward_net
        if freeze_forward:
            self.freeze_forward_net()  # 冻结前向网络参数
    
    def forward(self, target_spectrum):
        predicted_params = self.backward_net(target_spectrum)
        reconstructed_spectrum = self.forward_net(predicted_params)
        return predicted_params, reconstructed_spectrum
    
    def freeze_forward_net(self):
        for param in self.forward_net.parameters():
            param.requires_grad = False
        self.forward_net.eval()
```

---

## 5. 训练流程

### 5.1 前向网络训练 (train_forward.py)

#### 5.1.1 训练配置

```python
FORWARD_TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 1000,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_step_size": 200,        # 每200 epoch 学习率衰减
    "lr_gamma": 0.5,            # 衰减因子
    "early_stop_patience": 100,
    "print_every": 50,
}
```

#### 5.1.2 训练器实现

```python
class ForwardTrainer:
    def __init__(self):
        self.model = ForwardNet().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=200,
            gamma=0.5
        )
```

#### 5.1.3 训练 Epoch

```python
def train_epoch(self, train_loader):
    self.model.train()
    for params, spectra in train_loader:
        self.optimizer.zero_grad()
        predicted_spectra = self.model(params)
        loss = self.criterion(predicted_spectra, spectra)
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
```

#### 5.1.4 评估方法

**关键细节**：训练损失也在 eval 模式下计算，以便与测试损失公平比较（排除 Dropout 影响）。

```python
@torch.no_grad()
def evaluate_train(self, train_loader):
    self.model.eval()  # eval 模式
    # ... 计算损失
```

#### 5.1.5 早停机制

```python
class EarlyStopping:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 触发早停
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False
```

### 5.2 串联网络训练 (train_tandem.py)

#### 5.2.1 训练配置

```python
TANDEM_TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 5000,             # 更多 epochs
    "learning_rate": 5e-4,      # 更小的学习率
    "weight_decay": 1e-5,
    "lr_step_size": 500,
    "lr_gamma": 0.5,
    "early_stop_patience": 200,
    "print_every": 100,
}
```

#### 5.2.2 核心训练逻辑

```python
class TandemTrainer:
    def __init__(self):
        # 加载预训练的前向网络（必须先训练完成）
        self.forward_net = self._load_pretrained_forward_net()
        # 创建后向网络
        self.backward_net = BackwardNet().to(device)
        # 创建串联网络，冻结前向网络
        self.tandem_net = TandemNet(
            self.backward_net, 
            self.forward_net,
            freeze_forward=True  # 关键：冻结前向网络
        )
        # 优化器只优化后向网络参数
        self.optimizer = optim.Adam(
            self.backward_net.parameters(),  # 注意：只有后向网络
            lr=5e-4,
            weight_decay=1e-5
        )
    
    def train_epoch(self, train_loader):
        self.backward_net.train()
        self.forward_net.eval()  # 前向网络始终 eval
        
        for target_spectra, _ in train_loader:
            self.optimizer.zero_grad()
            # 串联前向传播
            predicted_params, reconstructed_spectra = self.tandem_net(target_spectra)
            # 损失：重建光谱与目标光谱的 MSE
            loss = self.criterion(reconstructed_spectra, target_spectra)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backward_net.parameters(), max_norm=1.0)
            self.optimizer.step()
```

#### 5.2.3 数据加载器

串联网络训练时，输入和目标都是光谱：

```python
def create_tandem_dataloaders(train_data, test_data, batch_size):
    spectra_train = torch.from_numpy(train_data['spectra']).float()
    # 输入和目标都是光谱
    train_dataset = TensorDataset(spectra_train, spectra_train)
    # ...
```

---

## 6. 模型评估 (evaluate.py)

### 6.1 评估流程

```python
class Evaluator:
    def __init__(self):
        # 加载训练好的模型
        self.forward_net = self._load_forward_net()
        self.backward_net = self._load_backward_net()
        self.tandem_net = TandemNet(
            self.backward_net, self.forward_net, freeze_forward=True
        )
        self.normalizers = load_normalizers()
```

### 6.2 前向网络评估

```python
@torch.no_grad()
def evaluate_forward(self, test_data):
    params = torch.from_numpy(test_data['params']).to(device)
    spectra_true = torch.from_numpy(test_data['spectra']).to(device)
    spectra_pred = self.forward_net(params)
    
    mse = torch.mean((spectra_pred - spectra_true) ** 2).item()
    mae = torch.mean(torch.abs(spectra_pred - spectra_true)).item()
    return {'mse': mse, 'mae': mae, ...}
```

### 6.3 逆向设计评估

```python
@torch.no_grad()
def evaluate_inverse(self, test_data):
    spectra = torch.from_numpy(test_data['spectra']).to(device)
    params_pred, spectra_recon = self.tandem_net(spectra)
    
    # 核心指标：光谱重建 MSE
    spectra_mse = torch.mean((spectra_recon - spectra) ** 2).item()
    return {'spectra_mse': spectra_mse, ...}
```

### 6.4 逆向设计推理

```python
def inverse_design(self, target_spectrum):
    # 输入：归一化后的光谱
    spectrum_tensor = torch.from_numpy(target_spectrum).float().to(device)
    
    with torch.no_grad():
        params_pred_norm, spectra_recon = self.tandem_net(spectrum_tensor)
    
    # 反标准化得到真实参数值
    params_pred_real = self.normalizers['params'].inverse_transform(
        params_pred_norm.cpu().numpy()
    )
    return {'params_real': params_pred_real, ...}
```

---

## 7. 运行指南

### 7.1 完整训练流程

```bash
# 方式1：运行 main.py（推荐，包含训练和评估）
python main.py

# 方式2：分步运行
python train_forward.py    # 先训练前向网络
python train_tandem.py     # 再训练串联网络
python evaluate.py         # 评估
```

### 7.2 命令行参数

```bash
python main.py              # 完整流程（训练+评估）
python main.py --train      # 仅训练
python main.py --evaluate   # 仅评估
python main.py --demo       # 逆向设计演示
```

### 7.3 输出文件

训练完成后，`checkpoints/` 目录包含：

| 文件 | 说明 |
|-----|------|
| `forward_net.pth` | 前向网络模型权重 |
| `backward_net.pth` | 后向网络模型权重 |
| `params_normalizer.pkl` | 参数标准化器 |
| `spectra_normalizer.pkl` | 光谱标准化器 |
| `forward_history.npy` | 前向网络训练历史 |
| `tandem_history.npy` | 串联网络训练历史 |

---

## 8. 关键实现细节

### 8.1 激活函数交替使用

隐藏层交替使用 LeakyReLU 和 Tanh：

```python
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
```

### 8.2 前向网络输出层

使用 Sigmoid 约束输出到 [0, 1]（与光谱 Min-Max 归一化匹配）：

```python
# 输出层
layers.append(nn.Linear(prev_dim, output_dim))
layers.append(nn.Sigmoid())  # 关键
```

### 8.3 后向网络输出层

无激活函数（参数 Z-Score 标准化后可为任意实数）：

```python
# 输出层（无激活函数）
layers.append(nn.Linear(prev_dim, output_dim))
```

### 8.4 梯度裁剪

防止梯度爆炸，两个训练器都使用：

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

### 8.5 设备支持

自动检测并使用可用设备：

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")
```

### 8.6 可重复性

设置随机种子确保结果可复现：

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## 9. 超参数汇总

### 9.1 数据配置

| 参数 | 值 | 说明 |
|-----|-----|------|
| INPUT_DIM | 3 | 结构参数维度 |
| OUTPUT_DIM | 200 | 吸收光谱维度 |
| TEST_RATIO | 0.1 | 测试集比例 |
| RANDOM_SEED | 42 | 随机种子 |

### 9.2 网络架构

| 网络 | 隐藏层维度 | Dropout | BatchNorm |
|-----|-----------|---------|-----------|
| ForwardNet | [128, 256, 512, 512, 256] | 0.1 | True |
| BackwardNet | [256, 512, 512, 256, 128] | 0.1 | True |

### 9.3 训练配置

| 参数 | 前向网络 | 串联网络 |
|-----|---------|---------|
| batch_size | 32 | 32 |
| epochs | 1000 | 5000 |
| learning_rate | 1e-3 | 5e-4 |
| weight_decay | 1e-5 | 1e-5 |
| lr_step_size | 200 | 500 |
| lr_gamma | 0.5 | 0.5 |
| early_stop_patience | 100 | 200 |

---

## 10. 模型保存格式

### 10.1 模型检查点

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_loss': best_loss,
    'config': config,  # 网络配置
}
torch.save(checkpoint, path)
```

### 10.2 标准化器

```python
# 保存
with open(path, 'wb') as f:
    pickle.dump({'mean': self.mean, 'std': self.std}, f)

# 加载
with open(path, 'rb') as f:
    data = pickle.load(f)
    self.mean = data['mean']
    self.std = data['std']
```

---

## 11. 训练历史

训练历史以字典形式保存为 `.npy` 文件：

```python
history = {
    'train_loss': [],  # 每个 epoch 的训练损失
    'test_loss': [],   # 每个 epoch 的测试损失
    'lr': []           # 每个 epoch 的学习率
}
np.save(history_path, history)

# 加载
history = np.load(history_path, allow_pickle=True).item()
```

---

## 12. 常见问题

### Q1: 为什么用串联网络而不是直接训练逆向网络？

逆向设计存在**多对一问题**：多组不同的结构参数可能产生相同的光谱。直接用 MSE 训练会导致网络学习到"平均"参数。串联网络通过重建光谱作为损失目标，绕过了这个问题。

### Q2: 为什么前向网络用 Sigmoid，后向网络无激活？

- 前向网络输出光谱，经 Min-Max 归一化后在 [0, 1] 范围，Sigmoid 自然匹配
- 后向网络输出参数，经 Z-Score 标准化后可为任意实数，无需约束

### Q3: 为什么训练损失也在 eval 模式下计算？

为了与测试损失**公平比较**。训练模式下 Dropout 和 BatchNorm 行为不同，会导致训练损失偏高。

### Q4: 标准化器必须保存吗？

**必须**。推理时需要：
1. 对输入光谱做 Min-Max 标准化
2. 对预测参数做 Z-Score 反标准化

---

## 13. 扩展建议

1. **数据增强**：对光谱添加噪声提高泛化能力
2. **网络架构**：可尝试 ResNet、Transformer 等结构
3. **损失函数**：结合物理约束或频域损失
4. **多目标优化**：同时优化重建误差和参数合理性

---

## 14. 代码依赖关系图

```
main.py
  ├── train_forward.py
  │     ├── config.py
  │     ├── data_loader.py
  │     └── models.py (ForwardNet)
  │
  ├── train_tandem.py
  │     ├── config.py
  │     ├── data_loader.py
  │     └── models.py (BackwardNet, TandemNet)
  │
  └── evaluate.py
        ├── config.py
        ├── data_loader.py
        └── models.py (ForwardNet, BackwardNet, TandemNet)
```

---

## 附录：完整配置文件参考 (config.py)

```python
# 路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

PARAMS_FILE = DATA_DIR / "input.csv"
SPECTRA_FILE = DATA_DIR / "output.csv"

FORWARD_MODEL_PATH = CHECKPOINT_DIR / "forward_net.pth"
BACKWARD_MODEL_PATH = CHECKPOINT_DIR / "backward_net.pth"

# 数据配置
INPUT_DIM = 3
OUTPUT_DIM = 200
TEST_RATIO = 0.1
RANDOM_SEED = 42

# 前向网络配置
FORWARD_CONFIG = {
    "input_dim": 3,
    "output_dim": 200,
    "hidden_dims": [128, 256, 512, 512, 256],
    "dropout": 0.1,
    "use_batch_norm": True,
}

# 后向网络配置
BACKWARD_CONFIG = {
    "input_dim": 200,
    "output_dim": 3,
    "hidden_dims": [256, 512, 512, 256, 128],
    "dropout": 0.1,
    "use_batch_norm": True,
}

# 前向网络训练配置
FORWARD_TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 1000,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_step_size": 200,
    "lr_gamma": 0.5,
    "early_stop_patience": 100,
    "print_every": 50,
}

# 串联网络训练配置
TANDEM_TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 5000,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "lr_step_size": 500,
    "lr_gamma": 0.5,
    "early_stop_patience": 200,
    "print_every": 100,
}
```

