"""
基于Mamba的光学器件神经网络正向/逆向设计项目 - 配置文件
数据集3: 6维结构参数 → 500维光谱
正向网络采用Mamba（选择性状态空间模型）架构，逆向设计采用串联网络
"""
from pathlib import Path

# ============== 路径配置 ==============
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

PARAMS_FILE = BASE_DIR / "input3.csv"
SPECTRA_FILE = BASE_DIR / "output3.csv"

FORWARD_MODEL_PATH = CHECKPOINT_DIR / "forward_mamba.pth"
BACKWARD_MODEL_PATH = CHECKPOINT_DIR / "backward_net.pth"
PARAMS_NORMALIZER_PATH = CHECKPOINT_DIR / "params_normalizer.pkl"
SPECTRA_NORMALIZER_PATH = CHECKPOINT_DIR / "spectra_normalizer.pkl"
FORWARD_HISTORY_PATH = CHECKPOINT_DIR / "forward_history.npy"
TANDEM_HISTORY_PATH = CHECKPOINT_DIR / "tandem_history.npy"

# ============== 数据配置 ==============
INPUT_DIM = 6       # 结构参数维度
OUTPUT_DIM = 500    # 光谱维度
TEST_RATIO = 0.1    # 测试集比例
RANDOM_SEED = 42    # 随机种子

# ============== Mamba前向网络配置 ==============
FORWARD_CONFIG = {
    "input_dim": 6,
    "output_dim": 500,
    "d_model": 64,          # 嵌入维度
    "d_state": 16,          # SSM 状态维度
    "d_conv": 4,            # 因果卷积核大小
    "expand": 2,            # 扩展因子 (d_inner = expand * d_model)
    "n_layers": 4,          # Mamba Block 层数
    "dropout": 0.1,
}

# ============== 后向网络配置（MLP） ==============
BACKWARD_CONFIG = {
    "input_dim": 500,
    "output_dim": 6,
    "hidden_dims": [256, 512, 512, 256, 128],
    "dropout": 0.1,
    "use_batch_norm": True,
}

# ============== Mamba前向网络训练配置 ==============
FORWARD_TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 2000,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_step_size": 400,
    "lr_gamma": 0.5,
    "early_stop_patience": 200,
    "print_every": 50,
}

# ============== 串联网络训练配置 ==============
TANDEM_TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 5000,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "lr_step_size": 500,
    "lr_gamma": 0.5,
    "early_stop_patience": 300,
    "print_every": 100,
}
