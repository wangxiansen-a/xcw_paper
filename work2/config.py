"""
基于LSTM的光学器件神经网络正向/逆向设计项目 - 配置文件
数据集2: 4维结构参数 → 200维吸收光谱
正向网络采用LSTM架构，逆向设计采用串联网络
"""
from pathlib import Path

# ============== 路径配置 ==============
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

PARAMS_FILE = BASE_DIR / "input2.0.csv"
SPECTRA_FILE = BASE_DIR / "output2.0.csv"

FORWARD_MODEL_PATH = CHECKPOINT_DIR / "forward_lstm.pth"
BACKWARD_MODEL_PATH = CHECKPOINT_DIR / "backward_net.pth"
PARAMS_NORMALIZER_PATH = CHECKPOINT_DIR / "params_normalizer.pkl"
SPECTRA_NORMALIZER_PATH = CHECKPOINT_DIR / "spectra_normalizer.pkl"
FORWARD_HISTORY_PATH = CHECKPOINT_DIR / "forward_history.npy"
TANDEM_HISTORY_PATH = CHECKPOINT_DIR / "tandem_history.npy"

# ============== 数据配置 ==============
INPUT_DIM = 4       # 结构参数维度（P, Lx, Ly, h）
OUTPUT_DIM = 200    # 吸收光谱维度
TEST_RATIO = 0.1    # 测试集比例
RANDOM_SEED = 42    # 随机种子

# ============== LSTM前向网络配置 ==============
# 参照论文: 4层LSTM, 隐藏层分别为10, 30, 50, 80个单元
FORWARD_CONFIG = {
    "input_dim": 4,          # 结构参数维度（序列长度）
    "output_dim": 200,       # 吸收光谱维度
    "lstm_hidden_sizes": [10, 30, 50, 80],  # 4层LSTM隐藏层大小
    "dropout": 0.1,
}

# ============== 后向网络配置（MLP） ==============
BACKWARD_CONFIG = {
    "input_dim": 200,
    "output_dim": 4,
    "hidden_dims": [256, 512, 512, 256, 128],
    "dropout": 0.1,
    "use_batch_norm": True,
}

# ============== LSTM前向网络训练配置 ==============
FORWARD_TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 2000,
    "learning_rate": 1e-2,       # 论文使用 10^-2
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
