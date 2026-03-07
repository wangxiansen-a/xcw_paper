"""
光学器件神经网络正向/逆向设计项目 - 配置文件
"""
from pathlib import Path

# ============== 路径配置 ==============
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

PARAMS_FILE = BASE_DIR / "input.csv"
SPECTRA_FILE = BASE_DIR / "output.csv"

FORWARD_MODEL_PATH = CHECKPOINT_DIR / "forward_net.pth"
BACKWARD_MODEL_PATH = CHECKPOINT_DIR / "backward_net.pth"
PARAMS_NORMALIZER_PATH = CHECKPOINT_DIR / "params_normalizer.pkl"
SPECTRA_NORMALIZER_PATH = CHECKPOINT_DIR / "spectra_normalizer.pkl"
FORWARD_HISTORY_PATH = CHECKPOINT_DIR / "forward_history.npy"
TANDEM_HISTORY_PATH = CHECKPOINT_DIR / "tandem_history.npy"

# ============== 数据配置 ==============
INPUT_DIM = 3       # 结构参数维度
OUTPUT_DIM = 200    # 吸收光谱维度
TEST_RATIO = 0.1    # 测试集比例
RANDOM_SEED = 42    # 随机种子

# ============== 前向网络配置 ==============
FORWARD_CONFIG = {
    "input_dim": 3,
    "output_dim": 200,
    "hidden_dims": [128, 256, 512, 512, 256],  # 5层隐藏层
    "dropout": 0.1,
    "use_batch_norm": True,
}

# ============== 后向网络配置 ==============
BACKWARD_CONFIG = {
    "input_dim": 200,
    "output_dim": 3,
    "hidden_dims": [256, 512, 512, 256, 128],  # 对称设计
    "dropout": 0.1,
    "use_batch_norm": True,
}

# ============== 前向网络训练配置 ==============
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

# ============== 串联网络训练配置 ==============
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

