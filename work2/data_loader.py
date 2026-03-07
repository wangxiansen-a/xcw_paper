"""
数据加载与预处理
数据集2: 4维结构参数, 200维吸收光谱, 约6129组数据
"""
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from config import (
    PARAMS_FILE, SPECTRA_FILE, INPUT_DIM, OUTPUT_DIM,
    TEST_RATIO, RANDOM_SEED, PARAMS_NORMALIZER_PATH,
    SPECTRA_NORMALIZER_PATH, CHECKPOINT_DIR
)


class ZScoreNormalizer:
    """Z-Score 标准化器（用于结构参数）"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std = np.where(self.std == 0, 1.0, self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, path):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mean = data['mean']
            self.std = data['std']


class MinMaxNormalizer:
    """Min-Max 标准化器（用于吸收光谱）"""
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.range_val = None

    def fit(self, data):
        self.min_val = np.min(data, axis=0, keepdims=True)
        self.max_val = np.max(data, axis=0, keepdims=True)
        self.range_val = self.max_val - self.min_val
        self.range_val = np.where(self.range_val == 0, 1.0, self.range_val)

    def transform(self, data):
        return (data - self.min_val) / self.range_val

    def inverse_transform(self, data):
        return data * self.range_val + self.min_val

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, path):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'min_val': self.min_val,
                'max_val': self.max_val,
                'range_val': self.range_val
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.min_val = data['min_val']
            self.max_val = data['max_val']
            self.range_val = data['range_val']


def load_raw_data():
    """
    加载原始数据
    数据按列存储，需转置后使用
    """
    params = pd.read_csv(PARAMS_FILE, header=None).values[:INPUT_DIM, :].T    # (N, 4)
    spectra = pd.read_csv(SPECTRA_FILE, header=None).values[:OUTPUT_DIM, :].T  # (N, 200)

    return params.astype(np.float32), spectra.astype(np.float32)


def prepare_data(save_normalizers=True):
    """
    准备训练和测试数据

    返回:
        train_data: 包含标准化后的训练数据
        test_data: 包含标准化后的测试数据
        normalizers: 标准化器字典
    """
    params, spectra = load_raw_data()
    print(f"数据加载完成: params shape = {params.shape}, spectra shape = {spectra.shape}")

    params_train, params_test, spectra_train, spectra_test = train_test_split(
        params, spectra,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED
    )
    print(f"数据划分完成: 训练集 {len(params_train)} 样本, 测试集 {len(params_test)} 样本")

    params_normalizer = ZScoreNormalizer()
    spectra_normalizer = MinMaxNormalizer()

    params_train_norm = params_normalizer.fit_transform(params_train)
    spectra_train_norm = spectra_normalizer.fit_transform(spectra_train)

    params_test_norm = params_normalizer.transform(params_test)
    spectra_test_norm = spectra_normalizer.transform(spectra_test)

    if save_normalizers:
        params_normalizer.save(PARAMS_NORMALIZER_PATH)
        spectra_normalizer.save(SPECTRA_NORMALIZER_PATH)
        print(f"标准化器已保存到 {CHECKPOINT_DIR}")

    train_data = {
        'params': params_train_norm,
        'spectra': spectra_train_norm,
        'params_raw': params_train,
        'spectra_raw': spectra_train,
    }

    test_data = {
        'params': params_test_norm,
        'spectra': spectra_test_norm,
        'params_raw': params_test,
        'spectra_raw': spectra_test,
    }

    normalizers = {
        'params': params_normalizer,
        'spectra': spectra_normalizer,
    }

    return train_data, test_data, normalizers


def load_normalizers():
    """加载已保存的标准化器"""
    params_normalizer = ZScoreNormalizer()
    spectra_normalizer = MinMaxNormalizer()

    params_normalizer.load(PARAMS_NORMALIZER_PATH)
    spectra_normalizer.load(SPECTRA_NORMALIZER_PATH)

    return {
        'params': params_normalizer,
        'spectra': spectra_normalizer,
    }


def create_forward_dataloaders(train_data, test_data, batch_size=64):
    """
    创建前向网络的数据加载器
    输入: 结构参数 (N, 4)
    输出: 吸收光谱 (N, 200)
    """
    params_train = torch.from_numpy(train_data['params']).float()
    spectra_train = torch.from_numpy(train_data['spectra']).float()
    params_test = torch.from_numpy(test_data['params']).float()
    spectra_test = torch.from_numpy(test_data['spectra']).float()

    train_dataset = TensorDataset(params_train, spectra_train)
    test_dataset = TensorDataset(params_test, spectra_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_tandem_dataloaders(train_data, test_data, batch_size=64):
    """
    创建串联网络的数据加载器
    输入和目标都是光谱（逆向设计场景）
    同时提供参数用于监控
    """
    spectra_train = torch.from_numpy(train_data['spectra']).float()
    spectra_test = torch.from_numpy(test_data['spectra']).float()

    train_dataset = TensorDataset(spectra_train, spectra_train)
    test_dataset = TensorDataset(spectra_test, spectra_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_data, test_data, normalizers = prepare_data()
    print(f"训练集参数范围: [{train_data['params'].min():.3f}, {train_data['params'].max():.3f}]")
    print(f"训练集光谱范围: [{train_data['spectra'].min():.3f}, {train_data['spectra'].max():.3f}]")
    print(f"原始参数范围: [{train_data['params_raw'].min():.6e}, {train_data['params_raw'].max():.6e}]")
