"""
基于LSTM的光学器件神经网络正向/逆向设计项目 - 主程序入口
数据集2: 4维结构参数 → 200维吸收光谱
"""
import argparse
import numpy as np
import torch

from config import RANDOM_SEED, CHECKPOINT_DIR
from utils import set_seed, get_device
from data_loader import prepare_data, load_normalizers
from train_forward import train_forward_network
from train_tandem import train_tandem_network
from evaluate import Evaluator, run_evaluation


def train():
    """训练所有模型"""
    print("\n" + "=" * 60)
    print("基于LSTM的光学器件正向/逆向设计系统 (数据集2)")
    print("=" * 60)

    set_seed(RANDOM_SEED)

    # 阶段1: 训练LSTM前向网络
    print("\n[阶段1] 训练LSTM前向网络")
    print("-" * 60)
    train_forward_network()

    # 阶段2: 训练串联网络（LSTM后向 + 冻结LSTM前向）
    print("\n[阶段2] 训练串联网络（逆向设计）")
    print("-" * 60)
    train_tandem_network()

    print("\n" + "=" * 60)
    print("所有模型训练完成！")
    print(f"模型保存位置: {CHECKPOINT_DIR}")
    print("=" * 60)


def evaluate():
    """评估模型"""
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    run_evaluation()


def demo():
    """逆向设计演示"""
    print("\n" + "=" * 60)
    print("逆向设计演示")
    print("=" * 60)

    evaluator = Evaluator()

    train_data, test_data, normalizers = prepare_data(save_normalizers=False)

    np.random.seed(42)
    num_demos = 5
    indices = np.random.choice(len(test_data['spectra']), num_demos, replace=False)

    param_names = ['P', 'Lx', 'Ly', 'h']

    print(f"\n演示 {num_demos} 个逆向设计案例:")
    print("-" * 60)

    for i, idx in enumerate(indices):
        target_spectrum = test_data['spectra'][idx:idx + 1]
        true_params_real = test_data['params_raw'][idx]

        result = evaluator.inverse_design(target_spectrum)

        recon_mse = np.mean((result['spectra_recon'] - target_spectrum) ** 2)

        print(f"\n案例 {i + 1} (样本索引 {idx}):")
        for j, name in enumerate(param_names):
            print(f"  {name}: 真实={true_params_real[j]:.4e}, "
                  f"预测={result['params_real'][0][j]:.4e}, "
                  f"误差={abs(result['params_real'][0][j] - true_params_real[j]):.4e}")
        print(f"  光谱重建 MSE: {recon_mse:.6f}")

    print("\n" + "=" * 60)
    print("演示完成！")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="基于LSTM的光学器件正向/逆向设计系统（数据集2）"
    )
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    parser.add_argument('--demo', action='store_true', help='逆向设计演示')

    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.demo]):
        print("\n未指定参数，执行完整流程（训练 + 评估 + 演示）")
        train()
        evaluate()
        demo()
    else:
        if args.train:
            train()
        if args.evaluate:
            evaluate()
        if args.demo:
            demo()


if __name__ == "__main__":
    main()
