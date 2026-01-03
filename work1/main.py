"""
光学器件神经网络正向/逆向设计项目 - 主程序入口
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
    print("光学器件神经网络正向/逆向设计系统")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 阶段1: 训练前向网络
    print("\n[阶段1] 训练前向网络")
    print("-" * 60)
    train_forward_network()
    
    # 阶段2: 训练串联网络
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
    
    # 加载评估器
    evaluator = Evaluator()
    
    # 准备测试数据
    train_data, test_data, normalizers = prepare_data(save_normalizers=False)
    
    # 随机选择几个测试样本
    np.random.seed(42)
    num_demos = 5
    indices = np.random.choice(len(test_data['spectra']), num_demos, replace=False)
    
    print(f"\n演示 {num_demos} 个逆向设计案例:")
    print("-" * 60)
    
    for i, idx in enumerate(indices):
        # 获取目标光谱
        target_spectrum = test_data['spectra'][idx:idx+1]
        true_params_norm = test_data['params'][idx]
        true_params_real = test_data['params_raw'][idx]
        
        # 执行逆向设计
        result = evaluator.inverse_design(target_spectrum)
        
        # 计算光谱重建误差
        recon_mse = np.mean((result['spectra_recon'] - target_spectrum) ** 2)
        
        print(f"\n案例 {i+1} (样本索引 {idx}):")
        print(f"  真实参数 (原始): {true_params_real}")
        print(f"  预测参数 (原始): {result['params_real'][0]}")
        print(f"  参数误差: {np.abs(result['params_real'][0] - true_params_real)}")
        print(f"  光谱重建 MSE: {recon_mse:.6f}")
    
    print("\n" + "=" * 60)
    print("演示完成！")


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="光学器件神经网络正向/逆向设计系统"
    )
    parser.add_argument(
        '--train', action='store_true',
        help='训练模型'
    )
    parser.add_argument(
        '--evaluate', action='store_true',
        help='评估模型'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='逆向设计演示'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，执行完整流程
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

