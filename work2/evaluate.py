"""
模型评估与可视化
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import (
    FORWARD_CONFIG, BACKWARD_CONFIG,
    FORWARD_MODEL_PATH, BACKWARD_MODEL_PATH,
    FORWARD_HISTORY_PATH, TANDEM_HISTORY_PATH, CHECKPOINT_DIR
)
from data_loader import prepare_data, load_normalizers
from models import LSTMForwardNet, LSTMBackwardNet, TandemNet
from utils import get_device


class Evaluator:
    """模型评估器"""

    def __init__(self):
        self.device = get_device()
        print(f"使用设备: {self.device}")

        self.forward_net = self._load_forward_net()
        self.backward_net = self._load_backward_net()
        self.tandem_net = TandemNet(
            self.backward_net, self.forward_net, freeze_forward=True
        )

        self.normalizers = load_normalizers()

    def _load_forward_net(self):
        """加载LSTM前向网络"""
        forward_net = LSTMForwardNet(FORWARD_CONFIG).to(self.device)
        checkpoint = torch.load(FORWARD_MODEL_PATH, map_location=self.device)
        forward_net.load_state_dict(checkpoint['model_state_dict'])
        forward_net.eval()
        print(f"LSTM前向网络加载成功")
        return forward_net

    def _load_backward_net(self):
        """加载LSTM后向网络"""
        backward_net = LSTMBackwardNet(BACKWARD_CONFIG).to(self.device)
        checkpoint = torch.load(BACKWARD_MODEL_PATH, map_location=self.device)
        backward_net.load_state_dict(checkpoint['model_state_dict'])
        backward_net.eval()
        print(f"后向网络加载成功")
        return backward_net

    @torch.no_grad()
    def evaluate_forward(self, test_data):
        """评估前向网络"""
        self.forward_net.eval()

        params = torch.from_numpy(test_data['params']).float().to(self.device)
        spectra_true = torch.from_numpy(test_data['spectra']).float().to(self.device)

        spectra_pred = self.forward_net(params)

        mse = torch.mean((spectra_pred - spectra_true) ** 2).item()
        mae = torch.mean(torch.abs(spectra_pred - spectra_true)).item()

        sample_mse = torch.mean((spectra_pred - spectra_true) ** 2, dim=1).cpu().numpy()

        # R^2 计算
        ss_res = torch.sum((spectra_true - spectra_pred) ** 2).item()
        ss_tot = torch.sum((spectra_true - spectra_true.mean()) ** 2).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        spectra_pred_real = self.normalizers['spectra'].inverse_transform(
            spectra_pred.cpu().numpy()
        )
        spectra_true_real = test_data['spectra_raw']

        results = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'sample_mse': sample_mse,
            'spectra_pred': spectra_pred_real,
            'spectra_true': spectra_true_real,
        }

        print(f"\n前向网络（LSTM）评估结果:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")

        return results

    @torch.no_grad()
    def evaluate_inverse(self, test_data):
        """评估逆向设计（串联网络）"""
        self.tandem_net.backward_net.eval()
        self.tandem_net.forward_net.eval()

        spectra = torch.from_numpy(test_data['spectra']).float().to(self.device)

        params_pred, spectra_recon = self.tandem_net(spectra)

        spectra_mse = torch.mean((spectra_recon - spectra) ** 2).item()
        spectra_mae = torch.mean(torch.abs(spectra_recon - spectra)).item()

        sample_mse = torch.mean((spectra_recon - spectra) ** 2, dim=1).cpu().numpy()

        # R^2
        ss_res = torch.sum((spectra - spectra_recon) ** 2).item()
        ss_tot = torch.sum((spectra - spectra.mean()) ** 2).item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        spectra_recon_real = self.normalizers['spectra'].inverse_transform(
            spectra_recon.cpu().numpy()
        )
        spectra_true_real = test_data['spectra_raw']

        # 反标准化预测参数
        params_pred_real = self.normalizers['params'].inverse_transform(
            params_pred.cpu().numpy()
        )
        params_true_real = test_data['params_raw']

        results = {
            'spectra_mse': spectra_mse,
            'spectra_mae': spectra_mae,
            'spectra_rmse': np.sqrt(spectra_mse),
            'spectra_r2': r2,
            'sample_mse': sample_mse,
            'params_pred': params_pred_real,
            'params_true': params_true_real,
            'spectra_recon': spectra_recon_real,
            'spectra_true': spectra_true_real,
        }

        print(f"\n逆向设计评估结果:")
        print(f"  光谱重建 MSE:  {spectra_mse:.6f}")
        print(f"  光谱重建 RMSE: {np.sqrt(spectra_mse):.6f}")
        print(f"  光谱重建 MAE:  {spectra_mae:.6f}")
        print(f"  光谱重建 R²:   {r2:.6f}")

        return results

    @torch.no_grad()
    def inverse_design(self, target_spectrum):
        """
        逆向设计推理

        Args:
            target_spectrum: 归一化后的目标光谱 (batch_size, 200) 或 (200,)

        Returns:
            dict: 包含预测参数和重建光谱
        """
        self.tandem_net.backward_net.eval()
        self.tandem_net.forward_net.eval()

        if target_spectrum.ndim == 1:
            target_spectrum = target_spectrum[np.newaxis, :]

        spectrum_tensor = torch.from_numpy(target_spectrum).float().to(self.device)

        params_pred_norm, spectra_recon = self.tandem_net(spectrum_tensor)

        params_pred_real = self.normalizers['params'].inverse_transform(
            params_pred_norm.cpu().numpy()
        )

        return {
            'params_real': params_pred_real,
            'params_norm': params_pred_norm.cpu().numpy(),
            'spectra_recon': spectra_recon.cpu().numpy(),
        }

    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if FORWARD_HISTORY_PATH.exists():
            history = np.load(FORWARD_HISTORY_PATH, allow_pickle=True).item()
            ax = axes[0]
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], label='Train Loss', alpha=0.8)
            ax.plot(epochs, history['test_loss'], label='Test Loss', alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('LSTM Forward Network Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        if TANDEM_HISTORY_PATH.exists():
            history = np.load(TANDEM_HISTORY_PATH, allow_pickle=True).item()
            ax = axes[1]
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], label='Train Loss', alpha=0.8)
            ax.plot(epochs, history['test_loss'], label='Test Loss', alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Tandem Network Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练历史图已保存到 {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_spectrum_comparison(self, spectra_true, spectra_pred,
                                  num_samples=5, title_prefix="", save_path=None):
        """绘制光谱对比图"""
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))

        if num_samples == 1:
            axes = [axes]

        np.random.seed(48)
        indices = np.random.choice(len(spectra_true), num_samples, replace=False)

        wavelengths = np.arange(1, 201)

        for i, idx in enumerate(indices):
            ax = axes[i]
            ax.plot(wavelengths, spectra_true[idx], 'b-', label='True', linewidth=1.5)
            ax.plot(wavelengths, spectra_pred[idx], 'r--', label='Predicted', linewidth=1.5)
            ax.set_xlabel('Wavelength Index')
            ax.set_ylabel('Absorption')
            ax.set_title(f'{title_prefix}Sample {idx}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"光谱对比图已保存到 {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_error_distribution(self, sample_mse, title="", save_path=None):
        """绘制误差分布图"""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(sample_mse, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(sample_mse), color='r', linestyle='--',
                   label=f'Mean: {np.mean(sample_mse):.6f}')
        ax.axvline(np.median(sample_mse), color='g', linestyle='--',
                   label=f'Median: {np.median(sample_mse):.6f}')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"误差分布图已保存到 {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_params_comparison(self, params_true, params_pred, save_path=None):
        """绘制逆向设计参数对比散点图"""
        param_names = ['P', 'Lx', 'Ly', 'h']
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i, (ax, name) in enumerate(zip(axes, param_names)):
            ax.scatter(params_true[:, i], params_pred[:, i], alpha=0.3, s=10, color='steelblue')
            lims = [
                min(params_true[:, i].min(), params_pred[:, i].min()),
                max(params_true[:, i].max(), params_pred[:, i].max()),
            ]
            ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'Parameter: {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"参数对比图已保存到 {save_path}")
        else:
            plt.show()

        plt.close()

    def export_results_csv(self, test_data, forward_results, inverse_results):
        """导出评估结果到CSV"""
        output_dir = CHECKPOINT_DIR

        # 前向网络预测结果
        forward_df = pd.DataFrame({
            'sample_mse': forward_results['sample_mse'],
        })
        forward_df.to_csv(output_dir / "forward_eval_results.csv", index=False)

        # 逆向设计结果
        param_names = ['P', 'Lx', 'Ly', 'h']
        inverse_data = {}
        for i, name in enumerate(param_names):
            inverse_data[f'true_{name}'] = inverse_results['params_true'][:, i]
            inverse_data[f'pred_{name}'] = inverse_results['params_pred'][:, i]
            inverse_data[f'error_{name}'] = np.abs(
                inverse_results['params_true'][:, i] - inverse_results['params_pred'][:, i]
            )
        inverse_data['spectra_mse'] = inverse_results['sample_mse']

        inverse_df = pd.DataFrame(inverse_data)
        inverse_df.to_csv(output_dir / "inverse_design_results.csv", index=False)

        print(f"评估结果已导出到 {output_dir}")


def run_evaluation():
    """运行完整评估"""
    print("正在准备数据...")
    train_data, test_data, _ = prepare_data(save_normalizers=False)

    evaluator = Evaluator()

    # 评估前向网络
    print("\n" + "=" * 60)
    forward_results = evaluator.evaluate_forward(test_data)

    # 评估逆向设计
    print("\n" + "=" * 60)
    inverse_results = evaluator.evaluate_inverse(test_data)

    # 创建输出目录
    output_dir = CHECKPOINT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制图表
    print("\n" + "=" * 60)
    print("生成可视化图表...")

    evaluator.plot_training_history(save_path=output_dir / "training_history.png")

    evaluator.plot_spectrum_comparison(
        forward_results['spectra_true'],
        forward_results['spectra_pred'],
        num_samples=5,
        title_prefix="Forward: ",
        save_path=output_dir / "forward_spectrum_comparison.png"
    )

    evaluator.plot_spectrum_comparison(
        inverse_results['spectra_true'],
        inverse_results['spectra_recon'],
        num_samples=5,
        title_prefix="Inverse: ",
        save_path=output_dir / "inverse_spectrum_comparison.png"
    )

    evaluator.plot_error_distribution(
        forward_results['sample_mse'],
        title="LSTM Forward Network",
        save_path=output_dir / "forward_error_distribution.png"
    )

    evaluator.plot_error_distribution(
        inverse_results['sample_mse'],
        title="Inverse Design (Tandem)",
        save_path=output_dir / "inverse_error_distribution.png"
    )

    evaluator.plot_params_comparison(
        inverse_results['params_true'],
        inverse_results['params_pred'],
        save_path=output_dir / "inverse_params_comparison.png"
    )

    # 导出CSV结果
    evaluator.export_results_csv(test_data, forward_results, inverse_results)

    print("\n" + "=" * 60)
    print("评估完成！")

    return evaluator, forward_results, inverse_results


if __name__ == "__main__":
    evaluator, forward_results, inverse_results = run_evaluation()
