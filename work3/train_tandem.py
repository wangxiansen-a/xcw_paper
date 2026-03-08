"""
串联网络训练（逆向设计）
后向网络（Mamba） + 前向网络（Mamba，冻结）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BACKWARD_CONFIG, TANDEM_TRAIN_CONFIG, FORWARD_CONFIG,
    FORWARD_MODEL_PATH, BACKWARD_MODEL_PATH,
    TANDEM_HISTORY_PATH, CHECKPOINT_DIR
)
from data_loader import prepare_data, create_tandem_dataloaders
from models import MambaForwardNet, MambaBackwardNet, TandemNet
from utils import set_seed, get_device, EarlyStopping


class TandemTrainer:
    """串联网络训练器"""

    def __init__(self, config=None, train_config=None):
        self.config = config or BACKWARD_CONFIG
        self.train_config = train_config or TANDEM_TRAIN_CONFIG

        self.device = get_device()
        print(f"使用设备: {self.device}")

        self.forward_net = self._load_pretrained_forward_net()

        self.backward_net = MambaBackwardNet(self.config).to(self.device)

        self.tandem_net = TandemNet(
            self.backward_net,
            self.forward_net,
            freeze_forward=True
        )

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.backward_net.parameters(),
            lr=self.train_config["learning_rate"],
            weight_decay=self.train_config["weight_decay"]
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.train_config["lr_step_size"],
            gamma=self.train_config["lr_gamma"]
        )

        self.early_stopping = EarlyStopping(
            patience=self.train_config["early_stop_patience"]
        )

        self.history = {
            'train_loss': [],
            'test_loss': [],
            'lr': []
        }

        self.best_loss = float('inf')

    def _load_pretrained_forward_net(self):
        print(f"正在加载预训练Mamba前向网络: {FORWARD_MODEL_PATH}")

        if not FORWARD_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"未找到预训练前向网络: {FORWARD_MODEL_PATH}\n"
                "请先运行 train_forward.py 训练前向网络"
            )

        forward_net = MambaForwardNet(FORWARD_CONFIG).to(self.device)
        checkpoint = torch.load(FORWARD_MODEL_PATH, map_location=self.device)
        forward_net.load_state_dict(checkpoint['model_state_dict'])

        print(f"Mamba前向网络加载成功，最佳损失: {checkpoint.get('best_loss', 'N/A')}")
        return forward_net

    def train_epoch(self, train_loader):
        self.backward_net.train()

        total_loss = 0.0
        num_batches = 0

        for target_spectra, _ in train_loader:
            target_spectra = target_spectra.to(self.device)

            self.optimizer.zero_grad()
            predicted_params, reconstructed_spectra = self.tandem_net(target_spectra)

            loss = self.criterion(reconstructed_spectra, target_spectra)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backward_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.backward_net.eval()

        total_loss = 0.0
        num_batches = 0

        for target_spectra, _ in data_loader:
            target_spectra = target_spectra.to(self.device)

            _, reconstructed_spectra = self.tandem_net(target_spectra)
            loss = self.criterion(reconstructed_spectra, target_spectra)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, test_loader):
        epochs = self.train_config["epochs"]
        print_every = self.train_config["print_every"]

        print(f"\n开始训练串联网络，共 {epochs} epochs")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            train_loss_raw = self.train_epoch(train_loader)

            train_loss = self.evaluate(train_loader)
            test_loss = self.evaluate(test_loader)

            current_lr = self.scheduler.get_last_lr()[0]

            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['lr'].append(current_lr)

            self.scheduler.step()

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.save_model()

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch [{epoch:4d}/{epochs}] | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Test Loss: {test_loss:.6f} | "
                      f"LR: {current_lr:.2e}")

            if self.early_stopping(test_loss):
                print(f"\n早停触发于 Epoch {epoch}")
                break

        print("=" * 60)
        print(f"训练完成！最佳测试损失: {self.best_loss:.6f}")

        self.save_history()

        return self.history

    def save_model(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.backward_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }
        torch.save(checkpoint, BACKWARD_MODEL_PATH)

    def save_history(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(TANDEM_HISTORY_PATH, self.history)
        print(f"训练历史已保存到 {TANDEM_HISTORY_PATH}")

    def load_model(self, path=None):
        path = path or BACKWARD_MODEL_PATH
        checkpoint = torch.load(path, map_location=self.device)
        self.backward_net.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"后向网络已从 {path} 加载")
        return self.backward_net


def train_tandem_network():
    from config import RANDOM_SEED
    set_seed(RANDOM_SEED)

    print("正在准备数据...")
    train_data, test_data, normalizers = prepare_data(save_normalizers=False)

    batch_size = TANDEM_TRAIN_CONFIG["batch_size"]
    train_loader, test_loader = create_tandem_dataloaders(
        train_data, test_data, batch_size
    )
    print(f"数据加载器创建完成: batch_size = {batch_size}")

    trainer = TandemTrainer()
    history = trainer.train(train_loader, test_loader)

    return trainer, history


if __name__ == "__main__":
    trainer, history = train_tandem_network()
