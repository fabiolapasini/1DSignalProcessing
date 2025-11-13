import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from networks.SimpleSignalNet import SimpleSignalNet
from networks.PeakDetectionModel import PeakDetectionModel


# ============================================================
#  TESTER
# ============================================================
class SignalTester:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion_pos = nn.BCELoss()
        self.criterion_height = nn.L1Loss()
        self.criterion_width = nn.L1Loss()

    def test(self, dataloader):
        self.model.eval()
        test_loss = []

        with torch.no_grad():
            for signal, true_pos, true_height, true_width in dataloader:
                signal = signal.to(self.device)
                true_pos = true_pos.to(self.device)
                true_height = true_height.to(self.device)
                true_width = true_width.to(self.device)

                pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))

                loss_p = self.criterion_pos(pred_pos, true_pos)
                loss_h = self.criterion_height(pred_height, true_height)
                loss_w = self.criterion_width(pred_width, true_width)

                total_loss = loss_p + loss_h + loss_w
                test_loss.append(total_loss.item())

        mean_loss = np.mean(test_loss)
        std_loss = np.std(test_loss)
        print(f"Test Loss: {mean_loss:.5f} ± {std_loss:.5f}")
        return mean_loss


# ============================================================
#  CUSTOM LOSS
# ============================================================
class SignalLoss(nn.Module):
    """
    More losses:
    - BCE for peack posiiton
    - L1 (MAE) for height un width
    """
    def __init__(self, w_pos=1.0, w_height=1.0, w_width=1.0):
        super().__init__()
        self.pos_loss = nn.BCELoss()
        self.height_loss = nn.L1Loss()
        self.width_loss = nn.L1Loss()
        self.w_pos = w_pos
        self.w_height = w_height
        self.w_width = w_width

    def forward(self, pred_pos, pred_height, pred_width, true_pos, true_height, true_width):
        loss_p = self.pos_loss(pred_pos, true_pos)
        loss_h = self.height_loss(pred_height, true_height)
        loss_w = self.width_loss(pred_width, true_width)
        total = (self.w_pos * loss_p) + (self.w_height * loss_h) + (self.w_width * loss_w)
        return total, {"pos": loss_p.item(), "height": loss_h.item(), "width": loss_w.item()}
    

# ============================================================
#  METRICS
# ============================================================
class SignalMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.details = []

    def update(self, total_loss, loss_dict):
        self.losses.append(total_loss)
        self.details.append(loss_dict)

    def avg(self):
        mean_loss = np.mean(self.losses)
        mean_details = {k: np.mean([d[k] for d in self.details]) for k in self.details[0].keys()}
        return mean_loss, mean_details


# ============================================================
#  TRAINER
# ============================================================
class SignalTrainer:
    def __init__(self, device, train_loader, val_loader=None, lr=1e-3, patience=7):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # self.model = SimpleSignalNet(dropout=0.4).to(self.device)
        self.model = PeakDetectionModel(dropout=0.4).to(self.device)

        self.criterion = SignalLoss()   # custom
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Early stopping
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.patience = patience

    # ------------------------------------------------------------
    def train(self, epochs=10):
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_metrics = SignalMetrics()

            for batch_idx, (signal, true_pos, true_height, true_width) in enumerate(self.train_loader):
                signal = signal.to(self.device)
                true_pos = true_pos.to(self.device)
                true_height = true_height.to(self.device)
                true_width = true_width.to(self.device)

                # Forward
                # pred_pos, pred_height, pred_width = self.model(signal)
                pred_pos, pred_heights, pred_width = self.model(signal.unsqueeze(1))

                # Compute loss
                loss, loss_dict = self.criterion(pred_pos, pred_heights, pred_width,
                                                 true_pos, true_height, true_width)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_metrics.update(loss.item(), loss_dict)

                if batch_idx % 10 == 0:
                    print(f"[Epoch {epoch} | Batch {batch_idx}] Total Loss: {loss.item():.4f}")

            # Epoch summary
            avg_loss, avg_details = train_metrics.avg()
            print(f"→ Epoch {epoch} | Train loss: {avg_loss:.4f} "
                  f"(pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f})")

            # Validation phase
            if self.val_loader:
                val_loss = self.validate(epoch)
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    
                    checkpoint_dir = "networks/checkpoints"
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(self.model.state_dict(), f"{checkpoint_dir}/best_model_epoch{epoch}.pth")

                    print(f"Saved best model (epoch {epoch}, val_loss={val_loss:.4f})")
                else:
                    self.early_stop_counter += 1
                    print(f"Early stop patience: {self.early_stop_counter}/{self.patience}")
                    if self.early_stop_counter >= self.patience:
                        print("🛑Early stopping triggered.")
                        break

    # ------------------------------------------------------------
    def validate(self, epoch):
        self.model.eval()
        val_metrics = SignalMetrics()

        with torch.no_grad():
            for signals, true_pos, true_height, true_width in self.val_loader:
                signals = signals.to(self.device)
                true_pos = true_pos.to(self.device)
                true_height = true_height.to(self.device)
                true_width = true_width.to(self.device)

                pred_pos, pred_height, pred_width = self.model(signals.unsqueeze(1))
                loss, loss_dict = self.criterion(pred_pos, pred_height, pred_width,
                                                 true_pos, true_height, true_width)
                val_metrics.update(loss.item(), loss_dict)

        avg_loss, avg_details = val_metrics.avg()
        print(f"[VALIDATION] Epoch {epoch} | Val loss: {avg_loss:.4f} "
              f"(pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f})")
        return avg_loss
