import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from networks.PeakDetectionModel import PeakDetectionModel
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
#  LOSS
# ============================================================
class SignalLoss(nn.Module):
    def __init__(self, neg_scale=0.3):
        super().__init__()
        self.neg_scale = neg_scale
        self.pos_scale = 1.0 - neg_scale

    def forward(self, true, pred):
        neg_mask = 1.0 - true[..., 0]
        pos_mask = true[..., 0]

        error = (true - pred) ** 2

        conf_error = error[..., 0]
        offset_error = error[..., 1]
        scale_error = error[..., 2]

        neg_loss = self.neg_scale * torch.sum(neg_mask * conf_error)
        pos_loss = self.pos_scale * torch.sum(
            pos_mask * (conf_error + offset_error + scale_error)
        )

        batch_size = true.shape[0]
        return (neg_loss + pos_loss) / batch_size
    

# ============================================================
#  METRICS
# ============================================================
class SignalMetrics:
    def __init__(self, n_cells, max_signal_length, anchors, iou_thresh=0.5):
        self.n_cells = n_cells
        self.max_signal_length = max_signal_length
        self.anchors = anchors
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0

    @torch.no_grad()
    def update(self, true, pred):
        # true, pred: (B, N, 3)
        true_c = true[..., 0]
        pred_c = (pred[..., 0] >= 0.5).float()

        B, N, _ = true.shape
        device = true.device

        cell_index = torch.arange(N, device=device).float()
        cell_index = cell_index.unsqueeze(0).expand(B, -1)

        true_x = (cell_index + true[..., 1]) / self.n_cells * self.max_signal_length
        pred_x = (cell_index + pred[..., 1]) / self.n_cells * self.max_signal_length

        true_w = true[..., 2] * self.anchors[0] * self.max_signal_length
        pred_w = pred[..., 2] * self.anchors[0] * self.max_signal_length

        true_xmin = true_x - true_w / 2
        true_xmax = true_xmin + true_w

        pred_xmin = pred_x - pred_w / 2
        pred_xmax = pred_xmin + pred_w

        inter_xmin = torch.maximum(true_xmin, pred_xmin)
        inter_xmax = torch.minimum(true_xmax, pred_xmax)
        inter = torch.clamp(inter_xmax - inter_xmin, min=0)

        union = (true_xmax - true_xmin) + (pred_xmax - pred_xmin) - inter
        iou = inter / torch.clamp(union, min=1e-6)

        tp = true_c * pred_c * (iou >= self.iou_thresh).float()
        fp = (
            true_c * pred_c * (iou < self.iou_thresh).float()
            + (1.0 - true_c) * pred_c
        )

        self.tp += tp.sum().item()
        self.fp += fp.sum().item()

    def compute(self):
        return self.tp / max(self.tp + self.fp, 1.0)


# ============================================================
#  TESTER
# ============================================================
class SignalTester:
    def __init__(self, model, device, threshold=0.5):
        self.model = model.to(device)
        self.device = device
        self.criterion = SignalLoss(w_pos=10.0, w_height=0.1, w_width=0.5)
        self.threshold = threshold
    
    def test(self, dataloader):
        self.model.eval()
        test_metrics = SignalMetrics()
        
        all_pred_pos = []
        all_true_pos = []

        with torch.no_grad():
            for batch in dataloader:
                signal = batch["signal"].to(self.device)
                peaks = batch["peaks"].to(self.device)

                pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))

                # Convert peaks continui in pos / height / width tensors
                true_pos = torch.zeros_like(pred_pos)
                true_height = torch.zeros(pred_height.shape, device=self.device)
                true_width = torch.zeros(pred_width.shape, device=self.device)

                for i, p in enumerate(peaks):
                    for j in range(p.shape[0]):
                        idx = int(round(p[j, 0].item()))
                        if 0 <= idx < true_pos.shape[1]:
                            true_pos[i, idx] = 1.0
                            true_height[i, j] = p[j, 1]
                            true_width[i, j] = p[j, 2]

                # Compute loss
                loss, loss_dict = self.criterion(pred_pos, pred_height, pred_width,
                                                 true_pos, true_height, true_width)

                test_metrics.update(loss.item(), loss_dict)

                all_pred_pos.append((pred_pos > self.threshold).cpu().numpy())
                all_true_pos.append(true_pos.cpu().numpy())

        # Flatten for global metrics
        pred_flat = np.concatenate(all_pred_pos).flatten()
        true_flat = np.concatenate(all_true_pos).flatten()

        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)

        avg_loss, avg_details, _, _, _ = test_metrics.avg()

        print(f"[TEST RESULTS]")
        print(f"Loss: {avg_loss:.4f} (pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f})")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        return avg_loss, precision, recall, f1


# ============================================================
#  TRAINER
# ============================================================
class SignalTrainer:
    def __init__(self, device, train_loader, val_loader=None, lr=1e-4, patience=7, threshold=0.5):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.threshold = threshold
        self.patience = patience

        # Model
        self.model = PeakDetectionModel().to(self.device)

        # Loss
        self.criterion = SignalLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Early stopping
        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.early_stop_counter = 0

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'val_precision': [], 'val_recall': [],
            'learning_rates': []
        }

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(self, epochs=10):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss, train_f1 = self._train_epoch(epoch)

            if self.val_loader:
                val_loss, val_precision, val_recall, val_f1 = self._validate_epoch(epoch)

                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_f1'].append(train_f1)
                self.history['val_f1'].append(val_f1)
                self.history['val_precision'].append(val_precision)
                self.history['val_recall'].append(val_recall)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

                # Scheduler step
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_f1 = val_f1
                    self.early_stop_counter = 0
                    self._save_checkpoint(epoch, val_loss, val_f1, is_best=True)
                else:
                    self.early_stop_counter += 1
                    print(f"Early stop counter: {self.early_stop_counter}/{self.patience}")
                    if self.early_stop_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s\n")

        print(f"Training finished. Best val loss: {self.best_val_loss:.4f}, Best val F1: {self.best_val_f1:.4f}")
        self._save_history()

    # -------------------------------
    # Training step
    # -------------------------------
    def _train_epoch(self, epoch):
        self.model.train()
        metrics = SignalMetrics(
        n_cells=32, 
        max_signal_length=1024, 
        anchors=[1.0], 
        iou_thresh=0.5
    )

        all_pred_pos = []
        all_true_pos = []

        for batch_idx, batch in enumerate(self.train_loader):
            signal = batch["signal"].to(self.device)
            peaks = batch["peaks"].to(self.device)

            # Forward
            pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))

            # Prepare targets: if using SignalLoss, split peaks into pos, height, width
            # Here we assume peaks are already aligned to max_peaks
            true_pos = torch.zeros_like(pred_pos)
            true_height = torch.zeros(pred_height.shape, device=self.device)
            true_width = torch.zeros(pred_width.shape, device=self.device)

            # Fill in true_pos / true_height / true_width from peaks
            for i, p in enumerate(peaks):
                for j in range(p.shape[0]):
                    idx = int(round(p[j, 0].item()))
                    if 0 <= idx < true_pos.shape[1]:
                        true_pos[i, idx] = 1.0
                        true_height[i, j] = p[j, 1]
                        true_width[i, j] = p[j, 2]

            # Compute loss
            loss, loss_dict = self.criterion(pred_pos, pred_height, pred_width,
                                             true_pos, true_height, true_width)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metrics.update(loss.item(), loss_dict)

            # Collect predictions for F1
            all_pred_pos.append((pred_pos > self.threshold).detach().cpu().numpy())
            all_true_pos.append(true_pos.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch:2d} | Batch {batch_idx:3d}] Loss: {loss.item():.4f}")

        # Average metrics
        avg_loss, avg_details, _, _, _ = metrics.avg()
        pred_flat = np.concatenate(all_pred_pos).flatten()
        true_flat = np.concatenate(all_true_pos).flatten()
        train_f1 = f1_score(true_flat, pred_flat, zero_division=0)

        print(f"\nEpoch {epoch} | Train Loss: {avg_loss:.4f} | F1: {train_f1:.4f}")
        print(f"   └─ pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f}")

        return avg_loss, train_f1

    # -------------------------------
    # Validation step
    # -------------------------------
    def _validate_epoch(self, epoch):
        self.model.eval()
        metrics = SignalMetrics()

        all_pred_pos = []
        all_true_pos = []

        with torch.no_grad():
            for batch in self.val_loader:
                signal = batch["signal"].to(self.device)
                peaks = batch["peaks"].to(self.device)

                pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))

                true_pos = torch.zeros_like(pred_pos)
                true_height = torch.zeros(pred_height.shape, device=self.device)
                true_width = torch.zeros(pred_width.shape, device=self.device)

                for i, p in enumerate(peaks):
                    for j in range(p.shape[0]):
                        idx = int(round(p[j, 0].item()))
                        if 0 <= idx < true_pos.shape[1]:
                            true_pos[i, idx] = 1.0
                            true_height[i, j] = p[j, 1]
                            true_width[i, j] = p[j, 2]

                loss, loss_dict = self.criterion(pred_pos, pred_height, pred_width,
                                                 true_pos, true_height, true_width)

                metrics.update(loss.item(), loss_dict)
                all_pred_pos.append((pred_pos > self.threshold).cpu().numpy())
                all_true_pos.append(true_pos.cpu().numpy())

        avg_loss, avg_details, _, _, _ = metrics.avg()
        pred_flat = np.concatenate(all_pred_pos).flatten()
        true_flat = np.concatenate(all_true_pos).flatten()

        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)

        print(f"Epoch {epoch} | Val Loss: {avg_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"   └─ pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f}")

        return avg_loss, precision, recall, f1

    # -------------------------------
    # Checkpoint
    # -------------------------------
    def _save_checkpoint(self, epoch, val_loss, val_f1, is_best=False):
        checkpoint_dir = "networks/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_f1': val_f1,
            'history': self.history
        }

        if is_best:
            path = f"{checkpoint_dir}/best_model_{self.timestamp}.pth"
            torch.save(checkpoint, path)
            print(f"Saved best model: epoch {epoch}, val_loss={val_loss:.4f}, F1={val_f1:.4f}")

    # -------------------------------
    # Save history
    # -------------------------------
    def _save_history(self):
        history_dir = "networks/history"
        os.makedirs(history_dir, exist_ok=True)

        history_path = f"{history_dir}/training_history_{self.timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"Training history saved to: {history_path}")


