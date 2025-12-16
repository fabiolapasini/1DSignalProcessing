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
        if true.dim() == 4:
            true = true.squeeze(1)
            pred = pred.squeeze(1)
        
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
        self.criterion = SignalLoss()
        self.threshold = threshold
    
    def test(self, dataloader):
        self.model.eval()
        test_metrics = SignalMetrics(
            n_cells=32, 
            max_signal_length=1024, 
            anchors=[1.0], 
            iou_thresh=0.5
        )
        
        all_pred_pos = []
        all_true_pos = []
        
        total_loss = 0.0
        total_loss_pos = 0.0
        total_loss_height = 0.0
        total_loss_width = 0.0
        num_batches = 0
        
        n_cells = 32
        max_signal_length = 1024
        cell_size = max_signal_length // n_cells
        
        with torch.no_grad():
            for batch in dataloader:
                signal = batch["signal"].to(self.device)
                peaks = batch["peaks"].to(self.device)
                n_peaks = batch["n_peaks"].to(self.device)
                
                pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))
                
                pred_pos = pred_pos.squeeze(1)
                pred_height = pred_height.squeeze(1)
                pred_width = pred_width.squeeze(1)
                
                true_pos = torch.zeros_like(pred_pos)
                true_height = torch.zeros_like(pred_height)
                true_width = torch.zeros_like(pred_width)
                
                for i, p in enumerate(peaks):
                    num_peaks_i = n_peaks[i].item()
                    if num_peaks_i == 0:
                        continue
                    if p.dim() == 1:
                        p = p.unsqueeze(0)
                    for j in range(min(num_peaks_i, p.shape[0])):
                        if p.shape[-1] < 3:
                            continue
                        
                        peak_position = p[j, 0].item()
                        cell_idx = int(peak_position / max_signal_length * n_cells)
                        
                        if 0 <= cell_idx < true_pos.shape[1]:
                            true_pos[i, cell_idx] = 1.0
                            true_height[i, cell_idx] = p[j, 1]
                            true_width[i, cell_idx] = p[j, 2]
                
                pred = torch.stack([pred_pos, pred_height, pred_width], dim=-1)
                true = torch.stack([true_pos, true_height, true_width], dim=-1)
                
                loss = self.criterion(true, pred)
                
                error = (true - pred) ** 2
                loss_dict = {
                    'pos': error[..., 0].mean().item(),
                    'height': error[..., 1].mean().item(),
                    'width': error[..., 2].mean().item()
                }
                
                test_metrics.update(true, pred)
                
                total_loss += loss.item()
                total_loss_pos += loss_dict['pos']
                total_loss_height += loss_dict['height']
                total_loss_width += loss_dict['width']
                num_batches += 1
                
                all_pred_pos.append((pred_pos > self.threshold).cpu().numpy())
                all_true_pos.append(true_pos.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        avg_details = {
            'pos': total_loss_pos / num_batches,
            'height': total_loss_height / num_batches,
            'width': total_loss_width / num_batches
        }
        
        pred_flat = np.concatenate([p.flatten() for p in all_pred_pos])
        true_flat = np.concatenate([t.flatten() for t in all_true_pos])
        
        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)
        
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

        self.model = PeakDetectionModel().to(self.device)
        self.criterion = SignalLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.early_stop_counter = 0

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

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_f1'].append(train_f1)
                self.history['val_f1'].append(val_f1)
                self.history['val_precision'].append(val_precision)
                self.history['val_recall'].append(val_recall)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

                self.scheduler.step(val_loss)

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
        
        total_loss = 0.0
        total_loss_pos = 0.0
        total_loss_height = 0.0
        total_loss_width = 0.0
        num_batches = 0
        
        n_cells = 32
        max_signal_length = 1024
        cell_size = max_signal_length // n_cells

        for batch_idx, batch in enumerate(self.train_loader):
            signal = batch["signal"].to(self.device)
            peaks = batch["peaks"].to(self.device)
            signal_lengths = batch["signal_lengths"].to(self.device)
            n_peaks = batch["n_peaks"].to(self.device)

            pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))
            
            pred_pos = pred_pos.squeeze(1)
            pred_height = pred_height.squeeze(1)
            pred_width = pred_width.squeeze(1)

            true_pos = torch.zeros_like(pred_pos)
            true_height = torch.zeros_like(pred_height)
            true_width = torch.zeros_like(pred_width)

            for i, p in enumerate(peaks):
                num_peaks = n_peaks[i].item() 
                if num_peaks == 0:
                    continue
                if p.dim() == 1:
                    p = p.unsqueeze(0)
                
                for j in range(min(num_peaks, p.shape[0])):
                    if p.shape[-1] < 3:
                        continue
                    
                    peak_position = p[j, 0].item()
                    cell_idx = int(peak_position / max_signal_length * n_cells)
                    
                    if 0 <= cell_idx < true_pos.shape[1]:
                        true_pos[i, cell_idx] = 1.0
                        true_height[i, cell_idx] = p[j, 1]
                        true_width[i, cell_idx] = p[j, 2]

            pred = torch.stack([pred_pos, pred_height, pred_width], dim=-1)
            true = torch.stack([true_pos, true_height, true_width], dim=-1)

            loss = self.criterion(true, pred)
            
            with torch.no_grad():
                error = (true - pred) ** 2
                loss_dict = {
                    'pos': error[..., 0].mean().item(),
                    'height': error[..., 1].mean().item(),
                    'width': error[..., 2].mean().item()
                }

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metrics.update(true, pred)
            
            total_loss += loss.item()
            total_loss_pos += loss_dict['pos']
            total_loss_height += loss_dict['height']
            total_loss_width += loss_dict['width']
            num_batches += 1

            all_pred_pos.append((pred_pos > self.threshold).detach().cpu().numpy())
            all_true_pos.append(true_pos.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch:2d} | Batch {batch_idx:3d}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        avg_details = {
            'pos': total_loss_pos / num_batches,
            'height': total_loss_height / num_batches,
            'width': total_loss_width / num_batches
        }
        
        pred_flat = np.concatenate([p.flatten() for p in all_pred_pos])
        true_flat = np.concatenate([t.flatten() for t in all_true_pos])
        train_f1 = f1_score(true_flat, pred_flat, zero_division=0)

        print(f"\nEpoch {epoch} | Train Loss: {avg_loss:.4f} | F1: {train_f1:.4f}")
        print(f"   └─ pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f}")

        return avg_loss, train_f1

    def _validate_epoch(self, epoch):
        self.model.eval()
        metrics = SignalMetrics(
            n_cells=32, 
            max_signal_length=1024, 
            anchors=[1.0], 
            iou_thresh=0.5
        )

        all_pred_pos = []
        all_true_pos = []
        
        total_loss = 0.0
        total_loss_pos = 0.0
        total_loss_height = 0.0
        total_loss_width = 0.0
        num_batches = 0
        
        n_cells = 32
        max_signal_length = 1024
        cell_size = max_signal_length // n_cells

        with torch.no_grad():
            for batch in self.val_loader:
                signal = batch["signal"].to(self.device)
                peaks = batch["peaks"].to(self.device)
                n_peaks = batch["n_peaks"].to(self.device)

                pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))
                
                pred_pos = pred_pos.squeeze(1)
                pred_height = pred_height.squeeze(1)
                pred_width = pred_width.squeeze(1)

                true_pos = torch.zeros_like(pred_pos)
                true_height = torch.zeros_like(pred_height)
                true_width = torch.zeros_like(pred_width)

                for i, p in enumerate(peaks):
                    num_peaks = n_peaks[i].item()
                    if num_peaks == 0:
                        continue
                    if p.dim() == 1:
                        p = p.unsqueeze(0)
                    for j in range(min(num_peaks, p.shape[0])):
                        if p.shape[-1] < 3:
                            continue
                        
                        peak_position = p[j, 0].item()
                        cell_idx = int(peak_position / max_signal_length * n_cells)
                        
                        if 0 <= cell_idx < true_pos.shape[1]:
                            true_pos[i, cell_idx] = 1.0
                            true_height[i, cell_idx] = p[j, 1]
                            true_width[i, cell_idx] = p[j, 2]

                pred = torch.stack([pred_pos, pred_height, pred_width], dim=-1)
                true = torch.stack([true_pos, true_height, true_width], dim=-1)

                loss = self.criterion(true, pred)
                
                error = (true - pred) ** 2
                loss_dict = {
                    'pos': error[..., 0].mean().item(),
                    'height': error[..., 1].mean().item(),
                    'width': error[..., 2].mean().item()
                }

                metrics.update(true, pred)
                
                total_loss += loss.item()
                total_loss_pos += loss_dict['pos']
                total_loss_height += loss_dict['height']
                total_loss_width += loss_dict['width']
                num_batches += 1
                
                all_pred_pos.append((pred_pos > self.threshold).cpu().numpy())
                all_true_pos.append(true_pos.cpu().numpy())

        avg_loss = total_loss / num_batches
        avg_details = {
            'pos': total_loss_pos / num_batches,
            'height': total_loss_height / num_batches,
            'width': total_loss_width / num_batches
        }
        
        pred_flat = np.concatenate([p.flatten() for p in all_pred_pos])
        true_flat = np.concatenate([t.flatten() for t in all_true_pos])

        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)

        print(f"Epoch {epoch} | Val Loss: {avg_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"   └─ pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f}")

        return avg_loss, precision, recall, f1

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

    def _save_history(self):
        history_dir = "networks/history"
        os.makedirs(history_dir, exist_ok=True)

        history_path = f"{history_dir}/training_history_{self.timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        print(f"Training history saved to: {history_path}")


