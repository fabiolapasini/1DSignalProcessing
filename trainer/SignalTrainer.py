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
    """
    Combined loss function for peak detection with weighted components:
    BCE for position classification, MSE for height and width regression.
    """
    def __init__(self, w_pos=10.0, w_height=0.1, w_width=0.5):
        super().__init__()
        self.pos_loss = nn.BCELoss()
        self.height_loss = nn.MSELoss()
        self.width_loss = nn.MSELoss()
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
    """
    Accumulates and computes average metrics during training/validation.
    Tracks total loss, individual loss components, and classification metrics (precision, recall, F1).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.details = []
        self.precisions = []
        self.recalls = []
        self.f1s = []

    def update(self, total_loss, loss_dict, precision=None, recall=None, f1=None):
        self.losses.append(total_loss)
        self.details.append(loss_dict)
        if precision is not None:
            self.precisions.append(precision)
        if recall is not None:
            self.recalls.append(recall)
        if f1 is not None:
            self.f1s.append(f1)

    def avg(self):
        if not self.losses:
            return 0.0, {}, 0.0, 0.0, 0.0
        mean_loss = np.mean(self.losses)
        mean_details = {k: np.mean([d[k] for d in self.details]) for k in self.details[0].keys()}
        mean_precision = np.mean(self.precisions) if self.precisions else 0.0
        mean_recall = np.mean(self.recalls) if self.recalls else 0.0
        mean_f1 = np.mean(self.f1s) if self.f1s else 0.0
        return mean_loss, mean_details, mean_precision, mean_recall, mean_f1


# ============================================================
#  TESTER
# ============================================================
class SignalTester:
    def __init__(self, model, device, threshold=0.3):
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
            for signal, true_pos, true_height, true_width in dataloader:
                signal = signal.to(self.device)
                true_pos = true_pos.to(self.device)
                true_height = true_height.to(self.device)
                true_width = true_width.to(self.device)
                
                pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))
                
                loss, loss_dict = self.criterion(pred_pos, pred_height, pred_width,
                                                 true_pos, true_height, true_width)
                
                all_pred_pos.append((pred_pos > self.threshold).cpu().numpy())
                all_true_pos.append(true_pos.cpu().numpy())
                
                test_metrics.update(loss.item(), loss_dict)
        
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
    def __init__(self, device, train_loader, val_loader=None, lr=0.001, patience=7, threshold=0.3):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.threshold = threshold

        self.model = PeakDetectionModel().to(self.device)

        # self.criterion = SignalLoss(w_pos=10.0, w_height=0.1, w_width=0.5)
        self.criterion = SignalLoss(w_pos=5.0, w_height=0.2, w_width=0.5)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        self.best_val_loss = float("inf")
        self.best_val_f1 = 0.0
        self.early_stop_counter = 0
        self.patience = patience
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rates': []
        }
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(self, epochs=10):
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
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
                    print(f"Early stop patience: {self.early_stop_counter}/{self.patience}")
                    
                    if self.early_stop_counter >= self.patience:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        break
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s\n")
        
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        self._save_history()

    def _train_epoch(self, epoch):
        self.model.train()
        train_metrics = SignalMetrics()
        
        all_pred_pos = []
        all_true_pos = []

        for batch_idx, (signal, true_pos, true_height, true_width) in enumerate(self.train_loader):
            signal = signal.to(self.device)
            true_pos = true_pos.to(self.device)
            true_height = true_height.to(self.device)
            true_width = true_width.to(self.device)

            pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))

            loss, loss_dict = self.criterion(pred_pos, pred_height, pred_width,
                                             true_pos, true_height, true_width)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_metrics.update(loss.item(), loss_dict)
            
            all_pred_pos.append((pred_pos > self.threshold).detach().cpu().numpy())
            all_true_pos.append(true_pos.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch:2d} | Batch {batch_idx:3d}] Loss: {loss.item():.4f}")

        avg_loss, avg_details, _, _, _ = train_metrics.avg()
        
        pred_flat = np.concatenate(all_pred_pos).flatten()
        true_flat = np.concatenate(all_true_pos).flatten()
        train_f1 = f1_score(true_flat, pred_flat, zero_division=0)
        
        print(f"\n Epoch {epoch} | Train Loss: {avg_loss:.4f} | F1: {train_f1:.4f}")
        print(f"   └─ pos={avg_details['pos']:.4f}, h={avg_details['height']:.4f}, w={avg_details['width']:.4f}")
        
        return avg_loss, train_f1

    def _validate_epoch(self, epoch):
        self.model.eval()
        val_metrics = SignalMetrics()
        
        all_pred_pos = []
        all_true_pos = []

        with torch.no_grad():
            for signal, true_pos, true_height, true_width in self.val_loader:
                signal = signal.to(self.device)
                true_pos = true_pos.to(self.device)
                true_height = true_height.to(self.device)
                true_width = true_width.to(self.device)

                pred_pos, pred_height, pred_width = self.model(signal.unsqueeze(1))
                loss, loss_dict = self.criterion(pred_pos, pred_height, pred_width,
                                                 true_pos, true_height, true_width)
                
                val_metrics.update(loss.item(), loss_dict)
                
                all_pred_pos.append((pred_pos > self.threshold).cpu().numpy())
                all_true_pos.append(true_pos.cpu().numpy())

        avg_loss, avg_details, _, _, _ = val_metrics.avg()
        
        pred_flat = np.concatenate(all_pred_pos).flatten()
        true_flat = np.concatenate(all_true_pos).flatten()
        
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