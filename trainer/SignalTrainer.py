import torch
import torch.nn as nn
import torch.optim as optim

from networks.models import SimpleSignalNet, SignalNet


class SignalTrainer:
    def __init__(self, train_loader, val_loader=None, device=None, lr=1e-3):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        self.model = SimpleSignalNet().to(self.device)  # test
        # self.model = SignalNet().to(self.device)

        # Loss Function
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=5):
        """Train the model for a given number of epochs."""
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch_idx, signals in enumerate(self.train_loader):
                signals = signals.to(self.device)

                # Dummy targets for now (mean of each signal)
                targets = signals.mean(dim=1, keepdim=True)

                # Forward
                preds = self.model(signals)

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # Backward - Compute the loss and its gradients
                loss = self.criterion(preds, targets)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if batch_idx % 10 == 0:
                    print(f"[Epoch {epoch}] Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(self.train_loader)
            print(f"→ Epoch {epoch} completed | Avg loss: {avg_loss:.4f}")

            # Optional validation
            if self.val_loader:
                self.validate()

    def validate(self):
        """Run a validation loop."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for signals in self.val_loader:
                signals = signals.to(self.device)
                targets = signals.mean(dim=1, keepdim=True)
                preds = self.model(signals)
                loss = self.criterion(preds, targets)
                total_loss += loss.item()

        print(f"[VALIDATION] Avg loss: {total_loss / len(self.val_loader):.4f}")
