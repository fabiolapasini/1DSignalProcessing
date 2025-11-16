import argparse
import torch
import os 
from torch.utils.data import DataLoader, random_split
from dataloader.SignalDataset import SignalDataset
from trainer.SignalTrainer import SignalTrainer, SignalTester

def argparse_config():
    parser = argparse.ArgumentParser(
        description="Run SignalDataset loading and training."
    )
    parser.add_argument("--signal_path", type=str, required=True, help="Path to signal binary file (.raw).")
    parser.add_argument("--info_path", type=str, required=True, help="Path to info binary file (.raw).")
    parser.add_argument("--chunks", type=int, default=1024, help="Signal length per sample.")
    parser.add_argument("--max_peaks", type=int, default=3, help="Maximum number of peaks to parse.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DataLoader.")
    parser.add_argument("--train_split", type=float, default=0.75, help="Train split ratio.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--normalize", action="store_true", help="Enable signal normalization.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate for the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

# ----------------------- PREPARE DATASET FUNCTION -----------------------
def prepare_dataset(args):
    """Prepare dataset splits and dataloaders for training."""
    dataset = SignalDataset(
        signal_path=args.signal_path,
        info_path=args.info_path,
        chunks=args.chunks,
        max_peaks=args.max_peaks,
        normalize=args.normalize
    )
    print(f"[INFO] Dataset loaded with {len(dataset)} signals (normalize={args.normalize})")
    
    if args.normalize:
        print(f"[INFO] Normalization stats - Mean: {dataset.signal_mean:.4f}, Std: {dataset.signal_std:.4f}")
    
    assert 0 < args.train_split < 1, "train_split must be between 0 and 1"
    assert 0 < args.val_split < 1, "val_split must be between 0 and 1"
    assert args.train_split + args.val_split < 1, "train_split + val_split must be < 1"
    
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    assert train_size > 0 and val_size > 0 and test_size > 0, \
        "Dataset too small or split ratios invalid. Check your splits!"
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"[INFO] Train set: {len(train_dataset)} | Val set: {len(val_dataset)} | Test set: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

# ----------------------- MAIN ENTRY POINT -----------------------
def main():
    args = argparse_config()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    train_loader, val_loader, test_loader = prepare_dataset(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    trainer = SignalTrainer(
        device, 
        train_loader, 
        val_loader, 
        lr=args.lr, 
        patience=args.patience
    )
    
    print(f"[INFO] Training with lr={args.lr}, patience={args.patience}, epochs={args.epochs}")
    trainer.train(epochs=args.epochs)
    
    print("Running test evaluation...")
    tester = SignalTester(trainer.model, device)
    tester.test(test_loader)
    
    checkpoint_dir = "networks"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    
    print("Saving the final model...")
    torch.save(trainer.model.state_dict(), checkpoint_path)
    print(f"✅ Model saved at {checkpoint_path}")

if __name__ == "__main__":
    main()