import argparse
import torch
from torch.utils.data import DataLoader, random_split

from dataloader.SignalDataset import SignalDataset
from trainer.SignalTrainer import SignalTrainer, SignalTester


def argparse_config():
    parser = argparse.ArgumentParser(
        description="Run SignalDataset loading and optional augmentation."
    )
    parser.add_argument("--signal_path", type=str, required=True, help="Path to signal binary file (.raw).")
    parser.add_argument("--info_path", type=str, required=True, help="Path to info binary file (.raw).")
    parser.add_argument("--chunks", type=int, default=1024, help="Signal length per sample.")
    parser.add_argument("--max_peaks", type=int, default=3, help="Maximum number of peaks to parse.")
    parser.add_argument("--augment", action="store_true", help="Apply signal augmentation (flip).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DataLoader.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data in the DataLoader.")
    parser.add_argument("--train_split", type=float, default=0.75, help="Train split ratio.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio.")
    return parser.parse_args()


# ----------------------- PREPARE DATASET FUNCTION -----------------------
def prepare_dataset(args):
    """Prepare dataset splits and dataloaders for training."""
    # Initialize dataset
    dataset = SignalDataset(
        signal_path=args.signal_path,
        info_path=args.info_path,
        chunks=args.chunks,
        max_peaks=args.max_peaks,
        augment=args.augment
    )

    print(f"[INFO] Dataset loaded with {len(dataset)} signals (augment={args.augment})")

    # --- Compute split sizes ---
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # --- Perform random split with reproducibility ---
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Train set: {len(train_dataset)} | Val set: {len(val_dataset)} | Test set: {len(test_dataset)}")

    # --- Iterate few batches to check shapes ---
    '''for i, batch in enumerate(train_loader):
        print(f"Train batch {i}: shape {batch.shape}")
        if i == 1:
            break'''

    return train_loader, val_loader, test_loader


# ----------------------- MAIN ENTRY POINT -----------------------
def main():
    args = argparse_config()

    # Prepare dataset
    train_loader, val_loader, test_loader = prepare_dataset(args)
    
    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = SignalTrainer(device, train_loader, val_loader)
    trainer.train(epochs=10)

    # Inference
    print("\nRunning test evaluation...")
    tester = SignalTester(trainer.model, device)
    tester.test(test_loader)

    # Save the model
    print("\nSaving the model...")
    torch.save(tester.model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
