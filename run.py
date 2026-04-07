import argparse
import torch
import os 
import onnx
import onnxruntime
from torch.utils.data import DataLoader, random_split
from dataloader.SignalDataset import SignalDataset
from trainer.SignalTrainer import SignalTrainer, SignalTester

SPLIT = 42

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
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--save_test_data", action="store_true", help="Save test set")
    return parser.parse_args()

# ----------------------- PREPARE DATASET FUNCTION -----------------------
def prepare_dataset(args):
    """Prepare dataset splits and dataloaders for training."""
    dataset = SignalDataset(
        signal_path=args.signal_path,
        info_path=args.info_path,
        chunks=args.chunks,
        max_peaks=args.max_peaks,
    )
    print(f"[INFO] Dataset loaded with {len(dataset)} signals")
 
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT)
    )

    if (args.save_test_data):
        torch.save(test_dataset, "data\\test_dataset.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"[INFO] Train set: {len(train_dataset)} | Val set: {len(val_dataset)} | Test set: {len(test_dataset)}")
    return train_loader, val_loader, test_loader 


# ----------------------- MAIN ENTRY POINT -----------------------
def main():
    args = argparse_config()
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    train_loader, val_loader, test_loader = prepare_dataset(args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    trainer = SignalTrainer(
        device, 
        train_loader, 
        val_loader
    )
    
    print(f"[INFO] Training for {args.epochs} epochs")
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

    # EXPORT TO ONNX
    # ============================================================
    torch_model = trainer.model
    torch_model.eval()  # disable dropout / batchnorm training mode
    torch_model.cpu()

    example_inputs = (torch.randn(1, 1, torch_model.signal_length),)
    onnx_path = os.path.join(checkpoint_dir, "peak_detection_model.onnx")
    onnx_program = torch.onnx.export(
        torch_model,
        example_inputs,
        onnx_path,
        input_names=["input"],
        output_names=["positions", "heights", "widths"],
        opset_version=17
    )
    onnx_program.save(onnx_path)
    print(f"ONNX model saved to: {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid ✅")
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("ONNX outputs:", [o.name for o in ort_session.get_outputs()])

if __name__ == "__main__":
    main()