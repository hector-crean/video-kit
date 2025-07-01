"""
Python utilities for converting SAM models to ONNX format.
"""

import torch
import argparse
import sys
from pathlib import Path


def download_sam_checkpoint(model_type: str, save_dir: Path) -> Path:
    """Download SAM checkpoint from official repository."""
    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    if model_type not in urls:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(urls.keys())}")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"sam_{model_type}.pth"
    
    if checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path
    
    print(f"Downloading {model_type} checkpoint...")
    try:
        import urllib.request
        urllib.request.urlretrieve(urls[model_type], checkpoint_path)
        print(f"Downloaded to: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        sys.exit(1)


def convert_sam_to_onnx(checkpoint_path: Path, output_path: Path, model_type: str):
    """Convert SAM model to ONNX format."""
    try:
        # Try to import segment_anything
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            print("Error: segment_anything package not found.")
            print("Please run './setup.sh' first to install dependencies.")
            sys.exit(1)
        
        print(f"Loading {model_type} model from {checkpoint_path}")
        
        # Load the model
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.eval()
        
        # Create dummy inputs
        # SAM expects images of size 1024x1024
        dummy_image = torch.randn(1, 3, 1024, 1024)
        
        print("Converting to ONNX...")
        
        # Export to ONNX (image encoder only for simplicity)
        # Note: Full SAM has multiple components (image encoder, prompt encoder, mask decoder)
        # For this example, we'll export just the image encoder
        torch.onnx.export(
            sam.image_encoder,
            dummy_image,
            str(output_path),
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['image_embeddings'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'image_embeddings': {0: 'batch_size'}
            },
            verbose=True
        )
        
        print(f"Successfully converted to: {output_path}")
        print(f"Model size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def main():
    """Main entry point for the convert-sam command."""
    parser = argparse.ArgumentParser(description="Convert SAM model to ONNX format")
    parser.add_argument(
        "--model-type", 
        choices=["vit_b", "vit_l", "vit_h"], 
        default="vit_b",
        help="SAM model variant to convert"
    )
    parser.add_argument(
        "--checkpoint", 
        type=Path,
        help="Path to SAM checkpoint file (if not provided, will download)"
    )
    parser.add_argument(
        "--output", 
        type=Path,
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to download checkpoints to"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = download_sam_checkpoint(args.model_type, args.download_dir)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(f"src/model/sam_{args.model_type}_01ec64.onnx")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to ONNX
    convert_sam_to_onnx(checkpoint_path, output_path, args.model_type)
    
    print("\nNext steps:")
    print(f"1. The ONNX model has been saved to: {output_path}")
    print("2. Update the model filename in build.rs and src/model/mod.rs if needed")
    print("3. Run 'cargo build' to generate the Rust model code")
    print("4. Test the model with: cargo run --bin sam -- --image your_image.jpg")


__all__ = ['main', 'download_sam_checkpoint', 'convert_sam_to_onnx'] 