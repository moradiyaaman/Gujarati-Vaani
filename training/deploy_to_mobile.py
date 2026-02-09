#!/usr/bin/env python3
"""
================================================================================
DEPLOY TO MOBILE - INT8 QUANTIZATION SCRIPT
================================================================================
Takes the fine-tuned pytorch_model.bin from Azure training and prepares it
for high-speed mobile inference (10-30 second target for large documents).

The Speed Secret: INT8 Dynamic Quantization
--------------------------------------------
Neural network weights are stored as 32-bit floats (FP32) by default.
INT8 quantization converts these to 8-bit integers:

    FP32 Weight: 0.123456789 (4 bytes)  →  INT8 Weight: 31 (1 byte)

Benefits:
    ✅ 4x smaller model size (200MB → ~50-75MB, with structure ~150MB)
    ✅ 2-3x faster inference on CPU (phones don't have powerful GPUs)
    ✅ Lower memory bandwidth requirements
    ✅ Minimal quality loss for TTS (< 2% MOS degradation)

Why "Dynamic" Quantization?
---------------------------
- Static: Requires calibration dataset, better for CNNs
- Dynamic: Quantizes weights offline, activations at runtime
- For Transformers/VITS: Dynamic is preferred (handles variable-length sequences)

Usage:
    python deploy_to_mobile.py \\
        --checkpoint ./model_weights/finetuned/final \\
        --output ./model_weights/mobile \\
        --verify

Output Structure:
    model_weights/mobile/
    ├── model_quantized.pt      (Quantized model for mobile)
    ├── config.json             (Model configuration)
    ├── tokenizer.json          (Tokenizer with any new vocabulary)
    ├── tokenizer_config.json   (Tokenizer settings)
    ├── special_tokens_map.json (Special tokens)
    └── deployment_info.json    (Metadata about quantization)

================================================================================
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

# For model loading
from transformers import VitsModel, AutoTokenizer

# Add parent for utils
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_model_size(model_path: Path) -> float:
    """Get model size in MB."""
    if model_path.is_file():
        return model_path.stat().st_size / (1024 * 1024)
    elif model_path.is_dir():
        total = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        return total / (1024 * 1024)
    return 0.0


def load_finetuned_model(
    checkpoint_path: Path,
    logger: logging.Logger
) -> Tuple[VitsModel, AutoTokenizer]:
    """
    Load the fine-tuned model from checkpoint.
    
    The checkpoint should contain:
    - pytorch_model.bin or model.safetensors
    - config.json
    - tokenizer files
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Check for required files
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")
    
    # Load model
    model = VitsModel.from_pretrained(str(checkpoint_path))
    model.eval()
    
    # Load tokenizer (may have updated vocabulary from fine-tuning)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  - Vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def quantize_model(
    model: VitsModel,
    logger: logging.Logger
) -> nn.Module:
    """
    Apply INT8 Dynamic Quantization to the model.
    
    We quantize:
    - nn.Linear layers (majority of compute)
    - nn.LSTM/GRU layers (if any)
    
    We DON'T quantize:
    - Conv1d layers (vocoder needs precision for audio quality)
    - Embedding layers (small, lookup-based)
    
    Technical Details:
    ------------------
    torch.quantize_dynamic() does the following:
    1. Converts weight matrices from FP32 to INT8
    2. Stores scale/zero-point for dequantization
    3. At runtime, dequantizes weights → computes → outputs FP32
    
    This is different from static quantization which also quantizes activations.
    """
    logger.info("Applying INT8 Dynamic Quantization...")
    logger.info("  - Target layers: nn.Linear, nn.LSTM, nn.GRU")
    logger.info("  - Preserving precision: nn.Conv1d (vocoder quality)")
    
    # Quantize
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
        dtype=torch.qint8  # INT8 quantization
    )
    
    # Count quantized parameters
    original_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # FP32 = 4 bytes
    
    logger.info(f"Quantization complete!")
    logger.info(f"  - Original param memory: ~{original_size:.1f} MB")
    logger.info(f"  - Expected reduction: 50-75%")
    
    return quantized_model


def verify_quantized_model(
    original_model: VitsModel,
    quantized_model: nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Quick verification that quantized model produces valid output.
    """
    logger.info("Verifying quantized model...")
    
    test_text = "નમસ્તે"  # "Hello" in Gujarati
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # Generate with original
    with torch.no_grad():
        original_output = original_model(**inputs)
        original_audio = original_output.waveform.squeeze().numpy()
    
    # Generate with quantized
    with torch.no_grad():
        quantized_output = quantized_model(**inputs)
        quantized_audio = quantized_output.waveform.squeeze().numpy()
    
    # Compare
    import numpy as np
    
    # Check shapes match
    if original_audio.shape != quantized_audio.shape:
        logger.warning(f"Shape mismatch: {original_audio.shape} vs {quantized_audio.shape}")
    
    # Check correlation (should be > 0.95 for good quantization)
    correlation = np.corrcoef(original_audio.flatten()[:1000], 
                               quantized_audio.flatten()[:1000])[0, 1]
    
    logger.info(f"  - Audio correlation: {correlation:.4f}")
    
    if correlation > 0.95:
        logger.info("  ✅ Quantization verification PASSED")
    elif correlation > 0.85:
        logger.warning("  ⚠️ Quantization shows some degradation (acceptable)")
    else:
        logger.error("  ❌ Quantization may have issues - please verify audio quality")
    
    # Save verification samples
    import scipy.io.wavfile as wav
    
    verify_dir = output_dir / "verification"
    verify_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    wav.write(verify_dir / "original_sample.wav", sample_rate, 
              (original_audio * 32767).astype(np.int16))
    wav.write(verify_dir / "quantized_sample.wav", sample_rate,
              (quantized_audio * 32767).astype(np.int16))
    
    logger.info(f"  - Verification samples saved to: {verify_dir}")


def save_for_mobile(
    quantized_model: nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    checkpoint_path: Path,
    logger: logging.Logger
):
    """
    Save the quantized model in a format ready for mobile deployment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save quantized model as .pt file
    model_path = output_dir / "model_quantized.pt"
    torch.save(quantized_model.state_dict(), model_path)
    logger.info(f"Saved quantized model: {model_path}")
    logger.info(f"  - Size: {get_model_size(model_path):.1f} MB")
    
    # Copy config.json
    config_src = checkpoint_path / "config.json"
    if config_src.exists():
        shutil.copy(config_src, output_dir / "config.json")
    
    # Save tokenizer (includes any vocabulary updates from fine-tuning)
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Saved tokenizer with vocab size: {len(tokenizer)}")
    
    # Copy training args if available
    training_args_src = checkpoint_path / "training_args.json"
    if training_args_src.exists():
        shutil.copy(training_args_src, output_dir / "training_args.json")
    
    # Create deployment info
    deployment_info = {
        "created_at": datetime.now().isoformat(),
        "source_checkpoint": str(checkpoint_path),
        "quantization": {
            "type": "INT8 Dynamic",
            "target_layers": ["nn.Linear", "nn.LSTM", "nn.GRU"],
            "preserved_layers": ["nn.Conv1d", "nn.Embedding"]
        },
        "model_file": "model_quantized.pt",
        "tokenizer_files": [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt"
        ],
        "sample_rate": 16000,
        "target_device": "mobile_cpu"
    }
    
    with open(output_dir / "deployment_info.json", 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info(f"\n✅ Mobile deployment package ready!")
    logger.info(f"   Location: {output_dir}")
    logger.info(f"   Total size: {get_model_size(output_dir):.1f} MB")


def deploy_to_mobile(
    checkpoint_path: Path,
    output_dir: Path,
    verify: bool,
    logger: logging.Logger
):
    """
    Main deployment pipeline.
    """
    logger.info("=" * 60)
    logger.info("GUJARATI VAANI - DEPLOY TO MOBILE")
    logger.info("=" * 60)
    
    # Step 1: Load fine-tuned model
    logger.info("\n📦 Step 1: Loading fine-tuned model...")
    original_model, tokenizer = load_finetuned_model(checkpoint_path, logger)
    
    original_size = get_model_size(checkpoint_path)
    logger.info(f"Original checkpoint size: {original_size:.1f} MB")
    
    # Step 2: Quantize
    logger.info("\n⚡ Step 2: Applying INT8 quantization...")
    quantized_model = quantize_model(original_model, logger)
    
    # Step 3: Verify (optional)
    if verify:
        logger.info("\n🔍 Step 3: Verifying quantized model...")
        verify_quantized_model(original_model, quantized_model, tokenizer, 
                               output_dir, logger)
    
    # Step 4: Save for mobile
    logger.info("\n💾 Step 4: Saving for mobile deployment...")
    save_for_mobile(quantized_model, tokenizer, output_dir, checkpoint_path, logger)
    
    # Summary
    final_size = get_model_size(output_dir)
    reduction = (1 - final_size / original_size) * 100
    
    logger.info("\n" + "=" * 60)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Original size:  {original_size:.1f} MB")
    logger.info(f"Final size:     {final_size:.1f} MB")
    logger.info(f"Reduction:      {reduction:.1f}%")
    logger.info(f"Output:         {output_dir}")
    logger.info("=" * 60)
    logger.info("\n📱 Next steps:")
    logger.info("   1. Copy model_weights/mobile/ to your app")
    logger.info("   2. Update app.py to load model_quantized.pt")
    logger.info("   3. Test on device with long documents")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize fine-tuned model for mobile deployment"
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default="./model_weights/finetuned/final",
        help="Path to fine-tuned checkpoint directory"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="./model_weights/mobile",
        help="Output directory for mobile-ready model"
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help="Run verification after quantization"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Run training first or provide correct path")
        sys.exit(1)
    
    deploy_to_mobile(checkpoint_path, output_dir, args.verify, logger)


if __name__ == "__main__":
    main()
