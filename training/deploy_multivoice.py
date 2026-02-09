#!/usr/bin/env python3
"""
================================================================================
GUJARATI VAANI - MULTI-VOICE DEPLOYMENT SCRIPT
================================================================================
Quantizes and packages both Male and Female voice models for mobile deployment.

Output Structure:
    model_weights/mobile/
    ├── male/
    │   ├── model_quantized.pt
    │   ├── config.json
    │   ├── vocab.json
    │   └── deployment_info.json
    └── female/
        ├── model_quantized.pt
        ├── config.json
        ├── vocab.json
        └── deployment_info.json

Usage:
    python deploy_multivoice.py              # Deploy both voices
    python deploy_multivoice.py --voice male # Deploy male only

Author: Gujarati Vaani Team
Date: January 2026
================================================================================
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.quantization
import soundfile as sf

from transformers import VitsModel, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def quantize_voice_model(
    checkpoint_path: Path,
    output_dir: Path,
    voice: str,
    verify: bool = True
) -> Path:
    """
    Quantize a trained voice model for mobile deployment.
    
    Uses INT8 dynamic quantization for optimal mobile performance.
    """
    logger.info(f"\n🔧 Quantizing {voice.upper()} voice model...")
    logger.info(f"   Input: {checkpoint_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    logger.info("   Loading trained model...")
    model = VitsModel.from_pretrained(str(checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    
    model.eval()
    
    # Count original parameters
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(f"   Original size: {original_size / 1024 / 1024:.2f} MB")
    
    # Apply INT8 dynamic quantization
    logger.info("   Applying INT8 dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv1d},
        dtype=torch.qint8
    )
    
    # Save quantized model
    quantized_path = output_dir / "model_quantized.pt"
    torch.save(quantized_model.state_dict(), quantized_path)
    
    # Get quantized size
    quantized_size = os.path.getsize(quantized_path)
    compression_ratio = original_size / quantized_size
    
    logger.info(f"   Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
    logger.info(f"   Compression ratio: {compression_ratio:.2f}x")
    
    # Copy tokenizer files
    logger.info("   Copying tokenizer files...")
    tokenizer_files = ['vocab.json', 'tokenizer_config.json', 'special_tokens_map.json']
    for file in tokenizer_files:
        src = checkpoint_path / file
        if src.exists():
            shutil.copy(src, output_dir / file)
    
    # Copy config
    config_src = checkpoint_path / "config.json"
    if config_src.exists():
        shutil.copy(config_src, output_dir / "config.json")
    
    # Create deployment info
    deployment_info = {
        'voice': voice,
        'model_type': 'VITS (INT8 Quantized)',
        'base_model': 'facebook/mms-tts-guj',
        'sample_rate': 16000,
        'quantization': 'INT8 Dynamic',
        'original_size_mb': original_size / 1024 / 1024,
        'quantized_size_mb': quantized_size / 1024 / 1024,
        'compression_ratio': compression_ratio,
        'created_at': datetime.now().isoformat(),
        'usage': {
            'python': f"model = torch.load('model_quantized.pt')",
            'android': "Load in assets/model_{voice}.pt"
        }
    }
    
    with open(output_dir / "deployment_info.json", 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    # Verification
    if verify:
        logger.info("   Verifying quantized model...")
        try:
            test_text = "નમસ્તે, આ ટેસ્ટ છે."
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                output = quantized_model(**inputs)
                waveform = output.waveform.squeeze().cpu().numpy()
            
            # Save verification sample
            verify_path = output_dir / f"verify_{voice}.wav"
            sf.write(verify_path, waveform, 16000)
            
            logger.info(f"   ✓ Verification passed! Sample: {verify_path.name}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Verification failed: {e}")
    
    logger.info(f"   ✅ {voice.upper()} model quantized successfully!")
    
    return output_dir


def deploy_multivoice(
    base_model_dir: Path,
    output_base_dir: Path,
    voices: list = ['male', 'female']
):
    """Deploy multiple voice models."""
    
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║       GUJARATI VAANI - MULTI-VOICE MOBILE DEPLOYMENT             ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    deployed_models = {}
    
    for voice in voices:
        checkpoint_path = base_model_dir / f"finetuned_{voice}" / "final"
        
        # Check if best model exists, use it instead
        best_path = base_model_dir / f"finetuned_{voice}" / "best"
        if best_path.exists():
            checkpoint_path = best_path
            logger.info(f"📦 Using BEST checkpoint for {voice}")
        
        if not checkpoint_path.exists():
            logger.warning(f"⚠️ No checkpoint found for {voice} at {checkpoint_path}")
            logger.warning(f"   Run: python train_multivoice.py --voice {voice}")
            continue
        
        output_dir = output_base_dir / voice
        
        try:
            quantize_voice_model(checkpoint_path, output_dir, voice, verify=True)
            deployed_models[voice] = output_dir
        except Exception as e:
            logger.error(f"❌ Failed to deploy {voice}: {e}")
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("📦 MULTI-VOICE DEPLOYMENT COMPLETE!")
    logger.info("=" * 70)
    
    for voice, path in deployed_models.items():
        size = os.path.getsize(path / "model_quantized.pt") / 1024 / 1024
        logger.info(f"   🎤 {voice.upper()}: {path} ({size:.1f} MB)")
    
    logger.info("")
    logger.info("📱 Mobile Integration:")
    logger.info("   Copy the following to your app's assets folder:")
    logger.info(f"      {output_base_dir}/")
    logger.info("      ├── male/model_quantized.pt")
    logger.info("      └── female/model_quantized.pt")
    logger.info("")
    logger.info("💻 Code Integration:")
    logger.info("   ```python")
    logger.info("   # Load male voice")
    logger.info("   male_model = torch.load('assets/male/model_quantized.pt')")
    logger.info("   ")
    logger.info("   # Load female voice")
    logger.info("   female_model = torch.load('assets/female/model_quantized.pt')")
    logger.info("   ```")
    logger.info("=" * 70)
    
    return deployed_models


def main():
    parser = argparse.ArgumentParser(
        description="Deploy multi-voice Gujarati TTS models for mobile"
    )
    parser.add_argument(
        '--voice', '-v',
        type=str,
        choices=['male', 'female', 'both'],
        default='both',
        help="Which voice to deploy"
    )
    parser.add_argument(
        '--model-dir', '-m',
        type=str,
        default='./model_weights',
        help="Base directory containing trained models"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./model_weights/mobile',
        help="Output directory for quantized models"
    )
    args = parser.parse_args()
    
    base_model_dir = Path(args.model_dir)
    output_base_dir = Path(args.output_dir)
    
    voices = ['male', 'female'] if args.voice == 'both' else [args.voice]
    
    deploy_multivoice(base_model_dir, output_base_dir, voices)


if __name__ == "__main__":
    main()
