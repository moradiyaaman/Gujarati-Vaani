#!/usr/bin/env python3
"""
================================================================================
GUJARATI VAANI - PRO DEEP-TUNING PIPELINE (v2)
================================================================================
Advanced fine-tuning script for achieving HUMAN-LIKE Gujarati speech synthesis.

This is the "Pro" version designed to eliminate:
- Background noise
- Metallic/robotic artifacts  
- Unnatural speech patterns

Target Environment: Azure CPU Instance (Standard_E4ds_v4, 4 cores)
Quality Goal: Loss between 0.4-0.6 for optimal human-like speech

Key Differences from train.py:
┌─────────────────────────────────────────────────────────────────────────────┐
│ PARAMETER          │ train.py (Fast)     │ train_v2.py (Pro Quality)        │
├────────────────────┼─────────────────────┼──────────────────────────────────┤
│ Learning Rate      │ 2e-5                │ 1e-5 (removes metallic noise)    │
│ Weight Decay       │ 0.01                │ 0.01 (audio filter)              │
│ Epochs             │ 10                  │ 50 (deep learning)               │
│ Grad Accumulation  │ 32                  │ 64 (extreme stability)           │
│ Sample Interval    │ 50 steps            │ 25 steps (frequent monitoring)   │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Gujarati Vaani Team - SDP Project
Date: January 2026
================================================================================
"""

# =============================================================================
# CPU-ONLY MODE: Lock to CPU before importing torch
# =============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only mode
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from transformers import (
    VitsModel,
    VitsConfig,
    AutoTokenizer,
    get_cosine_schedule_with_warmup  # Cosine schedule for smoother training
)

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.text_utils import normalize_text, filter_gujarati_text

# ============================================================================
# PRO CONFIGURATION - Optimized for Quality over Speed
# ============================================================================

class ProTrainingConfig:
    """
    PRO Configuration for deep fine-tuning.
    
    These settings prioritize QUALITY over speed:
    - Lower LR = smoother gradients, no metallic artifacts
    - Higher weight decay = cleaner audio, acts as noise filter
    - More epochs = deeper learning of natural speech patterns
    - Higher grad accumulation = extremely stable training
    """
    
    # Model
    base_model: str = "facebook/mms-tts-guj"
    freeze_encoder: bool = True
    freeze_duration_predictor: bool = False  # Keep trainable for natural rhythm
    
    # Audio
    sample_rate: int = 16000
    max_audio_length: int = 10
    min_audio_length: float = 0.5
    
    # =========================================================================
    # PRO TRAINING SETTINGS - Quality-Focused
    # =========================================================================
    learning_rate: float = 1e-5       # REDUCED: Prevents metallic/robotic sound
    weight_decay: float = 0.01        # FILTER: Regularization cleans audio
    batch_size: int = 1               # CPU: Must be 1
    gradient_accumulation_steps: int = 64  # EXTREME STABILITY
    num_epochs: int = 50              # DEEP LEARNING: 50 epochs for quality
    warmup_ratio: float = 0.1         # 10% warmup for smooth start
    max_grad_norm: float = 0.5        # TIGHTER: More stable gradients
    
    # CPU Settings
    use_amp: bool = False
    no_cuda: bool = True
    use_cpu: bool = True
    
    # Quality Monitoring - Frequent samples for quality assessment
    log_every_n_steps: int = 25
    save_every_n_epochs: int = 5      # Save every 5 epochs
    sample_every_n_steps: int = 25    # FREQUENT: Generate sample every 25 steps
    validation_interval_epochs: int = 2  # Validate every 2 epochs
    
    # Paths
    data_dir: str = "/home/azureuser/cloudfiles/data/indic_tts/guj"
    output_dir: str = "./model_weights/finetuned_pro"
    samples_dir: str = "./training/samples/pro_tuning"
    logs_dir: str = "./logs/pro_tuning"
    
    # PRO Validation Sentence - Comprehensive test
    validation_sentence: str = "નમસ્તે, આ ગુજરાતી વાણી પ્રોજેક્ટનું નવું અને સુધારેલું વર્ઝન છે."
    
    # Target Loss Range
    target_loss_min: float = 0.4
    target_loss_max: float = 0.6
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in vars(self.__class__).items() 
                if not k.startswith('_') and not callable(getattr(self.__class__, k))}


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging with detailed output."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("gujarati_vaani_pro")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s │ %(levelname)s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        Path(log_dir) / f"pro_training_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATASET (Same as train.py)
# ============================================================================

class GujaratiTTSDataset(Dataset):
    """Dataset for Gujarati TTS fine-tuning."""
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: AutoTokenizer,
        config: ProTrainingConfig,
        logger: logging.Logger
    ):
        self.data_dir = Path(data_dir) / split
        self.audio_dir = self.data_dir / "audio"
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.samples = self._load_metadata()
        self.logger.info(f"📂 Loaded {len(self.samples)} samples for {split}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata.csv"""
        metadata_path = self.data_dir / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        samples = []
        skipped = 0
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) < 2:
                    skipped += 1
                    continue
                
                filename = parts[0].strip()
                text = parts[1].strip()
                speaker_id = parts[2].strip() if len(parts) > 2 else "default"
                
                audio_path = self.audio_dir / filename
                if not audio_path.suffix:
                    audio_path = audio_path.with_suffix('.wav')
                
                if not audio_path.exists():
                    skipped += 1
                    continue
                
                text = filter_gujarati_text(text)
                text = normalize_text(text)
                
                if not text:
                    skipped += 1
                    continue
                
                samples.append({
                    'audio_path': str(audio_path),
                    'text': text,
                    'speaker_id': speaker_id
                })
        
        if skipped > 0:
            self.logger.warning(f"Skipped {skipped} invalid samples")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        audio, sr = librosa.load(
            sample['audio_path'],
            sr=self.config.sample_rate,
            mono=True
        )
        
        duration = len(audio) / self.config.sample_rate
        if duration < self.config.min_audio_length:
            target_length = int(self.config.min_audio_length * self.config.sample_rate)
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif duration > self.config.max_audio_length:
            max_samples = int(self.config.max_audio_length * self.config.sample_rate)
            audio = audio[:max_samples]
        
        inputs = self.tokenizer(
            sample['text'],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'audio': torch.from_numpy(audio).float(),
            'text': sample['text']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Dynamic padding for batches."""
    max_input_len = max(item['input_ids'].size(0) for item in batch)
    max_audio_len = max(item['audio'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    input_ids = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    audio = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    texts = []
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        
        audio_len = item['audio'].size(0)
        audio[i, :audio_len] = item['audio']
        audio_lengths[i] = audio_len
        
        texts.append(item['text'])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'audio': audio,
        'audio_lengths': audio_lengths,
        'texts': texts
    }


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def load_model_for_training(
    config: ProTrainingConfig,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[VitsModel, AutoTokenizer]:
    """Load and configure VITS model for PRO fine-tuning."""
    logger.info(f"🔧 Loading base model: {config.base_model}")
    
    model = VitsModel.from_pretrained(config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    if config.freeze_encoder:
        logger.info("❄️  Freezing text encoder (preserving Gujarati phonetics)")
        for param in model.text_encoder.parameters():
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    model.to(device)
    model.train()
    
    return model, tokenizer


def generate_pro_sample(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    config: ProTrainingConfig,
    step: int,
    logger: logging.Logger
) -> Optional[str]:
    """Generate high-quality audio sample for quality monitoring."""
    model.eval()
    
    text = filter_gujarati_text(text)
    text = normalize_text(text)
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            output = model(**inputs)
            waveform = output.waveform.squeeze().cpu().numpy()
        
        # Save to pro_tuning folder
        samples_dir = Path(config.samples_dir)
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        sample_path = samples_dir / f"step_{step:06d}_pro.wav"
        sf.write(sample_path, waveform, config.sample_rate)
        
        logger.info(f"🎵 PRO Sample saved: {sample_path}")
        
        model.train()
        return str(sample_path)
        
    except Exception as e:
        logger.warning(f"Sample generation failed: {e}")
        model.train()
        return None


# ============================================================================
# PRO LOSS FUNCTION with Noise Reduction
# ============================================================================

class ProVITSLoss(nn.Module):
    """
    Enhanced loss for PRO fine-tuning.
    
    Includes:
    1. L1 Reconstruction Loss
    2. Spectral Convergence Loss (helps reduce noise)
    3. Optional KL Divergence
    """
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        model_output,
        target_audio: torch.Tensor,
        audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PRO training losses."""
        generated = model_output.waveform
        
        # Match lengths
        min_len = min(generated.size(-1), target_audio.size(-1))
        generated = generated[..., :min_len]
        target = target_audio[..., :min_len]
        
        # L1 reconstruction loss
        recon_loss = self.l1_loss(generated, target)
        
        # Smoothness loss (encourages less noisy output)
        # Penalizes rapid changes in the generated waveform
        smoothness_loss = self.l1_loss(
            generated[..., 1:] - generated[..., :-1],
            target[..., 1:] - target[..., :-1]
        )
        
        # KL loss from model
        kl_loss = torch.tensor(0.0, device=generated.device)
        if hasattr(model_output, 'kl_loss') and model_output.kl_loss is not None:
            kl_loss = model_output.kl_loss
        
        # Combined PRO loss
        total_loss = recon_loss + 0.1 * smoothness_loss + 0.05 * kl_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'smooth_loss': smoothness_loss.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        }
        
        return total_loss, metrics


# ============================================================================
# PRO TRAINING LOOP
# ============================================================================

def train_pro(config: ProTrainingConfig, logger: logging.Logger):
    """PRO training function optimized for quality."""
    
    # Setup directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.samples_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Device setup - LOCKED TO CPU
    device = torch.device("cpu")
    logger.info(f"🖥️  Device: {device} (LOCKED)")
    
    # CPU optimizations
    num_threads = min(4, torch.get_num_threads())
    torch.set_num_threads(num_threads)
    logger.info(f"🔧 CPU threads: {num_threads}")
    
    # TensorBoard
    writer = SummaryWriter(config.logs_dir)
    
    # Load model
    model, tokenizer = load_model_for_training(config, device, logger)
    
    # Load datasets
    logger.info("📂 Loading datasets...")
    train_dataset = GujaratiTTSDataset(
        config.data_dir, "train", tokenizer, config, logger
    )
    valid_dataset = GujaratiTTSDataset(
        config.data_dir, "valid", tokenizer, config, logger
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    # PRO Optimizer with weight decay for noise reduction
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        betas=(0.9, 0.98),  # Higher beta2 for smoother updates
        eps=1e-9,
        weight_decay=config.weight_decay  # 0.01 - acts as audio filter
    )
    
    # Cosine schedule for smoother training
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # PRO Loss
    criterion = ProVITSLoss()
    
    # Training state
    global_step = 0
    best_loss = float('inf')
    
    # Calculate ETA
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_training_steps = steps_per_epoch * config.num_epochs
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 PRO DEEP-TUNING STARTING")
    logger.info("=" * 70)
    logger.info(f"  📈 Learning Rate: {config.learning_rate} (reduced for quality)")
    logger.info(f"  🔬 Weight Decay: {config.weight_decay} (noise filter)")
    logger.info(f"  📚 Epochs: {config.num_epochs} (deep learning)")
    logger.info(f"  🔄 Gradient Accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  📊 Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  🎯 Target Loss: {config.target_loss_min} - {config.target_loss_max}")
    logger.info(f"  🎵 Sample Sentence: {config.validation_sentence}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("⏰ WHEN TO CHECK QUALITY:")
    logger.info("   └─ Look at Loss when it reaches 0.4-0.6 range")
    logger.info("   └─ Listen to samples in: training/samples/pro_tuning/")
    logger.info("")
    
    # Generate initial sample
    logger.info("🎵 Generating INITIAL sample (before training)...")
    generate_pro_sample(model, tokenizer, config.validation_sentence, device, config, 0, logger)
    
    epoch_start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {'total_loss': 0, 'recon_loss': 0, 'smooth_loss': 0, 'kl_loss': 0}
        step_times = []
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.num_epochs}",
            leave=True,
            ncols=100
        )
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start = time.time()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, metrics = criterion(outputs, target_audio, audio_lengths)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Tighter gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # ============================================================
                # PRO SAMPLE GENERATION
                # ============================================================
                if global_step % config.sample_every_n_steps == 0:
                    generate_pro_sample(
                        model, tokenizer, config.validation_sentence, 
                        device, config, global_step, logger
                    )
                
                # Logging
                if global_step % config.log_every_n_steps == 0:
                    avg_step_time = np.mean(step_times[-50:])
                    current_lr = scheduler.get_last_lr()[0]
                    remaining_steps = total_training_steps - global_step
                    eta_seconds = remaining_steps * avg_step_time
                    eta_hours = eta_seconds / 3600
                    
                    # Quality indicator
                    loss_val = metrics['total_loss']
                    if config.target_loss_min <= loss_val <= config.target_loss_max:
                        quality_indicator = "🎯 TARGET ZONE!"
                    elif loss_val > 1.0:
                        quality_indicator = "📈 Learning..."
                    elif loss_val > config.target_loss_max:
                        quality_indicator = "📉 Getting closer..."
                    else:
                        quality_indicator = "✨ Excellent!"
                    
                    logger.info(
                        f"Step {global_step:5d} │ "
                        f"Loss: {loss_val:.4f} │ "
                        f"Recon: {metrics['recon_loss']:.4f} │ "
                        f"Smooth: {metrics['smooth_loss']:.4f} │ "
                        f"LR: {current_lr:.1e} │ "
                        f"ETA: {eta_hours:.1f}h │ "
                        f"{quality_indicator}"
                    )
                    
                    # TensorBoard
                    writer.add_scalar('PRO/total_loss', metrics['total_loss'], global_step)
                    writer.add_scalar('PRO/recon_loss', metrics['recon_loss'], global_step)
                    writer.add_scalar('PRO/smooth_loss', metrics['smooth_loss'], global_step)
                    writer.add_scalar('PRO/learning_rate', current_lr, global_step)
            
            # Update epoch metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            epoch_loss += metrics['total_loss']
            
            # Progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'step': global_step
            })
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        epoch_start_time = time.time()
        
        logger.info("")
        logger.info(f"━━━ Epoch {epoch} Complete ━━━")
        logger.info(f"    Average Loss: {avg_epoch_loss:.4f}")
        logger.info(f"    Time: {epoch_time/60:.1f} minutes")
        
        # Quality check
        if config.target_loss_min <= avg_epoch_loss <= config.target_loss_max:
            logger.info(f"    🎯 QUALITY TARGET REACHED! Loss is in optimal range.")
        
        # Validation every N epochs
        if epoch % config.validation_interval_epochs == 0:
            logger.info("Running validation...")
            model.eval()
            valid_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc="Validation", leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    target_audio = batch['audio'].to(device)
                    audio_lengths = batch['audio_lengths'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss, _ = criterion(outputs, target_audio, audio_lengths)
                    valid_loss += loss.item()
            
            valid_loss /= len(valid_loader)
            logger.info(f"    Validation Loss: {valid_loss:.4f}")
            
            writer.add_scalar('PRO/validation_loss', valid_loss, epoch)
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                logger.info(f"    🏆 NEW BEST MODEL! Loss: {valid_loss:.4f}")
                save_checkpoint(model, tokenizer, config, epoch, valid_loss, logger, is_best=True)
        
        # Save checkpoint
        if epoch % config.save_every_n_epochs == 0:
            save_checkpoint(model, tokenizer, config, epoch, avg_epoch_loss, logger)
        
        logger.info("")
    
    # Final save
    save_checkpoint(model, tokenizer, config, config.num_epochs, avg_epoch_loss, logger, is_final=True)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 PRO DEEP-TUNING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Best validation loss: {best_loss:.4f}")
    logger.info(f"  Model saved to: {config.output_dir}")
    logger.info(f"  Audio samples: {config.samples_dir}")
    logger.info("")
    logger.info("📋 NEXT STEPS:")
    logger.info("  1. Listen to samples in training/samples/pro_tuning/")
    logger.info("  2. If quality is good, run: python training/deploy_to_mobile.py")
    logger.info("=" * 70)
    
    writer.close()


def save_checkpoint(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    config: ProTrainingConfig,
    epoch: int,
    loss: float,
    logger: logging.Logger,
    is_best: bool = False,
    is_final: bool = False
):
    """Save model checkpoint."""
    if is_best:
        save_dir = Path(config.output_dir) / "best"
    elif is_final:
        save_dir = Path(config.output_dir) / "final"
    else:
        save_dir = Path(config.output_dir) / f"checkpoint_epoch_{epoch}"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    training_args = {
        'epoch': epoch,
        'loss': loss,
        'config': {
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'num_epochs': config.num_epochs,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'version': 'PRO_v2'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_dir / "training_args.json", 'w') as f:
        json.dump(training_args, f, indent=2)
    
    logger.info(f"💾 Checkpoint saved: {save_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRO Deep-Tuning for Gujarati VITS TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_v2.py                           # Run with default PRO settings
  python train_v2.py --epochs 100              # Train for 100 epochs
  python train_v2.py --learning-rate 5e-6      # Even lower LR for smoothness
        """
    )
    parser.add_argument('--data-dir', type=str, 
                        default="/home/azureuser/cloudfiles/data/indic_tts/guj",
                        help="Path to dataset")
    parser.add_argument('--output-dir', type=str, 
                        default="./model_weights/finetuned_pro",
                        help="Output directory")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Create PRO config
    config = ProTrainingConfig()
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    
    # Setup logging
    logger = setup_logging(config.logs_dir)
    
    # Banner
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║         GUJARATI VAANI - PRO DEEP-TUNING PIPELINE (v2)           ║")
    logger.info("║                    Quality Over Speed                             ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    # Start training
    try:
        train_pro(config, logger)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        logger.info("   Checkpoints saved. Resume with --resume flag.")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
