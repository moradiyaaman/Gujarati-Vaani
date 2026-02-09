#!/usr/bin/env python3
"""
================================================================================
GUJARATI VAANI - PROFESSIONAL FINE-TUNING PIPELINE
================================================================================
Fine-tunes the Meta MMS VITS model (facebook/mms-tts-guj) for high-fidelity
Gujarati speech synthesis.

Target Environment: Azure Standard_NC6 (Tesla T4 GPU, 16GB VRAM)
Dataset: Indic TTS Gujarati Dataset (mounted from Azure Blob Storage)

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE GITHUB-AZURE CLOUD LOOP                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. GitHub Repository                                                      │
│      └── Contains: train.py, utils/, requirements_train.txt                 │
│                         │                                                   │
│                         ▼ (git clone)                                       │
│   2. Azure ML Compute Instance (Standard_NC6)                               │
│      ├── T4 GPU with 16GB VRAM                                              │
│      ├── Clones this repository                                             │
│      └── Runs train.py                                                      │
│                         │                                                   │
│                         ▼ (reads from)                                      │
│   3. Azure Blob Storage (Mounted as /data)                                  │
│      └── Indic TTS Dataset                                                  │
│          ├── guj/                                                           │
│          │   ├── train/                                                     │
│          │   │   ├── audio/ (*.wav files at 16kHz)                          │
│          │   │   └── metadata.csv (filename|text|speaker)                   │
│          │   └── valid/                                                     │
│          └── ...                                                            │
│                         │                                                   │
│                         ▼ (saves to)                                        │
│   4. Output: model_weights/finetuned/                                       │
│      ├── pytorch_model.bin (trained weights)                                │
│      ├── config.json (model configuration)                                  │
│      └── training_args.json (training metadata)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

WHY THESE DESIGN CHOICES?
-------------------------
1. WHY 16kHz?
   The Meta MMS model's vocoder (HiFi-GAN) was trained on 16kHz audio.
   Using higher sample rates would require retraining the entire vocoder,
   which needs 100+ hours of compute. We match the native rate.

2. WHY FREEZE THE ENCODER?
   The text encoder already knows "how to read Gujarati" - it maps Gujarati
   Unicode characters to phonetic representations. We only want to teach
   the model "how to sound like this specific speaker/style." Freezing:
   - Saves 40% VRAM (encoder = ~30M parameters)
   - Prevents catastrophic forgetting of Gujarati phonetics
   - Speeds up training by 2x

3. WHY GRADIENT ACCUMULATION?
   The T4 has 16GB VRAM. VITS uses ~8GB for weights + optimizer states.
   With batch_size=4, we can fit ~2GB of audio data per step.
   Gradient accumulation (4 steps) gives us effective batch_size=16,
   which provides more stable gradients without OOM errors.

4. WHY MIXED PRECISION (FP16)?
   The T4 has Tensor Cores optimized for FP16 math.
   Using torch.cuda.amp gives us:
   - 2x faster forward/backward passes
   - 50% less memory usage
   - Negligible quality loss (< 0.1% in audio metrics)

Author: Gujarati Vaani Team
Date: January 2026
================================================================================
"""

# =============================================================================
# CPU-ONLY MODE: Disable CUDA before importing torch
# =============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only mode
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback for Apple Silicon

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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from transformers import (
    VitsModel,
    VitsConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.text_utils import normalize_text, filter_gujarati_text

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Central configuration for all training parameters."""
    
    # Model
    base_model: str = "facebook/mms-tts-guj"
    freeze_encoder: bool = True
    
    # Audio
    sample_rate: int = 16000  # Native MMS sample rate
    max_audio_length: int = 10  # seconds
    min_audio_length: float = 0.5  # seconds
    
    # Training - CPU OPTIMIZED for Standard_E4ds_v4 (4 cores)
    learning_rate: float = 2e-5
    batch_size: int = 1  # CPU: Must be 1 to prevent OOM
    gradient_accumulation_steps: int = 32  # CPU: High value for stable gradients
    num_epochs: int = 10  # CPU: Reduced for reasonable training time
    warmup_steps: int = 100  # CPU: Reduced warmup
    max_grad_norm: float = 1.0
    
    # CPU-specific settings
    use_amp: bool = False  # CPU: AMP requires CUDA - DISABLED
    no_cuda: bool = True  # Force CPU mode
    use_cpu: bool = True  # Explicit CPU flag
    fp16: bool = False  # CPU: FP16 not supported on CPU
    
    # Logging & Checkpointing
    log_every_n_steps: int = 50  # CPU: More frequent logging
    save_every_n_epochs: int = 1
    sample_every_n_steps: int = 50  # Generate audio sample every 50 steps
    validation_interval_seconds: int = 1800  # 30 minutes
    
    # Paths - Updated for FLEURS dataset location
    data_dir: str = "/home/azureuser/cloudfiles/data/indic_tts/guj"
    output_dir: str = "./model_weights/finetuned"
    samples_dir: str = "./training/samples"
    logs_dir: str = "./logs"
    
    # Validation sentence for sample generation
    validation_sentence: str = "નમસ્તે, આ ગુજરાતી વાણી પ્રોજેક્ટ છે."
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging for training."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("gujarati_vaani_training")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        Path(log_dir) / f"training_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATASET
# ============================================================================

class GujaratiTTSDataset(Dataset):
    """
    Dataset for Gujarati TTS fine-tuning.
    
    Expected directory structure:
    data_dir/
    ├── train/
    │   ├── audio/
    │   │   ├── sample_001.wav
    │   │   └── ...
    │   └── metadata.csv  (format: filename|text|speaker_id)
    └── valid/
        ├── audio/
        └── metadata.csv
    
    Audio requirements:
    - Sample rate: 16000 Hz (resampled if different)
    - Channels: Mono
    - Duration: 0.5s to 10s
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        logger: logging.Logger
    ):
        self.data_dir = Path(data_dir) / split
        self.audio_dir = self.data_dir / "audio"
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        
        # Load metadata
        self.samples = self._load_metadata()
        self.logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(self) -> List[Dict]:
        """Load and validate metadata.csv"""
        metadata_path = self.data_dir / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        samples = []
        skipped = 0
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) < 2:
                    self.logger.warning(f"Skipping line {line_num}: invalid format")
                    skipped += 1
                    continue
                
                filename = parts[0].strip()
                text = parts[1].strip()
                speaker_id = parts[2].strip() if len(parts) > 2 else "default"
                
                audio_path = self.audio_dir / filename
                if not audio_path.suffix:
                    audio_path = audio_path.with_suffix('.wav')
                
                if not audio_path.exists():
                    self.logger.debug(f"Audio file not found: {audio_path}")
                    skipped += 1
                    continue
                
                # Apply text preprocessing (CRITICAL: consistency with inference)
                text = filter_gujarati_text(text)
                text = normalize_text(text)
                
                if not text:
                    self.logger.debug(f"Empty text after filtering: line {line_num}")
                    skipped += 1
                    continue
                
                samples.append({
                    'audio_path': str(audio_path),
                    'text': text,
                    'speaker_id': speaker_id
                })
        
        if skipped > 0:
            self.logger.warning(f"Skipped {skipped} samples due to missing files or invalid format")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load audio
        audio, sr = librosa.load(
            sample['audio_path'],
            sr=self.config.sample_rate,
            mono=True
        )
        
        # Validate duration
        duration = len(audio) / self.config.sample_rate
        if duration < self.config.min_audio_length:
            # Pad short audio
            target_length = int(self.config.min_audio_length * self.config.sample_rate)
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif duration > self.config.max_audio_length:
            # Truncate long audio
            max_samples = int(self.config.max_audio_length * self.config.sample_rate)
            audio = audio[:max_samples]
        
        # Tokenize text
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
    """
    Dynamic padding for batches.
    
    This is crucial for GPU efficiency - we pad to the longest sequence
    in each batch, not a fixed maximum. This prevents wasting compute
    on silent gaps.
    """
    # Find max lengths
    max_input_len = max(item['input_ids'].size(0) for item in batch)
    max_audio_len = max(item['audio'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Pre-allocate tensors
    input_ids = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    audio = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    texts = []
    
    for i, item in enumerate(batch):
        # Pad input_ids
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        
        # Pad audio
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
    config: TrainingConfig,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[VitsModel, AutoTokenizer]:
    """
    Load and configure the VITS model for fine-tuning.
    
    Key optimizations:
    1. Freeze text encoder to preserve Gujarati phonetic knowledge
    2. Keep decoder trainable for voice adaptation
    """
    logger.info(f"Loading base model: {config.base_model}")
    
    # Load model and tokenizer
    model = VitsModel.from_pretrained(config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Freeze encoder if configured
    if config.freeze_encoder:
        logger.info("Freezing text encoder (preserving Gujarati phonetics)")
        
        # Freeze text encoder
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        logger.info(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    model.to(device)
    model.train()
    
    return model, tokenizer


def generate_validation_sample(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    config: TrainingConfig
) -> np.ndarray:
    """Generate audio for validation."""
    model.eval()
    
    # Preprocess text (same as training)
    text = filter_gujarati_text(text)
    text = normalize_text(text)
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model(**inputs)
        waveform = output.waveform.squeeze().cpu().numpy()
    
    model.train()
    return waveform


# ============================================================================
# TRAINING LOOP
# ============================================================================

class VITSLoss(nn.Module):
    """
    Combined loss for VITS fine-tuning.
    
    Components:
    1. Reconstruction Loss (L1): Measures audio quality
    2. KL Divergence: Measures how well the model learns speech "flow"
    """
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        model_output,
        target_audio: torch.Tensor,
        audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training losses.
        
        Note: VITS has an end-to-end architecture. In practice, we fine-tune
        by comparing generated waveforms to target waveforms using L1 loss,
        and use the model's internal KL divergence.
        """
        generated = model_output.waveform
        
        # Match lengths for comparison
        min_len = min(generated.size(-1), target_audio.size(-1))
        generated = generated[..., :min_len]
        target = target_audio[..., :min_len]
        
        # L1 reconstruction loss
        recon_loss = self.l1_loss(generated, target)
        
        # KL divergence (from model internals if available)
        # VITS computes this internally during forward pass
        kl_loss = torch.tensor(0.0, device=generated.device)
        if hasattr(model_output, 'kl_loss') and model_output.kl_loss is not None:
            kl_loss = model_output.kl_loss
        
        # Total loss
        total_loss = recon_loss + 0.1 * kl_loss
        
        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        }
        
        return total_loss, metrics


def train(config: TrainingConfig, logger: logging.Logger):
    """Main training function."""
    
    # Setup directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.samples_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Device setup - Force CPU for this configuration
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # CPU-specific optimizations
    num_threads = min(4, torch.get_num_threads())  # Limit threads to avoid overhead
    torch.set_num_threads(num_threads)
    logger.info(f"CPU threads: {num_threads}")
    logger.info("⚠️  CPU Training Mode: This will be slower than GPU training")
    logger.info("   Tip: Monitor memory with 'watch -n 1 free -h' in another terminal")
    
    # TensorBoard
    writer = SummaryWriter(config.logs_dir)
    
    # Load model
    model, tokenizer = load_model_for_training(config, device, logger)
    
    # Load datasets
    logger.info("Loading datasets...")
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
        num_workers=0,  # CPU: Reduced from 4 to avoid multiprocessing overhead
        collate_fn=collate_fn,
        pin_memory=False  # CPU: Disabled - only useful for GPU
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # CPU: Reduced from 2
        collate_fn=collate_fn,
        pin_memory=False  # CPU: Disabled
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision
    scaler = GradScaler() if config.use_amp else None
    
    # Loss function
    criterion = VITSLoss()
    
    # Training state
    global_step = 0
    best_loss = float('inf')
    last_validation_time = time.time()
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Total epochs: {config.num_epochs}")
    logger.info(f"Steps per epoch: {len(train_loader)}")
    logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info("=" * 60)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {'total_loss': 0, 'recon_loss': 0, 'kl_loss': 0}
        step_times = []
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.num_epochs}",
            leave=True
        )
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start = time.time()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            
            # Forward pass with mixed precision
            if config.use_amp:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss, metrics = criterion(outputs, target_audio, audio_lengths)
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, metrics = criterion(outputs, target_audio, audio_lengths)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # ============================================================
                # SAMPLE GENERATION: Generate audio every N steps
                # ============================================================
                if global_step % config.sample_every_n_steps == 0:
                    logger.info(f"🎵 Generating sample at step {global_step}...")
                    try:
                        sample_waveform = generate_validation_sample(
                            model, tokenizer, config.validation_sentence, device, config
                        )
                        sample_path = Path(config.samples_dir) / f"sample_step_{global_step:06d}.wav"
                        sf.write(sample_path, sample_waveform, config.sample_rate)
                        logger.info(f"✓ Sample saved: {sample_path}")
                    except Exception as e:
                        logger.warning(f"Sample generation failed: {e}")
                
                # Logging
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                if global_step % config.log_every_n_steps == 0:
                    avg_step_time = np.mean(step_times[-100:])
                    current_lr = scheduler.get_last_lr()[0]
                    
                    logger.info(
                        f"Step {global_step} | "
                        f"Loss: {metrics['total_loss']:.4f} | "
                        f"Recon: {metrics['recon_loss']:.4f} | "
                        f"KL: {metrics['kl_loss']:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {avg_step_time:.2f}s/step"
                    )
                    
                    # TensorBoard logging
                    writer.add_scalar('Loss/total', metrics['total_loss'], global_step)
                    writer.add_scalar('Loss/reconstruction', metrics['recon_loss'], global_step)
                    writer.add_scalar('Loss/kl_divergence', metrics['kl_loss'], global_step)
                    writer.add_scalar('Training/learning_rate', current_lr, global_step)
                    writer.add_scalar('Training/step_time', avg_step_time, global_step)
            
            # Update epoch metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            epoch_loss += metrics['total_loss']
            
            # Progress bar update
            progress_bar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'step': global_step
            })
            
            # Hourly validation sample
            current_time = time.time()
            if current_time - last_validation_time >= config.validation_interval_seconds:
                logger.info("Generating hourly validation sample...")
                
                waveform = generate_validation_sample(
                    model, tokenizer, config.validation_sentence, device, config
                )
                
                sample_path = Path(config.samples_dir) / f"validation_step_{global_step}.wav"
                sf.write(sample_path, waveform, config.sample_rate)
                logger.info(f"Saved validation sample: {sample_path}")
                
                last_validation_time = current_time
        
        # End of epoch
        num_batches = len(train_loader)
        avg_epoch_loss = epoch_loss / num_batches
        
        logger.info(f"Epoch {epoch} complete | Average Loss: {avg_epoch_loss:.4f}")
        
        # Validation
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
        logger.info(f"Validation Loss: {valid_loss:.4f}")
        
        writer.add_scalar('Loss/validation', valid_loss, epoch)
        
        # Save checkpoint
        if epoch % config.save_every_n_epochs == 0:
            save_checkpoint(model, tokenizer, config, epoch, valid_loss, logger)
        
        # Best model tracking
        if valid_loss < best_loss:
            best_loss = valid_loss
            logger.info(f"New best model! Validation Loss: {valid_loss:.4f}")
            save_checkpoint(model, tokenizer, config, epoch, valid_loss, logger, is_best=True)
    
    # Final save
    save_checkpoint(model, tokenizer, config, config.num_epochs, valid_loss, logger, is_final=True)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info("=" * 60)
    
    writer.close()
    
    # ========================================================================
    # AUTOMATIC MOBILE DEPLOYMENT - Quantize and prepare for phone
    # ========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPARING MOBILE DEPLOYMENT...")
    logger.info("=" * 60)
    
    try:
        from deploy_to_mobile import deploy_to_mobile
        
        final_checkpoint = Path(config.output_dir) / "final"
        mobile_output = Path(config.output_dir).parent / "mobile"
        
        deploy_to_mobile(
            checkpoint_path=final_checkpoint,
            output_dir=mobile_output,
            verify=True,
            logger=logger
        )
        
        logger.info("")
        logger.info("🎉 READY-TO-USE MOBILE MODEL CREATED!")
        logger.info(f"   Location: {mobile_output}")
        logger.info("   Copy this folder to your app's assets.")
        
    except ImportError as e:
        logger.warning(f"Could not auto-deploy to mobile: {e}")
        logger.info("Run manually: python deploy_to_mobile.py --checkpoint ./model_weights/finetuned/final")
    except Exception as e:
        logger.warning(f"Mobile deployment failed: {e}")
        logger.info("Training was successful. Run deploy_to_mobile.py manually.")


def save_checkpoint(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
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
    
    # Save model weights
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save training args
    training_args = {
        'epoch': epoch,
        'validation_loss': loss,
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_dir / "training_args.json", 'w') as f:
        json.dump(training_args, f, indent=2)
    
    logger.info(f"Checkpoint saved: {save_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gujarati VITS TTS model")
    parser.add_argument('--data-dir', type=str, default="/data/indic_tts/guj",
                        help="Path to Indic TTS dataset")
    parser.add_argument('--output-dir', type=str, default="./model_weights/finetuned",
                        help="Output directory for checkpoints")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Setup logging
    logger = setup_logging(config.logs_dir)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("GUJARATI VAANI - FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    logger.info("Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Start training
    try:
        train(config, logger)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
