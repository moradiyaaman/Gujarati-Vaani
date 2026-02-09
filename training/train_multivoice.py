#!/usr/bin/env python3
"""
================================================================================
GUJARATI VAANI - MULTI-VOICE PRO TRAINING PIPELINE
================================================================================
Sequential training for Male and Female voice models using PRO settings.

The "Two-Brain" Approach:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-VOICE TRAINING FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Session A: MALE VOICE                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Load: facebook/mms-tts-guj (base model)                            │   │
│   │  Train on: male/train/metadata.csv                                  │   │
│   │  Validate: male/valid/metadata.csv                                  │   │
│   │  Output: model_weights/finetuned_male/                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│   Session B: FEMALE VOICE                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Load: facebook/mms-tts-guj (base model) [FRESH START]              │   │
│   │  Train on: female/train/metadata.csv                                │   │
│   │  Validate: female/valid/metadata.csv                                │   │
│   │  Output: model_weights/finetuned_female/                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

PRO Settings (for both voices):
- Learning Rate: 1e-5 (no metallic noise)
- Weight Decay: 0.01 (clean audio)
- Gradient Accumulation: 64 (stable training)
- Epochs: 30 (balanced quality/time)

Usage:
    python train_multivoice.py --voice male     # Train male only
    python train_multivoice.py --voice female   # Train female only
    python train_multivoice.py --voice both     # Train both sequentially

Author: Gujarati Vaani Team
Date: January 2026
================================================================================
"""

# =============================================================================
# CPU-ONLY MODE
# =============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from transformers import (
    VitsModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.text_utils import normalize_text, filter_gujarati_text

# ============================================================================
# CONFIGURATION
# ============================================================================

class MultiVoiceConfig:
    """Configuration for multi-voice training."""
    
    # Model
    base_model: str = "facebook/mms-tts-guj"
    freeze_encoder: bool = True
    
    # Audio
    sample_rate: int = 16000
    max_audio_length: int = 10
    min_audio_length: float = 0.5
    
    # PRO Training Settings
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 1
    gradient_accumulation_steps: int = 64
    num_epochs: int = 30  # Reduced for two separate trainings
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5
    
    # CPU Settings
    use_amp: bool = False
    
    # Monitoring
    log_every_n_steps: int = 25
    save_every_n_epochs: int = 5
    sample_every_n_steps: int = 25
    
    # Base paths
    base_data_dir: str = "/home/azureuser/cloudfiles/data/indic_tts/guj"
    base_output_dir: str = "./model_weights"
    base_samples_dir: str = "./training/samples"
    base_logs_dir: str = "./logs"
    
    # Validation sentences
    validation_sentences = {
        'male': "નમસ્તે, હું ગુજરાતી વાણી છું. આ પુરુષ અવાજ છે.",
        'female': "નમસ્તે, હું ગુજરાતી વાણી છું. આ સ્ત્રી અવાજ છે."
    }
    
    def get_paths(self, voice: str) -> Dict[str, Path]:
        """Get paths for a specific voice (male/female)."""
        return {
            'data_dir': Path(self.base_data_dir) / voice,
            'output_dir': Path(self.base_output_dir) / f"finetuned_{voice}",
            'samples_dir': Path(self.base_samples_dir) / f"{voice}_tuning",
            'logs_dir': Path(self.base_logs_dir) / f"{voice}_tuning"
        }


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_dir: str, voice: str) -> logging.Logger:
    """Configure logging for training."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"gujarati_vaani_{voice}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        f'%(asctime)s │ {voice.upper()} │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        Path(log_dir) / f"{voice}_training_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# DATASET
# ============================================================================

class VoiceDataset(Dataset):
    """Dataset for single-voice training."""
    
    def __init__(
        self,
        data_dir: Path,
        split: str,
        tokenizer: AutoTokenizer,
        config: MultiVoiceConfig,
        logger: logging.Logger
    ):
        self.data_dir = data_dir / split
        self.audio_dir = self.data_dir / "audio"
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.samples = self._load_metadata()
        self.logger.info(f"   📂 Loaded {len(self.samples)} samples for {split}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata.csv"""
        metadata_path = self.data_dir / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        samples = []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) < 2:
                    continue
                
                filename = parts[0].strip()
                text = parts[1].strip()
                
                audio_path = self.audio_dir / filename
                if not audio_path.exists():
                    continue
                
                text = filter_gujarati_text(text)
                text = normalize_text(text)
                
                if text:
                    samples.append({
                        'audio_path': str(audio_path),
                        'text': text
                    })
        
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
# MODEL & LOSS
# ============================================================================

def load_model_for_training(
    config: MultiVoiceConfig,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[VitsModel, AutoTokenizer]:
    """Load fresh base model for training."""
    logger.info(f"   🔧 Loading base model: {config.base_model}")
    
    model = VitsModel.from_pretrained(config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    if config.freeze_encoder:
        logger.info("   ❄️  Freezing text encoder")
        for param in model.text_encoder.parameters():
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   📊 Trainable parameters: {trainable:,}")
    
    model.to(device)
    model.train()
    
    return model, tokenizer


class VoiceLoss(nn.Module):
    """Loss function for voice training."""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, model_output, target_audio, audio_lengths):
        generated = model_output.waveform
        
        min_len = min(generated.size(-1), target_audio.size(-1))
        generated = generated[..., :min_len]
        target = target_audio[..., :min_len]
        
        recon_loss = self.l1_loss(generated, target)
        smooth_loss = self.l1_loss(
            generated[..., 1:] - generated[..., :-1],
            target[..., 1:] - target[..., :-1]
        )
        
        total_loss = recon_loss + 0.1 * smooth_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'smooth_loss': smooth_loss.item()
        }


def generate_sample(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    config: MultiVoiceConfig,
    voice: str,
    step: int,
    paths: Dict[str, Path],
    logger: logging.Logger
) -> Optional[str]:
    """Generate audio sample."""
    model.eval()
    
    text = filter_gujarati_text(text)
    text = normalize_text(text)
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            output = model(**inputs)
            waveform = output.waveform.squeeze().cpu().numpy()
        
        paths['samples_dir'].mkdir(parents=True, exist_ok=True)
        sample_path = paths['samples_dir'] / f"step_{step:06d}_{voice}.wav"
        sf.write(sample_path, waveform, config.sample_rate)
        
        logger.info(f"   🎵 Sample saved: {sample_path.name}")
        
        model.train()
        return str(sample_path)
        
    except Exception as e:
        logger.warning(f"   Sample generation failed: {e}")
        model.train()
        return None


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_voice(voice: str, config: MultiVoiceConfig):
    """Train a single voice model."""
    
    paths = config.get_paths(voice)
    
    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(paths['logs_dir']), voice)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🎤 TRAINING {voice.upper()} VOICE MODEL")
    logger.info("=" * 70)
    
    # Device
    device = torch.device("cpu")
    logger.info(f"   🖥️  Device: {device}")
    
    num_threads = min(4, torch.get_num_threads())
    torch.set_num_threads(num_threads)
    
    # TensorBoard
    writer = SummaryWriter(str(paths['logs_dir']))
    
    # Load model (FRESH for each voice)
    logger.info("")
    logger.info("📦 Loading Model:")
    model, tokenizer = load_model_for_training(config, device, logger)
    
    # Load datasets
    logger.info("")
    logger.info("📂 Loading Datasets:")
    train_dataset = VoiceDataset(
        paths['data_dir'], "train", tokenizer, config, logger
    )
    valid_dataset = VoiceDataset(
        paths['data_dir'], "valid", tokenizer, config, logger
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss
    criterion = VoiceLoss()
    
    # Training state
    global_step = 0
    best_loss = float('inf')
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🚀 STARTING {voice.upper()} VOICE TRAINING")
    logger.info("=" * 70)
    logger.info(f"   📈 Learning Rate: {config.learning_rate}")
    logger.info(f"   📚 Epochs: {config.num_epochs}")
    logger.info(f"   🔄 Gradient Accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"   📊 Train samples: {len(train_dataset)}")
    logger.info(f"   📊 Valid samples: {len(valid_dataset)}")
    logger.info("=" * 70)
    
    # Generate initial sample
    validation_text = config.validation_sentences[voice]
    logger.info("")
    logger.info("🎵 Generating initial sample...")
    generate_sample(model, tokenizer, validation_text, device, config, voice, 0, paths, logger)
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
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
            batch_count = (epoch - 1) * len(train_loader) + batch_idx + 1
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, metrics = criterion(outputs, target_audio, audio_lengths)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Sample generation every N batch iterations (not optimizer steps)
            if batch_count % config.sample_every_n_steps == 0:
                generate_sample(
                    model, tokenizer, validation_text,
                    device, config, voice, batch_count, paths, logger
                )
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Logging
                if global_step % config.log_every_n_steps == 0:
                    avg_time = np.mean(step_times[-50:])
                    current_lr = scheduler.get_last_lr()[0]
                    
                    logger.info(
                        f"   Step {global_step:5d} │ "
                        f"Loss: {metrics['total_loss']:.4f} │ "
                        f"Recon: {metrics['recon_loss']:.4f} │ "
                        f"LR: {current_lr:.1e}"
                    )
                    
                    writer.add_scalar(f'{voice}/total_loss', metrics['total_loss'], global_step)
                    writer.add_scalar(f'{voice}/learning_rate', current_lr, global_step)
            
            epoch_loss += metrics['total_loss']
            progress_bar.set_postfix({'loss': f"{metrics['total_loss']:.4f}"})
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info("")
        logger.info(f"   ━━━ Epoch {epoch} Complete │ Avg Loss: {avg_epoch_loss:.4f} ━━━")
        
        # Validation
        if epoch % 2 == 0:
            model.eval()
            valid_loss = 0.0
            
            with torch.no_grad():
                for batch in valid_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    target_audio = batch['audio'].to(device)
                    audio_lengths = batch['audio_lengths'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss, _ = criterion(outputs, target_audio, audio_lengths)
                    valid_loss += loss.item()
            
            valid_loss /= len(valid_loader)
            logger.info(f"   Validation Loss: {valid_loss:.4f}")
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                logger.info(f"   🏆 NEW BEST MODEL!")
                save_checkpoint(model, tokenizer, config, voice, epoch, valid_loss, paths, logger, is_best=True)
        
        # Save checkpoint
        if epoch % config.save_every_n_epochs == 0:
            save_checkpoint(model, tokenizer, config, voice, epoch, avg_epoch_loss, paths, logger)
    
    # Final save
    save_checkpoint(model, tokenizer, config, voice, config.num_epochs, avg_epoch_loss, paths, logger, is_final=True)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✅ {voice.upper()} VOICE TRAINING COMPLETE!")
    logger.info(f"   Best Loss: {best_loss:.4f}")
    logger.info(f"   Model: {paths['output_dir']}")
    logger.info("=" * 70)
    
    writer.close()
    
    return paths['output_dir']


def save_checkpoint(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    config: MultiVoiceConfig,
    voice: str,
    epoch: int,
    loss: float,
    paths: Dict[str, Path],
    logger: logging.Logger,
    is_best: bool = False,
    is_final: bool = False
):
    """Save model checkpoint."""
    if is_best:
        save_dir = paths['output_dir'] / "best"
    elif is_final:
        save_dir = paths['output_dir'] / "final"
    else:
        save_dir = paths['output_dir'] / f"checkpoint_epoch_{epoch}"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    training_info = {
        'voice': voice,
        'epoch': epoch,
        'loss': loss,
        'config': {
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'num_epochs': config.num_epochs,
            'gradient_accumulation_steps': config.gradient_accumulation_steps
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"   💾 Saved: {save_dir.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Voice Gujarati TTS Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_multivoice.py --voice male      # Train male voice only
    python train_multivoice.py --voice female    # Train female voice only
    python train_multivoice.py --voice both      # Train both sequentially
        """
    )
    parser.add_argument(
        '--voice', '-v',
        type=str,
        choices=['male', 'female', 'both'],
        default='both',
        help="Which voice to train: male, female, or both"
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=30,
        help="Number of epochs per voice"
    )
    parser.add_argument(
        '--sample-every', '-s',
        type=int,
        default=200,
        help="Generate audio sample every N steps"
    )
    args = parser.parse_args()
    
    config = MultiVoiceConfig()
    config.num_epochs = args.epochs
    config.sample_every_n_steps = args.sample_every
    
    print("")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       GUJARATI VAANI - MULTI-VOICE TRAINING PIPELINE             ║")
    print("║                  The 'Two-Brain' Approach                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("")
    
    voices_to_train = ['male', 'female'] if args.voice == 'both' else [args.voice]
    trained_models = {}
    
    for voice in voices_to_train:
        print(f"\n{'='*70}")
        print(f"🎤 STARTING {voice.upper()} VOICE TRAINING")
        print(f"{'='*70}\n")
        
        try:
            model_path = train_voice(voice, config)
            trained_models[voice] = model_path
            print(f"\n✅ {voice.upper()} voice training complete!")
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted for {voice} voice")
            break
        except Exception as e:
            print(f"\n❌ Training failed for {voice} voice: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 MULTI-VOICE TRAINING SUMMARY")
    print("=" * 70)
    
    for voice, path in trained_models.items():
        print(f"   🎤 {voice.upper()}: {path}")
    
    print("")
    print("🚀 NEXT STEP: Run deployment script:")
    print("   python training/deploy_multivoice.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
