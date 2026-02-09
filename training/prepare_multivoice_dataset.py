#!/usr/bin/env python3
"""
================================================================================
GUJARATI VAANI - MULTI-VOICE DATASET PREPARATION
================================================================================
Splits the FLEURS Gujarati dataset by gender to create separate training sets
for Male and Female voice models.

Output Structure:
    /data/indic_tts/guj/
    ├── male/
    │   ├── train/
    │   │   ├── audio/
    │   │   └── metadata.csv
    │   └── valid/
    │       ├── audio/
    │       └── metadata.csv
    └── female/
        ├── train/
        │   ├── audio/
        │   └── metadata.csv
        └── valid/
            ├── audio/
            └── metadata.csv

Usage:
    python prepare_multivoice_dataset.py --output /home/azureuser/cloudfiles/data/indic_tts/guj

Author: Gujarati Vaani Team
Date: January 2026
================================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def prepare_multivoice_dataset(output_dir: Path):
    """
    Download and split FLEURS Gujarati dataset by gender.
    
    Gender mapping in FLEURS:
        0 = male
        1 = female
        2 = other (skipped)
    """
    from datasets import load_dataset, Audio
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎭 MULTI-VOICE DATASET PREPARATION")
    logger.info("=" * 70)
    
    # Create output directories for both genders
    genders = ['male', 'female']
    splits = ['train', 'valid']
    
    for gender in genders:
        for split in splits:
            audio_dir = output_dir / gender / split / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Load FLEURS Gujarati
    logger.info("📥 Loading FLEURS Gujarati dataset...")
    
    train_dataset = load_dataset("google/fleurs", "gu_in", split="train", trust_remote_code=True)
    valid_dataset = load_dataset("google/fleurs", "gu_in", split="validation", trust_remote_code=True)
    
    logger.info(f"   Total train samples: {len(train_dataset)}")
    logger.info(f"   Total validation samples: {len(valid_dataset)}")
    
    # Count samples by gender
    gender_counts_train = {'male': 0, 'female': 0, 'other': 0}
    gender_counts_valid = {'male': 0, 'female': 0, 'other': 0}
    
    for sample in train_dataset:
        gender_id = sample['gender']
        if gender_id == 0:
            gender_counts_train['male'] += 1
        elif gender_id == 1:
            gender_counts_train['female'] += 1
        else:
            gender_counts_train['other'] += 1
    
    for sample in valid_dataset:
        gender_id = sample['gender']
        if gender_id == 0:
            gender_counts_valid['male'] += 1
        elif gender_id == 1:
            gender_counts_valid['female'] += 1
        else:
            gender_counts_valid['other'] += 1
    
    logger.info("")
    logger.info("📊 Gender Distribution:")
    logger.info(f"   Train - Male: {gender_counts_train['male']}, Female: {gender_counts_train['female']}, Other: {gender_counts_train['other']}")
    logger.info(f"   Valid - Male: {gender_counts_valid['male']}, Female: {gender_counts_valid['female']}, Other: {gender_counts_valid['other']}")
    
    # Process and save by gender
    def process_split(dataset, split_name):
        """Process a dataset split and save by gender."""
        
        male_samples = []
        female_samples = []
        
        logger.info(f"\n📂 Processing {split_name} split...")
        
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            gender_id = sample['gender']
            
            # Skip 'other' gender for cleaner voice separation
            if gender_id not in [0, 1]:
                continue
            
            gender = 'male' if gender_id == 0 else 'female'
            
            # Get audio data
            audio_data = sample['audio']['array']
            sr = sample['audio']['sampling_rate']
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Get text
            text = sample['transcription'].strip()
            if not text:
                continue
            
            # Save audio file
            filename = f"{gender}_{split_name}_{idx:05d}.wav"
            audio_path = output_dir / gender / split_name / "audio" / filename
            
            sf.write(str(audio_path), audio_data.astype(np.float32), sr)
            
            # Add to appropriate list
            if gender == 'male':
                male_samples.append((filename, text))
            else:
                female_samples.append((filename, text))
        
        # Write metadata files
        for gender in ['male', 'female']:
            samples = male_samples if gender == 'male' else female_samples
            metadata_path = output_dir / gender / split_name / "metadata.csv"
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                for filename, text in samples:
                    f.write(f"{filename}|{text}|{gender}\n")
            
            logger.info(f"   ✓ {gender.capitalize()} {split_name}: {len(samples)} samples saved")
        
        return len(male_samples), len(female_samples)
    
    # Process both splits
    train_male, train_female = process_split(train_dataset, 'train')
    valid_male, valid_female = process_split(valid_dataset, 'valid')
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ MULTI-VOICE DATASET PREPARATION COMPLETE!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("📊 Final Statistics:")
    logger.info(f"   🎤 MALE Voice:")
    logger.info(f"      Train: {train_male} samples")
    logger.info(f"      Valid: {valid_male} samples")
    logger.info(f"      Location: {output_dir / 'male'}")
    logger.info("")
    logger.info(f"   🎤 FEMALE Voice:")
    logger.info(f"      Train: {train_female} samples")
    logger.info(f"      Valid: {valid_female} samples")
    logger.info(f"      Location: {output_dir / 'female'}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 Ready for multi-voice training!")
    logger.info("   Run: python training/train_multivoice.py")
    logger.info("=" * 70)
    
    return {
        'male': {'train': train_male, 'valid': valid_male},
        'female': {'train': train_female, 'valid': valid_female}
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-voice (male/female) Gujarati TTS dataset"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="/home/azureuser/cloudfiles/data/indic_tts/guj",
        help="Output directory for datasets"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║      GUJARATI VAANI - MULTI-VOICE DATASET PREPARATION            ║")
    logger.info("║               Creating Male & Female Datasets                     ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    try:
        stats = prepare_multivoice_dataset(output_dir)
        return stats
    except Exception as e:
        logger.exception(f"Dataset preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
