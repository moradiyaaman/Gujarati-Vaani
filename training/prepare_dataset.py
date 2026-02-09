#!/usr/bin/env python3
"""
================================================================================
DATASET PREPARATION SCRIPT
================================================================================
Prepares the Indic TTS dataset for Gujarati Vaani fine-tuning.

This script:
1. Downloads/processes the Indic TTS Gujarati dataset
2. Validates audio files (16kHz, mono)
3. Creates metadata.csv with proper format
4. Applies text normalization consistent with inference

Usage:
    python prepare_dataset.py --input /raw/data --output /processed/data

Dataset structure after processing:
    output_dir/
    ├── train/
    │   ├── audio/
    │   │   ├── sample_0001.wav
    │   │   └── ...
    │   └── metadata.csv
    └── valid/
        ├── audio/
        └── metadata.csv

================================================================================
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Add parent directory for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.text_utils import normalize_text, filter_gujarati_text

# Configuration
TARGET_SAMPLE_RATE = 16000
MIN_DURATION = 0.5  # seconds
MAX_DURATION = 15.0  # seconds
TRAIN_SPLIT = 0.95  # 95% train, 5% validation


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def validate_and_process_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Validate and resample audio file to target sample rate.
    
    Returns:
        Tuple of (success, duration, error_message)
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Check duration
        duration = len(audio) / sr
        if duration < MIN_DURATION:
            return False, None, f"Too short: {duration:.2f}s < {MIN_DURATION}s"
        if duration > MAX_DURATION:
            return False, None, f"Too long: {duration:.2f}s > {MAX_DURATION}s"
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio (-1 to 1)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, target_sr)
        
        return True, duration, None
        
    except Exception as e:
        return False, None, str(e)


def process_text(text: str) -> Optional[str]:
    """
    Process text using the same pipeline as inference.
    
    CRITICAL: This ensures training/inference consistency.
    """
    # Apply the same filters used in app.py
    text = filter_gujarati_text(text)
    text = normalize_text(text)
    
    # Validate
    if not text or len(text) < 2:
        return None
    
    return text


def parse_indic_tts_metadata(metadata_path: Path) -> List[dict]:
    """
    Parse Indic TTS metadata format.
    
    Expected formats:
    - CSV: filename|text|speaker
    - TSV: filename\ttext\tspeaker
    - JSON: [{"audio": "...", "text": "...", "speaker": "..."}]
    """
    samples = []
    
    if metadata_path.suffix == '.json':
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                samples.append({
                    'filename': item.get('audio', item.get('filename', '')),
                    'text': item.get('text', item.get('sentence', '')),
                    'speaker': item.get('speaker', 'default')
                })
    else:
        # CSV/TSV format
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try different delimiters
                if '|' in line:
                    parts = line.split('|')
                elif '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split(',')
                
                if len(parts) >= 2:
                    samples.append({
                        'filename': parts[0].strip(),
                        'text': parts[1].strip(),
                        'speaker': parts[2].strip() if len(parts) > 2 else 'default'
                    })
    
    return samples


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    num_workers: int = 4
):
    """
    Main dataset preparation function.
    """
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find metadata files
    metadata_files = list(input_dir.glob('**/*metadata*')) + \
                     list(input_dir.glob('**/*transcript*'))
    
    if not metadata_files:
        # Try common Indic TTS structure
        for pattern in ['*.csv', '*.tsv', '*.txt', '*.json']:
            metadata_files.extend(input_dir.glob(f'**/{pattern}'))
    
    if not metadata_files:
        logger.error("No metadata files found!")
        return
    
    logger.info(f"Found {len(metadata_files)} metadata file(s)")
    
    # Collect all samples
    all_samples = []
    
    for metadata_path in metadata_files:
        logger.info(f"Processing: {metadata_path}")
        samples = parse_indic_tts_metadata(metadata_path)
        
        # Resolve audio paths relative to metadata file
        audio_dir = metadata_path.parent
        
        for sample in samples:
            audio_path = audio_dir / sample['filename']
            if not audio_path.suffix:
                audio_path = audio_path.with_suffix('.wav')
            
            # Try to find the audio file
            if not audio_path.exists():
                # Check in subdirectories
                for subdir in ['audio', 'wavs', 'wav', 'clips']:
                    alt_path = audio_dir / subdir / sample['filename']
                    if not alt_path.suffix:
                        alt_path = alt_path.with_suffix('.wav')
                    if alt_path.exists():
                        audio_path = alt_path
                        break
            
            if audio_path.exists():
                sample['audio_path'] = audio_path
                all_samples.append(sample)
    
    logger.info(f"Found {len(all_samples)} samples with audio files")
    
    if not all_samples:
        logger.error("No valid samples found!")
        return
    
    # Shuffle and split
    np.random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_SPLIT)
    train_samples = all_samples[:split_idx]
    valid_samples = all_samples[split_idx:]
    
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Valid samples: {len(valid_samples)}")
    
    # Process splits
    for split_name, samples in [('train', train_samples), ('valid', valid_samples)]:
        logger.info(f"\nProcessing {split_name} split...")
        
        split_dir = output_dir / split_name
        audio_dir = split_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        valid_samples_list = []
        total_duration = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            
            for idx, sample in enumerate(samples):
                new_filename = f"sample_{idx:06d}.wav"
                output_path = audio_dir / new_filename
                
                future = executor.submit(
                    validate_and_process_audio,
                    sample['audio_path'],
                    output_path
                )
                futures[future] = (sample, new_filename)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=split_name):
                sample, new_filename = futures[future]
                success, duration, error = future.result()
                
                if success:
                    # Process text
                    processed_text = process_text(sample['text'])
                    
                    if processed_text:
                        valid_samples_list.append({
                            'filename': new_filename,
                            'text': processed_text,
                            'speaker': sample['speaker'],
                            'duration': duration
                        })
                        total_duration += duration
                    else:
                        logger.debug(f"Skipped (empty text): {sample['filename']}")
                else:
                    logger.debug(f"Skipped: {sample['filename']} - {error}")
        
        # Write metadata
        metadata_path = split_dir / 'metadata.csv'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("# Gujarati Vaani Training Data\n")
            f.write("# Format: filename|text|speaker_id\n")
            for sample in valid_samples_list:
                f.write(f"{sample['filename']}|{sample['text']}|{sample['speaker']}\n")
        
        hours = total_duration / 3600
        logger.info(f"{split_name}: {len(valid_samples_list)} samples, {hours:.2f} hours")
    
    # Write dataset info
    info = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'valid_samples': len(valid_samples),
        'sample_rate': TARGET_SAMPLE_RATE,
        'min_duration': MIN_DURATION,
        'max_duration': MAX_DURATION
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info("\n✅ Dataset preparation complete!")
    logger.info(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Indic TTS dataset for training")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Input directory with raw dataset")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Output directory for processed dataset")
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    logger = setup_logging()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    prepare_dataset(input_dir, output_dir, logger, args.workers)


if __name__ == "__main__":
    main()
