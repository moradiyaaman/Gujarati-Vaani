#!/usr/bin/env python3
"""
Curriculum Learning TTS Training for Gujarati
- Progressive difficulty: Easy → Medium → Hard → Expert
- Single voice training (run separately for male/female concurrently)
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import VitsModel, AutoTokenizer
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class CurriculumConfig:
    """Configuration for curriculum-based training"""
    voice: str = "male"  # or "female"
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs_per_stage: int = 2  # Epochs before moving to harder data
    total_stages: int = 4  # Easy, Medium, Hard, Expert
    gradient_accumulation_steps: int = 64
    sample_every_n_steps: int = 200
    save_every_n_epochs: int = 2
    data_dir: str = "/home/azureuser/cloudfiles/data/indic_tts/guj"
    output_dir: str = "/home/azureuser/cloudfiles/code/Gujarati-Vaani/training/curriculum_models"
    sample_dir: str = "/home/azureuser/cloudfiles/code/Gujarati-Vaani/training/samples"


class GujaratiComplexityScorer:
    """Score Gujarati text by pronunciation complexity"""
    
    # Complex consonant clusters in Gujarati
    COMPLEX_CLUSTERS = [
        'ક્ષ', 'જ્ઞ', 'શ્ર', 'ત્ર', 'પ્ર', 'ક્ર', 'દ્ર', 'બ્ર', 'ગ્ર', 'ફ્ર',
        'સ્ત', 'સ્થ', 'સ્પ', 'સ્ક', 'શ્ચ', 'ષ્ટ', 'ષ્ઠ', 'ક્ત', 'ત્ત', 'ત્ય',
        'દ્ય', 'ધ્ય', 'ન્ય', 'પ્ય', 'મ્ય', 'વ્ય', 'શ્ય', 'ષ્ય', 'સ્ય', 'હ્ય',
        'ક્ક', 'ગ્ગ', 'ચ્ચ', 'જ્જ', 'ટ્ટ', 'ડ્ડ', 'ત્ત', 'દ્દ', 'પ્પ', 'બ્બ',
        'મ્મ', 'લ્લ', 'વ્વ', 'સ્સ', 'ન્ન', 'ણ્ણ'
    ]
    
    # Rare/difficult phonemes (vocalic R, candra vowels)
    DIFFICULT_PHONEMES = ['ઋ', 'ૠ', 'ઌ', 'ૡ', 'ઍ', 'ઑ', 'ૐ']
    
    # Aspirated consonants (more difficult)
    ASPIRATED = ['ખ', 'ઘ', 'છ', 'ઝ', 'ઠ', 'ઢ', 'થ', 'ધ', 'ફ', 'ભ']
    
    # ===== VOWELS (સ્વર) - From image =====
    # Short vowels (હ્રસ્વ સ્વર) - easier
    SHORT_VOWELS = ['અ', 'ઇ', 'ઉ']
    SHORT_MATRAS = ['િ', 'ુ']  # Matras for short vowels
    
    # Long vowels (દીર્ઘ સ્વર) - harder (require longer articulation)
    LONG_VOWELS = ['આ', 'ઈ', 'ઊ', 'એ', 'ઐ', 'ઓ', 'ઔ']
    LONG_MATRAS = ['ા', 'ી', 'ૂ', 'ે', 'ૈ', 'ો', 'ૌ']  # Matras for long vowels
    
    # Nasalized vowels (અનુસ્વાર) - moderate difficulty
    ANUSVARA = ['ં']  # અં - nasalization marker
    
    # Visarga (વિસર્ગ) - breath release, moderate difficulty
    VISARGA = ['ઃ']  # અઃ
    
    # Diphthongs (સંયુક્ત સ્વર) - two vowels combined, harder
    DIPHTHONGS = ['ઐ', 'ઔ', 'ૈ', 'ૌ']
    
    # Retroflex consonants (મૂર્ધન્ય) - tongue curled back, harder
    RETROFLEX = ['ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 'ષ']
    
    # Sibilants and fricatives - require precise articulation
    SIBILANTS = ['શ', 'ષ', 'સ', 'હ']
    
    @classmethod
    def score(cls, text: str) -> float:
        """
        Score text complexity (0-100)
        Higher score = more difficult pronunciation
        """
        if not text:
            return 0
        
        score = 0
        
        # 1. Length complexity (longer = harder)
        words = text.split()
        word_count = len(words)
        char_count = len(text.replace(' ', ''))
        
        # Base score from length
        score += min(word_count * 2, 15)  # Up to 15 points for length
        
        # 2. Complex clusters (highest difficulty)
        cluster_count = sum(text.count(c) for c in cls.COMPLEX_CLUSTERS)
        score += min(cluster_count * 5, 25)  # Up to 25 points for clusters
        
        # 3. Difficult/rare phonemes
        rare_count = sum(text.count(p) for p in cls.DIFFICULT_PHONEMES)
        score += min(rare_count * 6, 15)  # Up to 15 points for rare phonemes
        
        # 4. Aspirated consonants
        aspirated_count = sum(text.count(a) for a in cls.ASPIRATED)
        score += min(aspirated_count * 1.5, 10)  # Up to 10 points for aspirated
        
        # 5. Long vowels and matras (require sustained articulation)
        long_vowel_count = sum(text.count(v) for v in cls.LONG_VOWELS + cls.LONG_MATRAS)
        score += min(long_vowel_count * 1, 10)  # Up to 10 points
        
        # 6. Diphthongs (two-vowel sounds, harder)
        diphthong_count = sum(text.count(d) for d in cls.DIPHTHONGS)
        score += min(diphthong_count * 2, 8)  # Up to 8 points
        
        # 7. Anusvara and Visarga (nasalization and breath)
        nasal_count = sum(text.count(n) for n in cls.ANUSVARA + cls.VISARGA)
        score += min(nasal_count * 1.5, 7)  # Up to 7 points
        
        # 8. Retroflex consonants (tongue position)
        retroflex_count = sum(text.count(r) for r in cls.RETROFLEX)
        score += min(retroflex_count * 1, 5)  # Up to 5 points
        
        # 9. Average word length (longer words = harder)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            score += min(avg_word_len * 1, 5)  # Up to 5 points
        
        return min(score, 100)  # Cap at 100
    
    @classmethod
    def categorize(cls, texts: List[str]) -> Dict[str, List[Tuple[int, str]]]:
        """
        Categorize texts into difficulty levels
        Returns dict with 'easy', 'medium', 'hard', 'expert' lists of (index, text)
        """
        scored = [(i, text, cls.score(text)) for i, text in enumerate(texts)]
        scored.sort(key=lambda x: x[2])  # Sort by complexity
        
        n = len(scored)
        quartile = n // 4
        
        categories = {
            'easy': [(s[0], s[1]) for s in scored[:quartile]],
            'medium': [(s[0], s[1]) for s in scored[quartile:2*quartile]],
            'hard': [(s[0], s[1]) for s in scored[2*quartile:3*quartile]],
            'expert': [(s[0], s[1]) for s in scored[3*quartile:]]
        }
        
        return categories


class CurriculumTTSDataset(Dataset):
    """Dataset that loads specific indices for curriculum learning"""
    
    def __init__(self, data_dir: str, indices: List[int], split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load metadata from CSV
        metadata_file = self.data_dir / split / "metadata.csv"
        all_metadata = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    all_metadata.append({
                        'audio_path': f"{split}/audio/{parts[0]}",
                        'text': parts[1]
                    })
        
        # Filter to only requested indices
        self.metadata = [all_metadata[i] for i in indices if i < len(all_metadata)]
        print(f"  Loaded {len(self.metadata)} samples for {split}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio using soundfile
        audio_path = self.data_dir / item['audio_path']
        waveform, sample_rate = sf.read(str(audio_path))
        
        # Convert to tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform_np = waveform.numpy()
            waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=16000)
            waveform = torch.tensor(waveform_np, dtype=torch.float32)
        
        # Ensure 1D
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        
        return {
            'text': item['text'],
            'waveform': waveform,
            'audio_path': str(audio_path)
        }


def get_stage_name(stage: int) -> str:
    """Get human-readable stage name"""
    names = ['easy', 'medium', 'hard', 'expert']
    return names[min(stage, len(names) - 1)]


def generate_sample(model, tokenizer, text: str, output_path: str, device: str = "cpu"):
    """Generate and save a TTS sample"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        output = model(**inputs)
        waveform = output.waveform[0].cpu().numpy()
        
        # Save as WAV
        import scipy.io.wavfile as wavfile
        wavfile.write(output_path, 16000, waveform)
    model.train()


def train_curriculum(config: CurriculumConfig):
    """Main curriculum-based training function"""
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    voice_upper = config.voice.upper()
    
    def log(msg):
        print(f"{datetime.now().strftime('%H:%M:%S')} │ {voice_upper} │ {msg}")
    
    log("=" * 60)
    log(f"🎓 CURRICULUM LEARNING - {voice_upper} VOICE")
    log("=" * 60)
    
    # Setup directories
    voice_dir = Path(config.data_dir) / config.voice
    output_dir = Path(config.output_dir) / config.voice
    sample_dir = Path(config.sample_dir) / f"{config.voice}_curriculum"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    
    # Load model and tokenizer
    log("Loading MMS-TTS model...")
    model = VitsModel.from_pretrained("facebook/mms-tts-guj").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-guj")
    
    # Enable training mode
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    # Load and categorize all training data by complexity
    log("Analyzing text complexity...")
    metadata_file = voice_dir / "train" / "metadata.csv"
    all_texts = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                all_texts.append(parts[1])
    categories = GujaratiComplexityScorer.categorize(all_texts)
    
    log(f"  📊 Easy samples: {len(categories['easy'])}")
    log(f"  📊 Medium samples: {len(categories['medium'])}")
    log(f"  📊 Hard samples: {len(categories['hard'])}")
    log(f"  📊 Expert samples: {len(categories['expert'])}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Test sentences for sample generation (progressive difficulty)
    test_sentences = {
        'easy': "નમસ્તે, કેમ છો?",
        'medium': "ગુજરાતી ભાષા ખૂબ સુંદર છે.",
        'hard': "આજનું હવામાન ખૂબ જ સારું છે અને આકાશ સ્વચ્છ છે.",
        'expert': "ભારતની સ્વતંત્રતા સંગ્રામમાં ગાંધીજીનું યોગદાન અવિસ્મરણીય છે."
    }
    
    # Generate initial baseline sample
    stage_name = get_stage_name(0)
    baseline_path = sample_dir / f"stage_00_{stage_name}_baseline.wav"
    generate_sample(model, tokenizer, test_sentences[stage_name], str(baseline_path), device)
    log(f"🎵 Baseline sample saved: {baseline_path.name}")
    
    # Training loop with curriculum stages
    total_epochs = config.epochs_per_stage * config.total_stages
    global_batch_count = 0
    best_loss = float('inf')
    
    for stage in range(config.total_stages):
        stage_name = get_stage_name(stage)
        log("")
        log(f"{'='*60}")
        log(f"📚 STAGE {stage + 1}/{config.total_stages}: {stage_name.upper()} DATA")
        log(f"{'='*60}")
        
        # Get indices for current stage (cumulative - include all easier stages)
        current_indices = []
        for s in range(stage + 1):
            s_name = get_stage_name(s)
            current_indices.extend([idx for idx, _ in categories[s_name]])
        
        log(f"Training on {len(current_indices)} samples (cumulative)")
        
        # Create dataset for current stage
        train_dataset = CurriculumTTSDataset(voice_dir, current_indices, "train")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Train for epochs_per_stage epochs
        start_epoch = stage * config.epochs_per_stage + 1
        end_epoch = (stage + 1) * config.epochs_per_stage
        
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_loss = 0
            batch_count = 0
            optimizer.zero_grad()
            
            pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch}/{total_epochs} [{stage_name}]",
                leave=True
            )
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    text = batch['text'][0]
                    target_waveform = batch['waveform'][0].to(device)
                    
                    # Tokenize
                    inputs = tokenizer(text, return_tensors="pt").to(device)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    generated_waveform = outputs.waveform[0]
                    
                    # Length matching
                    min_len = min(generated_waveform.shape[-1], target_waveform.shape[-1])
                    gen_wav = generated_waveform[..., :min_len]
                    tgt_wav = target_waveform[..., :min_len]
                    
                    # L1 Loss
                    loss = torch.nn.functional.l1_loss(gen_wav, tgt_wav)
                    loss = loss / config.gradient_accumulation_steps
                    loss.backward()
                    
                    epoch_loss += loss.item() * config.gradient_accumulation_steps
                    batch_count += 1
                    global_batch_count += 1
                    
                    # Gradient accumulation step
                    if batch_count % config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                        'stage': stage_name
                    })
                    
                    # Generate sample every N batches
                    if global_batch_count % config.sample_every_n_steps == 0:
                        sample_path = sample_dir / f"step_{global_batch_count:06d}_{stage_name}.wav"
                        generate_sample(
                            model, tokenizer, 
                            test_sentences[stage_name], 
                            str(sample_path), device
                        )
                        log(f"   🎵 Sample saved: {sample_path.name}")
                    
                except Exception as e:
                    continue
            
            # End of epoch
            avg_loss = epoch_loss / max(batch_count, 1)
            log("")
            log(f"   ━━━ Epoch {epoch} [{stage_name}] Complete │ Avg Loss: {avg_loss:.4f} ━━━")
            
            # Track best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / "best_model"
                model.save_pretrained(str(best_path))
                tokenizer.save_pretrained(str(best_path))
                log(f"   🏆 NEW BEST MODEL! (Loss: {best_loss:.4f})")
            
            # Save checkpoint every N epochs
            if epoch % config.save_every_n_epochs == 0:
                ckpt_path = output_dir / f"checkpoint_epoch_{epoch}"
                model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                log(f"   💾 Checkpoint saved: epoch {epoch}")
        
        # End of stage - generate comparison sample
        stage_sample_path = sample_dir / f"stage_{stage + 1:02d}_{stage_name}_complete.wav"
        generate_sample(model, tokenizer, test_sentences[stage_name], str(stage_sample_path), device)
        log(f"   🎵 Stage complete sample: {stage_sample_path.name}")
    
    # Final save
    final_path = output_dir / "final_model"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log("")
    log("=" * 60)
    log(f"✅ {voice_upper} VOICE CURRICULUM TRAINING COMPLETE!")
    log(f"   Best Loss: {best_loss:.4f}")
    log(f"   Model saved: {final_path}")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning TTS Training")
    parser.add_argument('--voice', type=str, required=True, choices=['male', 'female'],
                        help='Voice to train (male or female)')
    parser.add_argument('--epochs-per-stage', type=int, default=2,
                        help='Number of epochs per difficulty stage')
    parser.add_argument('--total-stages', type=int, default=4,
                        help='Total number of curriculum stages')
    parser.add_argument('--sample-every', type=int, default=200,
                        help='Generate sample every N batches')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    config = CurriculumConfig(
        voice=args.voice,
        epochs_per_stage=args.epochs_per_stage,
        total_stages=args.total_stages,
        sample_every_n_steps=args.sample_every,
        learning_rate=args.lr
    )
    
    train_curriculum(config)


if __name__ == "__main__":
    main()
