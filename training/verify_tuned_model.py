#!/usr/bin/env python3
"""
================================================================================
VERIFY TUNED MODEL - A/B COMPARISON TOOL
================================================================================
Compares the original Meta MMS model against your fine-tuned model.

This script generates audio from both models using the same sentence,
allowing you to hear the quality improvement immediately.

Output:
    verification/
    ├── original.wav      (Base Meta MMS model)
    ├── tuned.wav         (Your fine-tuned model)
    ├── comparison.json   (Metrics comparison)
    └── report.txt        (Human-readable summary)

Usage:
    python verify_tuned_model.py --tuned ./model_weights/finetuned/final

What to Listen For:
    ✅ Clearer pronunciation of Gujarati sounds
    ✅ More natural intonation/prosody
    ✅ Better handling of specific vocabulary from training data
    ✅ Consistent speaker identity (if trained on single speaker)

================================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import scipy.io.wavfile as wav

from transformers import VitsModel, AutoTokenizer

# Add parent for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test sentences - covering different aspects of Gujarati
TEST_SENTENCES = [
    {
        "id": "greeting",
        "text": "નમસ્તે, આ ગુજરાતી વાણી પ્રોજેક્ટ છે.",
        "translation": "Hello, this is the Gujarati Vaani project.",
        "purpose": "Basic greeting and project name"
    },
    {
        "id": "complex",
        "text": "ગુજરાતી ભાષા ખૂબ જ મધુર અને સમૃદ્ધ છે, તેની લિપિ દેવનાગરી જેવી સુંદર છે.",
        "translation": "Gujarati language is very sweet and rich, its script is beautiful like Devanagari.",
        "purpose": "Complex sentence with conjuncts"
    },
    {
        "id": "numbers",
        "text": "આજે તારીખ પાંચ જાન્યુઆરી બે હજાર છવ્વીસ છે.",
        "translation": "Today's date is January 5th, 2026.",
        "purpose": "Numbers and dates"
    },
    {
        "id": "technical",
        "text": "મશીન લર્નિંગ અને આર્ટિફિશિયલ ઇન્ટેલિજન્સ ટેક્નોલોજી ખૂબ ઝડપથી વિકસી રહી છે.",
        "translation": "Machine learning and artificial intelligence technology is developing very rapidly.",
        "purpose": "Technical terms (loanwords)"
    }
]

# Default sentence for quick test
DEFAULT_SENTENCE = TEST_SENTENCES[0]["text"]


def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_original_model(logger: logging.Logger):
    """Load the original Meta MMS model."""
    logger.info("Loading original Meta MMS model (facebook/mms-tts-guj)...")
    
    model = VitsModel.from_pretrained("facebook/mms-tts-guj")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-guj")
    
    model.eval()
    
    logger.info("  ✓ Original model loaded")
    return model, tokenizer


def load_tuned_model(checkpoint_path: Path, logger: logging.Logger):
    """Load the fine-tuned model."""
    logger.info(f"Loading fine-tuned model from: {checkpoint_path}")
    
    model = VitsModel.from_pretrained(str(checkpoint_path))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    
    model.eval()
    
    logger.info("  ✓ Fine-tuned model loaded")
    return model, tokenizer


def generate_audio(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu"
) -> np.ndarray:
    """Generate audio from text."""
    model = model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        audio = outputs.waveform.squeeze().cpu().numpy()
    
    return audio


def compute_audio_metrics(audio1: np.ndarray, audio2: np.ndarray) -> dict:
    """
    Compute comparison metrics between two audio signals.
    
    Note: These are technical metrics, not perceptual quality scores.
    Final quality judgment should be done by listening.
    """
    # Ensure same length for comparison
    min_len = min(len(audio1), len(audio2))
    a1, a2 = audio1[:min_len], audio2[:min_len]
    
    # Correlation (higher = more similar)
    correlation = np.corrcoef(a1, a2)[0, 1]
    
    # Mean Squared Difference
    msd = np.mean((a1 - a2) ** 2)
    
    # Energy ratio
    energy1 = np.sqrt(np.mean(a1 ** 2))
    energy2 = np.sqrt(np.mean(a2 ** 2))
    energy_ratio = energy2 / energy1 if energy1 > 0 else 0
    
    # Duration comparison
    duration1 = len(audio1) / 16000  # assuming 16kHz
    duration2 = len(audio2) / 16000
    
    return {
        "correlation": float(correlation),
        "mean_squared_difference": float(msd),
        "original_energy": float(energy1),
        "tuned_energy": float(energy2),
        "energy_ratio": float(energy_ratio),
        "original_duration_sec": float(duration1),
        "tuned_duration_sec": float(duration2),
        "duration_ratio": float(duration2 / duration1) if duration1 > 0 else 0
    }


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = 16000):
    """Save audio to WAV file."""
    # Normalize to int16 range
    audio_int16 = (audio * 32767).astype(np.int16)
    wav.write(path, sample_rate, audio_int16)


def run_verification(
    tuned_path: Path,
    output_dir: Path,
    sentence: str,
    full_suite: bool,
    logger: logging.Logger
):
    """
    Run the verification comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("GUJARATI VAANI - MODEL VERIFICATION")
    logger.info("=" * 60)
    
    # Load models
    original_model, original_tokenizer = load_original_model(logger)
    tuned_model, tuned_tokenizer = load_tuned_model(tuned_path, logger)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Get sentences to test
    if full_suite:
        sentences = TEST_SENTENCES
    else:
        sentences = [{"id": "custom", "text": sentence, "translation": "", "purpose": "User input"}]
    
    results = []
    
    for item in sentences:
        text = item["text"]
        sentence_id = item["id"]
        
        logger.info(f"\n📝 Testing: {text[:50]}...")
        
        # Generate audio from both models
        logger.info("  Generating with original model...")
        original_audio = generate_audio(original_model, original_tokenizer, text, device)
        
        logger.info("  Generating with fine-tuned model...")
        tuned_audio = generate_audio(tuned_model, tuned_tokenizer, text, device)
        
        # Save audio files
        original_path = output_dir / f"original_{sentence_id}.wav"
        tuned_path_out = output_dir / f"tuned_{sentence_id}.wav"
        
        save_audio(original_audio, original_path)
        save_audio(tuned_audio, tuned_path_out)
        
        logger.info(f"  Saved: {original_path.name} & {tuned_path_out.name}")
        
        # Compute metrics
        metrics = compute_audio_metrics(original_audio, tuned_audio)
        
        results.append({
            "sentence_id": sentence_id,
            "text": text,
            "translation": item.get("translation", ""),
            "purpose": item.get("purpose", ""),
            "original_file": str(original_path.name),
            "tuned_file": str(tuned_path_out.name),
            "metrics": metrics
        })
        
        logger.info(f"  Correlation: {metrics['correlation']:.4f}")
        logger.info(f"  Duration: {metrics['original_duration_sec']:.2f}s → {metrics['tuned_duration_sec']:.2f}s")
    
    # Also save as original.wav and tuned.wav for convenience
    if len(sentences) == 1:
        # Copy to standard names
        import shutil
        shutil.copy(output_dir / f"original_{sentences[0]['id']}.wav", output_dir / "original.wav")
        shutil.copy(output_dir / f"tuned_{sentences[0]['id']}.wav", output_dir / "tuned.wav")
    
    # Save comparison report
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "original_model": "facebook/mms-tts-guj",
        "tuned_model": str(tuned_path),
        "device": device,
        "results": results
    }
    
    with open(output_dir / "comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    # Generate human-readable report
    report = generate_report(comparison, sentences)
    with open(output_dir / "report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n📁 All outputs saved to: {output_dir}")
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\n🎧 Next steps:")
    logger.info("   1. Open the output folder")
    logger.info("   2. Listen to original.wav and tuned.wav")
    logger.info("   3. Compare pronunciation, clarity, and naturalness")
    logger.info("   4. Read report.txt for technical details")
    logger.info("")


def generate_report(comparison: dict, sentences: list) -> str:
    """Generate a human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("GUJARATI VAANI - MODEL COMPARISON REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Generated: {comparison['timestamp']}")
    lines.append(f"Original:  {comparison['original_model']}")
    lines.append(f"Tuned:     {comparison['tuned_model']}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("WHAT TO LISTEN FOR:")
    lines.append("-" * 60)
    lines.append("✓ Clearer consonant clusters (e.g., ક્ષ, જ્ઞ, શ્ર)")
    lines.append("✓ More natural rhythm and pauses")
    lines.append("✓ Better stress on important words")
    lines.append("✓ Consistent voice quality throughout")
    lines.append("✓ Reduced robotic artifacts")
    lines.append("")
    lines.append("-" * 60)
    lines.append("TEST RESULTS:")
    lines.append("-" * 60)
    
    for result in comparison['results']:
        lines.append("")
        lines.append(f"Sentence: {result['text']}")
        if result.get('translation'):
            lines.append(f"English:  {result['translation']}")
        lines.append(f"Purpose:  {result.get('purpose', 'N/A')}")
        lines.append(f"Files:    {result['original_file']} vs {result['tuned_file']}")
        
        m = result['metrics']
        lines.append(f"Metrics:")
        lines.append(f"  - Correlation:    {m['correlation']:.4f} (1.0 = identical)")
        lines.append(f"  - Duration:       {m['original_duration_sec']:.2f}s → {m['tuned_duration_sec']:.2f}s")
        lines.append(f"  - Energy ratio:   {m['energy_ratio']:.2f} (1.0 = same loudness)")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append("INTERPRETATION:")
    lines.append("-" * 60)
    lines.append("")
    lines.append("• Correlation < 0.5:  Model has learned significantly different")
    lines.append("                      characteristics. Listen carefully!")
    lines.append("")
    lines.append("• Correlation 0.5-0.8: Model has adapted while keeping base")
    lines.append("                       structure. Good fine-tuning result.")
    lines.append("")
    lines.append("• Correlation > 0.8:  Model is similar to original.")
    lines.append("                      May need more training or data.")
    lines.append("")
    lines.append("Remember: Lower correlation doesn't mean worse quality!")
    lines.append("The goal is BETTER pronunciation, which means DIFFERENT output.")
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs fine-tuned Gujarati TTS model"
    )
    parser.add_argument(
        '--tuned', '-t',
        type=str,
        default="./model_weights/finetuned/final",
        help="Path to fine-tuned model checkpoint"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="./verification",
        help="Output directory for comparison files"
    )
    parser.add_argument(
        '--sentence', '-s',
        type=str,
        default=DEFAULT_SENTENCE,
        help="Test sentence (default: standard greeting)"
    )
    parser.add_argument(
        '--full-suite',
        action='store_true',
        help="Run full test suite with multiple sentences"
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    
    tuned_path = Path(args.tuned)
    output_dir = Path(args.output)
    
    if not tuned_path.exists():
        logger.error(f"Fine-tuned model not found: {tuned_path}")
        logger.error("Train the model first or provide correct path")
        sys.exit(1)
    
    run_verification(tuned_path, output_dir, args.sentence, args.full_suite, logger)


if __name__ == "__main__":
    main()
