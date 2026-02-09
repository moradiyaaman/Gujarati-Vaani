"""
Gujarati Vaani - TTS API for Hugging Face Spaces
=================================================
FastAPI server that downloads fine-tuned model from Azure Blob Storage
"""

import sys
import io as stdlib_io
import os
import time
import logging
from pathlib import Path
from typing import Optional
import urllib.request

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.io import wavfile
from transformers import VitsModel, AutoTokenizer

# Configuration
SAMPLE_RATE = 16000
NUM_THREADS = 2  # Reduced for faster single-request processing
MODEL_DIR = Path("./gujarati_tts_model")

# Azure Blob Storage Configuration
AZURE_ACCOUNT = "gujarativaaniw1824455535"
AZURE_CONTAINER = "gujarati-tts-model"
AZURE_SAS_TOKEN = "se=2027-12-31&sp=rl&spr=https&sv=2022-11-02&sr=c&sig=frDJV7JiInrKE6L8BxIKUGawHjv08Hj0wiR8W95haPg%3D"
AZURE_BASE_URL = f"https://{AZURE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gujarati number conversion
GUJARATI_DIGITS = {
    '૦': 'શૂન્ય', '૧': 'એક', '૨': 'બે', '૩': 'ત્રણ', '૪': 'ચાર',
    '૫': 'પાંચ', '૬': 'છ', '૭': 'સાત', '૮': 'આઠ', '૯': 'નવ',
    '0': 'શૂન્ય', '1': 'એક', '2': 'બે', '3': 'ત્રણ', '4': 'ચાર',
    '5': 'પાંચ', '6': 'છ', '7': 'સાત', '8': 'આઠ', '9': 'નવ'
}

GUJARATI_TENS = {
    10: 'દસ', 11: 'અગિયાર', 12: 'બાર', 13: 'તેર', 14: 'ચૌદ',
    15: 'પંદર', 16: 'સોળ', 17: 'સત્તર', 18: 'અઢાર', 19: 'ઓગણીસ',
    20: 'વીસ', 21: 'એકવીસ', 22: 'બાવીસ', 23: 'તેવીસ', 24: 'ચોવીસ',
    25: 'પચ્ચીસ', 26: 'છવ્વીસ', 27: 'સત્તાવીસ', 28: 'અઠ્ઠાવીસ', 29: 'ઓગણત્રીસ',
    30: 'ત્રીસ', 40: 'ચાળીસ', 50: 'પચાસ', 60: 'સાઠ', 70: 'સિત્તેર',
    80: 'એંસી', 90: 'નેવું', 100: 'સો', 1000: 'હજાર', 100000: 'લાખ',
    10000000: 'કરોડ'
}

def convert_gujarati_digit_to_arabic(char):
    """Convert single Gujarati digit to Arabic numeral."""
    gujarati_to_arabic = {'૦': '0', '૧': '1', '૨': '2', '૩': '3', '૪': '4',
                          '૫': '5', '૬': '6', '૭': '7', '૮': '8', '૯': '9'}
    return gujarati_to_arabic.get(char, char)

def number_to_gujarati_words(num):
    """Convert number to Gujarati words."""
    if num == 0:
        return 'શૂન્ય'
    
    if num < 0:
        return 'ઋણ ' + number_to_gujarati_words(-num)
    
    if num in GUJARATI_TENS:
        return GUJARATI_TENS[num]
    
    result = []
    
    # Crore (10 million)
    if num >= 10000000:
        crore = num // 10000000
        result.append(number_to_gujarati_words(crore) + ' કરોડ')
        num %= 10000000
    
    # Lakh (100 thousand)
    if num >= 100000:
        lakh = num // 100000
        result.append(number_to_gujarati_words(lakh) + ' લાખ')
        num %= 100000
    
    # Thousand
    if num >= 1000:
        thousand = num // 1000
        result.append(number_to_gujarati_words(thousand) + ' હજાર')
        num %= 1000
    
    # Hundred
    if num >= 100:
        hundred = num // 100
        result.append(number_to_gujarati_words(hundred) + ' સો')
        num %= 100
    
    # Tens and ones
    if num > 0:
        if num in GUJARATI_TENS:
            result.append(GUJARATI_TENS[num])
        elif num < 10:
            result.append(GUJARATI_DIGITS[str(num)])
        else:
            tens = (num // 10) * 10
            ones = num % 10
            if tens in GUJARATI_TENS:
                word = GUJARATI_TENS[tens]
                if ones > 0:
                    word = GUJARATI_DIGITS[str(ones)] + word[1:] if tens == 20 else word
            result.append(word if ones == 0 else GUJARATI_DIGITS[str(ones)] + ' ' + GUJARATI_TENS.get(tens, ''))
    
    return ' '.join(result)

def preprocess_gujarati_text(text):
    """Convert Gujarati numerals to words for better TTS."""
    import re
    
    # Convert Gujarati digits to Arabic for processing
    converted = ''
    for char in text:
        converted += convert_gujarati_digit_to_arabic(char)
    
    # Find all numbers (including decimals) and replace with words
    def replace_number(match):
        num_str = match.group(0)
        
        # Handle decimal numbers
        if '.' in num_str:
            parts = num_str.split('.')
            try:
                integer_part = int(parts[0]) if parts[0] else 0
                decimal_part = parts[1] if len(parts) > 1 else ''
                
                result = number_to_gujarati_words(integer_part)
                if decimal_part:
                    result += ' દશાંશ '
                    for digit in decimal_part:
                        result += GUJARATI_DIGITS.get(digit, digit) + ' '
                return result.strip()
            except:
                return num_str
        else:
            try:
                num = int(num_str)
                return number_to_gujarati_words(num)
            except:
                return num_str
    
    # Replace numbers with Gujarati words
    result = re.sub(r'\d+\.?\d*', replace_number, converted)
    
    # Also handle percentage symbol
    result = result.replace('%', ' ટકા')
    
    logger.info(f"Preprocessed: '{text[:50]}...' -> '{result[:50]}...'")
    return result

# Global model variables
model = None
tokenizer = None

# FastAPI app
app = FastAPI(
    title="Gujarati Vaani TTS API",
    description="Text-to-Speech API for Gujarati language using fine-tuned MMS-TTS model",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Gujarati text to synthesize")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")

class BatchSynthesizeRequest(BaseModel):
    texts: list[str] = Field(..., description="List of Gujarati texts to synthesize")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier for all texts")

def download_from_azure():
    """Download model files from Azure Blob Storage."""
    logger.info("Downloading model from Azure Blob Storage...")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    original_dir = MODEL_DIR / "original"
    mobile_dir = MODEL_DIR / "mobile"
    tokenizer_dir = MODEL_DIR / "tokenizer"
    
    original_dir.mkdir(exist_ok=True)
    mobile_dir.mkdir(exist_ok=True)
    tokenizer_dir.mkdir(exist_ok=True)
    
    files_to_download = [
        # Original model
        ("original/config.json", original_dir / "config.json"),
        ("original/model.safetensors", original_dir / "model.safetensors"),
        # Mobile optimized
        ("mobile/model_quantized.pt", mobile_dir / "model_quantized.pt"),
        # Tokenizer
        ("tokenizer/config.json", tokenizer_dir / "config.json"),
        ("tokenizer/special_tokens_map.json", tokenizer_dir / "special_tokens_map.json"),
        ("tokenizer/tokenizer_config.json", tokenizer_dir / "tokenizer_config.json"),
        ("tokenizer/vocab.json", tokenizer_dir / "vocab.json"),
    ]
    
    for blob_path, local_path in files_to_download:
        if local_path.exists():
            logger.info(f"✓ {blob_path} (cached)")
            continue
            
        url = f"{AZURE_BASE_URL}/{blob_path}?{AZURE_SAS_TOKEN}"
        try:
            logger.info(f"Downloading {blob_path}...")
            urllib.request.urlretrieve(url, local_path)
            logger.info(f"✓ Downloaded {blob_path}")
        except Exception as e:
            logger.error(f"Failed to download {blob_path}: {e}")
    
    logger.info("Model download complete!")

def load_model():
    """Load the fine-tuned MMS-TTS Gujarati model."""
    global model, tokenizer
    
    logger.info("=" * 60)
    logger.info("Loading Fine-Tuned Gujarati TTS Model")
    logger.info("=" * 60)
    
    # Download from Azure
    download_from_azure()
    
    # Set CPU threads
    torch.set_num_threads(NUM_THREADS)
    logger.info(f"Using {NUM_THREADS} CPU threads")
    
    original_dir = MODEL_DIR / "original"
    tokenizer_dir = MODEL_DIR / "tokenizer"
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    logger.info("✓ Tokenizer loaded")
    
    # Load model
    logger.info(f"Loading model from {original_dir}")
    model = VitsModel.from_pretrained(str(original_dir))
    model.eval()
    logger.info("✓ Model loaded and set to eval mode")
    
    logger.info("=" * 60)
    logger.info("Model ready for inference!")
    logger.info("=" * 60)

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Gujarati Vaani TTS API",
        "status": "running",
        "model": "Fine-tuned MMS-TTS Gujarati",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "version": "1.0.0"
    }

def adjust_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Adjust audio playback speed."""
    if speed == 1.0:
        return audio
    
    from scipy import signal
    num_samples = int(len(audio) / speed)
    return signal.resample(audio, num_samples)

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Synthesize Gujarati text to speech."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Preprocess: convert Gujarati numerals to words
    text = preprocess_gujarati_text(text)
    
    logger.info(f"Synthesizing: '{text[:50]}...'")
    start_time = time.time()
    
    try:
        # Generate audio
        with torch.inference_mode():
            inputs = tokenizer(text, return_tensors="pt")
            output = model(**inputs)
            waveform = output.waveform[0].cpu().numpy()
        
        # Apply speed adjustment
        if request.speed != 1.0:
            waveform = adjust_speed(waveform, request.speed)
        
        # Normalize and convert to int16
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        waveform = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
        
        # Convert to WAV
        wav_buffer = stdlib_io.BytesIO()
        wavfile.write(wav_buffer, SAMPLE_RATE, waveform)
        wav_bytes = wav_buffer.getvalue()
        
        elapsed = time.time() - start_time
        audio_duration = len(waveform) / SAMPLE_RATE
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        
        logger.info(f"Generated {audio_duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f})")
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(audio_duration),
                "X-Processing-Time": str(elapsed),
                "X-RTF": str(rtf),
            }
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/batch_synthesize")
async def batch_synthesize(request: BatchSynthesizeRequest):
    """Synthesize multiple Gujarati texts to speech (optimized for bulk generation)."""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    logger.info(f"Batch synthesizing {len(request.texts)} texts")
    start_time = time.time()
    results = []
    
    try:
        for idx, text in enumerate(request.texts):
            text = text.strip()
            if not text:
                continue
            
            # Preprocess: convert Gujarati numerals to words
            text = preprocess_gujarati_text(text)
            
            # Generate audio (no speed adjustment for batch - faster)
            with torch.inference_mode():
                inputs = tokenizer(text, return_tensors="pt")
                output = model(**inputs)
                waveform = output.waveform[0].cpu().numpy()
            
            # Minimal processing for speed
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            waveform = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
            
            # Convert to base64 for JSON response
            wav_buffer = stdlib_io.BytesIO()
            wavfile.write(wav_buffer, SAMPLE_RATE, waveform)
            wav_bytes = wav_buffer.getvalue()
            
            import base64
            results.append({
                "index": idx,
                "text": text,
                "audio_base64": base64.b64encode(wav_bytes).decode('utf-8'),
                "duration": len(waveform) / SAMPLE_RATE
            })
        
        elapsed = time.time() - start_time
        total_duration = sum(r["duration"] for r in results)
        
        logger.info(f"Batch generated {total_duration:.2f}s audio in {elapsed:.2f}s ({len(results)} files)")
        
        return {
            "count": len(results),
            "total_audio_duration": total_duration,
            "processing_time": elapsed,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch synthesis failed: {str(e)}")
