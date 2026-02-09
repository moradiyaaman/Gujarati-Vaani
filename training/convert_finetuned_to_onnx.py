"""
Convert Fine-Tuned MMS-TTS Model to ONNX for Mobile Deployment
==============================================================
This script converts your fine-tuned PyTorch model to ONNX format
which can run on Android via ONNX Runtime.

Run this ONCE on your laptop before deploying to mobile.

Usage:
    python convert_finetuned_to_onnx.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "model_weights"
MOBILE_DIR = MODEL_WEIGHTS_DIR / "mobile"
ORIGINAL_DIR = MODEL_WEIGHTS_DIR / "original"
ONNX_OUTPUT = MOBILE_DIR / "model.onnx"

def load_finetuned_model():
    """Load the fine-tuned VITS model with quantized weights."""
    print("=" * 60)
    print("Loading Fine-Tuned MMS-TTS Gujarati Model")
    print("=" * 60)
    
    try:
        from transformers import VitsModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    
    # Step 1: Load base model architecture
    print("\n[1/4] Loading base model architecture...")
    if ORIGINAL_DIR.exists() and (ORIGINAL_DIR / "config.json").exists():
        print(f"  From local: {ORIGINAL_DIR}")
        model = VitsModel.from_pretrained(str(ORIGINAL_DIR))
    else:
        print("  From HuggingFace: facebook/mms-tts-guj")
        model = VitsModel.from_pretrained("facebook/mms-tts-guj")
    
    # Step 2: Load tokenizer
    print("\n[2/4] Loading tokenizer...")
    if MOBILE_DIR.exists() and (MOBILE_DIR / "vocab.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(MOBILE_DIR))
        print(f"  From: {MOBILE_DIR}")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-guj")
        print("  From: facebook/mms-tts-guj")
    
    # Step 3: Load fine-tuned weights
    weights_path = MOBILE_DIR / "model_quantized.pt"
    print(f"\n[3/4] Loading fine-tuned weights...")
    
    if not weights_path.exists():
        print(f"  ERROR: Weights file not found: {weights_path}")
        print("  Please ensure model_quantized.pt exists in model_weights/mobile/")
        sys.exit(1)
    
    print(f"  From: {weights_path}")
    print(f"  Size: {weights_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load state dict
    state_dict = torch.load(str(weights_path), map_location='cpu')
    
    # Handle different weight formats
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Load weights into model (allow partial loading)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded! (missing: {len(missing)}, unexpected: {len(unexpected)})")
    
    model.eval()
    print("\n[4/4] Model ready for ONNX conversion!")
    
    return model, tokenizer


class VitsONNXWrapper(torch.nn.Module):
    """Wrapper for VITS model for ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        """
        Forward pass for ONNX export.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
        
        Returns:
            waveform: Audio waveform [batch, 1, samples]
        """
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generate speech
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return outputs.waveform


def export_to_onnx(model, tokenizer, output_path):
    """Export the model to ONNX format."""
    print("\n" + "=" * 60)
    print("Exporting to ONNX Format")
    print("=" * 60)
    
    # Create wrapper
    wrapper = VitsONNXWrapper(model)
    wrapper.eval()
    
    # Sample input for tracing
    sample_text = "નમસ્તે"
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print(f"\nSample input: '{sample_text}'")
    print(f"Token IDs shape: {input_ids.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        test_output = wrapper(input_ids)
    print(f"Output shape: {test_output.shape}")
    print(f"Audio duration: {test_output.shape[-1] / 16000:.2f}s")
    
    # Export to ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to: {output_path}")
    print("This may take a few minutes...")
    
    try:
        torch.onnx.export(
            wrapper,
            input_ids,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['waveform'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'waveform': {0: 'batch', 2: 'samples'}
            },
            verbose=False
        )
        print(f"\n✅ ONNX export successful!")
        print(f"   File: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        print("\nTrying simplified export...")
        
        # Try simpler export
        try:
            torch.onnx.export(
                wrapper,
                input_ids,
                str(output_path),
                export_params=True,
                opset_version=11,
                input_names=['input_ids'],
                output_names=['waveform'],
            )
            print(f"\n✅ Simplified ONNX export successful!")
        except Exception as e2:
            print(f"❌ Simplified export also failed: {e2}")
            return False
    
    return True


def verify_onnx(onnx_path, tokenizer):
    """Verify the ONNX model works correctly."""
    print("\n" + "=" * 60)
    print("Verifying ONNX Model")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping verification.")
        print("Install with: pip install onnxruntime")
        return
    
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        print(f"ONNX file not found: {onnx_path}")
        return
    
    print(f"\nLoading ONNX model: {onnx_path}")
    
    # Create session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(str(onnx_path), sess_options)
    
    # Test inference
    test_text = "ગુજરાતી વાણી"
    inputs = tokenizer(test_text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    
    print(f"Test text: '{test_text}'")
    print(f"Input shape: {input_ids.shape}")
    
    # Run inference
    import time
    start = time.time()
    outputs = session.run(None, {"input_ids": input_ids})
    elapsed = time.time() - start
    
    waveform = outputs[0]
    print(f"\n✅ ONNX inference successful!")
    print(f"   Output shape: {waveform.shape}")
    print(f"   Audio duration: {waveform.shape[-1] / 16000:.2f}s")
    print(f"   Inference time: {elapsed * 1000:.0f}ms")
    
    # Save test audio
    test_audio_path = onnx_path.parent / "test_onnx_output.wav"
    audio = waveform.squeeze()
    audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)
    
    import wave
    with wave.open(str(test_audio_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio.tobytes())
    
    print(f"   Test audio saved: {test_audio_path}")


def main():
    print("\n" + "=" * 60)
    print("   FINE-TUNED MMS-TTS -> ONNX CONVERTER")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_finetuned_model()
    
    # Export to ONNX
    success = export_to_onnx(model, tokenizer, ONNX_OUTPUT)
    
    if success:
        # Verify
        verify_onnx(ONNX_OUTPUT, tokenizer)
        
        print("\n" + "=" * 60)
        print("   NEXT STEPS")
        print("=" * 60)
        print("""
1. Copy the ONNX model to your Android project:
   - Source: model_weights/mobile/model.onnx
   - Destination: standalone_apk/app/src/main/assets/

2. Rebuild the APK:
   cd standalone_apk
   ./gradlew assembleDebug

3. Install on device:
   adb install -r app/build/outputs/apk/debug/app-debug.apk
""")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
