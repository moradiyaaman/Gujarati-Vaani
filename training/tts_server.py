"""
Local TTS API Server for Fine-Tuned Gujarati Model
===================================================
Run this on your laptop to serve the fine-tuned model.
Your Android app can connect to this server over WiFi.

Usage:
    python tts_server.py
    
Then update your phone's API_URL to your laptop's IP address.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import threading

# Configuration
HOST = "0.0.0.0"
PORT = 8000
MODEL_DIR = Path(__file__).parent.parent / "model_weights"

# Global model
model = None
tokenizer = None


def load_model():
    """Load the fine-tuned MMS-TTS model."""
    global model, tokenizer
    
    print("=" * 60)
    print("Loading Fine-Tuned Gujarati TTS Model")
    print("=" * 60)
    
    from transformers import VitsModel, AutoTokenizer
    
    original_dir = MODEL_DIR / "original"
    mobile_dir = MODEL_DIR / "mobile"
    
    # Load base model
    print("\n[1/3] Loading base architecture...")
    if original_dir.exists():
        model = VitsModel.from_pretrained(str(original_dir))
    else:
        model = VitsModel.from_pretrained("facebook/mms-tts-guj")
    
    # Load tokenizer
    print("[2/3] Loading tokenizer...")
    if mobile_dir.exists() and (mobile_dir / "vocab.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(mobile_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-guj")
    
    # Load fine-tuned weights
    weights_path = mobile_dir / "model_quantized.pt"
    print(f"[3/3] Loading weights from {weights_path}...")
    
    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("  Fine-tuned weights loaded!")
    else:
        print("  WARNING: Fine-tuned weights not found, using base model")
    
    model.eval()
    
    # CPU optimizations
    torch.set_num_threads(4)
    
    print("\nModel ready!")
    print("=" * 60)


def synthesize(text: str) -> np.ndarray:
    """Synthesize speech from Gujarati text."""
    global model, tokenizer
    
    if model is None:
        load_model()
    
    # Split into sentences for faster processing
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '।.?!':
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    all_audio = []
    
    with torch.inference_mode():
        for sentence in sentences:
            if not sentence:
                continue
            
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs)
            audio = outputs.waveform.squeeze().numpy()
            all_audio.append(audio)
            
            # Small pause between sentences
            all_audio.append(np.zeros(int(16000 * 0.1)))
    
    if not all_audio:
        return np.zeros(16000)
    
    final_audio = np.concatenate(all_audio)
    final_audio = final_audio / (np.max(np.abs(final_audio)) + 1e-8)
    
    return final_audio


class TTSHandler(BaseHTTPRequestHandler):
    """HTTP request handler for TTS API."""
    
    def do_POST(self):
        """Handle POST requests for TTS synthesis."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get('text', '')
            
            if not text:
                self.send_error(400, "Missing 'text' field")
                return
            
            print(f"Synthesizing: {text[:50]}...")
            audio = synthesize(text)
            
            response = {
                'audio': audio.tolist(),
                'sample_rate': 16000,
                'duration': len(audio) / 16000
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
            print(f"  Generated {len(audio)/16000:.2f}s audio")
            
        except Exception as e:
            print(f"Error: {e}")
            self.send_error(500, str(e))
    
    def do_GET(self):
        """Health check endpoint."""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode('utf-8'))
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = """
            <html>
            <head><title>Gujarati TTS Server</title></head>
            <body style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px;">
                <h1>Gujarati TTS Server</h1>
                <p>Fine-tuned MMS-VITS model is running!</p>
                <h3>API Usage:</h3>
                <pre>
POST /tts
Content-Type: application/json

{"text": "નમસ્તે"}
                </pre>
                <p>Response: {"audio": [...], "sample_rate": 16000}</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def get_local_ip():
    """Get the local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


def main():
    # Pre-load model
    load_model()
    
    local_ip = get_local_ip()
    
    print("\n" + "=" * 60)
    print("TTS Server Started!")
    print("=" * 60)
    print(f"\nAccess from this computer: http://localhost:{PORT}")
    print(f"Access from your phone:    http://{local_ip}:{PORT}")
    print("\nUpdate API_URL in your Android app to:")
    print(f'  API_URL = "http://{local_ip}:{PORT}/tts"')
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    server = HTTPServer((HOST, PORT), TTSHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
