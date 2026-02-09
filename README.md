# ગુજરાતી વાણી (Gujarati Vaani)
## Intelligent Text-to-Speech & Document Reader

An AI-powered application that converts Gujarati text and PDF documents into natural-sounding speech using Meta's MMS (Massively Multilingual Speech) VITS model.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Technology Stack](#-technology-stack)
4. [System Architecture](#-system-architecture)
5. [Installation & Setup](#-installation--setup)
6. [Project Structure](#-project-structure)
7. [How It Works](#-how-it-works)
8. [The AI Model](#-the-ai-model)
9. [Text Processing Pipeline](#-text-processing-pipeline)
10. [Mobile Deployment](#-mobile-deployment)
11. [PWA (Progressive Web App)](#-pwa-progressive-web-app)
12. [OCR Setup](#-ocr-setup-for-scanned-pdfs)
13. [API Reference](#-api-reference)
14. [Configuration](#-configuration)
15. [Troubleshooting](#-troubleshooting)
16. [Future Enhancements](#-future-enhancements)

---

## 🎯 Project Overview

**Gujarati Vaani** is a complete Text-to-Speech (TTS) solution for the Gujarati language that:

- Converts typed Gujarati text to natural speech
- Extracts and reads text from PDF documents
- Uses OCR to read scanned/image-based PDFs
- Works offline after initial model download
- Runs on web browsers and mobile devices

### Why This Project?

- **Language Accessibility**: Provides TTS for Gujarati, an underserved language in TTS technology
- **Document Reading**: Helps visually impaired users access Gujarati documents
- **Offline Capability**: Works without internet after initial setup
- **Open Source**: Uses freely available AI models (no API costs)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Text Input** | Type or paste Gujarati Unicode text directly |
| **PDF Support** | Extract text from digital PDFs with selectable text |
| **OCR Support** | Read scanned PDFs using Tesseract OCR |
| **High-Quality TTS** | Natural speech using Meta MMS VITS neural network |
| **Audio Playback** | Built-in audio player in browser |
| **Download Audio** | Save generated speech as WAV file |
| **Turbo Mode** | Faster processing with FP16 optimization |
| **Offline Mode** | Works without internet after model download |
| **Mobile PWA** | Install as app on mobile devices |
| **Android App** | Native Android app via WebView |

---

## 🛠 Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Language** | Python | 3.9+ | Backend programming |
| **Web Framework** | Streamlit | 1.37+ | Web UI |
| **ML Framework** | PyTorch | 2.2+ | Neural network inference |
| **Model Hub** | Transformers | 4.33+ | Model loading |
| **TTS Model** | Meta MMS VITS | - | Speech synthesis |

### Additional Libraries

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `scipy` | Audio processing (WAV files) |
| `pypdf` | PDF text extraction |
| `pytesseract` | OCR for scanned PDFs |
| `pdf2image` | Convert PDF to images for OCR |
| `pdfplumber` | Alternative PDF extraction |
| `accelerate` | Model acceleration |

### Mobile Technologies

| Technology | Purpose |
|------------|---------|
| **Stlite** | Run Streamlit in browser (WebAssembly) |
| **Pyodide** | Python interpreter in browser |
| **Service Worker** | Offline caching for PWA |
| **Android WebView** | Native Android wrapper |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                         (Streamlit Web App)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│     Text Input Tab      │      PDF Upload Tab      │     Audio Output       │
│  ┌─────────────────┐   │  ┌─────────────────────┐ │  ┌─────────────────┐  │
│  │ Gujarati Text   │   │  │ Digital PDF         │ │  │ Audio Player    │  │
│  │ Input Area      │   │  │ Scanned PDF (OCR)   │ │  │ Download Button │  │
│  └─────────────────┘   │  └─────────────────────┘ │  └─────────────────┘  │
└────────────┬────────────┴────────────┬────────────┴────────────┬──────────┘
             │                         │                         │
             ▼                         ▼                         │
┌─────────────────────────────────────────────────────────────┐ │
│                   TEXT PROCESSING LAYER                      │ │
│                    (utils/text_utils.py)                     │ │
├─────────────────────────────────────────────────────────────┤ │
│  1. filter_gujarati_text() → Keep only Gujarati characters  │ │
│  2. normalize_text() → Clean whitespace and punctuation     │ │
│  3. chunk_text() → Split into 100-200 character chunks      │ │
└────────────────────────────────┬────────────────────────────┘ │
                                 │                               │
                                 ▼                               │
┌─────────────────────────────────────────────────────────────┐ │
│                      TTS MODEL LAYER                         │ │
│                       (utils/tts.py)                         │ │
├─────────────────────────────────────────────────────────────┤ │
│  Meta MMS VITS Model (facebook/mms-tts-guj)                 │ │
│  • Tokenize Gujarati text                                    │ │
│  • Neural network generates waveform                         │ │
│  • Optimizations: FP16, torch.compile                        │ │
└────────────────────────────────┬────────────────────────────┘ │
                                 │                               │
                                 ▼                               │
┌─────────────────────────────────────────────────────────────┐ │
│                       AUDIO OUTPUT                           │◄┘
│                  16-bit PCM WAV @ 16kHz                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation & Setup

### Prerequisites

- Windows 10/11 (or Linux/macOS)
- Python 3.9 or higher
- Internet connection (for first-time model download)
- ~2GB disk space (for model and dependencies)

### Step 1: Clone/Download Project

```powershell
# Navigate to your project folder
cd "D:\SEM 6\SDP"

# If using git
git clone <repository-url> "Gujarati Vaani"
cd "Gujarati Vaani"
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source .venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
pip install -U pip

# Install all requirements
pip install -r requirements.txt
```

### Step 4: Run the Application

```powershell
# Start the Streamlit app
streamlit run app.py
```

Open browser at: **http://localhost:8501**

### Step 5: (Optional) Download Model for Offline Use

```powershell
# Run this once to save model locally
python download_model.py
```

---

## 📁 Project Structure

```
Gujarati Vaani/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── download_model.py         # Script to download model for offline use
│
├── utils/                    # Core utility modules
│   ├── __init__.py
│   ├── text_utils.py         # Text processing functions
│   ├── tts.py                # TTS model loading & synthesis
│   └── tts_mobile.py         # Mobile-optimized TTS
│
├── model_weights/            # Saved model for offline use
│   ├── config.json           # Model configuration
│   ├── model_quantized.pt    # Quantized model (~150MB)
│   ├── original/             # Original model weights
│   │   ├── config.json
│   │   └── model.safetensors # Full weights (~200MB)
│   └── tokenizer/            # Tokenizer files
│       ├── vocab.json        # Gujarati vocabulary
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
│
├── tessdata/                 # OCR training data
│   ├── guj.traineddata       # Gujarati OCR model
│   ├── guj_fast.traineddata  # Fast Gujarati model
│   └── eng.traineddata       # English OCR model
│
├── stlite_pwa/               # Progressive Web App
│   ├── index.html            # Main PWA file (contains Python code)
│   ├── manifest.json         # PWA installation settings
│   ├── sw.js                 # Service Worker for offline
│   └── icons/                # App icons (72-512px)
│
├── mobile/                   # Android mobile app
│   └── android/
│       ├── app/
│       │   └── src/main/
│       │       ├── AndroidManifest.xml
│       │       └── java/.../MainActivity.kt
│       ├── build.gradle
│       └── settings.gradle
│
└── .venv/                    # Python virtual environment
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main entry point, Streamlit UI, handles user input |
| `utils/text_utils.py` | Gujarati text filtering, normalization, chunking, PDF extraction |
| `utils/tts.py` | Model loading, speech synthesis, audio conversion |
| `download_model.py` | Downloads and saves model for offline use |
| `tessdata/*.traineddata` | Tesseract OCR training data for Gujarati |

---

## ⚙ How It Works

### Complete Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              STEP 1: INPUT                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Option A: Text Input          Option B: PDF Upload                     │
│   ┌─────────────────────┐      ┌─────────────────────────────────────┐  │
│   │ User types Gujarati │      │ Digital PDF → pypdf extracts text   │  │
│   │ text in text area   │      │ Scanned PDF → Tesseract OCR         │  │
│   └──────────┬──────────┘      └──────────────────┬──────────────────┘  │
│              │                                     │                     │
│              └─────────────────┬───────────────────┘                     │
└────────────────────────────────┼─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         STEP 2: TEXT PROCESSING                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw Text: "Hello નમસ્તે! @#$ આ ટેસ્ટ છે..."                            │
│                          │                                               │
│                          ▼                                               │
│   filter_gujarati_text() → "નમસ્તે આ ટેસ્ટ છે"                           │
│   (Removes English, symbols, keeps Gujarati + punctuation)              │
│                          │                                               │
│                          ▼                                               │
│   normalize_text() → "નમસ્તે આ ટેસ્ટ છે"                                  │
│   (Cleans whitespace, fixes punctuation spacing)                        │
│                          │                                               │
│                          ▼                                               │
│   chunk_text(max_chars=100) → ["નમસ્તે આ ટેસ્ટ છે"]                       │
│   (Splits at sentence boundaries: . ! ? ; : newline ।)                  │
│                                                                          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         STEP 3: TTS SYNTHESIS                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   For each chunk:                                                        │
│                                                                          │
│   Text: "નમસ્તે"                                                          │
│           │                                                              │
│           ▼                                                              │
│   ┌─────────────────┐                                                    │
│   │    Tokenizer    │ → Convert text to token IDs                        │
│   └────────┬────────┘                                                    │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │  VITS Encoder   │ → Text representation                              │
│   └────────┬────────┘                                                    │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │ Duration Pred.  │ → Predict phoneme durations                        │
│   └────────┬────────┘                                                    │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │  Flow Decoder   │ → Generate mel-spectrogram                         │
│   └────────┬────────┘                                                    │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐                                                    │
│   │ HiFi-GAN Vocoder│ → Generate audio waveform                          │
│   └────────┬────────┘                                                    │
│            │                                                             │
│            ▼                                                             │
│   Audio Waveform (float32, -1 to 1)                                      │
│                                                                          │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         STEP 4: AUDIO OUTPUT                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Concatenate all chunk audio with 0.25s silence between             │
│   2. Convert float32 → int16 PCM (multiply by 32767)                    │
│   3. Create WAV file at 16000 Hz sample rate                             │
│   4. Display audio player in browser                                     │
│   5. Provide download button                                             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 The AI Model

### Model Information

| Property | Value |
|----------|-------|
| **Model ID** | `facebook/mms-tts-guj` |
| **Full Name** | Meta Massively Multilingual Speech - Gujarati TTS |
| **Architecture** | VITS (Variational Inference Text-to-Speech) |
| **Provider** | Meta AI (Facebook) |
| **Training Data** | Gujarati speech recordings |
| **Model Size** | ~200 MB |
| **Output** | 16kHz audio waveform |
| **Languages** | Part of MMS project (1,100+ languages) |

### VITS Architecture

```
                        VITS (Variational Inference TTS)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Input: "નમસ્તે"                                                         │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      TEXT ENCODER                                │   │
│  │  (Transformer-based)                                             │   │
│  │  • Multi-head attention                                          │   │
│  │  • Feed-forward layers                                           │   │
│  │  • Positional encoding                                           │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   STOCHASTIC DURATION PREDICTOR                  │   │
│  │  • Predicts how long each phoneme should last                    │   │
│  │  • Adds natural variation to speech rhythm                       │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FLOW-BASED DECODER                          │   │
│  │  • Normalizing flows for audio generation                        │   │
│  │  • High-quality spectrogram generation                           │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    HiFi-GAN VOCODER                              │   │
│  │  • Converts spectrogram to waveform                              │   │
│  │  • High-fidelity audio generation                                │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  Output: Audio Waveform (16kHz)                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Model Loading Options

| Mode | Description | Use Case |
|------|-------------|----------|
| **Online** | Download from Hugging Face | First run |
| **Offline** | Load from `model_weights/` | After setup |
| **Quantized** | INT8 quantized model | Mobile devices |
| **FP16** | Half-precision | Faster GPU inference |

### Why We Use Pre-trained Model (Not Fine-tuned)

- Pre-trained model is already high quality for Gujarati
- Fine-tuning requires 50+ hours of labeled audio data
- Needs GPU with 8GB+ VRAM for training
- Training takes 24-48 hours
- Not necessary for this project scope

---

## 📝 Text Processing Pipeline

### Gujarati Unicode Handling

```python
# Gujarati Unicode range: U+0A80 to U+0AFF
_GUJR_START = 0x0A80  # ઀
_GUJR_END = 0x0AFF    # ૿

def is_gujarati_char(ch: str) -> bool:
    """Check if character is Gujarati."""
    cp = ord(ch)
    return _GUJR_START <= cp <= _GUJR_END
```

### Text Chunking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_chars` (Turbo) | 100 | Smaller chunks, faster processing |
| `max_chars` (Standard) | 200 | Larger chunks, fewer API calls |
| `silence_s` | 0.25 sec | Pause between chunks |
| `sample_rate` | 16000 Hz | Audio samples per second |

### Sentence Delimiters

```python
# Characters that mark sentence boundaries
_SENTENCE_DELIMS = r"[\.\!\?\;\:\n\r\t\u0964\u0965\u0AF0]+"
#                    .  !  ?  ;  :  newlines  ।  ॥  ૰
#                                             (Gujarati punctuation)
```

---

## 📱 Mobile Deployment

### Option 1: Android WebView App

The Android app wraps the Streamlit web app in a native WebView.

**How it works:**
```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR PHONE                                │
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │  Android App    │   USB   │  Your PC                │   │
│  │  (WebView)      │◄───────►│  Streamlit Server       │   │
│  │                 │         │  (localhost:8502)       │   │
│  └─────────────────┘         └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Setup Steps:**
```powershell
# 1. Start Streamlit on your PC
streamlit run app.py --server.port 8502

# 2. Connect phone via USB (enable USB debugging)
# 3. Forward port using ADB
adb reverse tcp:8502 tcp:8502

# 4. Install and open the Android app
```

### Option 2: PWA (Progressive Web App)

Install the web app directly on your phone's home screen.

**Setup Steps:**
```powershell
# 1. Start a local server
cd stlite_pwa
python -m http.server 8080

# 2. Find your PC's IP address
ipconfig

# 3. Open on phone browser: http://<your-pc-ip>:8080
# 4. Click browser menu → "Add to Home Screen"
```

---

## 🌐 PWA (Progressive Web App)

### How Stlite PWA Works

The PWA runs Python directly in the browser using WebAssembly:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BROWSER                                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        index.html                                │   │
│  │                                                                  │   │
│  │   1. Stlite JavaScript (loads from CDN)                         │   │
│  │   2. Pyodide (Python compiled to WebAssembly)                   │   │
│  │   3. Python packages (numpy, scipy, pypdf)                      │   │
│  │   4. Streamlit UI framework                                      │   │
│  │   5. Your Python app code (embedded in HTML)                     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Service Worker (sw.js)                        │   │
│  │   • Caches all files for offline use                            │   │
│  │   • Intercepts network requests                                  │   │
│  │   • Serves from cache when offline                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### PWA Files

| File | Purpose |
|------|---------|
| `index.html` | Main app with embedded Python code |
| `manifest.json` | Makes app installable (name, icons, theme) |
| `sw.js` | Service Worker for offline caching |
| `icons/` | App icons in various sizes (72px to 512px) |

---

## 🔍 OCR Setup (for Scanned PDFs)

### Prerequisites

1. **Install Tesseract OCR**
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - During installation, select **Gujarati** language
   - Add to PATH: `C:\Program Files\Tesseract-OCR`

2. **Install Poppler** (for PDF to image conversion)
   - Download: https://github.com/oschwartz10612/poppler-windows/releases
   - Extract and add `bin` folder to PATH: `C:\poppler\bin`

3. **Restart Terminal** after PATH changes

### Verification

```powershell
# Check Tesseract installation
tesseract --version

# Check available languages (should show 'guj')
tesseract --list-langs
```

### OCR Processing Flow

```
Scanned PDF
    │
    ▼
pdf2image (convert_from_bytes)
    │ DPI: 300 for Gujarati
    ▼
Image Preprocessing
    │ • Convert to grayscale
    │ • Binary threshold (180)
    ▼
Tesseract OCR
    │ Language: 'guj'
    │ PSM: 3 (auto page segmentation)
    │ OEM: 1 (LSTM neural net)
    ▼
Extracted Gujarati Text
```

---

## 📚 API Reference

### utils/text_utils.py

```python
def is_gujarati_char(ch: str) -> bool:
    """Check if character is in Gujarati Unicode range (U+0A80-U+0AFF)."""

def filter_gujarati_text(text: str) -> str:
    """Remove non-Gujarati characters, keep punctuation and digits."""

def normalize_text(text: str) -> str:
    """Clean whitespace and normalize punctuation spacing."""

def chunk_text(text: str, max_chars: int = 200) -> List[str]:
    """Split text into chunks at sentence boundaries."""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from digital PDF using pypdf."""

def extract_text_from_pdf_ocr(file_bytes: bytes, lang: str = 'guj') -> str:
    """Extract text from scanned PDF using Tesseract OCR."""

def is_ocr_available() -> bool:
    """Check if OCR dependencies (Tesseract + Poppler) are installed."""
```

### utils/tts.py

```python
def load_mms_guj(
    device: str = "cpu",
    prefer_offline: bool = True,
    use_half_precision: bool = True,
    use_torch_compile: bool = True
) -> Tuple[VitsModel, AutoTokenizer]:
    """Load the Meta MMS Gujarati TTS model with optimizations."""

def synthesize_chunks(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    chunks: Iterable[str],
    device: str = "cpu",
    silence_s: float = 0.25
) -> Tuple[int, np.ndarray]:
    """Generate speech audio from text chunks."""

def synthesize_chunks_turbo(
    model: VitsModel,
    tokenizer: AutoTokenizer,
    chunks: List[str],
    device: str = "cpu",
    use_amp: bool = True,
    progress_callback = None
) -> Tuple[int, np.ndarray, dict]:
    """Optimized synthesis with progress tracking and statistics."""

def to_wav_bytes(sample_rate: int, pcm16: np.ndarray) -> bytes:
    """Convert audio numpy array to WAV file bytes."""

def is_offline_model_available() -> bool:
    """Check if model is saved locally in model_weights/ folder."""
```

---

## ⚙ Configuration

### Streamlit Theme Configuration

The app uses a dark theme with red accent. Create `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"

[server]
headless = true
port = 8501
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TESSDATA_PREFIX` | Path to Tesseract training data | `C:\Program Files\Tesseract-OCR\tessdata` |
| `STREAMLIT_SERVER_PORT` | Custom port for Streamlit | `8502` |

---

## 🔧 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Model download fails | Check internet connection, try VPN if blocked |
| OCR not working | Install Tesseract + Poppler, add to PATH, restart terminal |
| Audio not playing | Check browser permissions, try different browser |
| Slow processing | Enable Turbo mode, use GPU if available |
| Memory error | Reduce chunk size, close other applications |
| "Module not found" | Run `pip install -r requirements.txt` |

### Error Messages

```
"CUDA out of memory"
→ Solution: Switch to CPU mode (device = "cpu")

"Tesseract not found" / "TesseractNotFoundError"
→ Solution: Install Tesseract, add to PATH, restart terminal

"Model not found" / "OSError: Can't load model"
→ Solution: Run `python download_model.py` or check internet

"PDF extraction failed"
→ Solution: Try OCR mode for scanned PDFs

"No Gujarati text found"
→ Solution: Ensure input contains Gujarati Unicode characters
```

---

## 🚀 Future Enhancements

- [ ] Real-time streaming TTS
- [ ] Multiple voice options (male/female)
- [ ] Speech rate and pitch control
- [ ] SSML support for pronunciation control
- [ ] Cloud deployment (Streamlit Cloud, Hugging Face Spaces)
- [ ] True offline mobile app with ONNX runtime
- [ ] Hindi and other Indian language support
- [ ] Fine-tuning capability for custom voices

---

## 👨‍💻 Development Guide

### Running in Development Mode

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run with auto-reload on file changes
streamlit run app.py --server.runOnSave true
```

### Adding New Features

| To Modify | Edit This File |
|-----------|----------------|
| Text processing | `utils/text_utils.py` |
| TTS logic | `utils/tts.py` |
| Web UI | `app.py` |
| Mobile PWA | `stlite_pwa/index.html` |
| Android app | `mobile/android/app/src/main/java/.../MainActivity.kt` |

### Code Style

- Follow PEP 8 Python style guide
- Add docstrings to all functions
- Use type hints where possible
- Keep functions focused and modular

---

## 📄 License

This project is for educational purposes (SEM 6 SDP).

### Third-Party Licenses

| Component | License |
|-----------|---------|
| Meta MMS Model | CC-BY-NC 4.0 (Non-commercial) |
| Streamlit | Apache 2.0 |
| PyTorch | BSD-style |
| Transformers | Apache 2.0 |
| Tesseract OCR | Apache 2.0 |

---

## 🙏 Acknowledgments

- **Meta AI** for the MMS multilingual speech models
- **Hugging Face** for model hosting and Transformers library
- **Streamlit** for the excellent web framework
- **Tesseract OCR** for Gujarati text recognition
- **Stlite** for enabling Streamlit in browser

---

## 📞 Contact

For questions or issues, please open an issue in the repository.

---

*Created for SEM 6 - Software Development Project (SDP)*

*Last Updated: January 2026*
