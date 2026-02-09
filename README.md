# ગુજરાતી વાણી (Gujarati Vaani)
## Intelligent Gujarati Text-to-Speech Mobile Application

An AI-powered mobile application that converts Gujarati text into natural-sounding speech using a **fine-tuned** Meta MMS (Massively Multilingual Speech) VITS model.

![Flutter](https://img.shields.io/badge/Flutter-3.35+-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)
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
9. [API Reference](#-api-reference)
10. [Development Journey](#-development-journey)
11. [Future Enhancements](#-future-enhancements)

---

## 🎯 Project Overview

**Gujarati Vaani** is a complete Text-to-Speech (TTS) solution for the Gujarati language featuring:

- **Flutter Mobile App** - Native Android application with modern UI
- **Cloud TTS Backend** - Hugging Face Spaces running FastAPI server
- **Fine-tuned Model** - Custom trained on Gujarati speech data (275 MB)
- **Smart Text Processing** - Handles large texts with chunked processing
- **Gujarati Number Reading** - Converts ૧૨૩ to "એકસો ત્રેવીસ"

### Why This Project?

- **Language Accessibility**: High-quality TTS for Gujarati, an underserved language
- **Visual Accessibility**: Helps visually impaired users access Gujarati content
- **Mobile-First**: Native mobile experience with download & share features
- **Free & Open**: Uses Hugging Face free tier (no API costs)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Text Input** | Type or paste Gujarati Unicode text |
| **Large Text Support** | Processes texts of any length with chunked processing |
| **Number Reading** | Converts Gujarati numerals (૧, ૨, ૩) to spoken words |
| **High-Quality TTS** | Natural speech using fine-tuned MMS VITS model |
| **Audio Player** | Built-in player with seek bar and playback speed control |
| **Download Audio** | Save generated speech to device Downloads folder |
| **Share Audio** | Share audio via WhatsApp, email, or other apps |
| **Progress Tracking** | Real-time progress display during generation |
| **Playback Speed** | Adjust speed from 0.5x to 2.0x |

---

## 🛠 Technology Stack

### Backend (Hugging Face Space)

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Runtime** | Python | 3.10 | Server programming |
| **Framework** | FastAPI | 0.104+ | REST API server |
| **ML Framework** | PyTorch | 2.1.0 | Neural network inference |
| **Model Library** | Transformers | 4.36.0 | Model loading & inference |
| **TTS Model** | Fine-tuned MMS VITS | - | Speech synthesis |
| **Model Storage** | Azure Blob Storage | - | 275 MB model hosting |

### Mobile App (Flutter)

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Framework** | Flutter | 3.35+ | Cross-platform UI |
| **Language** | Dart | 3.9+ | App programming |
| **HTTP Client** | http | 1.1.0 | API communication |
| **Audio Player** | audioplayers | 5.2.1 | Audio playback |
| **File Handling** | path_provider | 2.1.1 | File system access |
| **Sharing** | share_plus | 7.2.1 | System share sheet |
| **Permissions** | permission_handler | 11.1.0 | Storage permissions |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUTTER MOBILE APP                                │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  Text Input     │  │  Progress Bar   │  │  Audio Player               │ │
│  │  (Large texts   │  │  (Elapsed time  │  │  • Seek bar                 │ │
│  │   supported)    │  │   per chunk)    │  │  • Speed control (0.5x-2x) │ │
│  └────────┬────────┘  └────────▲────────┘  │  • Download button          │ │
│           │                    │           │  • Share button             │ │
│           │                    │           └─────────────▲───────────────┘ │
│           │                    │                         │                  │
└───────────┼────────────────────┼─────────────────────────┼──────────────────┘
            │ HTTPS              │ Progress               │ Audio WAV
            │                    │                         │
            ▼                    │                         │
┌───────────────────────────────────────────────────────────────────────────┐
│                        HUGGING FACE SPACE                                  │
│                 (moradiyaaman-gujarati-vaani-tts.hf.space)                │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         FastAPI Server                               │  │
│  │                                                                      │  │
│  │  POST /synthesize         POST /batch_synthesize    GET /health     │  │
│  │  (Single text)            (Multiple chunks)         (Status check)  │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      Text Preprocessing                              │  │
│  │                                                                      │  │
│  │  1. Gujarati numeral to words (૧૨૩ → "એકસો ત્રેવીસ")                │  │
│  │  2. Text normalization                                               │  │
│  │  3. Sentence boundary detection                                      │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    Fine-tuned MMS-TTS Model                          │  │
│  │                    (275 MB, loaded from Azure Blob)                  │  │
│  │                                                                      │  │
│  │  Tokenizer → VITS Encoder → Duration Predictor → HiFi-GAN Vocoder  │  │
│  └──────────────────────────────┬──────────────────────────────────────┘  │
│                                 │                                          │
│                                 ▼                                          │
│                        Audio WAV (16kHz, 16-bit PCM)                       │
└────────────────────────────────────────────────────────────────────────────┘
            │
            │ Model download on startup
            ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         AZURE BLOB STORAGE                                  │
│                                                                             │
│  Account: gujarativaaniw1824455535                                         │
│  Container: gujarati-tts-model                                              │
│                                                                             │
│  Files:                                                                     │
│  ├── config.json (model configuration)                                     │
│  ├── model.safetensors (275 MB - model weights)                            │
│  ├── vocab.txt (Gujarati vocabulary)                                       │
│  └── tokenizer files                                                        │
│                                                                             │
│  Access: SAS Token (valid until 2027-12-31)                                │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation & Setup

### Option 1: Use the Pre-built APK

1. Download the APK from releases
2. Install on your Android device
3. Open the app and start converting text to speech!

### Option 2: Build Flutter App from Source

#### Prerequisites
- Flutter SDK 3.35+
- Android Studio or VS Code with Flutter extension
- Android device or emulator

#### Steps

```bash
# 1. Clone the repository
git clone https://github.com/moradiyaaman/Gujarati-Vaani.git
cd "Gujarati-Vaani/flutter_app"

# 2. Get dependencies
flutter pub get

# 3. Run on connected device
flutter run

# 4. Build release APK
flutter build apk --release
```

The APK will be at: `build/app/outputs/flutter-apk/app-release.apk`

### Option 3: Deploy Your Own Backend

#### Prerequisites
- Hugging Face account
- Azure Storage account (for model hosting)

#### Steps

1. **Create Hugging Face Space**
   - Go to huggingface.co/spaces
   - Create new Space with Docker SDK
   - Upload files from `huggingface_space/` folder

2. **Upload Model to Azure Blob Storage**
   - Create storage account and container
   - Upload fine-tuned model files
   - Generate SAS token for read access

3. **Configure Environment**
   - Set `AZURE_STORAGE_SAS_URL` secret in Hugging Face Space
   - Update API URL in Flutter app

---

## 📁 Project Structure

```
Gujarati-Vaani/
│
├── flutter_app/                    # Flutter mobile application
│   ├── lib/
│   │   └── main.dart               # Main app code (UI + logic)
│   ├── android/                    # Android-specific configuration
│   │   └── app/src/main/
│   │       └── AndroidManifest.xml # Permissions
│   ├── pubspec.yaml                # Flutter dependencies
│   └── README.md                   # Flutter app documentation
│
├── huggingface_space/              # Hugging Face Space (Backend)
│   ├── app.py                      # FastAPI server with TTS logic
│   ├── Dockerfile                  # Docker configuration
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Space documentation
│
├── training/                       # Model training scripts
│   ├── train.py                    # Training script
│   ├── train_curriculum.py         # Curriculum learning training
│   ├── prepare_dataset.py          # Dataset preparation
│   └── requirements_train.txt      # Training dependencies
│
├── DEVELOPMENT_JOURNEY.md          # Documentation of approaches tried
├── README.md                       # This file
└── logo.png                        # App logo
```

---

## ⚙ How It Works

### Text Processing Flow

```
User Input (Gujarati Text)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLUTTER APP - Text Chunking                   │
│                                                                  │
│  1. Split text into ~500 character chunks                        │
│  2. Split at sentence boundaries (. ! ? । ॥)                    │
│  3. Protect decimal numbers from splitting                       │
│  4. Send chunks to API sequentially                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼ (for each chunk)
┌─────────────────────────────────────────────────────────────────┐
│                   SERVER - Preprocessing                         │
│                                                                  │
│  1. Convert Gujarati numerals to words:                          │
│     ૨૦૨૫ → "બે હજાર પચીસ"                                        │
│     ૧,૨૩,૪૫૬ → "એક લાખ ત્રેવીસ હજાર ચારસો છપ્પન"                 │
│                                                                  │
│  2. Normalize whitespace and punctuation                         │
│                                                                  │
│  3. Handle special characters                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SERVER - TTS Synthesis                         │
│                                                                  │
│  1. Tokenize text using Gujarati vocabulary                      │
│  2. Run through VITS neural network:                             │
│     • Text Encoder (attention-based)                             │
│     • Duration Predictor (rhythm)                                │
│     • Flow Decoder (spectrogram)                                 │
│     • HiFi-GAN Vocoder (waveform)                                │
│  3. Return 16kHz WAV audio                                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FLUTTER APP - Audio Handling                   │
│                                                                  │
│  1. Receive audio bytes from server                              │
│  2. Concatenate chunks with silence gaps                         │
│  3. Save to temporary file                                       │
│  4. Play with built-in audio player                              │
│  5. Option to download to Downloads folder                       │
│  6. Option to share via system share sheet                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 The AI Model

### Model Information

| Property | Value |
|----------|-------|
| **Base Model** | facebook/mms-tts-guj |
| **Architecture** | VITS (Variational Inference TTS) |
| **Training Data** | Gujarati FLEURS dataset + custom data |
| **Model Size** | ~275 MB |
| **Output** | 16kHz mono WAV audio |
| **Fine-tuning** | Curriculum learning for improved quality |

### Why Fine-tuned?

The base MMS model was good but had issues with:
- Some matra (vowel sign) pronunciations
- Number reading (said digits separately)
- Punctuation handling

Our fine-tuned model improves:
- ✅ Better matra pronunciation
- ✅ Natural sentence flow
- ✅ Server-side number-to-word conversion

### VITS Architecture

```
                        VITS (Variational Inference TTS)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Input: "ગુજરાત"                                                         │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      TEXT ENCODER                                │   │
│  │  • Multi-head self-attention                                     │   │
│  │  • Feed-forward layers                                           │   │
│  │  • Learned character embeddings                                  │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   STOCHASTIC DURATION PREDICTOR                  │   │
│  │  • Predicts phoneme durations                                    │   │
│  │  • Adds natural rhythm variation                                 │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FLOW-BASED DECODER                          │   │
│  │  • Normalizing flows                                             │   │
│  │  • Generates mel-spectrogram                                     │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    HiFi-GAN VOCODER                              │   │
│  │  • Converts spectrogram to waveform                              │   │
│  │  • High-fidelity audio synthesis                                 │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  Output: Audio Waveform (16kHz, 16-bit PCM)                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 API Reference

### Base URL
```
https://moradiyaaman-gujarati-vaani-tts.hf.space
```

### Endpoints

#### `GET /health`
Check if the server and model are ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `POST /synthesize`
Convert text to speech.

**Request:**
```json
{
  "text": "નમસ્તે, કેમ છો?"
}
```

**Response:** WAV audio file (audio/wav)

#### `POST /batch_synthesize`
Convert multiple text chunks to speech.

**Request:**
```json
{
  "texts": ["નમસ્તે", "કેમ છો?"]
}
```

**Response:** Concatenated WAV audio file

---

## 🛤 Development Journey

See [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md) for detailed documentation of all approaches tried during development, including:

- Local Streamlit app
- Azure App Service deployment (500 errors)
- ONNX model conversion attempts
- Sherpa-ONNX for mobile
- Standalone Android APK
- PWA with Stlite

The final solution using **Hugging Face Spaces + Flutter** solved all previous challenges.

---

## 🚀 Future Enhancements

- [ ] Multiple voice options (male/female voices)
- [ ] Speech rate control on server side
- [ ] Offline mode with on-device model (ONNX)
- [ ] PDF text extraction and reading
- [ ] iOS app support
- [ ] Hindi and other Indian language support
- [ ] Real-time streaming TTS
- [ ] SSML support for fine-grained control

---

## 👨‍💻 Development

### Running the Flutter App in Development

```bash
cd flutter_app

# Get dependencies
flutter pub get

# Run with hot reload
flutter run

# Build debug APK
flutter build apk --debug
```

### Modifying the Backend

1. Edit `huggingface_space/app.py`
2. Push to Hugging Face Space repository
3. Space will auto-rebuild

### Key Files to Modify

| To Modify | Edit This File |
|-----------|----------------|
| Mobile UI | `flutter_app/lib/main.dart` |
| API Logic | `huggingface_space/app.py` |
| Number Conversion | `huggingface_space/app.py` (number_to_gujarati_words) |
| Chunking Logic | `flutter_app/lib/main.dart` (_splitTextIntoChunks) |

---

## 📄 License

This project is for educational purposes (SEM 6 SDP).

### Third-Party Licenses

| Component | License |
|-----------|---------|
| Meta MMS Model | CC-BY-NC 4.0 (Non-commercial) |
| Flutter | BSD-3-Clause |
| FastAPI | MIT |
| PyTorch | BSD-style |
| Transformers | Apache 2.0 |

---

## 🙏 Acknowledgments

- **Meta AI** for the MMS multilingual speech models
- **Hugging Face** for free model hosting and Spaces
- **Azure** for blob storage
- **Flutter** team for the excellent mobile framework

---

## 📞 Contact

For questions or issues, please open an issue in the repository.

**Repository:** https://github.com/moradiyaaman/Gujarati-Vaani

---

Created for **SEM 6 - Software Development Project (SDP)**

**Last Updated:** February 2026
