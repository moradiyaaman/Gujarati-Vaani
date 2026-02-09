# Gujarati Vaani - Development Journey

This document summarizes the different approaches explored during development and the challenges encountered before arriving at the final solution.

## Final Solution: Hugging Face Spaces + Flutter App

The application now uses:
- **Backend**: Hugging Face Spaces (Docker) running FastAPI with fine-tuned MMS-TTS Gujarati model
- **Model Storage**: Azure Blob Storage (275 MB model files)
- **Frontend**: Flutter mobile app with chunked text processing

---

## Approaches Explored

### 1. Local Streamlit App (app.py)
**Description**: Streamlit web app running locally with the TTS model.

**Issues**:
- Required local GPU/powerful CPU for inference
- Model loading took significant time on each startup
- Not deployable as a mobile app
- Heavy resource requirements made it impractical for end users

---

### 2. Azure App Service Deployment (azure_server/)
**Description**: Deployed FastAPI server on Azure App Service with the model.

**Issues**:
- Persistent 500 Internal Server Error despite successful deployment
- Model files (275 MB) exceeded Azure App Service storage limits
- Cold start issues - model loading took too long
- GPU instances on Azure were expensive
- Debugging was difficult due to limited log access
- Multiple deployment attempts with different configurations all failed

---

### 3. ONNX Model Conversion (mms_conversion/)
**Description**: Converted PyTorch model to ONNX format for faster inference and smaller size.

**Issues**:
- Conversion process was complex due to MMS-TTS architecture
- Some model components didn't convert properly
- Quality degradation in some cases
- Still required significant compute resources for inference

---

### 4. Sherpa-ONNX Export (export_to_sherpa_onnx.py)
**Description**: Attempted to export model to Sherpa-ONNX format for mobile deployment.

**Issues**:
- MMS-TTS model architecture not fully compatible with Sherpa-ONNX
- Export scripts produced incomplete models
- Would have required significant modifications to work

---

### 5. Standalone Android APK (standalone_apk/)
**Description**: Native Android app with embedded model.

**Issues**:
- APK size would be too large (275+ MB for model alone)
- Memory constraints on mobile devices
- Inference speed too slow on mobile CPUs
- Complex build process

---

### 6. Stlite PWA (stlite_pwa/)
**Description**: Progressive Web App using Stlite (Streamlit in browser).

**Issues**:
- Limited browser capabilities for audio processing
- Model couldn't run efficiently in browser
- Still needed a backend for actual TTS inference

---

### 7. Mobile WebView Approach (mobile/)
**Description**: Mobile app wrapping a web interface.

**Issues**:
- Same backend problems as other approaches
- Additional complexity without solving core issues
- Poor user experience compared to native app

---

## Why Hugging Face Spaces Works

The final solution using Hugging Face Spaces solved all previous issues:

1. **Free Hosting**: Hugging Face provides free Docker Spaces with sufficient resources
2. **Persistent Model**: Model loads once and stays in memory (no cold starts after initial load)
3. **Azure Blob Storage**: Large model files stored separately, downloaded at startup
4. **SAS Token Access**: Secure, time-limited access to model files (valid until 2027)
5. **FastAPI**: Lightweight, fast API for TTS requests
6. **Flutter App**: Native mobile experience with clean API integration
7. **Chunked Processing**: Handles large texts by splitting into manageable chunks

---

## Key Learnings

1. **Model Size Matters**: 275 MB models are too large for mobile embedding
2. **Backend is Essential**: TTS models need server-side processing for good performance
3. **Free Tiers Have Limits**: Azure free tier wasn't suitable for ML workloads
4. **Simple Architecture Wins**: FastAPI + Flutter is simpler and more reliable than complex setups
5. **Preprocessing Helps**: Gujarati number-to-word conversion improved output quality

---

## Current Architecture

```
┌─────────────────┐     HTTPS      ┌─────────────────────────┐
│   Flutter App   │ ──────────────►│   Hugging Face Space    │
│   (Mobile)      │◄────────────── │   (FastAPI + MMS-TTS)   │
└─────────────────┘    Audio WAV   └───────────┬─────────────┘
                                               │
                                               │ Model Download
                                               │ (on startup)
                                               ▼
                                   ┌─────────────────────────┐
                                   │   Azure Blob Storage    │
                                   │   (275 MB Model Files)  │
                                   └─────────────────────────┘
```

---

## Repository Structure (Final)

```
Gujarati-Vaani/
├── flutter_app/          # Flutter mobile application
├── huggingface_space/    # Hugging Face Space (FastAPI server)
├── training/             # Model training scripts and data
├── utils/                # Utility functions
├── model_weights/        # Local model weights (for development)
└── README.md             # Project documentation
```
