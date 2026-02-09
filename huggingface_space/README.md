---
title: Gujarati Vaani TTS
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Gujarati Vaani - Text-to-Speech API

FastAPI-based TTS service for Gujarati language using fine-tuned MMS-TTS model.

## Features
- High-quality Gujarati speech synthesis
- Adjustable speech speed (0.5x - 2.0x)
- Fast inference with CPU optimization
- REST API interface

## API Endpoints

### POST /synthesize
Generate speech from Gujarati text.

**Request:**
```json
{
  "text": "તમે કેમ છો?",
  "speed": 1.0
}
```

**Response:** WAV audio file (16kHz, 16-bit PCM)

### GET /health
Check API health status.

## Model
Uses fine-tuned facebook/mms-tts-guj model stored in Azure Blob Storage.
