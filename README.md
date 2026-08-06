# AI Speech Intelligence System

## Overview

AI Speech Intelligence is a FastAPI-based REST API that analyzes speech recordings using OpenAI Whisper. The system converts speech to text and provides useful insights such as language detection, text summarization, keyword extraction, and audio quality analysis.

---

## Features

- Speech-to-Text Transcription using OpenAI Whisper
- Language Detection
- Automatic Text Summarization
- Keyword Extraction
- Audio Duration Detection
- Silence Detection
- Audio Energy Analysis
- Basic Speaker Analysis
- REST API with Interactive Swagger Documentation

---

## Project Structure

```
AI-Speech-Intelligence/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── benchmark_report.md
├── sample_audio/
├── outputs/
├── tests/
└── docs/
```

---

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
uvicorn app:app --reload
```

Open your browser and visit:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST /predict

Upload an audio file (`.wav`, `.mp3`, `.m4a`, or `.aac`).

### Sample Response

```json
{
  "transcript": "...",
  "language": "en",
  "content_generation": {
    "summary": "...",
    "keywords": [
      "AI",
      "Speech"
    ],
    "word_count": 120
  },
  "audio_analysis": {
    "duration_seconds": 12.7,
    "silence_percentage": 18.5,
    "average_energy": 0.034,
    "quality": "Good"
  },
  "speaker_analysis": {
    "speaker_count": 1,
    "speaker": "Speaker_00"
  }
}
```

---

## Technologies Used

- Python
- FastAPI
- OpenAI Whisper
- Sumy
- KeyBERT
- Librosa
- NLTK
- NumPy
- Uvicorn

---

## Future Improvements

- Speaker Diarization
- Confidence Scores
- Emotion Detection
- Noise Classification
- Multi-language Translation
- Real-time Audio Streaming

---

## Author

Developed as part of the AI/ML Engineering Internship Project.