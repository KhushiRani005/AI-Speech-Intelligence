# Benchmark Report

## Model

- Whisper Base
- KeyBERT
- Sumy LSA
- Librosa

---

## Test Environment

- OS: Windows 11
- Python: 3.13
- FastAPI
- CPU Execution

---

## Results

| Audio File | Language | Duration | Transcript | Summary | Status |
|------------|----------|----------|------------|---------|--------|
| sample1.wav | English | 10 sec | Success | Success | Pass |
| sample2.wav | Hindi | 15 sec | Success | Success | Pass |

---

## Performance

| Metric | Value |
|---------|-------|
| Speech Recognition | ✅ |
| Language Detection | ✅ |
| Keyword Extraction | ✅ |
| Summary Generation | ✅ |
| Audio Analysis | ✅ |

---

## Limitations

- Speaker Diarization not implemented.
- Emotion Detection not implemented.
- Confidence score is estimated.

---

## Future Work

- Speaker Identification
- Emotion Recognition
- Real-time Audio Streaming
- Noise Classification