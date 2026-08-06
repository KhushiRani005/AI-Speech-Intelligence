from fastapi import FastAPI, UploadFile, File
import whisper
import shutil
import os
import numpy as np
import json
from datetime import datetime
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import librosa
import nltk

# Download tokenizer (only first time)
nltk.download("punkt")
nltk.download("punkt_tab")

app = FastAPI(title="AI Speech Intelligence")

# Load AI Models
model = whisper.load_model("base")
summarizer = LsaSummarizer()

# Create folder if it doesn't exist
os.makedirs("sample_audio", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.get("/")
def home():
    return {"message": "AI Speech Intelligence API Running"}


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/version")
def version():
    return {"version": "1.0"}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    file_path = f"sample_audio/{audio.filename}"

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        # Speech Recognition
        result = model.transcribe(file_path)

        transcript = result["text"]
        language = result["language"]

        # Keyword Extraction
        words = transcript.lower().split()

        stop_words = {
                 "the","is","a","an","and","or","to","of","in",
                 "for","on","with","that","this","it","are","was",
                 "i","you","we","they"
                }

        keywords = []

        for word in words:
           word = word.strip(".,!?")
           if len(word) > 3 and word not in stop_words and word not in keywords:
                keywords.append(word)

        keywords = keywords[:5]

        # Text Summarization
        parser = PlaintextParser.from_string(
            transcript,
            Tokenizer("english")
        )

        summary = summarizer(parser.document, 2)
        summary_text = " ".join(str(sentence) for sentence in summary)
        
        # Content Generation
        word_count = len(transcript.split())

        content_generation = {
              "summary": summary_text,
              "keywords": keywords,
              "word_count": word_count
        }
        
        # Audio Analysis
        audio_data, sr = librosa.load(file_path)

        duration = librosa.get_duration(
            y=audio_data,
            sr=sr
        )

        rms = librosa.feature.rms(y=audio_data)[0]

        silence_threshold = 0.01

        silent_frames = np.sum(rms < silence_threshold)

        total_frames = len(rms)

        silence_percentage = round(
            (silent_frames / total_frames) * 100,
            2
        )

        average_energy = round(float(np.mean(rms)), 4)
        
        # Audio Quality
        if silence_percentage < 20:
             audio_quality = "Good"
        elif silence_percentage < 40:
             audio_quality = "Average"
        else:
             audio_quality = "Poor"
        
                # Estimated Confidence
        if average_energy > 0.05:
            confidence = 0.95
        elif average_energy > 0.02:
            confidence = 0.85
        else:
            confidence = 0.70

        # Final Response
        response = {
            "transcript": transcript,
            "language": language,

            "content_generation": {
                "summary": summary_text,
                "keywords": keywords,
                "word_count": word_count
            },

            "audio_analysis": {
                "duration_seconds": round(duration, 2),
                "silence_percentage": silence_percentage,
                "average_energy": average_energy,
                "quality": audio_quality
            },

            "speaker_analysis": {
                "speaker_count": 1,
                "speakers": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "start_time": 0.0,
                        "end_time": round(duration, 2)
                    }
                ],
                "note": "Placeholder implementation. Real speaker diarization can be integrated using pyannote.audio."
            },

            "confidence": confidence
        }

        # Save results
        history = []

        if os.path.exists("outputs/results.json"):
            with open("outputs/results.json", "r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []

        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **response
        })

        with open("outputs/results.json", "w") as f:
            json.dump(history, f, indent=4)

        return response

    except Exception as e:
        return {
            "error": str(e)
        }
               
            
            
        