# siraj_assistant.py
import os
import re
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wavfile
import pandas as pd
from fuzzywuzzy import process
from loguru import logger
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk

# ─── Load configuration from .env ─────────────────────────────────────────────
load_dotenv()
AZURE_KEY    = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION", "uaenorth").strip()
CSV_PATH     = os.getenv("CSV_PATH", "restaurants2_neighborhood_stations_paths6.csv")

if not AZURE_KEY:
    logger.error("Missing AZURE_SPEECH_KEY in .env")
    exit(1)

# ─── Prepare Azure Speech Configs ─────────────────────────────────────────────
speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_KEY,
    region=AZURE_REGION
)
speech_config.speech_recognition_language = "ar-SA"  # or "ar-EG" for Egyptian
speech_config.speech_synthesis_voice_name = "ar-SA-HamedNeural"

audio_out_cfg = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# Singletons
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config,
    audio_config=audio_out_cfg
)

# ─── Load & preprocess CSV ───────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
# Some files have Arabic name in `Name`, English fallback in `English_name`
df["Display_Name"] = df["Name"].fillna(df.get("English_name", ""))

# ─── Utility Functions ───────────────────────────────────────────────────────

def record_audio(duration: int = 5, fs: int = 16000) -> str:
    """Record from default mic for `duration` seconds, save to temp WAV, return filepath."""
    logger.info(f"🎤 تسجيل لمدة {duration} ثانية...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    fd, path = tempfile.mkstemp(suffix=".wav")
    wavfile.write(path, fs, audio)
    os.close(fd)
    logger.info("✅ انتهى التسجيل")
    return path

def azure_transcribe(wav_path: str) -> str:
    """Send a WAV file to Azure and return the recognized Arabic text."""
    audio_input = speechsdk.AudioConfig(filename=wav_path)
    rec   = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    result = rec.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        logger.info(f"📥 «{result.text}»")
        return result.text
    else:
        logger.warning(f"تعذر التعرف على الكلام: {result.reason}")
        return ""

def azure_speak(text: str):
    """Speak out `text` via Azure TTS."""
    if not text:
        return
    logger.info(f"🗣️ {text}")
    result = synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        logger.error(f"TTS failed: {result.reason}")

def extract_destination(text: str) -> str:
    """Extract the place name after إلى / لـ / ل."""
    match = re.search(r"(?:إلى|لـ|ل)\s*([\u0600-\u06FF\s]+)", text)
    if match:
        return match.group(1).strip()
    return text.strip()

def find_best_match(query: str) -> (str, int):
    """Fuzzy-match `query` against our Display_Name column."""
    names = df["Display_Name"].dropna().tolist()
    best, score = process.extractOne(query, names)
    return best, score or 0

def get_path(name: str) -> str:
    """Return the 'Path' field for a given restaurant display name."""
    row = df[df["Display_Name"] == name]
    if row.empty:
        return ""
    return row.iloc[0]["Path"]

# ─── Main interactive loop ────────────────────────────────────────────────────

WAKE_WORD = "سراج"

def main():
    azure_speak("سراجْ جاهز للاستماع، قل «سرااجْ» عندما تريد السؤال عن مطعم.")
    while True:
        # 1) listen for wake word
        wav = record_audio(duration=4)
        text = azure_transcribe(wav)
        os.remove(wav)

        if WAKE_WORD in text:
            azure_speak("تفضل، ما سؤالك؟")

            # 2) listen for the actual question
            wav2 = record_audio(duration=6)
            question = azure_transcribe(wav2)
            os.remove(wav2)

            # 3) extract destination
            dest_raw = extract_destination(question)
            best, score = find_best_match(dest_raw)

            if score < 60:
                azure_speak("عذراً، لم أتعرف على اسم المطعم. حاول مرة أخرى.")
                continue

            # 4) fetch path and speak
            path = get_path(best)
            if not path:
                azure_speak("للأسف لا توجد توجيهات متوفرة لهذه الوجهة.")
            else:
                azure_speak(f"أقرب مسار لمطعم {best} هو: {path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("👋 انتهاء البرنامج")  
