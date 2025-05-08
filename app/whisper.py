from difflib import SequenceMatcher
import os
import re
import tempfile
from openai import OpenAI
from fastapi import UploadFile


async def transcribe_audio(audio_file: UploadFile) -> str:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    temp_path = temp_file.name
    
    try:
        content = await audio_file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        await audio_file.seek(0)
        
        with open(temp_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                file=audio,
                model="gpt-4o-mini-transcribe"
            )
        
        print(f"Transcription completed for {audio_file.filename}, result is {transcription.text}")
        
        return transcription.text
    except Exception as error:
        print(f"Error transcribing audio: {error}")
        raise error


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def calculate_speech_score(expected_phrase: str, transcription: str) -> float:
    if not expected_phrase or not transcription:
        return 0.0

    expected_normalized = normalize_text(expected_phrase)
    transcription_normalized = normalize_text(transcription)

    matcher = SequenceMatcher(None, expected_normalized, transcription_normalized)
    similarity = matcher.ratio()

    expected_words = set(expected_normalized.split())
    transcription_words = set(transcription_normalized.split())

    if not expected_words:
        return 0.0

    word_overlap = len(expected_words.intersection(transcription_words)) / len(expected_words)

    final_score = 0.7 * similarity + 0.3 * word_overlap

    return round(final_score, 2)
