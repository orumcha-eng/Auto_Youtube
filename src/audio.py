# src/audio.py
# -*- coding: utf-8 -*-

import os
from google.cloud import texttospeech
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

class AudioGenerator:
    def __init__(self):
        # Look for the key file in the project root
        key_path = "gcp-tts-key.json"
        if os.path.exists(key_path):
            creds = service_account.Credentials.from_service_account_file(key_path)
            self.client = texttospeech.TextToSpeechClient(credentials=creds)
        else:
            # Fallback to env var or default auth
            self.client = texttospeech.TextToSpeechClient()

    def list_voices(self, lang="ko-KR") -> list:
        try:
            resp = self.client.list_voices(language_code=lang)
            # Filter for Neural2 or Studio voices if possible, or just return all names
            # Sort to put Neural/Studio text at top if possible
            voices = sorted([v.name for v in resp.voices])
            # prioritize Neural2
            neural = [v for v in voices if "Neural2" in v]
            studio = [v for v in voices if "Studio" in v]
            standard = [v for v in voices if "Standard" in v]
            wavenet = [v for v in voices if "WaveNet" in v]
            
            return studio + neural + wavenet + standard
        except Exception as e:
            print(f"List Voices Failed: {e}")
            return ["ko-KR-Neural2-c", "ko-KR-Neural2-A"] # Fallback

    def generate(self, text: str, voice_name: str, out_path: str, speed: float = 1.0) -> str:
        # Input
        s_input = texttospeech.SynthesisInput(text=text)
        
        # Voice
        # Parse language from voice name (e.g. ko-KR-Neural2-C -> ko-KR)
        lang_code = "-".join(voice_name.split("-")[:2])
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name
        )
        
        # Audio Config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speed
        )

        try:
            resp = self.client.synthesize_speech(
                input=s_input, 
                voice=voice_params, 
                audio_config=audio_config
            )
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(resp.audio_content)
            
            return out_path
            
        except Exception as e:
            print(f"Google TTS Failed: {e}")
            return None

