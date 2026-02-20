# src/audio.py
# -*- coding: utf-8 -*-

import os
import base64
import subprocess
import tempfile
import re
import wave
from google.cloud import texttospeech
from google.oauth2 import service_account
from google import genai
from google.genai import types
from dotenv import load_dotenv
import imageio_ffmpeg

load_dotenv()

class AudioGenerator:
    AI_STUDIO_VOICES = [
        "Kore", "Puck", "Charon", "Fenrir", "Orus", "Zephyr",
        "Aoede", "Autonoe", "Callirrhoe", "Despina", "Erinome",
        "Iapetus", "Laomedeia", "Leda", "Melpomene", "Nereid",
        "Orpheus", "Rasalgethi", "Schedar", "Sulafat", "Umbriel", "Vindemiatrix",
    ]

    def __init__(self, api_key: str = None, provider: str = "auto"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.provider = (provider or "auto").lower()
        self.cloud_client = None
        self.ai_client = None
        self.ai_tts_model = "models/gemini-2.5-flash-preview-tts"
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        if self.provider == "auto":
            self.provider = "aistudio" if self.api_key else "cloud"

        if self.provider == "aistudio":
            try:
                self.ai_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"AI Studio client init failed, fallback to cloud: {e}")
                self.provider = "cloud"

        if self.provider == "cloud":
            self.cloud_client = self._init_cloud_client()

    def _init_cloud_client(self):
        key_path = "gcp-tts-key.json"
        if os.path.exists(key_path):
            creds = service_account.Credentials.from_service_account_file(key_path)
            return texttospeech.TextToSpeechClient(credentials=creds)
        return texttospeech.TextToSpeechClient()

    def list_voices(self, lang="ko-KR") -> list:
        if self.provider == "aistudio":
            return self.AI_STUDIO_VOICES

        try:
            resp = self.cloud_client.list_voices(language_code=lang)
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
            return ["ko-KR-Neural2-C", "ko-KR-Neural2-A"] # Fallback

    def generate(self, text: str, voice_name: str, out_path: str, speed: float = 1.0) -> str:
        if self.provider == "aistudio":
            res = self._generate_aistudio(text, voice_name, out_path, speed)
            if res:
                return res
            # Fallback to Cloud TTS when AI Studio returns empty audio or errors.
            if self.cloud_client is None:
                try:
                    self.cloud_client = self._init_cloud_client()
                except Exception as e:
                    print(f"Cloud fallback init failed: {e}")
                    return None
            cloud_voice = voice_name if "-" in (voice_name or "") else "ko-KR-Neural2-C"
            return self._generate_cloud(text, cloud_voice, out_path, speed)
        return self._generate_cloud(text, voice_name, out_path, speed)

    def _generate_cloud(self, text: str, voice_name: str, out_path: str, speed: float = 1.0) -> str:
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
            resp = self.cloud_client.synthesize_speech(
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

    def _extract_inline_audio(self, resp):
        # New SDK TTS often returns audio in resp.parts
        for part in (getattr(resp, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                mime = getattr(inline, "mime_type", "") or ""
                data = inline.data
                if isinstance(data, str):
                    data = base64.b64decode(data)
                return mime, data

        # Fallback path for nested candidate parts
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    mime = getattr(inline, "mime_type", "") or ""
                    data = inline.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    return mime, data
        return "", b""

    def _pcm_l16_to_wav(self, pcm_bytes: bytes, wav_path: str, sample_rate: int = 24000, channels: int = 1):
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

    def _wav_to_mp3(self, wav_path: str, mp3_path: str) -> bool:
        cmd = [
            self.ffmpeg, "-y",
            "-i", wav_path,
            "-ar", "44100", "-ac", "2",
            "-b:a", "192k",
            mp3_path,
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode == 0 and os.path.exists(mp3_path)

    def _generate_aistudio(self, text: str, voice_name: str, out_path: str, speed: float = 1.0) -> str:
        if not self.ai_client:
            print("AI Studio client not initialized.")
            return None
        try:
            resp = self.ai_client.models.generate_content(
                model=self.ai_tts_model,
                contents=(text or "").strip(),
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name or "Kore"
                            )
                        )
                    ),
                ),
            )

            mime, audio_bytes = self._extract_inline_audio(resp)
            if not audio_bytes:
                print("AI Studio TTS returned no audio bytes.")
                return None

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            mime_l = (mime or "").lower()
            if "mpeg" in mime_l or out_path.lower().endswith(".mp3"):
                # If already mp3, write directly.
                if "mpeg" in mime_l:
                    with open(out_path, "wb") as f:
                        f.write(audio_bytes)
                    return out_path
                # If mime is pcm/wav and target is mp3, convert.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp_wav = tmp.name
                if "l16" in mime_l:
                    m = re.search(r"rate=(\d+)", mime_l)
                    rate = int(m.group(1)) if m else 24000
                    self._pcm_l16_to_wav(audio_bytes, tmp_wav, sample_rate=rate, channels=1)
                else:
                    with open(tmp_wav, "wb") as f:
                        f.write(audio_bytes)
                ok = self._wav_to_mp3(tmp_wav, out_path)
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass
                return out_path if ok else None

            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            return out_path
        except Exception as e:
            print(f"AI Studio TTS Failed: {e}")
            return None

