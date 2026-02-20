# src/audio.py
# -*- coding: utf-8 -*-

import os
import base64
import subprocess
import tempfile
import re
import wave
import time
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
    AI_STUDIO_VOICE_GENDER = {
        # Heuristic labels for UI guidance only.
        "Kore": "Female",
        "Puck": "Male",
        "Charon": "Male",
        "Fenrir": "Male",
        "Orus": "Male",
        "Zephyr": "Female",
        "Aoede": "Female",
        "Autonoe": "Female",
        "Callirrhoe": "Female",
        "Despina": "Female",
        "Erinome": "Female",
        "Iapetus": "Male",
        "Laomedeia": "Female",
        "Leda": "Female",
        "Melpomene": "Female",
        "Nereid": "Female",
        "Orpheus": "Male",
        "Rasalgethi": "Male",
        "Schedar": "Female",
        "Sulafat": "Female",
        "Umbriel": "Male",
        "Vindemiatrix": "Female",
    }

    def __init__(self, api_key: str = None, provider: str = "auto", backup_api_key: str = None):
        primary_key = (api_key or os.getenv("GOOGLE_API_KEY", "") or "").strip()
        backup_key = (backup_api_key or os.getenv("GOOGLE_API_KEY_2", "") or "").strip()
        keys = [k for k in [primary_key, backup_key] if k]
        # De-duplicate while preserving order.
        self.api_keys = list(dict.fromkeys(keys))
        self.ai_key_idx = 0
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.provider = (provider or "auto").lower()
        self.cloud_client = None
        self.ai_client = None
        self.ai_tts_model = "models/gemini-2.5-flash-preview-tts"
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.last_provider_used = self.provider
        self.last_error = ""
        self.voice_gender_map = {}

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

    def _rotate_aistudio_key(self, reason_msg: str = "") -> bool:
        if self.provider != "aistudio":
            return False
        if len(self.api_keys) <= 1:
            return False
        nxt = self.ai_key_idx + 1
        while nxt < len(self.api_keys):
            cand = self.api_keys[nxt]
            try:
                self.ai_client = genai.Client(api_key=cand)
                self.ai_key_idx = nxt
                self.api_key = cand
                print(f"AI Studio key rotated to backup key index {self.ai_key_idx} due to: {reason_msg[:120]}")
                return True
            except Exception as e:
                print(f"Backup AI Studio key init failed at index {nxt}: {e}")
                nxt += 1
        return False

    def _init_cloud_client(self):
        key_path = "gcp-tts-key.json"
        if os.path.exists(key_path):
            creds = service_account.Credentials.from_service_account_file(key_path)
            return texttospeech.TextToSpeechClient(credentials=creds)
        return texttospeech.TextToSpeechClient()

    def list_voices(self, lang="ko-KR") -> list:
        if self.provider == "aistudio":
            self.voice_gender_map = dict(self.AI_STUDIO_VOICE_GENDER)
            return self.AI_STUDIO_VOICES

        try:
            resp = self.cloud_client.list_voices(language_code=lang)
            # Filter for Neural2 or Studio voices if possible, or just return all names
            # Sort to put Neural/Studio text at top if possible
            voices = sorted([v.name for v in resp.voices])
            gmap = {}
            for v in resp.voices:
                g = str(getattr(v, "ssml_gender", "NEUTRAL")).upper()
                if "FEMALE" in g:
                    gg = "Female"
                elif "MALE" in g:
                    gg = "Male"
                else:
                    gg = "Neutral"
                gmap[v.name] = gg
            self.voice_gender_map = gmap
            # prioritize Neural2
            neural = [v for v in voices if "Neural2" in v]
            studio = [v for v in voices if "Studio" in v]
            standard = [v for v in voices if "Standard" in v]
            wavenet = [v for v in voices if "WaveNet" in v]
            
            return studio + neural + wavenet + standard
        except Exception as e:
            print(f"List Voices Failed: {e}")
            self.last_error = str(e)
            return ["ko-KR-Neural2-C", "ko-KR-Neural2-A"] # Fallback

    def get_voice_gender(self, voice_name: str) -> str:
        if not voice_name:
            return "Unknown"
        if self.provider == "aistudio":
            return self.AI_STUDIO_VOICE_GENDER.get(voice_name, "Unknown")
        if voice_name in self.voice_gender_map:
            return self.voice_gender_map[voice_name]
        if self.provider == "cloud":
            # Heuristic fallback for cloud custom names.
            if "-A" in voice_name or "-C" in voice_name or "-E" in voice_name:
                return "Female"
            if "-B" in voice_name or "-D" in voice_name:
                return "Male"
        return "Unknown"

    def generate(
        self,
        text: str,
        voice_name: str,
        out_path: str,
        speed: float = 1.0,
        style_instruction: str = "",
        allow_fallback: bool = True,
    ) -> str:
        self.last_error = ""
        if self.provider == "aistudio":
            res = self._generate_aistudio(text, voice_name, out_path, speed, style_instruction=style_instruction)
            if res:
                self.last_provider_used = f"aistudio_k{self.ai_key_idx+1}"
                return res
            if not allow_fallback:
                self.last_provider_used = "aistudio_failed"
                return None
            # Fallback to Cloud TTS when AI Studio returns empty audio or errors.
            if self.cloud_client is None:
                try:
                    self.cloud_client = self._init_cloud_client()
                except Exception as e:
                    print(f"Cloud fallback init failed: {e}")
                    self.last_error = str(e)
                    return None
            cloud_voice = voice_name if "-" in (voice_name or "") else "ko-KR-Neural2-C"
            r = self._generate_cloud(text, cloud_voice, out_path, speed)
            self.last_provider_used = "cloud_fallback" if r else "aistudio_failed"
            return r
        r = self._generate_cloud(text, voice_name, out_path, speed)
        self.last_provider_used = "cloud" if r else "cloud_failed"
        return r

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
            self.last_error = str(e)
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

    def _generate_aistudio(self, text: str, voice_name: str, out_path: str, speed: float = 1.0, style_instruction: str = "") -> str:
        if not self.ai_client:
            print("AI Studio client not initialized.")
            return None

        def _retry_delay_from_error(msg: str) -> float:
            m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", msg, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
            m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?([0-9]+)s", msg, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
            return 6.0

        def _normalize_style(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"\s+", " ", s)
            s = s.replace("\"", "").replace("'", "")
            if len(s) > 160:
                s = s[:160].rstrip() + "."
            return s

        def _build_prompt_variants(transcript: str, style: str) -> list:
            if style:
                return [
                    f"Generate speech only in Korean. Read this transcript in this style: {style}\nTranscript: \"{transcript}\"",
                    f"In Korean, speak clearly and steadily like a financial news presenter.\nTranscript: \"{transcript}\"",
                    f"In Korean, say exactly this transcript naturally.\nTranscript: \"{transcript}\"",
                ]
            return [
                f"Generate speech only in Korean. Read this transcript naturally.\nTranscript: \"{transcript}\"",
                f"In Korean, say exactly this transcript.\nTranscript: \"{transcript}\"",
            ]

        t = (text or "").strip()
        if not t:
            self.last_error = "empty_text"
            return None

        sys_instr = _normalize_style(style_instruction)
        prompt_variants = _build_prompt_variants(t, sys_instr)
        last_msg = "AI Studio returned empty audio bytes"

        for idx, tts_prompt in enumerate(prompt_variants):
            for attempt in range(2):
                try:
                    resp = self.ai_client.models.generate_content(
                        model=self.ai_tts_model,
                        contents=tts_prompt,
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
                        last_msg = "AI Studio returned empty audio bytes"
                        if attempt == 0:
                            time.sleep(0.8)
                            continue
                        break

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
                    msg = str(e)
                    print(f"AI Studio TTS Failed: {msg}")
                    last_msg = msg
                    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                        ml = msg.lower()
                        is_daily_quota = (
                            "per_day" in ml
                            or "per model per day" in ml
                            or "generaterequestsperday" in ml
                            or "limit: 0" in ml
                        )
                        if self._rotate_aistudio_key(msg):
                            # Retry immediately with backup key.
                            continue
                        if is_daily_quota:
                            # Daily cap won't recover by waiting; fail fast.
                            self.last_error = msg
                            return None
                        wait_s = _retry_delay_from_error(msg) + 0.5
                        time.sleep(wait_s)
                        continue
                    if "INTERNAL" in msg or "500" in msg:
                        if attempt == 0:
                            time.sleep(0.9)
                            continue
                        break
                    if "INVALID_ARGUMENT" in msg and "generate text" in msg.lower():
                        break
                    return None
            if idx < len(prompt_variants) - 1:
                time.sleep(0.3)
        self.last_error = last_msg
        return None

