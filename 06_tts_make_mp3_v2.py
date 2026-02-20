# 06_tts_make_mp3_v2.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import List, Optional

from google.cloud import texttospeech
from google.api_core.exceptions import InvalidArgument
from xml.sax.saxutils import escape as xml_escape
import xml.etree.ElementTree as ET


ACRONYMS = ["FOMC", "CPI", "PPI", "BEA", "BLS", "GDP", "PMI", "ETF", "FX", "VIX"]


def normalize_ws(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00a0", " ")
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def time_et_to_spoken(text: str) -> str:
    # "08:30 ET" -> "8:30 A M Eastern"
    def repl(m):
        hh = int(m.group(1))
        mm = int(m.group(2))
        if hh == 0:
            hh12, ap = 12, "A M"
        elif 1 <= hh < 12:
            hh12, ap = hh, "A M"
        elif hh == 12:
            hh12, ap = 12, "P M"
        else:
            hh12, ap = hh - 12, "P M"
        return f"{hh12}:{mm:02d} {ap} Eastern"

    return re.sub(r"\b(\d{2}):(\d{2})\s*ET\b", repl, text)


def script_to_ssml(script: str) -> str:
    """
    SSML 안전 빌더:
    1) 원문 텍스트에서 약어를 placeholder로 바꾼 뒤
    2) XML escape 처리 (&, <, > 등)
    3) placeholder를 say-as 태그로 되돌림
    """
    s = normalize_ws(script)
    s = time_et_to_spoken(s)

    # 1) 약어 placeholder 처리
    for a in ACRONYMS:
        s = re.sub(rf"\b{a}\b", f"__ACR_{a}__", s)

    # 2) XML escape (중요: & 같은 문자 때문에 400 나는 걸 막음)
    s = xml_escape(s)

    # 3) placeholder -> say-as 태그 복원
    for a in ACRONYMS:
        s = s.replace(f"__ACR_{a}__", f'<say-as interpret-as="characters">{a}</say-as>')

    # 가독성 위해 문장 사이 짧은 쉬어가기(선택)
    # 너무 과하면 어색해져서 150ms 정도만
    s = re.sub(r"([.!?])\s+", r'\1 <break time="150ms"/> ', s)

    ssml = f"<speak>{s}</speak>"

    # SSML 유효성 로컬 검증 (깨져 있으면 여기서 바로 잡힘)
    try:
        ET.fromstring(ssml)
    except Exception as e:
        raise ValueError(f"SSML parse failed: {e}\nSSML:\n{ssml}")

    return ssml


def list_voices(client: texttospeech.TextToSpeechClient, language_code: str) -> List[str]:
    resp = client.list_voices(language_code=language_code)
    names = []
    for v in resp.voices:
        names.append(v.name)
    return sorted(set(names))


def pick_voice(client: texttospeech.TextToSpeechClient, language_code: str, preferred: Optional[str]) -> str:
    voices = list_voices(client, language_code)
    if not voices:
        raise RuntimeError(f"No voices returned for language_code={language_code}")

    if preferred:
        if preferred in voices:
            return preferred
        # 사용자가 준 voice가 없으면, 힌트 출력
        print(f"⚠️ Requested voice not found: {preferred}")
        print("   Available examples:", ", ".join(voices[:8]), "...")
        # fallback 계속

    # 기본 선택 우선순위: Neural2 > Wavenet > 첫번째
    for key in ["Neural2", "Wavenet"]:
        for v in voices:
            if key in v:
                return v
    return voices[0]


def synthesize(client: texttospeech.TextToSpeechClient,
               input_obj: texttospeech.SynthesisInput,
               voice_name: str,
               language_code: str,
               out_path: str,
               speaking_rate: float,
               pitch: float) -> None:
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch
    )

    response = client.synthesize_speech(
        input=input_obj,
        voice=voice,
        audio_config=audio_config
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(response.audio_content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", required=True, help="Path to out/brief_YYYY-MM-DD.json")
    ap.add_argument("--out", help="Output mp3 path (default: out/audio_YYYY-MM-DD.mp3)")
    ap.add_argument("--voice", default=None, help="Preferred voice name (optional)")
    ap.add_argument("--lang", default="en-US", help="Language code (default: en-US)")
    ap.add_argument("--speaking-rate", type=float, default=1.02)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--no-ssml", action="store_true", help="Use plain text input instead of SSML")
    ap.add_argument("--list-voices", action="store_true", help="Print available voices and exit")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    client = texttospeech.TextToSpeechClient()

    if args.list_voices:
        names = list_voices(client, args.lang)
        print(f"[Voices for {args.lang}] count={len(names)}")
        for n in names[:60]:
            print("-", n)
        if len(names) > 60:
            print("... (truncated)")
        return

    with open(args.brief, "r", encoding="utf-8") as f:
        data = json.load(f)

    script = data.get("script", "")
    if not script:
        raise ValueError("No 'script' field found in brief JSON.")

    date_et = data.get("date_et") or datetime.now().date().isoformat()
    out_path = args.out or os.path.join("out", f"audio_{date_et}.mp3")

    voice_name = pick_voice(client, args.lang, args.voice)

    if args.debug:
        print(f"[debug] voice={voice_name} lang={args.lang} speaking_rate={args.speaking_rate}")

    # 1) 기본은 SSML
    if not args.no_ssml:
        try:
            ssml = script_to_ssml(script)
            if args.debug:
                print("[debug] SSML OK (length chars):", len(ssml))
            input_obj = texttospeech.SynthesisInput(ssml=ssml)
            synthesize(client, input_obj, voice_name, args.lang, out_path, args.speaking_rate, args.pitch)
            print(f"✅ TTS saved: {out_path}")
            return
        except (ValueError, InvalidArgument) as e:
            print("⚠️ SSML mode failed. Falling back to plain text.")
            if args.debug:
                print("   error:", repr(e))

    # 2) fallback: plain text (SSML 깨져도 이건 거의 항상 됨)
    clean = normalize_ws(script)
    input_obj = texttospeech.SynthesisInput(text=clean)
    synthesize(client, input_obj, voice_name, args.lang, out_path, args.speaking_rate, args.pitch)
    print(f"✅ TTS saved (text fallback): {out_path}")


if __name__ == "__main__":
    main()
