# 07_render_shorts.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
from mutagen.mp3 import MP3
import imageio_ffmpeg

W, H = 1080, 1920  # Shorts 9:16


def norm(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_font() -> str:
    candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def wrap_text_by_width(draw, text: str, font, max_w: int) -> List[str]:
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = (" ".join(cur + [w])).strip()
        if draw.textlength(test, font=font) <= max_w:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                lines.append(w)
                cur = []
    if cur:
        lines.append(" ".join(cur))
    return lines


def make_events_table_png(events: List[dict], out_path: str) -> None:
    table_w, table_h = 620, 460
    img = Image.new("RGBA", (table_w, table_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    font_path = find_font()
    font_title = ImageFont.truetype(font_path, 40) if font_path else ImageFont.load_default()
    font_head = ImageFont.truetype(font_path, 26) if font_path else ImageFont.load_default()
    font_body = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()

    d.rounded_rectangle([0, 0, table_w, table_h], radius=24, fill=(0, 0, 0, 155), outline=(255, 255, 255, 60), width=2)

    d.text((24, 16), "Today Events (ET)", font=font_title, fill=(255, 255, 255, 235))
    y0 = 82
    d.line([24, y0, table_w - 24, y0], fill=(255, 255, 255, 70), width=2)

    x_time, x_event = 24, 170
    d.text((x_time, y0 + 12), "TIME", font=font_head, fill=(255, 255, 255, 210))
    d.text((x_event, y0 + 12), "EVENT", font=font_head, fill=(255, 255, 255, 210))

    rows = events[:3]
    while len(rows) < 3:
        rows.append({"time_et": "", "title": ""})

    y = y0 + 56
    row_h = 118
    for i, ev in enumerate(rows):
        t = (ev.get("time_et") or "").strip() or "—"
        title = norm(ev.get("title") or "")
        if not title:
            title = "No major releases" if i == 0 else ""

        if i > 0:
            d.line([24, y - 14, table_w - 24, y - 14], fill=(255, 255, 255, 55), width=2)

        d.text((x_time, y), t, font=font_body, fill=(255, 255, 255, 235))

        max_w = table_w - x_event - 24
        lines = wrap_text_by_width(d, title, font_body, max_w=max_w)
        lines = lines[:2]
        if len(lines) == 2 and len(" ".join(lines)) < len(title):
            if not lines[1].endswith("..."):
                lines[1] = (lines[1][:max(0, len(lines[1]) - 3)] + "...") if len(lines[1]) > 8 else (lines[1] + "...")

        for li, line in enumerate(lines):
            d.text((x_event, y + li * 34), line, font=font_body, fill=(255, 255, 255, 235))

        y += row_h

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def estimate_audio_seconds(mp3_path: str) -> float:
    audio = MP3(mp3_path)
    return float(audio.info.length)


def wrap_subtitle_lines(text: str, max_chars: int = 34, max_lines: int = 2) -> str:
    """SRT 한 캡션을 1~2줄로 강제 줄바꿈(너무 길면 잘라서 ...)."""
    text = norm(text)
    if not text:
        return text

    words = text.split()
    lines, cur = [], []
    for w in words:
        test = (" ".join(cur + [w])).strip()
        if len(test) <= max_chars:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
            if len(lines) >= max_lines:
                break

    if len(lines) < max_lines and cur:
        lines.append(" ".join(cur))

    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return ""

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    # 너무 길면 마지막 줄에 ...
    joined = " ".join(lines)
    if len(joined) < len(text) and not lines[-1].endswith("..."):
        lines[-1] = (lines[-1][:max(0, max_chars - 3)] + "...") if len(lines[-1]) > 8 else (lines[-1] + "...")

    return "\n".join(lines)


def script_to_srt(script: str, duration: float, out_path: str) -> None:
    s = norm(script)
    chunks = re.split(r"(?<=[\.\!\?])\s+", s)
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        chunks = [s] if s else [""]

    weights = [max(1, len(c.split())) for c in chunks]
    total = sum(weights)

    def fmt(t: float) -> str:
        ms = int(round((t - int(t)) * 1000))
        sec = int(t)
        hh = sec // 3600
        mm = (sec % 3600) // 60
        ss = sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    cur = 0.0
    out_lines = []
    for i, (c, w) in enumerate(zip(chunks, weights), start=1):
        seg = duration * (w / total)
        start = cur
        end = min(duration, cur + seg)
        cur = end

        caption = wrap_subtitle_lines(c, max_chars=34, max_lines=2)

        out_lines.append(str(i))
        out_lines.append(f"{fmt(start)} --> {fmt(end)}")
        out_lines.append(caption)
        out_lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


def run_ffmpeg(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed\nSTDERR:\n{p.stderr}")


def print_video_info(ffmpeg: str, path: str) -> None:
    p = subprocess.run([ffmpeg, "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"(\d{2,5})x(\d{2,5})", p.stderr)
    if m:
        print(f"📐 Output resolution: {m.group(1)}x{m.group(2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--bg", required=True)
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-assets", action="store_true")
    args = ap.parse_args()

    with open(args.brief, "r", encoding="utf-8") as f:
        data = json.load(f)

    date_et = data.get("date_et") or datetime.now().date().isoformat()
    brand = norm(data.get("brand") or "Macro Minute Brief")
    script = data.get("script") or ""
    today_events = data.get("today_events") or []

    events_norm = [{"time_et": e.get("time_et") or "", "title": e.get("title") or ""} for e in today_events[:3]]

    out_mp4 = args.out or os.path.join("out", f"shorts_{date_et}.mp4")
    table_png = os.path.join("out", f"events_{date_et}.png")
    srt_path = os.path.join("out", f"subs_{date_et}.srt")

    make_events_table_png(events_norm, table_png)
    dur = estimate_audio_seconds(args.audio)
    script_to_srt(script, dur, srt_path)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # 좌표(템플릿 고정)
    table_x, table_y = 410, 220
    anchor_x, anchor_y = 70, 330

    try:
        dt = datetime.fromisoformat(date_et)
        date_label = dt.strftime("%a %b %d")
    except Exception:
        date_label = date_et

    # drawtext에서 따옴표 문제 방지
    brand_safe = brand.replace("'", r"\'")
    date_safe = date_label.replace("'", r"\'")

    # subtitles 스타일(자막 크기/위치/외곽선)
    srt_ff = srt_path.replace("\\", "\\\\")
    force_style = "FontName=Segoe UI,FontSize=44,Outline=3,Shadow=0,MarginV=140,Alignment=2"

    filters = []

    # ✅ 핵심: cover/crop로 9:16 꽉 채우기 (쇼츠 비율 보장)
    # + 살짝 어둡게 해서 가독성 올림
    filters.append(
        "[0:v]"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "eq=brightness=-0.06:saturation=1.05,"
        "format=rgba[base]"
    )

    vout = "[base]"
    input_index = 1

    if args.anchor:
        filters.append(f"[{input_index}:v]scale=420:-1,format=rgba[anchor]")
        filters.append(f"{vout}[anchor]overlay={anchor_x}:{anchor_y}:format=auto[v1]")
        vout = "[v1]"
        input_index += 1

    filters.append(f"[{input_index}:v]format=rgba[table]")
    filters.append(f"{vout}[table]overlay={table_x}:{table_y}:format=auto[v2]")
    vout = "[v2]"
    input_index += 1

    # 상단 바
    filters.append(
        f"{vout}"
        "drawbox=x=0:y=0:w=1080:h=120:color=black@0.55:t=fill,"
        f"drawtext=text='{brand_safe}  |  {date_safe} ET':x=42:y=34:fontsize=44:fontcolor=white@0.95"
        "[v3]"
    )
    vout = "[v3]"

    # 자막 영역 배경 + 자막 burn-in (작고 깔끔하게)
    filters.append(
        f"{vout}"
        "drawbox=x=0:y=1460:w=1080:h=460:color=black@0.42:t=fill,"
        f"subtitles='{srt_ff}':force_style='{force_style}'"
        "[vfinal]"
    )

    filter_complex = ";".join(filters)

    cmd = [ffmpeg, "-y"]
    cmd += ["-loop", "1", "-i", args.bg]
    if args.anchor:
        cmd += ["-i", args.anchor]
    cmd += ["-i", table_png]
    cmd += ["-i", args.audio]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vfinal]",
        "-map", f"{input_index}:a",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-t", f"{dur:.3f}",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        out_mp4
    ]

    run_ffmpeg(cmd)
    print(f"✅ Video saved: {out_mp4}")
    print_video_info(ffmpeg, out_mp4)


if __name__ == "__main__":
    main()
