# app_v3.py
# -*- coding: utf-8 -*-

import streamlit as st
import os
import json
import time
import re
import subprocess
import copy
import difflib
import shutil
from datetime import datetime
from dotenv import load_dotenv
from src.visual_gen import VisualGenerator
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from src.audio import AudioGenerator
from src.media_processor import render_video, get_audio_duration, compute_scene_durations_from_audio_segments

# ---- CONFIG & SETUP ----
st.set_page_config(layout="wide", page_title="Auto Youtube V3 (Bean World)")
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "out", "v3")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PROJECTS_DIR = os.path.join(OUT_DIR, "projects")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(PROJECTS_DIR, exist_ok=True)

# ---- STYLE CONSTANTS ----
BEAN_STYLE_PROMPT = "A white blob character, bean shape, tall, simple stick limbs, cute face, vector art, thick outlines, isolated on white"
THUMB_STYLE_PRESETS = {
    "Popular Macro (Red Chaos)": (
        "Aggressive Korean macro thumbnail style. High-energy red/orange chaos background, "
        "explosion-like lighting, dramatic contrast, crowded symbolic elements (flags, charts, factories, currency signs), "
        "wide scene with a clear focal point, heavy cinematic depth, gritty atmosphere."
    ),
    "Dark Finance Studio": (
        "Dark studio macro style. Deep black/blue background, neon chart glow, reflective surfaces, "
        "focused central subject, premium and sharp."
    ),
    "Minimal Tension": (
        "Minimal but tense style. Two to three major objects only, strong negative space, "
        "high contrast key light, urgent mood."
    ),
}
MIN_WORDS_PER_MIN = 105
EST_WORDS_PER_MIN = 120

# ---- STATE MANAGEMENT ----
if "v3_data" not in st.session_state:
    st.session_state.v3_data = {
        "step": 1, 
        "topic": "Global Economy",
        "target_duration_min": 10,
        "thumbnail_data": None, # {image_path, title_text, full_prompt}
        "thumbnail_copy_options": {},
        "selected_thumbnail_copy": "",
        "script_data": None,    # {full_text, segments: [{title, body}]}
        "scenes": [],           # List of scene dicts
        "audio_segments": [],
        "audio_path": None,
        "video_path": None
    }

def next_step(): st.session_state.v3_data["step"] += 1
def prev_step(): st.session_state.v3_data["step"] -= 1


def jump_to_step(step_num: int):
    st.session_state.v3_data["step"] = step_num
    st.rerun()


def split_script_sentences(text: str) -> list:
    src = (text or "").strip()
    if not src:
        return []
    src = re.sub(r"\s+", " ", src)
    parts = re.split(r"(?<=[\.\!\?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", src)
    out = [p.strip() for p in parts if p and len(p.strip()) > 2]
    return out


def auto_tts_style_instruction(text: str) -> str:
    # User requested: disable dynamic style instruction to keep voice stable.
    return ""


def _norm_match_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[^0-9a-z가-힣]", "", t)
    return t


def map_segments_to_scenes_by_text(scenes: list, subtitle_segments: list) -> list:
    if not scenes or not subtitle_segments:
        return []
    scene_norms = [_norm_match_text(s.get("text") or "") for s in scenes]
    indices = []
    start_idx = 0
    n = len(scenes)
    m = len(subtitle_segments)

    for i, seg in enumerate(subtitle_segments):
        seg_norm = _norm_match_text(seg.get("text") or "")
        if not seg_norm:
            # Keep monotonic flow for empty/malformed segment text.
            idx = min(n - 1, int(i * n / max(1, m)))
            idx = max(start_idx, idx)
            indices.append(idx)
            start_idx = idx
            continue

        best_idx = start_idx
        best_score = -1.0
        for j in range(start_idx, n):
            sn = scene_norms[j]
            if not sn:
                continue
            if seg_norm in sn or sn in seg_norm:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, seg_norm, sn).ratio()
            if score > best_score:
                best_score = score
                best_idx = j
                if score >= 0.995:
                    break
        indices.append(best_idx)
        start_idx = best_idx
    return indices


def min_script_words(duration_min: int) -> int:
    return max(600, int(duration_min * MIN_WORDS_PER_MIN))


def is_terminal_tts_quota_error(msg: str) -> bool:
    m = (msg or "").lower()
    if not m:
        return False
    if "resource_exhausted" not in m and "429" not in m and "quota" not in m:
        return False
    return (
        "per_day" in m
        or "per model per day" in m
        or "generaterequestsperday" in m
        or "limit: 0" in m
        or "daily" in m
    )


def merge_mp3_files(input_files: list, out_path: str) -> str:
    files = [f for f in input_files if f and os.path.exists(f)]
    if not files:
        return ""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    list_path = os.path.join(OUT_DIR, f"concat_{int(time.time())}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"file '{os.path.abspath(p).replace('\\\\', '/')}'\n")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-ar", "44100", "-ac", "2",
        "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _serialize_v3_data(v3: dict) -> dict:
    return copy.deepcopy(v3 or {})


def _save_project_snapshot(v3: dict, label: str = "manual") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(PROJECTS_DIR, f"project_{label}_{ts}.json")
    payload = {
        "saved_at": datetime.now().isoformat(),
        "v3_data": _serialize_v3_data(v3),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _list_project_snapshots() -> list:
    files = [os.path.join(PROJECTS_DIR, x) for x in os.listdir(PROJECTS_DIR) if x.lower().endswith(".json")]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _load_project_snapshot(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("v3_data")
        if not isinstance(data, dict):
            return False
        st.session_state.v3_data = data
        return True
    except Exception:
        return False


def _move_file_to_dir(src_path: str, target_dir: str) -> str:
    """Move file to target directory. Returns final moved path."""
    src = (src_path or "").strip()
    dst_dir = (target_dir or "").strip()
    if not src or not dst_dir:
        return src
    if not os.path.exists(src):
        return src

    # Relative paths are resolved from project base directory.
    if not os.path.isabs(dst_dir):
        dst_dir = os.path.join(BASE_DIR, dst_dir)

    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dst = os.path.join(dst_dir, base)
    n = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{name}_{n}{ext}")
        n += 1

    shutil.move(src, dst)
    return dst


def _recover_latest_outputs_best_effort() -> dict:
    out = {
        "step": 5,
        "topic": "Recovered Session",
        "target_duration_min": 10,
        "thumbnail_data": None,
        "thumbnail_copy_options": {},
        "selected_thumbnail_copy": "",
        "script_data": None,
        "scenes": [],
        "audio_segments": [],
        "audio_path": None,
        "video_path": None,
    }
    # Latest final video/audio
    vids = []
    for root, _, files in os.walk(OUT_DIR):
        for x in files:
            if x.lower().endswith(".mp4"):
                vids.append(os.path.join(root, x))
    vids.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if vids:
        out["video_path"] = vids[0]

    auds = [os.path.join(OUT_DIR, x) for x in os.listdir(OUT_DIR) if x.lower().endswith(".mp3")]
    auds.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if auds:
        out["audio_path"] = auds[0]

    # Recover scenes from generated scene images/videos
    scene_imgs = [os.path.join(OUT_DIR, x) for x in os.listdir(OUT_DIR) if x.startswith("scene_") and x.lower().endswith(".png")]
    scene_vids = [os.path.join(OUT_DIR, x) for x in os.listdir(OUT_DIR) if x.startswith("scene_vid_") and x.lower().endswith(".mp4")]
    scene_imgs.sort(key=lambda p: os.path.getmtime(p))
    scene_vids.sort(key=lambda p: os.path.getmtime(p))
    max_n = max(len(scene_imgs), len(scene_vids))
    for i in range(max_n):
        img = scene_imgs[i] if i < len(scene_imgs) else None
        vid = scene_vids[i] if i < len(scene_vids) else None
        media = vid or img
        if not media:
            continue
        out["scenes"].append({
            "text": f"Recovered scene {i+1}",
            "image_prompt": "",
            "video_prompt": "",
            "image_path": img,
            "video_path": vid,
            "generated_media_path": media,
            "media_type": "video" if vid else "image",
        })

    # Do not recover script text from SRT because encoding-corrupted SRT can pollute script with gibberish.
    out["script_data"] = {"full_text": "Recovered session. Please regenerate/refine script in Step 2.", "segments": []}
    return out


def fmt_mmss(sec: float) -> str:
    s = max(0, int(round(sec)))
    mm = s // 60
    ss = s % 60
    return f"{mm}:{ss:02d}"


def build_timeline_items(v3: dict) -> list:
    audio_segments = v3.get("audio_segments") or []
    scenes = v3.get("last_render_scenes") or v3.get("scenes") or []
    script_data = v3.get("script_data") or {}
    audio_path = v3.get("audio_path")

    total_audio = get_audio_duration(audio_path) if (audio_path and os.path.exists(audio_path)) else 0.0
    if total_audio <= 0:
        total_audio = sum(
            get_audio_duration(seg.get("audio_path"))
            for seg in audio_segments
            if seg.get("audio_path") and os.path.exists(seg.get("audio_path"))
        )
    if total_audio <= 0:
        total_audio = 600.0

    # Prefer chapter-level timeline from Step 2 script segments.
    # Use total audio duration to place chapter timestamps across the full video.
    script_segments = script_data.get("segments") or []
    if script_segments:
        chapter_titles = []
        chapter_weights = []
        for i, seg in enumerate(script_segments):
            title = (seg.get("title") or "").strip() or f"챕터 {i+1}"
            body = (seg.get("body") or "").strip()
            wc = max(1, len(re.findall(r"\S+", body)))
            chapter_titles.append(title)
            chapter_weights.append(wc)

        total_w = max(1, sum(chapter_weights))
        intro_sec = min(20.0, max(5.0, total_audio * 0.03))
        usable = max(1.0, total_audio - intro_sec)

        items = [("0:00", "영상 시작")]
        acc_w = 0
        for title, w in zip(chapter_titles, chapter_weights):
            t = intro_sec + (acc_w / total_w) * usable
            items.append((fmt_mmss(t), title))
            acc_w += w

        # de-dup title while preserving order
        out = []
        seen = set()
        for t, h in items:
            key = (t, h)
            if key in seen:
                continue
            seen.add(key)
            out.append((t, h))
        return out[:12]

    scene_durs = compute_scene_durations_from_audio_segments(scenes, audio_segments, total_audio)
    if len(scene_durs) != len(scenes):
        # fallback: even split
        n = max(1, len(scenes))
        scene_durs = [total_audio / n] * n

    def _label_from_scene(sc: dict, idx: int) -> str:
        txt = (sc.get("text") or "").strip()
        if not txt:
            return f"핵심 이슈 {idx+1}"
        # Use first phrase to keep timeline readable.
        parts = re.split(r"[.!?]|다\\.|요\\.", txt)
        head = (parts[0] or txt).strip()
        head = re.sub(r"\\s+", " ", head)
        return head[:36] + ("..." if len(head) > 36 else "")

    items = []
    cur = 0.0
    for i, sc in enumerate(scenes):
        t = "0:00" if i == 0 else fmt_mmss(cur)
        items.append((t, _label_from_scene(sc, i)))
        cur += max(0.0, float(scene_durs[i]))

    if not items:
        items = [("0:00", "오프닝"), (fmt_mmss(total_audio * 0.33), "핵심 이슈"), (fmt_mmss(total_audio * 0.8), "마무리")]
    return items[:12]


def build_issue_summary(v3: dict) -> list:
    script_data = v3.get("script_data") or {}
    issues = []
    for seg in script_data.get("segments", []):
        title = (seg.get("title") or "").strip()
        body = (seg.get("body") or "").strip()
        if title and title not in ("오프닝 훅", "마무리"):
            issues.append(title)
        elif body:
            sentence = re.split(r"(?<=[.!?])\\s+|(?<=다\\.)\\s+|(?<=요\\.)\\s+", body)[0].strip()
            if sentence:
                issues.append(sentence[:42] + ("..." if len(sentence) > 42 else ""))
    if not issues:
        for i, sc in enumerate(v3.get("last_render_scenes") or v3.get("scenes") or []):
            txt = (sc.get("text") or "").strip()
            if txt:
                issues.append(txt[:42] + ("..." if len(txt) > 42 else ""))
            if len(issues) >= 5:
                break
    # de-dup preserve order
    out = []
    seen = set()
    for it in issues:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out[:5]


def build_publish_package(v3: dict) -> dict:
    topic = (v3.get("topic") or "오늘의 시장 브리핑").strip()
    date_str = datetime.now().strftime("%Y-%m-%d")
    title = f"[{date_str}] {topic} | 콩이의 경제/시장 핵심 브리핑"

    tags = [
        "경제", "시장", "주식", "미국증시", "나스닥", "S&P500", "금리", "인플레이션",
        "환율", "유가", "FOMC", "투자전략", "경제뉴스", "콩이", topic.replace(" ", "")
    ]
    tags = [t for t in tags if t]
    seen = set()
    uniq_tags = []
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        uniq_tags.append(t)
    tags_csv = ", ".join(uniq_tags[:15])

    timeline = build_timeline_items(v3)
    tl_lines = [f"{t} {h}" for t, h in timeline]
    issues = build_issue_summary(v3)
    issue_lines = "\n".join([f"- {x}" for x in issues]) if issues else "- 오늘 시장 핵심 이슈 요약"
    desc = (
        f"오늘 영상은 {topic}를 중심으로, 장 마감 후 꼭 확인해야 할 이슈만 압축 정리했습니다.\n"
        f"숫자와 맥락 위주로 핵심 시나리오를 정리했으니 투자 판단 전에 체크해보세요.\n\n"
        f"[오늘의 핵심 이슈]\n{issue_lines}\n\n"
        f"[타임라인]\n" + "\n".join(tl_lines) + "\n\n"
        f"[안내]\n본 영상은 투자 권유가 아닌 정보 제공 목적입니다."
    )

    return {"title": title, "tags_csv": tags_csv, "description": desc}

# ---- UI: SIDEBAR ----
st.sidebar.title("Bean World Studio")
api_key = st.sidebar.text_input("Gemini API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
api_key_backup = st.sidebar.text_input("Gemini API Key (Backup, Optional)", value=os.getenv("GOOGLE_API_KEY_2", ""), type="password")
st.session_state["luma_key_input"] = st.sidebar.text_input("Luma API Key (Optional for Real Video)", value=os.getenv("LUMA_API_KEY", ""), type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("Project State")
if st.sidebar.button("Save Snapshot", use_container_width=True):
    p = _save_project_snapshot(st.session_state.v3_data, label="manual")
    st.sidebar.success(f"Saved: {os.path.basename(p)}")

if st.sidebar.button("Load Latest Snapshot", use_container_width=True):
    snaps = _list_project_snapshots()
    if not snaps:
        st.sidebar.warning("No snapshots found.")
    else:
        ok = _load_project_snapshot(snaps[0])
        if ok:
            st.sidebar.success(f"Loaded: {os.path.basename(snaps[0])}")
            st.rerun()
        else:
            st.sidebar.error("Failed to load latest snapshot.")

if st.sidebar.button("Recover Latest Outputs (Best Effort)", use_container_width=True):
    st.session_state.v3_data = _recover_latest_outputs_best_effort()
    st.sidebar.success("Recovered from latest outputs.")
    st.rerun()

# Sidebar bookmark navigation
st.sidebar.markdown("---")
st.sidebar.subheader("Bookmarks")
current_step = st.session_state.v3_data["step"]
st.sidebar.caption(f"Current Step: {current_step}")
if st.sidebar.button("Step 1: Thumbnail", use_container_width=True, type=("primary" if current_step == 1 else "secondary")):
    jump_to_step(1)
if st.sidebar.button("Step 2: Script", use_container_width=True, type=("primary" if current_step == 2 else "secondary")):
    jump_to_step(2)
if st.sidebar.button("Step 3: Storyboard", use_container_width=True, type=("primary" if current_step == 3 else "secondary")):
    jump_to_step(3)
if st.sidebar.button("Step 4: Audio", use_container_width=True, type=("primary" if current_step == 4 else "secondary")):
    jump_to_step(4)
if st.sidebar.button("Step 5: Render", use_container_width=True, type=("primary" if current_step == 5 else "secondary")):
    jump_to_step(5)
if st.sidebar.button("Step 6: Publish Pack", use_container_width=True, type=("primary" if current_step == 6 else "secondary")):
    jump_to_step(6)

from src.content import ContentModule

# Top bookmark bar
top_nav = st.columns(6)
if top_nav[0].button("1. Thumbnail", use_container_width=True, type=("primary" if current_step == 1 else "secondary")):
    jump_to_step(1)
if top_nav[1].button("2. Script", use_container_width=True, type=("primary" if current_step == 2 else "secondary")):
    jump_to_step(2)
if top_nav[2].button("3. Storyboard", use_container_width=True, type=("primary" if current_step == 3 else "secondary")):
    jump_to_step(3)
if top_nav[3].button("4. Audio", use_container_width=True, type=("primary" if current_step == 4 else "secondary")):
    jump_to_step(4)
if top_nav[4].button("5. Render", use_container_width=True, type=("primary" if current_step == 5 else "secondary")):
    jump_to_step(5)
if top_nav[5].button("6. Publish", use_container_width=True, type=("primary" if current_step == 6 else "secondary")):
    jump_to_step(6)
st.caption(f"Bookmark Navigation | Current Step: {st.session_state.v3_data['step']}")

# ---- STEP 1: TOPIC & THUMBNAIL ----
def step_thumbnail():
    st.header("Step 1: Check Market & Create Thumbnail")
    
    if not api_key:
        st.error("Please enter Gemini API Key in sidebar first.")
        return

    # Auto-Analyze Section
    if not st.session_state.v3_data["topic"] or st.session_state.v3_data["topic"] == "Global Economy":
         if st.button("Analyze Today's Market News (Auto)"):
             with st.spinner("Reading global news feeds & identifying keywords..."):
                 cm = ContentModule()
                 res = cm.identify_daily_theme(api_key)
                 st.session_state.v3_data["topic"] = res["theme_short"]
                 st.session_state.v3_data["topic_summary"] = res["summary"]
                 st.rerun()
    
    # Helper: generation logic (Clean Image Only)
    def run_clean_gen(sel_obj, prompt_text: str):
        vg = VisualGenerator(api_key, backup_api_key=api_key_backup)
        path = os.path.join(OUT_DIR, f"thumb_{int(time.time())}.png")
        with st.spinner("Rendering Base Thumbnail..."):
            res = vg.generate_image(prompt_text, path)
            if res:
                st.session_state.v3_data["thumbnail_data"] = {
                    "image_path": res, 
                    "title": sel_obj['title'],
                    "visual": sel_obj['visual'],
                    "title_text": st.session_state.v3_data.get("selected_thumbnail_copy", ""),
                    "is_final": False
                }
                st.success("Base Image Generated!")
                st.rerun()

    def compose_thumbnail_text(base_image_path: str, copy_text: str) -> str:
        """Overlay text in a high-CTR Korean finance thumbnail style."""
        if not base_image_path or not os.path.exists(base_image_path):
            return ""
        text = (copy_text or "").strip()
        if not text:
            return ""

        img_src = Image.open(base_image_path).convert("RGB")
        # Enforce YouTube thumbnail aspect for legacy square sources.
        try:
            from PIL import ImageOps
            img_src = ImageOps.fit(img_src, (1280, 720), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        except Exception:
            pass
        img = img_src.convert("RGBA")
        d = ImageDraw.Draw(img)
        w, h = img.size

        lines = [x.strip() for x in text.split(" / ") if x.strip()]
        if not lines:
            lines = [text]
        if len(lines) == 1 and len(lines[0]) > 12:
            parts = lines[0].split()
            mid = max(1, len(parts) // 2)
            lines = [" ".join(parts[:mid]), " ".join(parts[mid:])]
        lines = lines[:2]

        font_candidates = [
            r"C:\Windows\Fonts\malgunbd.ttf",
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\impact.ttf",
        ]
        font_path = next((p for p in font_candidates if os.path.exists(p)), None)
        base_size = max(76, int(h * 0.125))
        font = ImageFont.truetype(font_path, base_size) if font_path else ImageFont.load_default()

        # Fit text width
        max_w = int(w * 0.92)
        while True:
            widths = []
            for ln in lines:
                bb = d.textbbox((0, 0), ln, font=font, stroke_width=8)
                widths.append(bb[2] - bb[0])
            if max(widths) <= max_w or base_size <= 52:
                break
            base_size -= 2
            font = ImageFont.truetype(font_path, base_size) if font_path else ImageFont.load_default()

        line_gap = int(base_size * 0.12)
        line_heights = []
        for ln in lines:
            bb = d.textbbox((0, 0), ln, font=font, stroke_width=8)
            line_heights.append(bb[3] - bb[1])
        total_h = sum(line_heights) + line_gap * (len(lines) - 1)

        # Place near lower third like reference style (no heavy black plate).
        y = h - total_h - int(h * 0.14)

        # Add top-left emergency badge.
        badge_text = "긴급"
        badge_font = ImageFont.truetype(font_path, max(30, int(base_size * 0.4))) if font_path else ImageFont.load_default()
        bx1, by1 = int(w * 0.04), int(h * 0.05)
        bx2, by2 = bx1 + int(w * 0.16), by1 + int(h * 0.075)
        d.rounded_rectangle([bx1, by1, bx2, by2], radius=12, fill=(210, 0, 0, 230), outline=(255, 255, 255, 220), width=3)
        bt = d.textbbox((0, 0), badge_text, font=badge_font, stroke_width=2)
        bw, bh = bt[2] - bt[0], bt[3] - bt[1]
        d.text(((bx1 + bx2 - bw) // 2, (by1 + by2 - bh) // 2), badge_text, font=badge_font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))

        # Red arrow element for urgency.
        ax1, ay1 = int(w * 0.84), int(h * 0.18)
        ax2, ay2 = int(w * 0.70), int(h * 0.28)
        d.line([ax1, ay1, ax2, ay2], fill=(255, 20, 20, 255), width=14)
        d.polygon([(ax2, ay2), (ax2 + 26, ay2 - 10), (ax2 + 4, ay2 - 32)], fill=(255, 20, 20, 255))

        colors = [(255, 230, 0, 255), (255, 60, 60, 255)]
        for i, ln in enumerate(lines):
            bb = d.textbbox((0, 0), ln, font=font, stroke_width=8)
            lw = bb[2] - bb[0]
            lh = bb[3] - bb[1]
            x = (w - lw) // 2

            # Shadow layer
            d.text(
                (x + 5, y + 6),
                ln,
                font=font,
                fill=(0, 0, 0, 200),
                stroke_width=8,
                stroke_fill=(0, 0, 0, 220),
            )
            d.text(
                (x, y),
                ln,
                font=font,
                fill=colors[i % len(colors)],
                stroke_width=8,
                stroke_fill=(0, 0, 0, 255),
            )
            y += lh + line_gap

        out_path = os.path.join(OUT_DIR, f"thumb_text_{int(time.time())}.png")
        img.convert("RGB").save(out_path)
        return out_path

    # Show Topic Editor (User can override)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Today's Theme")
        topic = st.text_input("Main Item", st.session_state.v3_data["topic"])
        if "topic_summary" in st.session_state.v3_data:
            st.info(f"Why? {st.session_state.v3_data['topic_summary']}")
        
        st.session_state.v3_data["topic"] = topic

        style_name = st.selectbox(
            "Thumbnail Style Direction",
            options=list(THUMB_STYLE_PRESETS.keys()),
            index=0,
        )
        style_brief = st.text_area(
            "Style Brief (Editable)",
            THUMB_STYLE_PRESETS[style_name],
            height=90,
            key="thumb_style_brief",
        )
        
        st.markdown("---")
        # Button to (Re)Generate Ideas
        if st.button("Generate New Ideas" if "thumbnail_candidates" in st.session_state.v3_data else "Generate Thumbnail Ideas"):
            if not api_key:
                st.error("Please provide Gemini API Key in sidebar.")
            else:
                with st.spinner("Brainstorming Catchy Titles & Visuals..."):
                    cm = ContentModule()
                    ideas = cm.generate_thumbnail_ideas(api_key, topic, style_direction=style_brief)
                    if ideas:
                        st.session_state.v3_data["thumbnail_candidates"] = ideas
                        st.rerun() # Refresh to show new options
                    else:
                        st.error("Failed to generate ideas.")
                
        if "thumbnail_candidates" in st.session_state.v3_data:
            thumb_sel = st.radio("Select Concept", 
                                [c["title"] for c in st.session_state.v3_data["thumbnail_candidates"]])
            
            # Find selected
            sel_obj = next(x for x in st.session_state.v3_data["thumbnail_candidates"] if x["title"] == thumb_sel)
            
            st.write(f"**Visual Concept:** {sel_obj['visual']}")

            default_thumb_prompt = (
                f"YouTube thumbnail, 16:9, ultra clear composition. "
                f"{BEAN_STYLE_PROMPT}. "
                f"Style direction: {style_brief}. "
                f"Scene concept: {sel_obj['visual']}. "
                "Genre: macro economy / market news. "
                "Subject: Bean character must be clearly visible and emotionally expressive. "
                "Use one strong focal point, clean foreground/background separation, dramatic but readable lighting, "
                "high contrast, cinematic color grading, realistic depth, crisp outlines, no clutter. "
                "Background elements can include chart screens, candlestick glow, index tickers, newsroom mood. "
                "No text, no letters, no watermark, no logo, no gibberish typography, no extra faces, no deformed hands."
            )
            thumb_prompt = st.text_area(
                "Thumbnail Image Prompt (Editable)",
                default_thumb_prompt,
                height=180,
                key=f"thumb_prompt_{thumb_sel}"
            )

            st.markdown("---")
            st.write("**Thumbnail Text Copy (5~10 options)**")
            if st.button("Generate Thumbnail Copy Options", key=f"copy_btn_{thumb_sel}"):
                with st.spinner("Analyzing proven copy patterns and generating options..."):
                    cm = ContentModule()
                    copy_opts = cm.generate_thumbnail_copy_options(
                        api_key=api_key,
                        topic=topic,
                        concept_title=sel_obj.get("title", ""),
                        concept_visual=sel_obj.get("visual", ""),
                        n_options=8,
                    )
                    if copy_opts:
                        st.session_state.v3_data["thumbnail_copy_options"][thumb_sel] = copy_opts
                        st.session_state.v3_data["selected_thumbnail_copy"] = copy_opts[0]
                        st.success(f"Generated {len(copy_opts)} copy options.")
                    else:
                        st.warning("Failed to generate copy options.")

            copy_opts = st.session_state.v3_data["thumbnail_copy_options"].get(thumb_sel, [])
            if copy_opts:
                selected_copy = st.radio(
                    "Select thumbnail copy",
                    options=copy_opts,
                    key=f"copy_radio_{thumb_sel}",
                )
                st.session_state.v3_data["selected_thumbnail_copy"] = selected_copy
                st.code(selected_copy.replace(" / ", "\n"))

            # Button to (Re)Generate Base Image (Col 1)
            if st.button("Generate Base Image" if not st.session_state.v3_data.get("thumbnail_data") else "Regenerate Base Image"):
                run_clean_gen(sel_obj, thumb_prompt)

    with col2:
        # Preview Area
        if st.session_state.v3_data["thumbnail_data"]:
            td = st.session_state.v3_data["thumbnail_data"]
            
            # Show Final if exists, else Base
            show_path = td.get("image_path_final", td["image_path"])
            
            st.image(show_path, caption=td["title"], use_container_width=True)
            if td.get("title_text"):
                st.caption("Selected Thumbnail Copy")
                st.code(td["title_text"].replace(" / ", "\n"))

            if td.get("title_text"):
                if st.button("Compose Thumbnail Text"):
                    out_img = compose_thumbnail_text(td["image_path"], td["title_text"])
                    if out_img:
                        st.session_state.v3_data["thumbnail_data"]["image_path_final"] = out_img
                        st.success("Composited thumbnail generated.")
                        st.rerun()
                    else:
                        st.warning("Failed to compose thumbnail text.")
            
            st.info("Image ready. Download and add text manually if needed.")
            
            with open(show_path, "rb") as f:
                st.download_button(
                    label="Download High-Res Image",
                    data=f,
                    file_name=os.path.basename(show_path),
                    mime="image/png"
                )
            
            st.divider()

            if st.button("Confirm Thumbnail & Go to Script"):
                next_step()
                st.rerun()

# ---- STEP 2: SCRIPT GENERATION (DYNAMIC LENGTH) ----
def step_script():
    st.header("Step 2: Script & Structure")
    
    st.info(f"Topic: {st.session_state.v3_data['topic']}")
    
    # Options for length (Min 10m as requested)
    length_map = {"Standard (10m)": 10, "Deep Dive (20m)": 20, "Marathon (30m)": 30}
    length_opt = st.select_slider("Target Duration", options=list(length_map.keys()))
    duration_val = length_map[length_opt]
    st.session_state.v3_data["target_duration_min"] = duration_val
    
    if st.button("Generate Full Script" if not st.session_state.v3_data.get("script_data") else "Regenerate Script (Overwrite)"):
        if not api_key:
             st.error("API Key needed.")
        else:
             with st.spinner(f"Researching & Writing {duration_val} min script... (This takes a moment)"):
                cm = ContentModule()
                # 1. Gather Data
                data = cm.generate(datetime.now().isoformat(), query=st.session_state.v3_data["topic"], api_key=api_key)
                
                # 2. Write Long Script
                curated = data.get("curated_data", {})
                if not curated:
                    # Fallback if curation failed or plain generate used
                    curated = {"top_news": data["headlines"], "upcoming_events": data["today_events"]}
                
                script_res = cm.generate_longform_script_llm(api_key, datetime.now(), curated, duration_min=duration_val)
                
                st.session_state.v3_data["script_data"] = script_res
                wc = len((script_res.get("full_text") or "").split())
                st.success(f"Long-form Script Generated! ({wc} words)")
                min_wc = min_script_words(duration_val)
                if wc < min_wc:
                    st.warning(f"Script is shorter than target for {duration_val} min. (current: {wc}, target min: {min_wc})")

    if st.session_state.v3_data["script_data"]:
        cur_text = (st.session_state.v3_data["script_data"].get("full_text") or "")
        cur_wc = len(cur_text.split())
        est_min = cur_wc / EST_WORDS_PER_MIN if cur_wc > 0 else 0.0
        st.caption(
            f"Script length: {cur_wc} words | Estimated narration: {est_min:.1f} min "
            f"(target: {duration_val} min)"
        )

        # Show Segments
        tabs = st.tabs(["Full Text"] + [s.get("title", f"Seg {i}") for i, s in enumerate(st.session_state.v3_data["script_data"].get("segments", []))])
        
        # Tab 0: Full Text
        with tabs[0]:
            edited_full = st.text_area("Full Script (Editable)", st.session_state.v3_data["script_data"].get("full_text", ""), height=500)
            st.session_state.v3_data["script_data"]["full_text"] = edited_full
            
        # Segment Tabs
        segments = st.session_state.v3_data["script_data"].get("segments", [])
        for i, seg in enumerate(segments):
            with tabs[i+1]:
                st.subheader(seg.get("title", ""))
                st.write(seg.get("body", ""))
                st.caption("(Edit global text in first tab to apply changes)")
        
        c1, c2 = st.columns(2)
        with c1: 
            pass # Placeholder if we needed specific alignment
        with c2: 
            if st.button("Recalculate Scenes & Go Next"): 
                min_wc = min_script_words(duration_val)
                wc_now = len((st.session_state.v3_data["script_data"].get("full_text") or "").split())
                if wc_now < min_wc:
                    st.warning(
                        f"Script too short for {duration_val}m target. "
                        f"Current {wc_now} words, recommended minimum {min_wc}. Continuing anyway."
                    )
                next_step()
                st.rerun()

    st.markdown("---")
    if st.button("Back to Thumbnail"): 
        prev_step()
        st.rerun()

# ---- STEP 3: STORYBOARD (HYBRID VIDEO/IMAGE) ----
def step_storyboard():
    st.header("Step 3: Storyboard Editor")
    st.caption("Image-first mode: scenes are generated as images first. Video prompts are pre-generated for optional manual conversion.")

    if not st.session_state.v3_data.get("script_data"):
        st.warning("No script found. Please go back to Step 2.")
        if st.button("Back to Script"):
            prev_step()
            st.rerun()
        return

    if not st.session_state.v3_data["scenes"]:
        st.info("Let's break down your script into visual scenes.")
        if st.button("Analyze Script & Create Scenes"):
            if not api_key:
                st.error("API Key required.")
            else:
                with st.spinner("Director AI is planning shots..."):
                    vg = VisualGenerator(api_key, backup_api_key=api_key_backup)
                    full_script = st.session_state.v3_data["script_data"]["full_text"]
                    scenes = vg.analyze_script(full_script, BEAN_STYLE_PROMPT)
                    if scenes:
                        for scene in scenes:
                            scene["media_type"] = "image"
                            scene["image_path"] = scene.get("image_path")
                            scene["video_path"] = scene.get("video_path")
                            scene["generated_media_path"] = scene.get("generated_media_path") or scene.get("image_path")
                            if not scene.get("video_prompt"):
                                scene["video_prompt"] = f"Subtle cinematic motion. {scene.get('image_prompt', '')}"
                        st.session_state.v3_data["scenes"] = scenes
                        st.success(f"Generated {len(scenes)} scenes!")
                        st.rerun()
                    else:
                        st.error("Failed to analyze script.")
    else:
        scenes = st.session_state.v3_data["scenes"]

        c_act1, c_act2 = st.columns([2, 1])
        with c_act1:
            st.caption(f"Total Scenes: {len(scenes)}")
        with c_act2:
            if st.button("Re-Analyze (Reset All)"):
                st.session_state.v3_data["scenes"] = []
                st.rerun()

        if st.button("Generate ALL Missing Images", type="primary"):
            vg = VisualGenerator(api_key, backup_api_key=api_key_backup)
            progress_bar = st.progress(0)
            targets = [i for i, s in enumerate(scenes) if not s.get("image_path") or not os.path.exists(s["image_path"])]
            if not targets:
                st.info("All scenes already have images.")
            else:
                for idx, scene_idx in enumerate(targets):
                    scene = scenes[scene_idx]
                    fname = f"scene_{int(time.time())}_{scene_idx}.png"
                    out_path = os.path.join(OUT_DIR, fname)
                    final_prompt = vg.build_cinematic_image_prompt(
                        scene=scene,
                        character_desc=BEAN_STYLE_PROMPT,
                        style_hint="high-energy Korean macro channel look, rich environment details",
                    )
                    res = vg.generate_image(final_prompt, out_path)
                    if res:
                        scene["image_path"] = res
                        scene["generated_media_path"] = res
                        scene["media_type"] = "image"
                    progress_bar.progress((idx + 1) / len(targets))
                st.success("Batch generation complete.")
                st.rerun()

        for i, scene in enumerate(scenes):
            scene_title = (scene.get("text") or "")[:40]
            with st.expander(f"Scene {i+1}: {scene_title}...", expanded=(i == 0)):
                c_text, c_prompt, c_video_prompt = st.columns(3)
                with c_text:
                    scene["text"] = st.text_area(
                        f"Narrator (Scene {i+1})",
                        scene.get("text", ""),
                        height=100,
                        key=f"txt_{i}"
                    )
                with c_prompt:
                    scene["image_prompt"] = st.text_area(
                        f"Image Prompt (Scene {i+1})",
                        scene.get("image_prompt", ""),
                        height=100,
                        key=f"prm_{i}"
                    )
                with c_video_prompt:
                    scene["video_prompt"] = st.text_area(
                        f"Video Prompt (Scene {i+1})",
                        scene.get("video_prompt", ""),
                        height=100,
                        key=f"vprm_{i}"
                    )

                c_img, c_btn = st.columns([2, 1])
                with c_img:
                    current_img = scene.get("image_path")
                    if current_img and os.path.exists(current_img):
                        st.image(current_img, use_container_width=True)
                    else:
                        st.warning("No image yet.")
                    if scene.get("video_path") and os.path.exists(scene["video_path"]):
                        st.video(scene["video_path"])
                    st.caption(f"Current media_type: {scene.get('media_type', 'image')}")

                with c_btn:
                    if st.button(f"Generate Image {i+1}", key=f"btn_gen_{i}"):
                        vg = VisualGenerator(api_key, backup_api_key=api_key_backup)
                        fname = f"scene_{int(time.time())}_{i}.png"
                        out_path = os.path.join(OUT_DIR, fname)
                        with st.spinner("Drawing..."):
                            final_prompt = vg.build_cinematic_image_prompt(
                                scene=scene,
                                character_desc=BEAN_STYLE_PROMPT,
                                style_hint="high-energy Korean macro channel look, rich environment details",
                            )
                            res = vg.generate_image(final_prompt, out_path)
                            if res:
                                scene["image_path"] = res
                                scene["generated_media_path"] = res
                                scene["media_type"] = "image"
                                st.rerun()

                    if st.button(f"Make Video #{i+1}", key=f"btn_vid_{i}"):
                        current_img = scene.get("image_path")
                        if not current_img or not os.path.exists(current_img):
                            st.error("Generate image first.")
                        else:
                            vg = VisualGenerator(api_key, backup_api_key=api_key_backup)
                            fname = f"scene_vid_{int(time.time())}_{i}.mp4"
                            out_path = os.path.join(OUT_DIR, fname)
                            veo_success = False

                            if api_key:
                                with st.spinner("Generating AI Video (Google Veo)..."):
                                    video_prompt = vg.build_cinematic_video_prompt(
                                        scene=scene,
                                        character_desc=BEAN_STYLE_PROMPT,
                                        style_hint="high-energy Korean macro channel look, rich background details",
                                    )
                                    res = vg.generate_video_veo(current_img, video_prompt, out_path)
                                    if res:
                                        scene["video_path"] = res
                                        scene["generated_media_path"] = res
                                        scene["media_type"] = "video"
                                        veo_success = True
                                        st.rerun()

                            if not veo_success:
                                luma_key = os.getenv("LUMA_API_KEY") or st.session_state.get("luma_key_input")
                                if luma_key:
                                    with st.spinner("Veo failed, trying Luma..."):
                                        video_prompt = vg.build_cinematic_video_prompt(
                                            scene=scene,
                                            character_desc=BEAN_STYLE_PROMPT,
                                            style_hint="high-energy Korean macro channel look, rich background details",
                                        )
                                        res = vg.generate_video_luma(current_img, video_prompt, out_path, luma_key)
                                        if res:
                                            scene["video_path"] = res
                                            scene["generated_media_path"] = res
                                            scene["media_type"] = "video"
                                            st.rerun()
                                        else:
                                            st.warning("Luma generation failed. Trying zoom fallback.")
                                            with st.spinner("Applying zoom effect..."):
                                                zres = vg.generate_video_motion(current_img, out_path)
                                                if zres:
                                                    scene["video_path"] = zres
                                                    scene["generated_media_path"] = zres
                                                    scene["media_type"] = "video"
                                                    st.rerun()
                                                else:
                                                    st.error("Video generation failed on Veo, Luma, and zoom fallback.")
                                else:
                                    st.warning("Veo/Luma unavailable. Using zoom fallback.")
                                    with st.spinner("Applying zoom effect..."):
                                        res = vg.generate_video_motion(current_img, out_path)
                                        if res:
                                            scene["video_path"] = res
                                            scene["generated_media_path"] = res
                                            scene["media_type"] = "video"
                                            st.rerun()
                                        else:
                                            st.error("Zoom fallback failed. Check logs for details.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Script"):
            prev_step()
            st.rerun()
    with c2:
        all_ready = st.session_state.v3_data["scenes"] and all(s.get("image_path") for s in st.session_state.v3_data["scenes"])
        if st.button("Go to Audio (Step 4)" if all_ready else "Go to Audio (Skip missing images)"):
            next_step()
            st.rerun()


# ---- STEP 4: AUDIO (SENTENCE TTS) ----
def step_audio():
    st.header("Step 4: Audio (Sentence TTS)")

    script_data = st.session_state.v3_data.get("script_data") or {}
    full_text = (script_data.get("full_text") or "").strip()
    if not full_text:
        st.warning("No script found. Go back to Step 2 first.")
        if st.button("Back to Script"):
            st.session_state.v3_data["step"] = 2
            st.rerun()
        return

    provider_label = st.selectbox(
        "TTS Provider",
        ["AI Studio (Gemini TTS)", "Google Cloud TTS"],
        index=0,
        help="AI Studio uses Gemini TTS voice set. Cloud uses ko-KR-Neural2/Studio voices."
    )
    tts_provider = "aistudio" if provider_label.startswith("AI Studio") else "cloud"
    ag = AudioGenerator(api_key=api_key, provider=tts_provider, backup_api_key=api_key_backup)
    recommended_voice = "Kore" if tts_provider == "aistudio" else "ko-KR-Neural2-C"

    voice_cache_key = f"voice_list_{tts_provider}"
    if voice_cache_key not in st.session_state.v3_data:
        st.session_state.v3_data[voice_cache_key] = []

    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("Load Voices"):
            voices = ag.list_voices(lang="ko-KR")
            st.session_state.v3_data[voice_cache_key] = voices
            if ag.last_error and tts_provider == "cloud":
                st.warning(f"Cloud TTS credential issue: {ag.last_error}")
            st.success(f"Loaded {len(voices)} voices")
    with c2:
        speed = st.slider("Speed", 0.85, 1.20, 1.00, 0.01)
        strict_voice = st.checkbox(
            "Keep one voice only (disable fallback)",
            value=True,
            help="If checked, AI Studio failure does NOT fallback to Cloud. This prevents voice mixing."
        )

    voices = st.session_state.v3_data.get(voice_cache_key) or []
    if voices:
        default_idx = 0
        if recommended_voice in voices:
            default_idx = voices.index(recommended_voice)
        elif tts_provider == "cloud":
            for i, v in enumerate(voices):
                if "Neural2" in v:
                    default_idx = i
                    break
        voice_name = st.selectbox("Voice", voices, index=default_idx)
    else:
        default_voice = recommended_voice
        voice_name = st.text_input("Voice", value=default_voice)
    st.caption(f"Recommended default for this channel: {recommended_voice}")
    voice_gender = ag.get_voice_gender(voice_name)
    st.caption(f"Selected voice gender: {voice_gender}")

    if st.button("Prepare Sentence Segments"):
        sents = split_script_sentences(full_text)
        segs = []
        for i, s in enumerate(sents):
            segs.append({
                "idx": i,
                "text": s,
                "audio_path": None,
                "tts_status": "pending",
                "tts_provider_used": None,
                "tts_voice_used": None,
                "tts_voice_gender": None,
                "tts_style_used": None,
                "tts_error": None,
            })
        st.session_state.v3_data["audio_segments"] = segs
        st.success(f"Prepared {len(segs)} sentence segments.")

    segs = st.session_state.v3_data.get("audio_segments") or []
    if segs:
        # Detect likely mojibake/gibberish before costly TTS batch runs.
        bad_like = 0
        for s in segs[: min(30, len(segs))]:
            t = (s.get("text") or "")
            if not t:
                continue
            q_ratio = t.count("?") / max(1, len(t))
            if q_ratio > 0.2 or "�" in t:
                bad_like += 1
        if bad_like >= 3:
            st.error("Current script/segments look encoding-corrupted (many ?/�). Regenerate script in Step 2 before TTS.")

        failed_count = len([s for s in segs if s.get("tts_status") == "failed"])
        ready_count = len([s for s in segs if s.get("audio_path") and os.path.exists(s.get("audio_path"))])
        st.caption(f"Ready: {ready_count} | Failed: {failed_count} | Total: {len(segs)}")
        min_interval_sec = 6.2 if tts_provider == "aistudio" else 0.0

        if st.button("Generate ALL Missing Sentence Audio", type="primary"):
            pb = st.progress(0)
            done = 0
            mixed_count = 0
            last_req_at = 0.0
            quota_stop_msg = ""
            for i, seg in enumerate(segs):
                if seg.get("audio_path") and os.path.exists(seg["audio_path"]):
                    done += 1
                    pb.progress(done / len(segs))
                    continue
                if min_interval_sec > 0 and last_req_at > 0:
                    gap = time.time() - last_req_at
                    if gap < min_interval_sec:
                        time.sleep(min_interval_sec - gap)
                out_p = os.path.join(OUT_DIR, f"tts_seg_{i:04d}_{int(time.time())}.mp3")
                style_instr = auto_tts_style_instruction(seg["text"])
                res = ag.generate(
                    seg["text"],
                    voice_name,
                    out_p,
                    speed=speed,
                    style_instruction=style_instr,
                    allow_fallback=not strict_voice,
                )
                last_req_at = time.time()
                if res and os.path.exists(res):
                    seg["audio_path"] = res
                    seg["tts_provider_used"] = ag.last_provider_used
                    seg["tts_voice_used"] = voice_name
                    seg["tts_voice_gender"] = ag.get_voice_gender(voice_name)
                    seg["tts_style_used"] = style_instr
                    seg["tts_status"] = "ok"
                    seg["tts_error"] = None
                    if tts_provider == "aistudio" and ag.last_provider_used != "aistudio":
                        mixed_count += 1
                else:
                    seg["tts_status"] = "failed"
                    seg["tts_provider_used"] = ag.last_provider_used
                    seg["tts_voice_used"] = voice_name
                    seg["tts_voice_gender"] = ag.get_voice_gender(voice_name)
                    seg["tts_style_used"] = style_instr
                    seg["tts_error"] = ag.last_error or "unknown_error"
                    if is_terminal_tts_quota_error(seg["tts_error"]):
                        quota_stop_msg = seg["tts_error"]
                done += 1
                pb.progress(done / len(segs))
                if quota_stop_msg:
                    break
            st.success("Sentence TTS generation done.")
            if mixed_count > 0:
                st.warning(
                    f"Voice source mixed detected: {mixed_count} segment(s) used fallback engine. "
                    "Check each segment provider and regenerate if needed."
                )
            if quota_stop_msg:
                if tts_provider == "aistudio" and strict_voice:
                    st.error(
                        "AI Studio daily quota reached. Batch stopped early. "
                        "Turn OFF 'Keep one voice only' to allow Cloud fallback, or switch provider to Google Cloud TTS."
                    )
                elif tts_provider == "aistudio":
                    st.error(
                        "AI Studio daily quota reached. Batch stopped early. "
                        "Cloud fallback may be unavailable if credentials are missing."
                    )
                else:
                    st.error("TTS quota limit detected. Batch stopped early. Try again later or switch account/project.")

        if st.button("Retry FAILED Segments with Current Provider"):
            failed = [s for s in segs if s.get("tts_status") == "failed"]
            if not failed:
                st.info("No failed segments.")
            else:
                pb = st.progress(0)
                last_req_at = 0.0
                quota_stop_msg = ""
                for i, seg in enumerate(failed):
                    if min_interval_sec > 0 and last_req_at > 0:
                        gap = time.time() - last_req_at
                        if gap < min_interval_sec:
                            time.sleep(min_interval_sec - gap)
                    out_p = os.path.join(OUT_DIR, f"tts_seg_{seg['idx']:04d}_{int(time.time())}.mp3")
                    style_instr = auto_tts_style_instruction(seg["text"])
                    res = ag.generate(
                        seg["text"],
                        voice_name,
                        out_p,
                        speed=speed,
                        style_instruction=style_instr,
                        allow_fallback=not strict_voice,
                    )
                    last_req_at = time.time()
                    if res and os.path.exists(res):
                        seg["audio_path"] = res
                        seg["tts_status"] = "ok"
                        seg["tts_provider_used"] = ag.last_provider_used
                        seg["tts_voice_used"] = voice_name
                        seg["tts_voice_gender"] = ag.get_voice_gender(voice_name)
                        seg["tts_style_used"] = style_instr
                        seg["tts_error"] = None
                    else:
                        seg["tts_status"] = "failed"
                        seg["tts_provider_used"] = ag.last_provider_used
                        seg["tts_voice_used"] = voice_name
                        seg["tts_voice_gender"] = ag.get_voice_gender(voice_name)
                        seg["tts_style_used"] = style_instr
                        seg["tts_error"] = ag.last_error or "unknown_error"
                        if is_terminal_tts_quota_error(seg["tts_error"]):
                            quota_stop_msg = seg["tts_error"]
                    pb.progress((i + 1) / len(failed))
                    if quota_stop_msg:
                        break
                st.success("Retry complete.")
                if quota_stop_msg:
                    if tts_provider == "aistudio" and strict_voice:
                        st.error(
                            "AI Studio daily quota reached. Retry stopped early. "
                            "Turn OFF 'Keep one voice only' to allow Cloud fallback, or switch provider to Google Cloud TTS."
                        )
                    elif tts_provider == "aistudio":
                        st.error(
                            "AI Studio daily quota reached. Retry stopped early. "
                            "Cloud fallback may be unavailable if credentials are missing."
                        )
                    else:
                        st.error("TTS quota limit detected. Retry stopped early. Try again later or switch account/project.")

        st.caption(f"Segments: {len(segs)}")
        for seg in segs[:40]:
            with st.expander(f"Seg {seg['idx']+1}: {seg['text'][:60]}..."):
                st.write(seg["text"])
                if st.button(f"Generate Seg {seg['idx']+1}", key=f"gen_seg_{seg['idx']}"):
                    if min_interval_sec > 0:
                        time.sleep(min_interval_sec)
                    out_p = os.path.join(OUT_DIR, f"tts_seg_{seg['idx']:04d}_{int(time.time())}.mp3")
                    style_instr = auto_tts_style_instruction(seg["text"])
                    res = ag.generate(
                        seg["text"],
                        voice_name,
                        out_p,
                        speed=speed,
                        style_instruction=style_instr,
                        allow_fallback=not strict_voice,
                    )
                    if res and os.path.exists(res):
                        seg["audio_path"] = res
                        seg["tts_provider_used"] = ag.last_provider_used
                        seg["tts_voice_used"] = voice_name
                        seg["tts_voice_gender"] = ag.get_voice_gender(voice_name)
                        seg["tts_style_used"] = style_instr
                        seg["tts_status"] = "ok"
                        seg["tts_error"] = None
                        if tts_provider == "aistudio" and ag.last_provider_used != "aistudio":
                            st.warning(
                                f"Seg {seg['idx']+1} used fallback provider: {ag.last_provider_used}. "
                                "Voice may sound different."
                            )
                    else:
                        seg["tts_status"] = "failed"
                        seg["tts_provider_used"] = ag.last_provider_used
                        seg["tts_voice_used"] = voice_name
                        seg["tts_voice_gender"] = ag.get_voice_gender(voice_name)
                        seg["tts_style_used"] = style_instr
                        seg["tts_error"] = ag.last_error or "unknown_error"
                        st.error(f"TTS failed: {seg['tts_error']}")
                        if is_terminal_tts_quota_error(seg["tts_error"]):
                            if tts_provider == "aistudio" and strict_voice:
                                st.warning(
                                    "Daily quota reached. Turn OFF 'Keep one voice only' to allow Cloud fallback, "
                                    "or switch provider to Google Cloud TTS."
                                )
                            elif tts_provider == "aistudio":
                                st.warning(
                                    "Daily quota reached on AI Studio. Cloud fallback may fail if Cloud credentials are missing."
                                )
                    st.rerun()
                if seg.get("audio_path") and os.path.exists(seg["audio_path"]):
                    st.audio(seg["audio_path"])
                if seg.get("tts_provider_used"):
                    st.caption(f"Provider used: {seg.get('tts_provider_used')}")
                if seg.get("tts_voice_used") or seg.get("tts_voice_gender"):
                    st.caption(f"Voice used: {seg.get('tts_voice_used')} | Gender: {seg.get('tts_voice_gender')}")
                if seg.get("tts_style_used"):
                    st.caption(f"Style used: {seg.get('tts_style_used')}")
                if seg.get("tts_status") == "failed":
                    st.error(f"Status: failed | reason: {seg.get('tts_error') or 'unknown_error'}")

        if len(segs) > 40:
            st.caption(f"... {len(segs)-40} more segments not expanded")

        if st.button("Merge Sentence Audio to Final MP3"):
            paths = [s.get("audio_path") for s in segs]
            if not all(p and os.path.exists(p) for p in paths):
                st.error("Some segments are missing audio. Generate all first.")
            else:
                out_mp3 = os.path.join(OUT_DIR, f"audio_final_{int(time.time())}.mp3")
                try:
                    merged = merge_mp3_files(paths, out_mp3)
                    merged_d = get_audio_duration(merged)
                    if effective_target_min > 0 and merged_d < (effective_target_min * 60 * 0.85):
                        st.error(
                            f"Merged audio is too short ({merged_d/60:.2f} min) for target "
                            f"{effective_target_min} min. Regenerate script/TTS before render."
                        )
                    st.session_state.v3_data["audio_path"] = merged
                    st.success(f"Merged audio: {merged}")
                except Exception as e:
                    st.error(f"Audio merge failed: {e}")

    script_wc = len((script_data.get("full_text") or "").split())
    est_script_min = (script_wc / 130.0) if script_wc > 0 else 0.0
    cfg_target_min = int(st.session_state.v3_data.get("target_duration_min") or 0)
    effective_target_min = cfg_target_min if cfg_target_min > 0 else int(round(est_script_min))

    if st.session_state.v3_data.get("audio_path") and os.path.exists(st.session_state.v3_data["audio_path"]):
        st.subheader("Final Audio")
        st.audio(st.session_state.v3_data["audio_path"])
        aud_d = get_audio_duration(st.session_state.v3_data["audio_path"])
        if effective_target_min > 0:
            st.caption(f"Merged audio length: {aud_d/60:.2f} min (target: {effective_target_min} min)")
            if aud_d < (effective_target_min * 60 * 0.85):
                st.warning(
                    "Merged audio is much shorter than target duration. "
                    "Please regenerate/expand script or fill missing TTS before rendering."
                )

    st.markdown("---")
    n1, n2 = st.columns(2)
    with n1:
        if st.button("Back to Storyboard"):
            prev_step()
            st.rerun()
    with n2:
        if st.button("Go to Render (Step 5)"):
            next_step()
            st.rerun()


# ---- STEP 5: RENDER ----
def step_render():
    st.header("Step 5: Final Render")
    scenes = st.session_state.v3_data.get("scenes") or []
    audio_path = st.session_state.v3_data.get("audio_path")

    if not scenes:
        st.warning("No scenes found. Go back to Step 3.")
        if st.button("Back to Storyboard"):
            st.session_state.v3_data["step"] = 3
            st.rerun()
        return
    if not audio_path or not os.path.exists(audio_path):
        st.warning("No merged audio found. Go back to Step 4.")
        if st.button("Back to Audio"):
            st.session_state.v3_data["step"] = 4
            st.rerun()
        return
    script_data = st.session_state.v3_data.get("script_data") or {}
    script_wc = len((script_data.get("full_text") or "").split())
    est_script_min = (script_wc / 130.0) if script_wc > 0 else 0.0
    target_min = int(st.session_state.v3_data.get("target_duration_min") or 0)
    if target_min <= 0 and est_script_min > 0:
        target_min = int(round(est_script_min))
    audio_dur = get_audio_duration(audio_path)
    if target_min > 0:
        st.caption(f"Audio length: {audio_dur/60:.2f} min (target: {target_min} min)")
        if audio_dur < (target_min * 60 * 0.85):
            st.error(
                "Current merged audio is too short for selected target duration. "
                "Go back to Step 4 and regenerate TTS/script, then merge again."
            )
            if st.button("Back to Audio to Fix Duration"):
                st.session_state.v3_data["step"] = 4
                st.rerun()
            return

    # Longform mode: do not use old frame template layers.
    overlay_png = None
    fixed_char = None

    default_out = os.path.join(OUT_DIR, f"final_longform_{int(time.time())}.mp4")
    default_move_dir = os.getenv("FINAL_VIDEO_MOVE_DIR", os.path.join(OUT_DIR, "final_videos"))
    out_path = st.text_input("Output Path", value=default_out)
    move_after_render_dir = st.text_input("Move Folder After Render", value=default_move_dir)
    # Longform 16:9 full-frame defaults
    news_box = (0, 0, 1920, 1080)
    sub_margin_v = 60
    sub_font_size = 34
    image_zoom_max = st.slider("Image Motion Zoom Max", 1.02, 1.20, 1.10, 0.01)
    st.caption("Longform mode (16:9): no template frame / no overlay / no fixed character")

    # Keep scene count/order fixed to avoid AV timing drift.
    # Missing media is replaced with nearest valid media or placeholder instead of skipping.
    render_scenes = []
    last_valid_media = None
    missing_count = 0
    placeholder_path = os.path.join(ASSETS_DIR, "placeholder.jpg")
    for s in scenes:
        s2 = dict(s)
        # Prefer real video when available to avoid stale image path causing unexpected zoom motion.
        media_p = s2.get("video_path") or s2.get("generated_media_path") or s2.get("image_path")
        if media_p and os.path.exists(media_p):
            s2["generated_media_path"] = media_p
            last_valid_media = media_p
        else:
            missing_count += 1
            fallback_p = last_valid_media if last_valid_media and os.path.exists(last_valid_media) else placeholder_path
            s2["generated_media_path"] = fallback_p
            if not (fallback_p.lower().endswith(".mp4") or fallback_p.lower().endswith(".mov")):
                s2["media_type"] = "image"
        render_scenes.append(s2)

    subtitle_segments = st.session_state.v3_data.get("audio_segments", []) or []
    render_scenes_final = render_scenes
    if subtitle_segments and render_scenes:
        # Always build one render scene per subtitle sentence.
        # Media is mapped by scene-text <-> sentence-text matching (monotonic).
        n = len(render_scenes)
        m = len(subtitle_segments)
        mapped_indices = map_segments_to_scenes_by_text(render_scenes, subtitle_segments)
        expanded = []
        for i, seg in enumerate(subtitle_segments):
            if i < len(mapped_indices):
                si = mapped_indices[i]
            else:
                si = min(n - 1, int(i * n / max(1, m)))
            base = dict(render_scenes[si])
            if seg.get("text"):
                base["text"] = seg.get("text")
            seg_ap = seg.get("audio_path")
            base["_segment_audio_path"] = seg_ap
            base["_segment_duration"] = get_audio_duration(seg_ap) if seg_ap and os.path.exists(seg_ap) else 0.0
            expanded.append(base)
        render_scenes_final = expanded
        # Deterministic sync guard: every sentence must have measurable TTS duration.
        bad_idx = [i for i, s in enumerate(render_scenes_final) if float(s.get("_segment_duration") or 0.0) <= 0.0]
        if bad_idx:
            st.error(
                f"Render blocked: missing segment duration for sentence index {bad_idx[0]+1}. "
                "Regenerate missing TTS sentence audio first."
            )
            return

    st.caption(f"Renderable scenes: {len(render_scenes_final)} / {len(scenes)}")
    if missing_count > 0:
        st.warning(
            f"{missing_count} scene(s) had missing media. Render keeps timeline using previous media/placeholder (no scene skip)."
        )

    if st.button("Render Final Video", type="primary"):
        if not render_scenes_final:
            st.error("No renderable scenes. Generate scene media first.")
        else:
            try:
                progress_slot = st.progress(0.0)
                status_slot = st.empty()
                def _on_progress(v: float, msg: str):
                    progress_slot.progress(v)
                    status_slot.caption(f"Render Progress: {int(v * 100)}% | {msg}")

                with st.spinner("Rendering final video... this can take several minutes."):
                    layout = {
                        "news_box": news_box,
                        "fixed_char_path": fixed_char,
                        "sub_margin_v": sub_margin_v,
                        "sub_font_size": sub_font_size,
                        "subtitle_font_name": "Malgun Gothic Bold",
                        "subtitle_bold": 1,
                        "subtitle_max_chars": 20,
                        "image_motion": "zoom_in_center",
                        "image_zoom_max": image_zoom_max,
                        "subtitle_segments": subtitle_segments,
                    }
                    final_path = render_video(
                        scenes=render_scenes_final,
                        audio_path=audio_path,
                        overlay_png=overlay_png,
                        out_path=out_path,
                        layout_config=layout,
                        width=1920,
                        height=1080,
                        progress_cb=_on_progress,
                    )
                    if not os.path.exists(final_path) or os.path.getsize(final_path) < 1024:
                        raise RuntimeError("Render output missing/empty. Check ffmpeg logs and source scene media.")

                    try:
                        moved_path = _move_file_to_dir(final_path, move_after_render_dir)
                        if moved_path != final_path:
                            final_path = moved_path
                            st.info(f"Final video moved to: {final_path}")
                    except Exception as move_err:
                        st.warning(f"Render done, but move failed: {move_err}")

                    progress_slot.progress(1.0)
                    status_slot.caption("Render Progress: 100% | Done")
                    st.session_state.v3_data["last_render_scenes"] = render_scenes_final
                    st.session_state.v3_data["video_path"] = final_path
                    _save_project_snapshot(st.session_state.v3_data, label="post_render")
                    st.success(f"Render complete: {final_path}")
            except Exception as e:
                st.error(f"Render failed: {e}")

    final_video = st.session_state.v3_data.get("video_path")
    if final_video and os.path.exists(final_video):
        st.video(final_video)
        with open(final_video, "rb") as f:
            st.download_button(
                "Download Final Video",
                data=f,
                file_name=os.path.basename(final_video),
                mime="video/mp4",
            )

    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Back to Audio"):
            prev_step()
            st.rerun()
    with b2:
        if final_video and os.path.exists(final_video):
            if st.button("Go to Publish Pack (Step 6)"):
                st.session_state.v3_data["step"] = 6
                st.rerun()
    with b3:
        if st.button("Restart"):
            st.session_state.v3_data["step"] = 1
            st.rerun()


# ---- STEP 6: PUBLISH PACKAGE ----
def step_publish_pack():
    st.header("Step 6: Publish Package")
    final_video = st.session_state.v3_data.get("video_path")
    if not final_video or not os.path.exists(final_video):
        st.warning("Final video not found. Complete Step 5 render first.")
        if st.button("Back to Render"):
            st.session_state.v3_data["step"] = 5
            st.rerun()
        return

    if st.button("Generate Title/Tags/Description", type="primary") or "publish_pack" not in st.session_state.v3_data:
        st.session_state.v3_data["publish_pack"] = build_publish_package(st.session_state.v3_data)

    pack = st.session_state.v3_data.get("publish_pack") or build_publish_package(st.session_state.v3_data)

    st.subheader("1) Video Title")
    title_txt = st.text_area("Copy Title", value=pack.get("title", ""), height=80)

    st.subheader("2) Tags (Comma Separated, 10+)")
    tags_txt = st.text_area("Copy Tags", value=pack.get("tags_csv", ""), height=90)
    st.caption("예: 태그1, 태그2, 태그3 ... 형태로 바로 붙여넣기")

    st.subheader("3) Description (with Timeline)")
    desc_txt = st.text_area("Copy Description", value=pack.get("description", ""), height=260)

    st.download_button(
        "Download Publish Text (.txt)",
        data=f"[TITLE]\\n{title_txt}\\n\\n[TAGS]\\n{tags_txt}\\n\\n[DESCRIPTION]\\n{desc_txt}\\n",
        file_name=f"publish_pack_{int(time.time())}.txt",
        mime="text/plain",
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Render"):
            st.session_state.v3_data["step"] = 5
            st.rerun()
    with c2:
        if st.button("Restart Workflow"):
            st.session_state.v3_data["step"] = 1
            st.rerun()

# ---- MAIN ROUTER ----
step = st.session_state.v3_data["step"]

if step == 1:
    step_thumbnail()
elif step == 2:
    step_script()
elif step == 3:
    step_storyboard()
elif step == 4:
    step_audio()
elif step == 5:
    step_render()
elif step == 6:
    step_publish_pack()
else:
    st.write("Unknown step.")
    if st.button("Restart"):
        st.session_state.v3_data["step"] = 1

