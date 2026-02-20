# src/media_processor.py
# -*- coding: utf-8 -*-

import os
import subprocess
import re
import tempfile
from typing import List, Optional, Callable
import imageio_ffmpeg
from mutagen.mp3 import MP3

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

def get_audio_duration(path: str) -> float:
    try:
        return float(MP3(path).info.length)
    except:
        return 0.0

def create_srt(scenes: List[dict], total_audio_dur: float, out_path: str):
    # Establish timing per scene based on word count (rough approx)
    # better approach: The caller should provide audio segment durations.
    # For MVP V2, we assume 1 big audio file. We split duration proportionally to text length.
    
    total_len = sum([len(s["text"]) for s in scenes])
    if total_len == 0: total_len = 1
    
    current_time = 0.0
    lines = []
    
    def fmt_time(t):
        ms = int((t % 1) * 1000)
        s = int(t)
        hh = s // 3600
        mm = (s % 3600) // 60
        ss = s % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    for i, s in enumerate(scenes):
        txt_len = len(s["text"])
        duration = total_audio_dur * (txt_len / total_len)
        start = current_time
        end = start + duration
        current_time = end
        
        lines.append(f"{i+1}")
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(s["text"]) # text might be long, wrapping handled by ffmpeg subtitles filter usually? 
        # Actually standard SRT doesn't wrap automatically in all players, but ffmpeg force_style can.
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _split_caption_chunks(text: str, max_chars: int = 20) -> List[str]:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return []

    words = s.split(" ")
    chunks = []
    cur = ""
    for w in words:
        if not cur:
            if len(w) <= max_chars:
                cur = w
            else:
                # Hard-cut very long tokens to keep single-line guarantee.
                for i in range(0, len(w), max_chars):
                    chunks.append(w[i : i + max_chars])
                cur = ""
            continue
        cand = f"{cur} {w}"
        if len(cand) <= max_chars:
            cur = cand
        else:
            chunks.append(cur)
            if len(w) <= max_chars:
                cur = w
            else:
                for i in range(0, len(w), max_chars):
                    chunks.append(w[i : i + max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def create_srt_from_audio_segments(
    audio_segments: List[dict],
    out_path: str,
    max_chars: int = 20,
    target_total_dur: float = 0.0,
) -> str:
    def fmt_time(t):
        ms = int((t % 1) * 1000)
        s = int(t)
        hh = s // 3600
        mm = (s % 3600) // 60
        ss = s % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    entries = []  # [(caption_text, duration)]
    for seg in audio_segments or []:
        text = (seg.get("text") or "").strip()
        ap = seg.get("audio_path")
        dur = get_audio_duration(ap) if ap and os.path.exists(ap) else 0.0
        if dur <= 0:
            continue
        chunks = _split_caption_chunks(text, max_chars=max_chars) or [text]

        # Split one sentence duration into multiple one-line captions.
        weights = [max(1, len(c)) for c in chunks]
        total_w = sum(weights)
        for c, w in zip(chunks, weights):
            local_dur = dur * (w / total_w)
            entries.append((c, local_dur))

    # Re-scale caption durations to match merged audio total length and reduce drift.
    sum_dur = sum(d for _, d in entries)
    scale = 1.0
    if target_total_dur > 0 and sum_dur > 0:
        scale = target_total_dur / sum_dur

    cur = 0.0
    lines = []
    idx = 1
    for cap, d in entries:
        d2 = d * scale
        start = cur
        end = cur + d2
        cur = end
        lines.append(str(idx))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(cap)
        lines.append("")
        idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _compute_scene_durations_from_audio_segments(
    scenes: List[dict],
    subtitle_segments: List[dict],
    total_audio_dur: float,
) -> List[float]:
    if not scenes or not subtitle_segments:
        return []
    seg_durs = []
    for seg in subtitle_segments:
        ap = seg.get("audio_path")
        d = get_audio_duration(ap) if ap and os.path.exists(ap) else 0.0
        if d > 0:
            seg_durs.append(d)
    if not seg_durs:
        return []

    n = len(scenes)
    m = len(seg_durs)

    # Stable mapping: allocate subtitle/audio segments by index order.
    # This avoids tiny flashes from text-length weighting and keeps pacing predictable.
    by_scene = [0.0] * n
    for i, d in enumerate(seg_durs):
        si = min(n - 1, int(i * n / max(1, m)))
        by_scene[si] += d

    # Ensure every scene gets visible non-zero time.
    min_d = 0.8
    by_scene = [x if x > 0 else min_d for x in by_scene]
    sum_scene = sum(by_scene)
    target = total_audio_dur if total_audio_dur > 0 else sum(seg_durs)
    if sum_scene > 0 and target > 0:
        scale = target / sum_scene
        by_scene = [x * scale for x in by_scene]
    return by_scene


def compute_scene_durations_from_audio_segments(
    scenes: List[dict],
    subtitle_segments: List[dict],
    total_audio_dur: float,
) -> List[float]:
    """Public helper for timeline/reporting: mirrors render scene-duration allocation."""
    return _compute_scene_durations_from_audio_segments(scenes, subtitle_segments, total_audio_dur)

def render_video(
    scenes: List[dict],
    audio_path: str,
    overlay_png: Optional[str],
    out_path: str,
    layout_config: dict = None,
    width=1920, height=1080,
    progress_cb: Optional[Callable[[float, str], None]] = None,
):
    """
    Stitches scenes together.
    """
    if layout_config is None:
        layout_config = {
            "title_x": 42, "title_y": 34, "title_font_size": 44,
            "sub_y": 1460, "sub_margin_v": 100, "sub_font_size": 48
        }

    def _progress(v: float, msg: str):
        if progress_cb:
            try:
                progress_cb(max(0.0, min(1.0, v)), msg)
            except Exception:
                pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _progress(0.02, "Preparing render...")
    total_dur = get_audio_duration(audio_path)
    srt_path = out_path.replace(".mp4", ".srt")
    subtitle_segments = (layout_config or {}).get("subtitle_segments", [])
    if subtitle_segments:
        max_chars = int((layout_config or {}).get("subtitle_max_chars", 20))
        create_srt_from_audio_segments(
            subtitle_segments,
            srt_path,
            max_chars=max_chars,
            target_total_dur=total_dur,
        )
        if not os.path.exists(srt_path) or os.path.getsize(srt_path) == 0:
            create_srt(scenes, total_dur, srt_path)
    else:
        create_srt(scenes, total_dur, srt_path)
    _progress(0.08, "Subtitles prepared.")
    
    # ... (rest of prep logic same as before until filter_complex) ...
    
    # 1. Create concat list
    total_chars = sum([len(s["text"]) for s in scenes])
    if total_chars == 0: total_chars = 1
    
    # Get Layout
    news_x, news_y, news_w, news_h = layout_config.get("news_box", (0, 0, width, height)) if layout_config else (0, 0, width, height)

    # 1. Process Scenes (Resize to News Box size)
    temp_files = []
    image_motion = (layout_config or {}).get("image_motion", "zoom_in_center")
    image_zoom_max = float((layout_config or {}).get("image_zoom_max", 1.10))
    
    scene_durations = _compute_scene_durations_from_audio_segments(scenes, subtitle_segments, total_dur)
    for i, s in enumerate(scenes):
        txt_len = len(s["text"])
        dur = scene_durations[i] if len(scene_durations) == len(scenes) else (total_dur * (txt_len / total_chars))
        dur = max(0.25, float(dur))
        media_file = (
            s.get("generated_media_path")
            or s.get("video_path")
            or s.get("image_path")
        )
        
        if not media_file: 
             media_file = "assets/placeholder.jpg" 
             
        is_video = s.get("media_type") == "video"
        seg_out = f"out/temp_seg_{i:03d}.mp4"
        
        seg_cmd = [FFMPEG_BIN, "-y"]
        
        # Scale & Crop to News Box Size
        vf_scale = f"scale={news_w}:{news_h}:force_original_aspect_ratio=increase,crop={news_w}:{news_h},setsar=1"
        
        if is_video:
             seg_cmd += ["-stream_loop", "-1", "-i", media_file, "-t", f"{dur:.3f}", "-vf", vf_scale]
        else:
             if image_motion == "zoom_in_center":
                 # Ken Burns style center zoom for still images.
                 fps = 30
                 total_frames = max(1, int(round(dur * fps)))
                 step = max(0.0002, (image_zoom_max - 1.0) / total_frames)
                 vf_img = (
                     f"scale={news_w}:{news_h}:force_original_aspect_ratio=increase,"
                     f"crop={news_w}:{news_h},"
                     f"zoompan="
                     f"z='if(lte(on,1),1.0,min(zoom+{step:.7f},{image_zoom_max:.4f}))':"
                     f"x='iw/2-(iw/zoom/2)':"
                     f"y='ih/2-(ih/zoom/2)':"
                     f"d=1:s={news_w}x{news_h}:fps={fps},"
                     f"setsar=1"
                 )
                 seg_cmd += ["-loop", "1", "-i", media_file, "-t", f"{dur:.3f}", "-vf", vf_img]
             else:
                 seg_cmd += ["-loop", "1", "-i", media_file, "-t", f"{dur:.3f}", "-vf", vf_scale]
             
        seg_cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-an", seg_out]
        
        subprocess.run(seg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_files.append(seg_out)
        if scenes:
            _progress(0.10 + 0.65 * ((i + 1) / len(scenes)), f"Rendering scene {i+1}/{len(scenes)}...")

    concat_list_path = os.path.join("out", "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for tf in temp_files:
            f.write(f"file '{os.path.abspath(tf)}'\n")
    _progress(0.78, "Scene segments ready.")
            
    # Final Stitch
    # Inputs:
    # 0: Concat Scenes (News Stream in NewsBox Size)
    # 1: Audio
    # 2: Fixed Character (Optional)
    # 3: Overlay Template (Optional)
    
    input_args = ["-f", "concat", "-safe", "0", "-i", concat_list_path, "-i", audio_path]
    audio_idx = 1
    next_idx = 2
    
    # Filter Chain Construction
    filter_chain = []
    
    # [bg] base canvas
    filter_chain.append(f"color=c=black:s={width}x{height}[bg]")
    
    # Overlay News Stream [0:v] onto [bg]
    filter_chain.append(f"[bg][0:v]overlay={news_x}:{news_y}:shortest=1[v_canvas]")
    current_v = "[v_canvas]"
    
    # 1. Fixed Character Layer
    fixed_char_path = layout_config.get("fixed_char_path") if layout_config else None
    char_input_idx = None
    if fixed_char_path and os.path.exists(fixed_char_path):
        char_input_idx = next_idx
        next_idx += 1
        input_args += ["-i", fixed_char_path]

        # Scale Char to fit canvas width and preserve aspect, align bottom
        filter_chain.append(f"[{char_input_idx}:v]scale={width}:-1[v_char]") 
        filter_chain.append(f"{current_v}[v_char]overlay=(W-w)/2:H-h[v_w_char]")
        current_v = "[v_w_char]"

    # 2. Template Overlay Layer
    ov_idx = None
    if overlay_png:
        ov_idx = next_idx
        next_idx += 1
        input_args += ["-i", overlay_png]
        
    if overlay_png:
        filter_chain.append(f"{current_v}[{ov_idx}:v]overlay=0:0[v_ov]")
        current_v = "[v_ov]"
        
    # 3. Subtitles
    # FFmpeg subtitles filter can fail on some non-ASCII paths on Windows.
    # Copy SRT to a temp ASCII-safe path and use that path in filter.
    srt_filter_path = os.path.join(tempfile.gettempdir(), f"bean_subs_{os.getpid()}.srt")
    with open(srt_path, "r", encoding="utf-8") as sf, open(srt_filter_path, "w", encoding="utf-8") as tf:
        tf.write(sf.read())
    srt_abs = os.path.abspath(srt_filter_path)
    srt_safe = srt_abs.replace("\\", "/").replace(":", "\\:").replace("'", r"\'")
    margin_v = layout_config.get("sub_margin_v", 100) if layout_config else 100
    font_size = layout_config.get("sub_font_size", 48) if layout_config else 48
    font_name = layout_config.get("subtitle_font_name", "Malgun Gothic") if layout_config else "Malgun Gothic"
    subtitle_bold = int(layout_config.get("subtitle_bold", 1)) if layout_config else 1
    
    # Thumbnail-like subtitle style: no black box plate, strong outline only.
    style = (
        f"FontName={font_name},Bold={subtitle_bold},FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Shadow=0,"
        f"BorderStyle=1,MarginV={margin_v},Alignment=2"
    )
    filter_chain.append(f"{current_v}subtitles='{srt_safe}':charenc=UTF-8:force_style='{style}'[v_final]")
    
    # Construct final command
    stitch_cmd = [FFMPEG_BIN, "-y"] + input_args
    stitch_cmd += ["-filter_complex", ";".join(filter_chain)]
    stitch_cmd += ["-map", "[v_final]", "-map", f"{audio_idx}:a"]
    stitch_cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest", out_path]
    
    print(f"Running FFmpeg: {' '.join(stitch_cmd)}")
    _progress(0.82, "Stitching final video...")
    proc = subprocess.Popen(
        stitch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    err_lines = []
    time_re = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
    while True:
        line = proc.stderr.readline() if proc.stderr else ""
        if line:
            err_lines.append(line)
            m = time_re.search(line)
            if m and total_dur > 0:
                hh = int(m.group(1)); mm = int(m.group(2)); ss = float(m.group(3))
                sec = hh * 3600 + mm * 60 + ss
                frac = max(0.0, min(1.0, sec / total_dur))
                _progress(0.82 + 0.16 * frac, f"Stitching... {int(frac * 100)}%")
        elif proc.poll() is not None:
            break
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc})\n{''.join(err_lines)}")
    _progress(1.0, "Render complete.")
    return out_path
