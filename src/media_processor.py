# src/media_processor.py
# -*- coding: utf-8 -*-

import os
import subprocess
import json
import re
from typing import List, Optional
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

def render_video(
    scenes: List[dict],
    audio_path: str,
    overlay_png: Optional[str],
    out_path: str,
    layout_config: dict = None,
    width=1080, height=1920
):
    """
    Stitches scenes together.
    """
    if layout_config is None:
        layout_config = {
            "title_x": 42, "title_y": 34, "title_font_size": 44,
            "sub_y": 1460, "sub_margin_v": 100, "sub_font_size": 48
        }

    total_dur = get_audio_duration(audio_path)
    srt_path = out_path.replace(".mp4", ".srt")
    create_srt(scenes, total_dur, srt_path)
    
    # ... (rest of prep logic same as before until filter_complex) ...
    
    # 1. Create concat list
    total_chars = sum([len(s["text"]) for s in scenes])
    if total_chars == 0: total_chars = 1
    
    # Get Layout
    news_x, news_y, news_w, news_h = layout_config.get("news_box", (0, 0, 1080, 1920)) if layout_config else (0, 0, 1080, 1920)

    # 1. Process Scenes (Resize to News Box size)
    temp_files = []
    
    for i, s in enumerate(scenes):
        txt_len = len(s["text"])
        dur = total_dur * (txt_len / total_chars)
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
             seg_cmd += ["-loop", "1", "-i", media_file, "-t", f"{dur:.3f}", "-vf", vf_scale]
             
        seg_cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-an", seg_out]
        
        subprocess.run(seg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_files.append(seg_out)

    with open("out/concat_list.txt", "w") as f:
        for tf in temp_files:
            f.write(f"file '{os.path.abspath(tf)}'\n")
            
    # Final Stitch
    # Inputs:
    # 0: Concat Scenes (News Stream in NewsBox Size)
    # 1: Audio
    # 2: Fixed Character (Optional)
    # 3: Overlay Template (Optional)
    
    input_args = ["-f", "concat", "-safe", "0", "-i", "out/concat_list.txt", "-i", audio_path]
    
    # Filter Chain Construction
    filter_chain = []
    
    # [bg] base 1080x1920
    filter_chain.append(f"color=c=black:s=1080x1920[bg]")
    
    # Overlay News Stream [0:v] onto [bg]
    filter_chain.append(f"[bg][0:v]overlay={news_x}:{news_y}:shortest=1[v_canvas]")
    current_v = "[v_canvas]"
    
    # 1. Fixed Character Layer
    fixed_char_path = layout_config.get("fixed_char_path") if layout_config else None
    char_input_idx = -1
    if fixed_char_path and os.path.exists(fixed_char_path):
        input_args += ["-i", fixed_char_path]
        char_input_idx = len(input_args) // 2 - 1 
        
        # Scale Char to fit width (1080) and preserve aspect, align bottom
        filter_chain.append(f"[{char_input_idx}:v]scale=1080:-1[v_char]") 
        filter_chain.append(f"{current_v}[v_char]overlay=(W-w)/2:H-h[v_w_char]")
        current_v = "[v_w_char]"

    # 2. Template Overlay Layer
    ov_idx = -1
    if overlay_png:
        input_args += ["-i", overlay_png]
        ov_idx = len(input_args) // 2 - 1
        
    if overlay_png:
        filter_chain.append(f"{current_v}[{ov_idx}:v]overlay=0:0[v_ov]")
        current_v = "[v_ov]"
        
    # 3. Subtitles
    srt_safe = srt_path.replace("\\", "/").replace(":", "\\:")
    margin_v = layout_config.get("sub_margin_v", 100) if layout_config else 100
    font_size = layout_config.get("sub_font_size", 48) if layout_config else 48
    
    # Use PrimaryColour for text (White), BackColour (Black outline/shadow)
    style = f"FontName=Arial,FontSize={font_size},PrimaryColour=&H00FFFFFF,Outline=2,BackColour=&H80000000,BorderStyle=3,MarginV={margin_v},Alignment=2"
    filter_chain.append(f"{current_v}subtitles='{srt_safe}':force_style='{style}'[v_final]")
    
    # Construct final command
    stitch_cmd = [FFMPEG_BIN, "-y"] + input_args
    stitch_cmd += ["-filter_complex", ";".join(filter_chain)]
    stitch_cmd += ["-map", "[v_final]", "-map", "1:a"]
    stitch_cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest", out_path]
    
    print(f"Running FFmpeg: {' '.join(stitch_cmd)}")
    subprocess.run(stitch_cmd, check=True)
    return out_path
