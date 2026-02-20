# app_v2.py
# -*- coding: utf-8 -*-

import streamlit as st
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from src.content import ContentModule
from src.visual_gen import VisualGenerator
from src.audio import AudioGenerator
from src.media_processor import render_video

# ---- CONFIG ----
st.set_page_config(layout="wide", page_title="Auto Youtube V2")
load_dotenv() # Load .env if exists

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "out")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---- AUTH (Google Cloud TTS) ----
# Automatically set credentials for Google Cloud if the key file exists
tts_key_path = os.path.join(BASE_DIR, "gcp-tts-key.json")
if os.path.exists(tts_key_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tts_key_path
elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    st.warning("⚠️ Warning: 'gcp-tts-key.json' not found. TTS might fail.")

# ---- STATE ----
if "script_data" not in st.session_state:
    st.session_state.script_data = None
if "scenes" not in st.session_state:
    st.session_state.scenes = []
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# ---- SIDEBAR ----
st.sidebar.title("Configuration")

# Auto-load key
default_key = os.getenv("GOOGLE_API_KEY", "")
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password")

# Generic Asset Helper
def get_assets(ext=".png"):
    return [f for f in os.listdir(ASSETS_DIR) if f.endswith(ext)]

# Character Mode
st.sidebar.subheader("Character Settings")
char_mode = st.sidebar.radio("Character Mode", ["AI Generated (Description)", "Fixed Image (Asset)"])

character_desc = ""
fixed_char_path = None

if char_mode == "AI Generated (Description)":
    character_desc = st.sidebar.text_area("Character Description", "A professional news anchor, futuristic studio, blue suit, 8k resolution.")
else:
    # Asset Selection
    char_assets = get_assets(".png")
    char_selection = st.sidebar.selectbox("Select Character Image", ["(Upload New)"] + char_assets)
    
    if char_selection == "(Upload New)":
        uploaded_char = st.sidebar.file_uploader("Upload Character PNG", type=["png"])
        if uploaded_char:
            # Save with original name
            save_path = os.path.join(ASSETS_DIR, uploaded_char.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_char.getbuffer())
            st.sidebar.success(f"Saved: {uploaded_char.name}")
            st.rerun() # Refresh to show in list
    else:
        fixed_char_path = os.path.join(ASSETS_DIR, char_selection)
        st.sidebar.image(fixed_char_path, caption="Selected Character", width=100)

st.sidebar.subheader("Overlay & Layout")

# Overlay Selection
ov_assets = get_assets(".png")
ov_options = ["(None)", "(Upload New)"] + ov_assets

# Session state for selection index
if "ov_index" not in st.session_state:
    st.session_state.ov_index = 0

def update_ov():
    # Update index based on selection
    # But streamlit handles this if we don't force index?
    # If we want to set it programmatically, we need a key.
    pass

ov_selection = st.sidebar.selectbox(
    "Select Overlay Template", 
    ov_options, 
    index=st.session_state.ov_index,
    key="ov_selectbox"
)

overlay_path = None
if ov_selection == "(Upload New)":
    uploaded_overlay = st.sidebar.file_uploader("Upload Overlay PNG", type=["png"])
    if uploaded_overlay:
        save_path = os.path.join(ASSETS_DIR, uploaded_overlay.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_overlay.getbuffer())
        
        # Find new index
        try:
            # Re-read assets to be sure
            new_assets = get_assets(".png")
            new_options = ["(None)", "(Upload New)"] + new_assets
            new_idx = new_options.index(uploaded_overlay.name)
            st.session_state.ov_index = new_idx
        except:
            st.session_state.ov_index = 0 # Fallback
            
        st.sidebar.success(f"Saved: {uploaded_overlay.name}")
        st.rerun()
        
elif ov_selection != "(None)":
    # Update session state to match current selection logic (if user changes manually)
    # Actually, if user changes formatting, 'ov_index' stays old?
    # We should sync them.
    # Simple fix: Just use the key value.
    if ov_selection in ov_assets:
         idx = ov_options.index(ov_selection)
         st.session_state.ov_index = idx

    overlay_path = os.path.join(ASSETS_DIR, ov_selection)
    st.sidebar.image(overlay_path, caption="Selected Overlay", width=100)

# ---- LAYOUT CONFIG ----
with st.sidebar.expander("Layout Settings (Preview)"):
    st.write("Adjust Title & Subtitle Positions")

    # Title Config
    title_y = st.number_input("Title Y Position (from Top)", value=200, step=10)
    title_font_size = st.number_input("Title Font Size", value=60, step=5)
    
    # Subtitle Config
    sub_margin_v = st.number_input("Subtitle Margin (from Bottom)", value=380, step=10)
    sub_font_size = st.number_input("Subtitle Font Size", value=50, step=2)
    
    st.markdown("---")
    st.write("News Screen Area")
    news_x = st.number_input("News X", value=60, step=10)
    news_y = st.number_input("News Y", value=250, step=10)
    news_w = st.number_input("News W", value=960, step=10)
    news_h = st.number_input("News H", value=600, step=10)
    
    st.markdown("---")
    st.write("News Screen Area (Where images/videos play)")
    # Default fully covering top part? Or a specific PIP box?
    # Shorts full screen is 1080x1920.
    # Let's default to full screen for now but allow adjustment.
    news_x = st.number_input("News X", value=0, step=10)
    news_y = st.number_input("News Y", value=0, step=10)
    news_w = st.number_input("News Width", value=1080, step=10)
    news_h = st.number_input("News Height", value=1000, step=10)
    
    show_preview = st.toggle("Show Live Preview", value=True)
    
    if show_preview:
        from PIL import Image, ImageDraw, ImageFont
        
        # 1. Base Canvas (Black)
        base = Image.new("RGBA", (1080, 1920), (0,0,0,255))
        
        # 0. News Box
        d = ImageDraw.Draw(base)
        d.rectangle([(news_x, news_y), (news_x+news_w, news_y+news_h)], fill="#000033", outline="#0000FF", width=3)
        
        # 2. Add Fixed Character (if any)
        
        # 0. Draw News Box Placeholder (Blue)
        d = ImageDraw.Draw(base)
        d.rectangle([(news_x, news_y), (news_x+news_w, news_y+news_h)], fill="#000033", outline="#0000FF", width=3)
        d.text((news_x + news_w//2, news_y + news_h//2), "NEWS CONTENT AREA", fill="#00AAFF", anchor="mm", font_size=40)
        
        # 2. Add Fixed Character (if any)
        
        # 2. Add Fixed Character (if any)
        if fixed_char_path and os.path.exists(fixed_char_path):
            try:
                char_img = Image.open(fixed_char_path).convert("RGBA")
                
                # Smart Scaling
                # If char is "small" relative to canvas, maybe don't resize?
                # But usually users upload high-res. 
                # Let's scale to width 1080 if it's wider or significantly smaller?
                # For preview consistnecy with FFmpeg (which we set to scale=1080:-1), do same here.
                cw, ch = char_img.size
                scale_ratio = 1080 / cw
                new_h = int(ch * scale_ratio)
                char_img = char_img.resize((1080, new_h))
                
                # Align bottom center
                x_pos = 0 # since width is 1080
                y_pos = 1920 - new_h
                base.paste(char_img, (x_pos, y_pos), char_img)
            except: pass

        # 3. Add Overlay Template
        if overlay_path and os.path.exists(overlay_path):
            try:
                ov_img = Image.open(overlay_path).convert("RGBA")
                base.paste(ov_img, (0,0), ov_img)
            except: pass
            
        d = ImageDraw.Draw(base)
        
        # 4. Draw Layout Guides (Bright Colors)
        
        # TITLE Box (Green)
        d.rectangle([(50, title_y), (1030, title_y + title_font_size + 20)], outline="#00FF00", width=5)
        d.text((540, title_y), "TITLE TEXT HERE", fill="#00FF00", anchor="mt", font_size=title_font_size)
        
        # SUBTITLE Box (Yellow) - MarginV from bottom
        sub_y = 1920 - sub_margin_v
        d.rectangle([(50, sub_y - sub_font_size - 10), (1030, sub_y + 10)], outline="#FFFF00", width=5)
        d.text((540, sub_y), "자막이 여기에 표시됩니다 (Subtitle)", fill="#FFFF00", anchor="ms", font_size=sub_font_size)
        
        # Center Line
        d.line([(540, 0), (540, 1920)], fill="#333333", width=2)
        
        st.image(base, caption="Layout Preview (Green=Title, Yellow=Subtitles)", use_container_width=True)

layout_config = {
    "title_y": title_y,
    "title_font_size": title_font_size,
    "sub_margin_v": sub_margin_v,
    "sub_font_size": sub_font_size,
    "fixed_char_path": fixed_char_path,
    "news_box": (news_x, news_y, news_w, news_h)
}

# ---- MAIN TABS ----
tab1, tab2, tab3 = st.tabs(["1. Content & Script", "2. Storyboard & Visuals", "3. Audio & Render"])

# ==== TAB 1: CONTENT ====
with tab1:
    st.header("Step 1: Generate News Script")
    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("Target Date", datetime.now())
        query = st.text_input("News Query", "markets tech")
    
    if st.button("Generate Script"):
        with st.spinner("Fetching news & Writing script (AI)..."):
            cm = ContentModule()
            data = cm.generate(target_date.isoformat(), query, api_key=api_key)
            
            # PARSING LOGIC: Extract only [55초 대본] and [다음 일정]
            full_text = data["script"]
            
            # Simple Regex or split to find relevant sections
            import re
            
            # Extract [55초 대본] section
            script_match = re.search(r"\[55초 대본\](.*?)(?=\[|$)", full_text, re.DOTALL)
            schedule_match = re.search(r"\[다음 일정\(한국시간\)\](.*?)(?=\[|$)", full_text, re.DOTALL)
            
            clean_script = ""
            if script_match:
                clean_script += script_match.group(1).strip()
            
            if schedule_match:
                clean_script += "\n\n" + schedule_match.group(1).strip()
            
            if not clean_script:
                # Fallback if parsing fails (e.g. AI format didn't match perfectly)
                clean_script = full_text
            
            # Clean markers
            # Markers observed: (오프닝), (반응/리스크), (훅), (팩트 1), (해석 1) ...
            # Regex to remove these patterns
            marker_pattern = r"\((오프닝|훅|팩트|해석|체크포인트|마무리|반응|리스크|작게).*?\)"
            clean_script = re.sub(marker_pattern, "", clean_script)
            
            # Clean up extra spaces/newlines
            clean_script = re.sub(r"\n{3,}", "\n\n", clean_script).strip()
            
            data["full_analysis"] = full_text 
            data["script"] = clean_script 
            
            st.session_state.script_data = data
            st.success("Korean Script generated! (Parsed for TTS)")

    if st.session_state.script_data:
        st.subheader("Edit Script (TTS Only)")
        st.info("💡 Below is the parsed text for TTS. Edit if needed.")
        edited_script = st.text_area("Script", st.session_state.script_data["script"], height=300)
        st.session_state.script_data["script"] = edited_script
        
        with st.expander("View Full AI Analysis (Checklists, Key Points, Sources)"):
            st.text(st.session_state.script_data.get("full_analysis", "No full analysis available"))
        
        st.subheader("News Source Info")
        st.json(st.session_state.script_data["headlines"])


# ==== TAB 2: STORYBOARD ====
with tab2:
    st.header("Step 2: Storyboard & Visualization")
    
    if not api_key:
        st.warning("Please enter Gemini API Key in Sidebar to use AI Visuals.")
    
    if st.button("Analyze Script -> Create Scenes"):
        if not st.session_state.script_data:
            st.error("Go to Step 1 and generate script first.")
        elif not api_key:
            st.error("API Key needed.")
        else:
            with st.spinner("Analyzing scenes & Generating Initial Images..."):
                vg = VisualGenerator(api_key)
                full_script = st.session_state.script_data["script"]
                scenes = vg.analyze_script(full_script, character_desc)
                
                # Auto-generate images first
                for i, s in enumerate(scenes):
                    s["media_type"] = "image"
                    p = s.get("image_prompt", "")
                    fname = f"scene_{s['seq']:03d}_{int(time.time())}.png"
                    out_path = os.path.join(OUT_DIR, fname)
                    res = vg.generate_image(p, out_path)
                    s["generated_media_path"] = res
                
                st.session_state.scenes = scenes
                st.success(f"Generated {len(scenes)} scenes with Images!")

    # Display Scenes
    if st.session_state.scenes:
        st.write("---")
        for i, scene in enumerate(st.session_state.scenes):
            with st.container():
                cols = st.columns([1, 2, 2])
                
                # Col 1: Text & Controls
                with cols[0]:
                    st.markdown(f"**Scene {i+1}**")
                    scene["text"] = st.text_area(f"Text #{i+1}", scene["text"], height=80)
                    
                    # Current Type Indicator
                    ctype = scene.get("media_type", "image")
                    st.info(f"Type: {ctype.upper()}")
                
                # Col 2: Prompt & Actions
                with cols[1]:
                    start_prompt = scene.get("image_prompt", "")
                    if scene.get("media_type") == "video":
                        start_prompt = scene.get("video_prompt", "")
                        
                    user_prompt = st.text_area(f"Prompt #{i+1}", start_prompt, height=80)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(f"Regen Image #{i+1}"):
                            vg = VisualGenerator(api_key)
                            fname = f"scene_{i+1:03d}_{int(time.time())}.png"
                            out_path = os.path.join(OUT_DIR, fname)
                            scene["image_prompt"] = user_prompt
                            res = vg.generate_image(user_prompt, out_path)
                            scene["generated_media_path"] = res
                            scene["media_type"] = "image"
                            st.rerun()
                            
                    with c2:
                        if st.button(f"Make Video #{i+1}"):
                            # Placeholder for Video Generation
                            st.warning("Video Gen waiting for model...")
                            # vg.generate_video(...)
                            scene["media_type"] = "video"
                            # For now just switch type so render knows?
                            # But we need a video file.
                            # If no video file, render uses image looped.
                            st.rerun()

                # Col 3: Preview
                with cols[2]:
                    path = scene.get("generated_media_path")
                    if path and os.path.exists(path):
                        if scene.get("media_type") == "video" and path.endswith(".mp4"):
                            st.video(path)
                        else:
                            st.image(path)
                    else:
                        st.info("No media")
                
                st.divider()
        
        if st.button("Save Storyboard State"):
            st.success("State saved.")


# ==== TAB 3: RENDER ====
with tab3:
    st.header("Step 3: Audio & Final Render")
    
    ag = AudioGenerator()
    voices = ag.list_voices()
    
    sel_voice = st.selectbox("Select Voice", voices, index=0 if voices else 0)
    
    if st.button("Generate TTS"):
        if not st.session_state.script_data:
            st.error("No script")
        else:
            with st.spinner("Synthesizing Audio..."):
                final_script = st.session_state.script_data["script"]
                # Or reconstruct from scenes? 
                # Better to reconstruct from scenes to match order if user edited text in storyboard?
                # For now let's stick to the main script or allow user to choose.
                # Actually, synchronizing text edits is hard. 
                # Let's simple re-join scene texts for the final audio to ensure consistency.
                if st.session_state.scenes:
                    final_script = " ".join([s["text"] for s in st.session_state.scenes])
                
                fname = f"audio_{int(time.time())}.mp3"
                p = os.path.join(OUT_DIR, fname)
                st.session_state.audio_path = ag.generate(final_script, sel_voice, p)
                st.success("Audio Created!")
    
    if st.session_state.audio_path:
        st.audio(st.session_state.audio_path)
        
    st.write("---")
    
    if st.button("Render Final Video"):
        if not st.session_state.audio_path:
            st.error("No audio generated.")
        elif not st.session_state.scenes:
            st.error("No scenes defined.")
        else:
            # Check if all scenes have media
            missing = [i+1 for i, s in enumerate(st.session_state.scenes) if "generated_media_path" not in s]
            if missing:
                st.warning(f"Scenes {missing} have no media. Placeholder will be used.")
            
            with st.spinner("Rendering Video with FFmpeg..."):
                fname = f"final_{int(time.time())}.mp4"
                p = os.path.join(OUT_DIR, fname)
                
                try:
                    res = render_video(
                        st.session_state.scenes,
                        st.session_state.audio_path,
                        overlay_path,
                        p,
                        layout_config=layout_config
                    )
                    st.session_state.video_path = res
                    st.success("Video Rendered!")
                except Exception as e:
                    st.error(f"Render failed: {e}")

    if st.session_state.video_path:
        st.video(st.session_state.video_path)
