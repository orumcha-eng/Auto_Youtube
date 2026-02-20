# app_v3.py
# -*- coding: utf-8 -*-

import streamlit as st
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from src.visual_gen import VisualGenerator
from PIL import Image, ImageDraw, ImageFont

# ---- CONFIG & SETUP ----
st.set_page_config(layout="wide", page_title="Auto Youtube V3 (Bean World)")
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "out", "v3")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

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

# ---- STATE MANAGEMENT ----
if "v3_data" not in st.session_state:
    st.session_state.v3_data = {
        "step": 1, 
        "topic": "Global Economy",
        "thumbnail_data": None, # {image_path, title_text, full_prompt}
        "thumbnail_copy_options": {},
        "selected_thumbnail_copy": "",
        "script_data": None,    # {full_text, segments: [{title, body}]}
        "scenes": [],           # List of scene dicts
        "audio_path": None,
        "video_path": None
    }

def next_step(): st.session_state.v3_data["step"] += 1
def prev_step(): st.session_state.v3_data["step"] -= 1

# ---- UI: SIDEBAR ----
st.sidebar.title("Bean World Studio")
api_key = st.sidebar.text_input("Gemini API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
st.session_state["luma_key_input"] = st.sidebar.text_input("Luma API Key (Optional for Real Video)", value=os.getenv("LUMA_API_KEY", ""), type="password")

from src.content import ContentModule

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
        vg = VisualGenerator(api_key)
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

        img = Image.open(base_image_path).convert("RGBA")
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
        base_size = max(64, int(h * 0.105))
        font = ImageFont.truetype(font_path, base_size) if font_path else ImageFont.load_default()

        # Fit text width
        max_w = int(w * 0.92)
        while True:
            widths = []
            for ln in lines:
                bb = d.textbbox((0, 0), ln, font=font, stroke_width=8)
                widths.append(bb[2] - bb[0])
            if max(widths) <= max_w or base_size <= 40:
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
                min_wc = max(600, int(duration_val * 115))
                if wc < min_wc:
                    st.warning(f"Script is shorter than target for {duration_val} min. (current: {wc}, target min: {min_wc})")

    if st.session_state.v3_data["script_data"]:
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
                    vg = VisualGenerator(api_key)
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
            vg = VisualGenerator(api_key)
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
                        vg = VisualGenerator(api_key)
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
                            vg = VisualGenerator(api_key)
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
        if st.button("Go to Render (Step 4)" if all_ready else "Go to Render (Skip missing images)"):
            next_step()
            st.rerun()

# ---- MAIN ROUTER ----
step = st.session_state.v3_data["step"]

if step == 1:
    step_thumbnail()
elif step == 2:
    step_script()
elif step == 3:
    step_storyboard()
else:
    st.write("Render Page Placeholder")
    if st.button("Restart"):
        st.session_state.v3_data["step"] = 1

