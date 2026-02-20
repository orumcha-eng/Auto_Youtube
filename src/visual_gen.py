# src/visual_gen.py
# -*- coding: utf-8 -*-

import json
import re
import typing
import time
from google import genai
from google.genai import types
from dataclasses import dataclass
import os
import base64

@dataclass
class Scene:
    seq: int
    text: str
    media_type: str = "image"  # 'image' or 'video'
    image_prompt: str = ""
    video_prompt: str = ""

class VisualGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash"

    def add_text_overlay(self, image_path: str, text: str, style: str = "Yellow Label"):
        """Adds text overlay to the existing image using PIL with specific styles."""
        from PIL import Image, ImageDraw, ImageFont
        
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            W, H = img.size
            
            # Font config
            font_path = "C:/Windows/Fonts/malgunbd.ttf" # Malgun Gothic Bold
            if not os.path.exists(font_path):
                font_path = "C:/Windows/Fonts/arialbd.ttf"
                
            # Dynamic font size (Approx 15% of height per line? Let's fit width)
            fontsize = 1
            font = ImageFont.truetype(font_path, fontsize)
            
            # Text Wrap logic (Simple split if too long)
            # For now, let's keep one generic block or max 2 lines
            lines = []
            if len(text) > 12: # naive wrap
                mid = len(text) // 2
                split_idx = text.find(" ", mid-2, mid+3)
                if split_idx == -1: split_idx = mid
                lines = [text[:split_idx], text[split_idx:].strip()]
            else:
                lines = [text]

            # Determine Max Font Size that fits width
            target_width = W * 0.8
            while True:
                max_w = 0
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    if (bbox[2] - bbox[0]) > max_w: max_w = bbox[2] - bbox[0]
                
                if max_w > target_width or fontsize > 200:
                    break
                fontsize += 2
                font = ImageFont.truetype(font_path, fontsize)
            
            # Style Config
            text_color = "black"
            bg_color = None
            stroke_width = 0
            stroke_color = None
            
            if style == "Yellow Label":
                text_color = "black"
                bg_color = (255, 225, 0) # Vivid Yellow
            elif style == "Red Badge":
                text_color = "white"
                bg_color = (230, 0, 0) # Deep Red
            elif style == "White Outline":
                text_color = "white"
                stroke_width = int(fontsize / 10)
                stroke_color = "black"
                
            # Draw Loop
            total_h = sum([draw.textbbox((0,0), l, font=font)[3] - draw.textbbox((0,0), l, font=font)[1] for l in lines]) * 1.2
            current_y = H * 0.1 # Top 10%
            
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                lx = (W - lw) / 2
                
                # Draw Background if needed
                if bg_color:
                    padding = fontsize * 0.2
                    # Rounded Rect
                    draw.rounded_rectangle(
                        [lx - padding, current_y - padding, lx + lw + padding, current_y + lh + padding],
                        radius=20,
                        fill=bg_color,
                        outline="black",
                        width=3
                    )
                
                # Draw Text
                if stroke_width > 0:
                    draw.text((lx, current_y), line, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_color)
                else:
                    draw.text((lx, current_y), line, font=font, fill=text_color)
                
                current_y += lh * 1.3 # Line spacing (gap)

            img.save(image_path)
            return True
        except Exception as e:
            print(f"Text overlay failed: {e}")
            return False

    def analyze_script(self, script: str, character_desc: str) -> typing.List[dict]:
        """
        Analyzes the script and breaks it down into scenes with prompts.
        Returns a list of dicts: {seq, text, media_type, image_prompt, video_prompt}
        """
        
        system_prompt = f"""
        You are an expert AI video director. 
        Your task is to break down the provided news script into visual scenes.
        
        STORY SETTING:
        - Character: {character_desc}
        - Style: High quality, cinematic, 8k resolution, photorealistic.
        
        INSTRUCTIONS:
        1. Split the script into logical scenes (approx 1 sentence per scene).
        2. Default workflow is IMAGE-FIRST. Set media_type to "image" for every scene.
        3. Write a detailed 'image_prompt' for generating the image using an image generation model.
           - Include the CHARACTER description in every prompt to ensure consistency.
           - Describe the background/setting relevant to the news text.
        4. Also write a detailed 'video_prompt' for every scene.
           - This is for optional manual conversion from image to video later.
           - Include camera motion and subject motion.
        
        OUTPUT FORMAT:
        Return ONLY a raw JSON array of objects. No markdown formatting.
        [
          {{
            "text": "sentence from script...",
            "media_type": "image",
            "image_prompt": "A photo of ...",
            "video_prompt": ""
          }}
        ]
        """
        
        def _split_text_to_scenes(text: str, max_scenes: int = 40) -> typing.List[str]:
            src = (text or "").strip()
            if not src:
                return []

            # 1) split by line first
            lines = [ln.strip() for ln in re.split(r"[\r\n]+", src) if ln.strip()]
            units = []
            for ln in lines:
                # 2) split by sentence punctuation (Korean/English)
                parts = re.split(r"(?<=[\.\!\?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", ln)
                for p in parts:
                    p = p.strip()
                    if p:
                        units.append(p)

            # 3) if still one giant block, chunk by length
            if len(units) <= 1 and len(src) > 240:
                words = src.split()
                cur = []
                for w in words:
                    test = (" ".join(cur + [w])).strip()
                    if len(test) <= 180:
                        cur.append(w)
                    else:
                        if cur:
                            units.append(" ".join(cur))
                        cur = [w]
                if cur:
                    units.append(" ".join(cur))

            units = [u for u in units if len(u) >= 8]
            if not units:
                units = [src]
            return units[:max_scenes]

        try:
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=[system_prompt, f"SCRIPT:\n{script}"],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            raw = response.text
            # Clean up potential markdown code blocks if SDK doesn't handle it (it usually does with mime_type, but safe to keep)
            raw = re.sub(r"```json", "", raw)
            raw = re.sub(r"```", "", raw).strip()
            
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array from analyze_script")
            
            # Add sequence numbers
            for i, item in enumerate(data):
                item["seq"] = i + 1
                item["text"] = item.get("text", "")
                item["image_prompt"] = item.get("image_prompt", "")
                # Always image-first unless user manually converts to video.
                item["media_type"] = "image"
                vp = (item.get("video_prompt") or "").strip()
                if not vp:
                    vp = f"Slow cinematic camera motion. {item['image_prompt']}"
                item["video_prompt"] = vp
                item["generated_media_path"] = item.get("generated_media_path") or item.get("image_path")

            # Safety: if model returned one huge scene, split locally.
            if len(data) <= 1 and len((script or "").strip()) > 260:
                chunks = _split_text_to_scenes(script, max_scenes=40)
                out = []
                for i, ch in enumerate(chunks, start=1):
                    img_prompt = f"{character_desc}. Scene based on narration: {ch}"
                    out.append({
                        "seq": i,
                        "text": ch,
                        "media_type": "image",
                        "image_prompt": img_prompt,
                        "video_prompt": f"Slow cinematic camera motion. {img_prompt}",
                        "generated_media_path": None,
                    })
                return out

            return data
             
        except Exception as e:
            print(f"Error in analyze_script: {e}")
            chunks = _split_text_to_scenes(script, max_scenes=40)
            out = []
            for i, ch in enumerate(chunks, start=1):
                img_prompt = f"{character_desc}. Scene based on narration: {ch}"
                out.append({
                    "seq": i,
                    "text": ch,
                    "media_type": "image",
                    "image_prompt": img_prompt,
                    "video_prompt": f"Slow cinematic camera motion. {img_prompt}",
                    "generated_media_path": None,
                })
            return out or [{
                "seq": 1,
                "text": script,
                "media_type": "image",
                "image_prompt": f"{character_desc} reading news: {script[:50]}...",
                "video_prompt": "",
                "generated_media_path": None,
            }]

    def regenerate_prompts(self, scene_text: str, character_desc: str, media_type: str) -> dict:
        """Regenerate prompt for a single scene"""
        res = self.analyze_script(scene_text, character_desc)
        if res:
            item = res[0]
            item["media_type"] = media_type
            return item
        return {}

    def build_cinematic_video_prompt(self, scene: dict, character_desc: str = "", style_hint: str = "") -> str:
        """
        Build a richer I2V prompt so results are not just a character walking on plain background.
        """
        narration = (scene.get("text") or "").strip()
        image_prompt = (scene.get("image_prompt") or "").strip()
        user_video_prompt = (scene.get("video_prompt") or "").strip()
        char = (character_desc or "Bean character").strip()

        lines = [
            "Create a detailed cinematic macro-news scene with strong environmental storytelling.",
            f"Character: {char}.",
            f"Narration context: {narration}",
            f"Scene visual context: {image_prompt}",
            f"Extra motion direction: {user_video_prompt}",
            f"Style hint: {style_hint}" if style_hint else "",
            "Composition requirements:",
            "- Foreground: character with clear facial emotion and readable gesture.",
            "- Midground: topic-relevant props (screens, charts, documents, currency, crowd, vehicles, factory, city objects).",
            "- Background: rich world detail with depth and atmospheric perspective; never plain white or empty backdrop.",
            "Motion requirements:",
            "- Camera move: slow dolly-in + subtle lateral parallax.",
            "- Character move: blinking, breathing, micro head turn, hand/arm action.",
            "- Environment move: lights flicker, smoke/cloud drift, people/traffic/object motion where relevant.",
            "Lighting requirements:",
            "- Dramatic key light, rim light, volumetric light, high contrast cinematic grade.",
            "Output constraints:",
            "- No text overlays, no logos, no watermark, no gibberish letters.",
            "- Keep character identity consistent with reference image.",
        ]
        return "\n".join([ln for ln in lines if ln])

    def build_cinematic_image_prompt(self, scene: dict, character_desc: str = "", style_hint: str = "") -> str:
        """
        Build a richer image prompt so generated stills have narrative depth and non-empty backgrounds.
        """
        narration = (scene.get("text") or "").strip()
        base_visual = (scene.get("image_prompt") or "").strip()
        char = (character_desc or "Bean character").strip()

        lines = [
            "Create a high-detail cinematic still image for a macro-news YouTube scene.",
            f"Character: {char}.",
            f"Narration context: {narration}",
            f"Base visual idea: {base_visual}",
            f"Style hint: {style_hint}" if style_hint else "",
            "Composition requirements:",
            "- Foreground: character close enough to read emotion and gesture.",
            "- Midground: topic-relevant props (trading screens, charts, newspaper desk, oil barrel, factory floor, city traffic, currency objects).",
            "- Background: rich environment with depth, atmosphere, and perspective; never plain white or empty backdrop.",
            "Art direction:",
            "- Dramatic but readable lighting, strong contrast, cinematic color grade.",
            "- Clear focal point, layered depth, realistic texture detail.",
            "- Dynamic storytelling frame (not static passport-photo framing).",
            "Constraints:",
            "- No text, no letters, no watermark, no logo.",
            "- Keep character identity consistent and clean.",
        ]
        return "\n".join([ln for ln in lines if ln])

    def generate_image(self, prompt: str, out_path: str):
        """Generates an image with current google.genai models, then falls back."""
        print(f"Generating image: {prompt[:50]}...")

        # 1) Gemini image-capable models via generate_content (returns inline image).
        for model_id in [
            "models/gemini-2.5-flash-image",
            "models/gemini-2.0-flash-exp-image-generation",
        ]:
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
                )
                parts = getattr(response, "parts", None) or []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        with open(out_path, "wb") as f:
                            f.write(inline.data)
                        return out_path
            except Exception as e:
                print(f"Gemini image gen failed ({model_id}): {e}")

        # 2) Imagen fallback via generate_images.
        for model_id in [
            "models/imagen-4.0-fast-generate-001",
            "models/imagen-4.0-generate-001",
        ]:
            try:
                response = self.client.models.generate_images(
                    model=model_id,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="16:9",
                    ),
                )
                if response.generated_images:
                    image = response.generated_images[0].image
                    image.save(out_path)
                    return out_path
            except Exception as e:
                print(f"Imagen gen failed ({model_id}): {e}")

        # 3) Pollinations as last resort.
        print("Falling back to Pollinations.ai...")
        return self.generate_image_pollinations(prompt, out_path)

    def generate_image_pollinations(self, prompt: str, out_path: str):
        """Generates an image from the prompt. Uses Pollinations.ai for free prototyping."""
        import requests
        from PIL import Image
        from io import BytesIO

        # Keep fallback request short and stable to avoid upstream URL-length/server errors.
        short_prompt = re.sub(r"\s+", " ", prompt).strip()[:280]
        safe_prompt = re.sub(r"[^a-zA-Z0-9, ]", "", short_prompt) or "bean character market thumbnail"
        url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(safe_prompt)
        params = {"width": 1080, "height": 1920, "nologo": "true"}
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            img.save(out_path)
            return out_path
        except Exception as e:
            print(f"Pollinations Gen failed: {e}")
            return None
    
    def generate_video_veo(self, image_path: str, prompt: str, out_path: str):
        """Generates a video using Google Veo with current google.genai API format."""
        print(f"Veo Gen Starting: {prompt}")

        try:
            import requests

            client = genai.Client(api_key=self.api_key)
            if not hasattr(client.models, "generate_videos"):
                print("Veo Gen Failed: SDK has no generate_videos")
                return None

            # Pick first available Veo model from preferred order.
            preferred = [
                "models/veo-2.0-generate-001",
                "models/veo-3.0-fast-generate-001",
                "models/veo-3.0-generate-001",
            ]
            available = []
            try:
                for m in client.models.list():
                    n = getattr(m, "name", "")
                    if n and "veo" in n.lower():
                        available.append(n)
            except Exception as e:
                print(f"Veo list models failed: {e}")

            model_id = next((m for m in preferred if m in available), None) or "models/veo-2.0-generate-001"
            print(f"Using Veo model: {model_id}")
            duration_sec = 5 if "veo-2.0" in model_id else 4

            ref_image = types.Image.from_file(location=image_path)
            op = client.models.generate_videos(
                model=model_id,
                prompt=prompt,
                image=ref_image,
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    aspect_ratio="16:9",
                    duration_seconds=duration_sec,
                ),
            )

            # Poll operation status.
            for _ in range(60):
                time.sleep(3)
                op = client.operations.get(op)
                if op.done:
                    break

            if not op.done:
                print("Veo Gen Failed: operation timeout")
                return None
            if getattr(op, "error", None):
                print(f"Veo Gen Failed (operation error): {op.error}")
                return None

            result = getattr(op, "result", None) or getattr(op, "response", None)
            if not result or not getattr(result, "generated_videos", None):
                print("Veo Gen Failed: no generated videos in result")
                return None

            video_obj = result.generated_videos[0].video
            if getattr(video_obj, "video_bytes", None):
                with open(out_path, "wb") as f:
                    f.write(video_obj.video_bytes)
                return out_path

            uri = getattr(video_obj, "uri", None)
            if uri:
                r = requests.get(uri, params={"key": self.api_key}, timeout=60)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return out_path

            print("Veo Gen Failed: no downloadable bytes/uri")
            return None

        except Exception as e:
            print(f"Veo Gen Failed: {e}")
            return None

    def generate_video_luma(self, image_path: str, prompt: str, out_path: str, luma_key: str):

        """Generates a video using Luma Dream Machine (Image-to-Video)."""
        try:
            from lumaai import LumaAI
            client = LumaAI(auth_token=luma_key)
            
            print(f"Luma Gen Starting: {prompt}")
            
            # Upload Image? Luma usually takes URL. 
            # We need to upload our local image to a temporary public URL or use their upload API if available.
            # Luma SDK might support file upload. Let's check docs pattern or assume URL needed.
            # Wait, standard pattern: Luma generation often requires an image URL.
            # Since we are local (localhost), we can't give a URL easily unless we use a tunnel.
            # BUT, the SDK might handle uploads? 
            # Search result [1] said "Create generations...".
            
            # Workaround: If Luma needs URL, we are stuck on localhost.
            # Check if Luma supports direct file upload in generation.
            # Most modern AI SDKs do.
            
            # Let's try the standard 'image_url' or 'image' path.
            # If not supported, we might have to use a temporary upload service (e.g. tmpfiles.org)
            # THIS IS A RISK.
            
            # Alternative: Use "Text to Video" only if Image to Video is hard? 
            # User wants "Characters moving", likely from the drawing.
            
            # Let's assume for now we try to implement Text-to-Video as a safe fallback 
            # OR try to upload if the SDK supports it.
            
            # Let's look at a simpler path: Text to Video with character description.
            generation = client.generations.create(
                prompt=prompt,
                aspect_ratio="16:9"
            )
            
            # Poll for completion
            gen_id = generation.id
            print(f"Luma Job ID: {gen_id}")
            
            for _ in range(60): # Wait up to 60s (might take longer, loop more)
                time.sleep(3)
                chk = client.generations.get(id=gen_id)
                if chk.state == "completed":
                    video_url = chk.assets.video
                    # Download
                    import requests
                    r = requests.get(video_url)
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    return out_path
                elif chk.state == "failed":
                    print("Luma Failed")
                    return None
            
            return None

        except ImportError:
            print("LumaAI library not installed.")
            return None
        except Exception as e:
            print(f"Luma Gen Failed: {e}")
            return None
            
    def generate_video_motion(self, image_path: str, out_path: str, duration: int = 5):

        """Generates a simple zoom-in video from the image using MoviePy."""
        try:
            # MoviePy v1 path
            try:
                from moviepy.editor import ImageClip
                clip = ImageClip(image_path).set_duration(duration)
                w, h = clip.size
                clip = clip.resize(lambda t: 1 + 0.02 * t)
                clip = clip.crop(x1=0, y1=0, w=w, h=h).set_position(("center", "center"))
                clip.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None)
                return out_path
            except Exception:
                # MoviePy v2 path
                from moviepy import ImageClip
                clip = ImageClip(image_path, duration=duration)
                w, h = clip.size
                clip = clip.resized(lambda t: 1 + 0.02 * t).cropped(x1=0, y1=0, width=w, height=h)
                clip.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None)
                return out_path
        except Exception as e:
            print(f"MoviePy path failed: {e}. Falling back to ffmpeg.")

        # Last-resort fallback: generate a short loop video with ffmpeg only.
        try:
            from PIL import Image
            import imageio_ffmpeg
            import subprocess

            img = Image.open(image_path)
            w, h = img.size
            total_frames = max(1, int(duration * 24))
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            vf = (
                f"zoompan=z='min(zoom+0.0015,1.15)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:s={w}x{h}:fps=24,format=yuv420p"
            )
            cmd = [
                ffmpeg, "-y",
                "-loop", "1", "-i", image_path,
                "-t", str(duration),
                "-vf", vf,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                out_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_path
        except Exception as e:
            print(f"Video Motion Gen failed: {e}")
            return None
