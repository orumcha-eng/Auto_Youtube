# 08_generate_character_candidates.py
import os
from src.visual_gen import VisualGenerator
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # Try to find key in app_v2.py logic or just ask user if missing
    # For now assume env is set or we fail
    print("API KEY not found in env.")

vg = VisualGenerator(api_key)

# Prompts trying to capture "Viral Stick Figure / Bean" style
prompts = [
    "A cute simple minimalist long bean-shaped character standing, thick black marker lines, white background, hand-drawn stick figure style, cartoon, expressive big eyes, no shading, flat 2d",
    "A long pill-shaped character with stick arms and legs, simple doodle style, black ink on white paper, funny expression, minimalist viral youtube animation style",
    "A white blob character, bean shape, tall, simple stick limbs, cute face, vector art, thick outlines, isolated on white",
    "Hand drawn sketch of a tall bean man, simple eyes and mouth, stick figure arms, very simple, funny, viral animation style"
]

os.makedirs("assets/candidates", exist_ok=True)

print("Generating character candidates...")
for i, p in enumerate(prompts):
    path = f"assets/candidates/char_candidate_{i+1}.png"
    print(f"Gen {i+1}: {p}")
    vg.generate_image(p, path)
    print(f"Saved to {path}")

print("Done. Check assets/candidates folder.")
