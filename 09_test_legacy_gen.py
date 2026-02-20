# 09_test_legacy_gen.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("No key")
    exit()

genai.configure(api_key=api_key)

prompt = "A cute simple minimalist long bean-shaped character standing, thick black marker lines, white background, hand-drawn stick figure style, cartoon, expressive big eyes, no shading, flat 2d"

print("Trying legacy generate_content with image model...")
try:
    # Some legacy models use generate_content for images too? No, usually it's different.
    # But gemini-2.5-flash-image might work with generate_content?
    # Or maybe it is not supported in the old SDK python wrapper for inference yet?
    # Let's try the standard way if it exists, otherwise prompt specifically.
    
    # Actually, for standard Gemini 1.5/2.0, image generation is via tools or specific methods.
    # But let's try a simple GenerateContent with text if it supports creation? 
    # No, usually need 'imagen' model.
    # Let's try to find an 'imagen' model in the list list again.
    
    # The list had: models/gemini-2.5-flash-image
    model = genai.GenerativeModel("models/gemini-2.5-flash-image")
    response = model.generate_content(prompt)
    
    # Check if it has parts with image
    print("Response received.")
    if response.parts:
        print("Parts found.")
        # Usually it returns an image object?
        # If it returns text, it failed.
        print(response.text)
    else:
        print("No text? Check for images.")
        # This wrapper might not handle image binary return well if it's raw.
        # But let's inspect response.
        print(response)

except Exception as e:
    print(f"Error: {e}")
