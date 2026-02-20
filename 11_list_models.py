
import os
import google.generativeai as genai_legacy
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print("--- Checking Legacy SDK Models ---")
try:
    genai_legacy.configure(api_key=api_key)
    for m in genai_legacy.list_models():
        if 'generateContent' in m.supported_generation_methods or 'generateImage' in m.supported_generation_methods:
            print(f"Legacy Model: {m.name} | Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Legacy List Failed: {e}")

print("\n--- Checking New SDK Models ---")
try:
    client = genai.Client(api_key=api_key)
    # The new SDK list_models might need iteration or different call
    # Checking client.models.list() if it exists or similar
    # Based on docs it might be client.models.list()
    for m in client.models.list():
        print(f"New Model: {m.name} | Display: {m.display_name}")
except Exception as e:
    print(f"New SDK List Failed: {e}")
