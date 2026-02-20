
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def test_gemini_video():
    client = genai.Client(api_key=api_key)
    print("Attempting to generate video with gemini-2.0-flash-exp...")
    
    try:
        # Check if generate_videos exists or similar
        # Based on docs, it might be separate tool or model
        # Let's try standard generate_content first with "generate a video" prompt
        # But usually specific method is needed for file output
        
        # Hypothetical method check
        if hasattr(client.models, 'generate_videos'):
            print("Client has generate_videos method!")
            response = client.models.generate_videos(
                model='gemini-2.0-flash-exp', 
                prompt='A cute white bean character waving hand, 4 seconds, high quality',
                config=types.GenerateVideosConfig(number_of_videos=1)
            )
            print("Video Gen Success?")
            print(response)
        else:
            print("No generate_videos method found in NEW SDK.")
            
    except Exception as e:
        print(f"New SDK Video failed: {e}")

if __name__ == "__main__":
    if not api_key:
        print("No API Key")
    else:
        test_gemini_video()
