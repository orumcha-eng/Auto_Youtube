from youtube_transcript_api import YouTubeTranscriptApi
import json
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

video_id = "o7x-Y940VJU"

try:
    # Try to fetch transcript in Korean, fallback to English
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
    
    # Combine text
    full_text = " ".join([entry['text'] for entry in transcript])
    
    # Return as JSON
    print(json.dumps({"success": True, "text": full_text}, ensure_ascii=False))
    
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}, ensure_ascii=False))
