from youtube_transcript_api import YouTubeTranscriptApi
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

video_id = "o7x-Y940VJU"

try:
    # Instantiate the API class
    api = YouTubeTranscriptApi()
    
    # Fetch transcript
    fetched = api.fetch(video_id, languages=['ko', 'en'])
    
    full_text = " ".join([entry['text'] for entry in fetched])
    print(json.dumps({"success": True, "text": full_text}, ensure_ascii=False))
    
except Exception as e:
    import traceback
    print(json.dumps({"success": False, "error": str(e), "trace": traceback.format_exc()}, ensure_ascii=False))
