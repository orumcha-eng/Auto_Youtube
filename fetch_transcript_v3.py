from youtube_transcript_api import YouTubeTranscriptApi
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
video_id = "o7x-Y940VJU"

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    
    # Try to find 'ko' or 'en' first
    t = None
    try:
        t = transcript_list.find_transcript(['ko', 'en'])
    except:
        # If not, take the first available one
        # convert to list first just in case it's an iterator
        ts = list(transcript_list)
        if ts:
            t = ts[0]
            
    if t:
        fetched = t.fetch()
        language_code = t.language_code
        
        full_text = " ".join([entry['text'] for entry in fetched])
        
        print(json.dumps({
            "success": True, 
            "text": full_text[:100000], # Limit char count just in case
            "language": language_code,
            "is_generated": t.is_generated
        }, ensure_ascii=False))
    else:
        print(json.dumps({"success": False, "error": "No transcripts found at all."}, ensure_ascii=False))

except Exception as e:
    import traceback
    print(json.dumps({"success": False, "error": str(e), "trace": traceback.format_exc()}, ensure_ascii=False))
