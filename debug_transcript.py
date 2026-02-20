from youtube_transcript_api import YouTubeTranscriptApi
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    print(dir(YouTubeTranscriptApi))
except Exception as e:
    print(f"Error: {e}")
