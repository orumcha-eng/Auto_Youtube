import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Version:", getattr(youtube_transcript_api, '__version__', 'unknown'))
print("File:", getattr(youtube_transcript_api, '__file__', 'unknown'))

try:
    print("\nHELP list:")
    help(YouTubeTranscriptApi.list)
except Exception as e:
    print(f"No help for list: {e}")

try:
    print("\nHELP fetch:")
    help(YouTubeTranscriptApi.fetch) 
except Exception as e:
    print(f"No help for fetch: {e}")
