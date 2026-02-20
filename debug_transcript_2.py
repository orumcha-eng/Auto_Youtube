from youtube_transcript_api import YouTubeTranscriptApi
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("DIR:", dir(YouTubeTranscriptApi))
try:
    print("HELP list:")
    help(YouTubeTranscriptApi.list_transcripts)
except:
    print("No list_transcripts")

try:
    print("HELP get_transcript:")
    help(YouTubeTranscriptApi.get_transcript) 
except:
    print("No get_transcript")
