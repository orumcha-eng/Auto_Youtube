import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly",
]

CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"


def get_youtube_client():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_private(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    yt = get_youtube_client()

    body = {
        "snippet": {
            "title": "TEST - Macro Minute Brief (Private)",
            "description": "Test upload from automation.\n\nNot financial advice.",
            "tags": ["macro", "market", "test"],
            "categoryId": "25",  # News & Politics
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, chunksize=5 * 1024 * 1024, resumable=True)
    request = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    print("\n✅ Upload complete!")
    print("videoId:", response["id"])
    return response["id"]


if __name__ == "__main__":
    # 방법 1) 프로젝트 폴더에 test.mp4 넣었으면:
    upload_private(r".\test.mp4")

    # 방법 2) 다른 경로면 아래처럼:
    # upload_private(r"D:\path\to\test.mp4")
