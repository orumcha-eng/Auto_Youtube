import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

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

    # 현재 토큰에 실제로 들어간 스코프 출력(디버깅용)
    print("Token scopes:", getattr(creds, "scopes", None))

    return build("youtube", "v3", credentials=creds)


def main():
    yt = get_youtube_client()

    mine = yt.channels().list(part="snippet,contentDetails", mine=True).execute()
    items = mine.get("items", [])

    print("\n=== channels.list(mine=True) ===")
    if not items:
        print("No channel returned. (이 계정에 유튜브 채널이 없거나 권한 문제가 있을 수 있어요.)")
        return

    for ch in items:
        title = ch["snippet"]["title"]
        channel_id = ch["id"]
        uploads_pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
        print(f"- title: {title}")
        print(f"  channelId: {channel_id}")
        print(f"  uploadsPlaylistId: {uploads_pl}")

    print("\n✅ 여기 출력된 title이 'Macro Minute Brief'면 채널 잡힌 거고, 다음 단계로 업로드 테스트 진행하면 됩니다.")


if __name__ == "__main__":
    main()
