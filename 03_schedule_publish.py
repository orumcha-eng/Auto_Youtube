import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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

    return build("youtube", "v3", credentials=creds)


def next_us_morning_publish_at(et_hour=8, et_minute=0, buffer_minutes=20) -> str:
    """
    다음 미국 동부시간(ET) 기준 et_hour:et_minute 에 공개되도록 publishAt(UTC ISO8601) 문자열 생성.
    buffer_minutes: 지금 시간이 이미 너무 가까우면 다음날로 넘기기 위한 안전 버퍼.
    """
    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)

    target_et = now_et.replace(hour=et_hour, minute=et_minute, second=0, microsecond=0)
    # target이 이미 지났거나, 너무 촉박하면 다음날로
    if target_et <= now_et + timedelta(minutes=buffer_minutes):
        target_et = target_et + timedelta(days=1)

    # YouTube API는 RFC3339(ISO8601) 형식, UTC로 변환해서 Z로
    target_utc = target_et.astimezone(timezone.utc)
    return target_utc.isoformat().replace("+00:00", "Z")


def schedule_publish(video_id: str, publish_at_utc: str):
    yt = get_youtube_client()

    # status 업데이트 (private + publishAt)
    body = {
        "id": video_id,
        "status": {
            "privacyStatus": "private",  # publishAt 설정에 필요
            "publishAt": publish_at_utc,
            "selfDeclaredMadeForKids": False,
        },
    }

    resp = yt.videos().update(part="status", body=body).execute()

    print("\n✅ Scheduled!")
    print("videoId:", resp["id"])
    print("privacyStatus:", resp["status"]["privacyStatus"])
    print("publishAt(UTC):", resp["status"].get("publishAt"))


if __name__ == "__main__":
    # 방금 업로드한 videoId
    vid = "QamT__03sEU"

    # 기본: ET 08:00 공개, 버퍼 20분
    publish_at = next_us_morning_publish_at(et_hour=8, et_minute=0, buffer_minutes=20)
    print("Will schedule publishAt(UTC):", publish_at)

    schedule_publish(vid, publish_at)
