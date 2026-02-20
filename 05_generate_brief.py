import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict, Tuple
from urllib.parse import quote_plus

import requests
import feedparser
from dateutil import parser as dtparser
from icalendar import Calendar

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---- Official schedules (no API key) ----
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
BEA_RELEASE_DATES_JSON_URL = "https://apps.bea.gov/API/signup/release_dates.json"

# ---- Headline feeds (optional but useful) ----
# Fed RSS list is on the official page; these URLs usually work from a normal browser/python requests.
FED_RSS_FEEDS: List[Tuple[str, str]] = [
    ("Fed — Monetary Policy Press Releases", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("Fed — Chair Powell (Speeches/Testimony)", "https://www.federalreserve.gov/feeds/s_t_powell.xml"),
]

# Google News RSS (unofficial but widely used); you can remove if you dislike it.
GOOGLE_NEWS_RSS_TEMPLATE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MacroMinuteBrief/1.0"

@dataclass
class Headline:
    source: str
    title: str
    link: str
    published_utc: Optional[str] = None

@dataclass
class EventItem:
    source: str
    title: str
    time_et: Optional[str] = None  # "HH:MM"
    datetime_et: Optional[str] = None  # ISO
    link: Optional[str] = None

def http_get(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    return r.content

def to_et(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # assume ET if timezone missing
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)

def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()

def iso_et(dt: datetime) -> str:
    return to_et(dt).isoformat()

def hhmm_et(dt: datetime) -> str:
    d = to_et(dt)
    return d.strftime("%H:%M")

def normalize_title(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fetch_bls_events(target_date_et: date) -> List[EventItem]:
    cal_bytes = http_get(BLS_ICS_URL)
    cal = Calendar.from_ical(cal_bytes)

    items: List[EventItem] = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        summary = str(component.get("summary", "")).strip()
        dtstart = component.get("dtstart")
        if not dtstart:
            continue

        dt = dtstart.dt
        if isinstance(dt, date) and not isinstance(dt, datetime):
            # all-day event; treat as midnight ET
            dt = datetime(dt.year, dt.month, dt.day, 0, 0, tzinfo=ET)
        elif isinstance(dt, datetime):
            dt = to_et(dt)

        if dt.date() != target_date_et:
            continue

        items.append(EventItem(
            source="BLS",
            title=normalize_title(summary),
            time_et=hhmm_et(dt),
            datetime_et=iso_et(dt),
            link="https://www.bls.gov/schedule/"
        ))

    # sort by time
    items.sort(key=lambda x: x.time_et or "99:99")
    return items

def fetch_bea_events(target_date_et: date) -> List[EventItem]:
    raw = http_get(BEA_RELEASE_DATES_JSON_URL)
    data = json.loads(raw.decode("utf-8", errors="ignore"))

    items: List[EventItem] = []
    for release_name, payload in data.items():
        if not isinstance(payload, dict):
            continue
        dates = payload.get("release_dates", [])
        if not isinstance(dates, list):
            continue

        for dstr in dates:
            try:
                dt_utc = dtparser.isoparse(dstr)  # usually has +00:00
                dt_et = to_et(dt_utc)
            except Exception:
                continue

            if dt_et.date() != target_date_et:
                continue

            items.append(EventItem(
                source="BEA",
                title=normalize_title(release_name),
                time_et=dt_et.strftime("%H:%M"),
                datetime_et=dt_et.isoformat(),
                link="https://www.bea.gov/news/schedule"
            ))

    # dedupe (some BEA items can be duplicated)
    dedup: Dict[str, EventItem] = {}
    for it in items:
        key = f"{it.title}|{it.datetime_et}"
        dedup[key] = it

    out = list(dedup.values())
    out.sort(key=lambda x: x.time_et or "99:99")
    return out

def fetch_rss_headlines(url: str, source_name: str, hours_back: int) -> List[Headline]:
    # feedparser can parse bytes directly
    content = http_get(url)
    feed = feedparser.parse(content)

    cutoff_utc = datetime.now(tz=UTC) - timedelta(hours=hours_back)
    items: List[Headline] = []

    for e in feed.entries[:50]:
        title = normalize_title(getattr(e, "title", "") or "")
        link = getattr(e, "link", "") or ""
        published_utc = None

        # best-effort date parsing
        dt = None
        if hasattr(e, "published"):
            try:
                dt = dtparser.parse(e.published)
            except Exception:
                dt = None
        if dt is None and hasattr(e, "updated"):
            try:
                dt = dtparser.parse(e.updated)
            except Exception:
                dt = None

        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            dt_utc = dt.astimezone(UTC)
            if dt_utc < cutoff_utc:
                continue
            published_utc = dt_utc.isoformat()

        if title and link:
            items.append(Headline(source=source_name, title=title, link=link, published_utc=published_utc))

    return items

def fetch_google_news(query: str, hours_back: int) -> List[Headline]:
    url = GOOGLE_NEWS_RSS_TEMPLATE.format(q=quote_plus(query))
    return fetch_rss_headlines(url, f"Google News: {query}", hours_back)

def generate_script(date_et: date, headlines: List[Headline], events: List[EventItem]) -> str:
    # pick top headlines (limit 3)
    top_h = headlines[:3]

    # pick top events (limit 4)
    top_e = events[:4]

    dow = datetime(date_et.year, date_et.month, date_et.day, tzinfo=ET).strftime("%A")
    month_day = datetime(date_et.year, date_et.month, date_et.day, tzinfo=ET).strftime("%B %d")
    lines: List[str] = []

    lines.append(f"Good morning — {dow}, {month_day}. This is Macro Minute Brief.")
    if top_h:
        lines.append("Overnight highlights:")
        for i, h in enumerate(top_h, 1):
            # Keep it TTS-friendly and short
            t = re.sub(r"[\[\]•|]+", " ", h.title)
            lines.append(f"{i}. {t}.")
    else:
        lines.append("Overnight was relatively quiet on major macro headlines.")

    if top_e:
        lines.append("Today’s key data and events (Eastern Time):")
        for e in top_e:
            if e.time_et:
                lines.append(f"- {e.time_et} — {e.title}.")
            else:
                lines.append(f"- {e.title}.")
    else:
        lines.append("No major scheduled U.S. releases on the official calendars today.")

    lines.append("That’s the macro setup — trade safe, and I’ll see you tomorrow.")

    # TTS pacing: ensure not too long
    script = " ".join(lines)
    # Hard trim if needed (roughly)
    words = script.split()
    if len(words) > 165:
        script = " ".join(words[:165]) + "…"
    return script

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-et", help="Target date in ET, e.g. 2026-01-06. Default: today ET")
    ap.add_argument("--hours-back", type=int, default=18, help="How many hours back to look for headlines")
    ap.add_argument("--no-google-news", action="store_true", help="Disable Google News RSS")
    ap.add_argument("--google-query", default="US economy market futures Fed CPI", help="Google News query")
    ap.add_argument("--out", default="out", help="Output folder")
    args = ap.parse_args()

    now_et = datetime.now(tz=ET)
    if args.date_et:
        target = dtparser.isoparse(args.date_et).date()
    else:
        target = now_et.date()

    # 1) schedules
    bls = fetch_bls_events(target)
    bea = fetch_bea_events(target)
    events = bls + bea
    events.sort(key=lambda x: (x.time_et or "99:99", x.source, x.title))

    # 2) headlines
    headlines: List[Headline] = []
    for name, url in FED_RSS_FEEDS:
        try:
            headlines.extend(fetch_rss_headlines(url, name, args.hours_back))
        except Exception as e:
            print(f"[WARN] RSS failed: {name} ({url}) -> {e}")

    if not args.no_google_news:
        try:
            headlines.extend(fetch_google_news(args.google_query, args.hours_back))
        except Exception as e:
            print(f"[WARN] Google News RSS failed -> {e}")

    # sort headlines by published_utc desc when available; otherwise keep order
    def h_key(h: Headline):
        if h.published_utc:
            try:
                return dtparser.isoparse(h.published_utc)
            except Exception:
                pass
        return datetime(1970, 1, 1, tzinfo=UTC)

    headlines.sort(key=h_key, reverse=True)

    # 3) script
    script = generate_script(target, headlines, events)

    # 4) write output
    out_obj = {
        "brand": "Macro Minute Brief",
        "target_date_et": target.isoformat(),
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "headlines": [asdict(x) for x in headlines[:10]],
        "events": [asdict(x) for x in events],
        "script": script,
    }

    import os
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"brief_{target.isoformat()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # print summary for quick debug
    print(f"\n=== Macro Minute Brief (ET) {target.isoformat()} ===")
    print("\n[Top Headlines]")
    for h in headlines[:5]:
        print(f"- ({h.source}) {h.title}")

    print("\n[Today Events]")
    for e in events[:10]:
        t = e.time_et or "--:--"
        print(f"- {t} ({e.source}) {e.title}")

    print("\n[Script]")
    print(script)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
