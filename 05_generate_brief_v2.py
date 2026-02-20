# 05_generate_brief_v2.py
# -*- coding: utf-8 -*-

import argparse
import calendar
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from icalendar import Calendar

# ---- Timezone helpers (Windows-safe) ----
def get_tz(name: str):
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return ZoneInfo(name)
    except Exception:
        # fallback
        import pytz
        return pytz.timezone(name)

TZ_ET = get_tz("America/New_York")
TZ_UTC = get_tz("UTC")

UA = {
    "User-Agent": "Mozilla/5.0 (MacroMinuteBrief/1.0; +https://example.com) AppleWebKit/537.36"
}

# ---- RSS / Calendar sources ----
GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_BUSINESS = "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"

# BEA provides an official ICS subscription calendar
BEA_ICS_URL = "https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics"

# Fed Board monthly calendar pages (e.g., https://www.federalreserve.gov/newsevents/2026-january.htm)
FED_CAL_URL_TMPL = "https://www.federalreserve.gov/newsevents/{year}-{monthname}.htm"


@dataclass
class Headline:
    title: str
    source: str
    link: str
    published_utc: Optional[str] = None


@dataclass
class EventItem:
    title: str
    time_et: Optional[str]  # "HH:MM" or None
    source: str             # "FED", "BEA"
    details: Optional[str] = None


def now_et() -> datetime:
    return datetime.now(TZ_ET)


def parse_time_ampm_to_24h(s: str) -> Optional[str]:
    # examples: "4:10 p.m.", "9:15 a.m."
    s = s.strip().lower()
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.)$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ap = m.group(3)
    if ap == "p.m." and hh != 12:
        hh += 12
    if ap == "a.m." and hh == 12:
        hh = 0
    return f"{hh:02d}:{mm:02d}"


def safe_get(url: str, timeout: int = 25) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text


# ---------------- Headlines ----------------
def fetch_google_headlines(hours_back: int, query: str, debug: bool = False) -> List[Headline]:
    cutoff = datetime.now(TZ_UTC) - timedelta(hours=hours_back)

    def pull(url: str) -> List[Headline]:
        feed = feedparser.parse(url)
        out: List[Headline] = []
        for e in getattr(feed, "entries", []) or []:
            title_raw = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            published = None
            published_dt = None

            # published_parsed may exist
            if e.get("published"):
                try:
                    published_dt = dtparser.parse(e["published"])
                except Exception:
                    published_dt = None
            if published_dt is not None and published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=TZ_UTC)

            if published_dt is not None:
                published = published_dt.astimezone(TZ_UTC).isoformat()
                if published_dt.astimezone(TZ_UTC) < cutoff:
                    continue

            # Google News title is often "headline - Source"
            source = "Google News"
            if " - " in title_raw:
                parts = title_raw.rsplit(" - ", 1)
                if len(parts) == 2:
                    title_raw, source = parts[0].strip(), parts[1].strip()

            if title_raw and link:
                out.append(Headline(title=title_raw, source=source, link=link, published_utc=published))
        return out

    url_q = GOOGLE_NEWS_SEARCH.format(q=requests.utils.quote(query))
    h = pull(url_q)
    if debug:
        print(f"[debug] Google search feed: {url_q}")
        print(f"[debug] headlines(search) = {len(h)} within last {hours_back}h")

    # fallback to Business topic feed if empty
    if not h:
        h = pull(GOOGLE_NEWS_BUSINESS)
        if debug:
            print(f"[debug] fallback business feed: {GOOGLE_NEWS_BUSINESS}")
            print(f"[debug] headlines(business) = {len(h)} within last {hours_back}h")

    # de-dup by title
    seen = set()
    uniq = []
    for x in h:
        key = x.title.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)

    return uniq[:8]


# ---------------- BEA ICS events ----------------
def fetch_bea_events_for_date(target: date, debug: bool = False) -> List[EventItem]:
    try:
        ics_text = safe_get(BEA_ICS_URL)
    except Exception as e:
        if debug:
            print(f"[debug] BEA ICS fetch failed: {e}")
        return []

    cal = Calendar.from_ical(ics_text)
    items: List[EventItem] = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        summary = str(component.get("SUMMARY", "")).strip()
        dtstart = component.get("DTSTART")
        if not summary or not dtstart:
            continue

        dt = dtstart.dt
        if isinstance(dt, datetime):
            dt_et = dt.astimezone(TZ_ET)
            if dt_et.date() != target:
                continue
            t_et = dt_et.strftime("%H:%M")
        else:
            # DATE only
            if dt != target:
                continue
            t_et = None

        items.append(EventItem(title=summary, time_et=t_et, source="BEA"))

    # sort by time
    items.sort(key=lambda x: x.time_et or "99:99")
    return items


# ---------------- FED monthly calendar events ----------------
def fetch_fed_events_for_date(target: date, debug: bool = False) -> List[EventItem]:
    monthname = calendar.month_name[target.month].lower()
    url = FED_CAL_URL_TMPL.format(year=target.year, monthname=monthname)

    try:
        html = safe_get(url)
    except Exception as e:
        if debug:
            print(f"[debug] FED calendar fetch failed: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Track which section we are in
    sections = {"Speeches", "Testimony", "FOMC Meetings", "Beige Book", "Statistical Releases", "Other"}
    current_section = None

    # We parse blocks ending with a day list line like: "2, 8, 15, 22, 29" or "18"
    day_line_re = re.compile(r"^\d{1,2}(\s*,\s*\d{1,2})*$")

    # collect raw blocks: (section, block_lines, day_nums)
    blocks: List[Tuple[str, List[str], List[int]]] = []

    i = 0
    while i < len(lines):
        ln = lines[i]

        if ln in sections:
            current_section = ln
            i += 1
            continue

        if current_section is None:
            i += 1
            continue

        # Start of an event block:
        # - can start with a time ("4:10 p.m.") OR a title line ("Holiday - ...") OR a release title line.
        # We'll read forward until we hit a day line.
        block = [ln]
        j = i + 1
        while j < len(lines) and not day_line_re.match(lines[j]):
            # stop if we enter another section header abruptly
            if lines[j] in sections:
                break
            block.append(lines[j])
            j += 1

        # If we found a day line, attach it and create a block
        if j < len(lines) and day_line_re.match(lines[j]):
            days = [int(x.strip()) for x in lines[j].split(",")]
            blocks.append((current_section, block, days))
            i = j + 1
        else:
            i += 1

    # Filter blocks for target day
    day = target.day
    picked: List[EventItem] = []
    for section, block, days in blocks:
        if day not in days:
            continue

        # Turn block into EventItem
        # Heuristic:
        # if first line is time -> next line is title
        t24 = parse_time_ampm_to_24h(block[0])
        if t24:
            title = block[1] if len(block) > 1 else "Fed event"
            details = " ".join(block[2:]).strip() if len(block) > 2 else None
            item = EventItem(title=title, time_et=t24, source="FED", details=(details if details else None))
        else:
            # no explicit time; treat first line as title
            title = block[0]
            details = " ".join(block[1:]).strip() if len(block) > 1 else None
            item = EventItem(title=title, time_et=None, source="FED", details=(details if details else None))

        # Keep only "major-ish" items:
        major_sections = {"Speeches", "Testimony", "FOMC Meetings", "Beige Book"}
        major_stats_keywords = [
            "Industrial Production", "Consumer Credit", "Financial Accounts", "Z.1", "SLOOS", "Senior Loan Officer"
        ]
        if section in major_sections:
            picked.append(item)
        elif section == "Statistical Releases":
            if any(k.lower() in (item.title or "").lower() for k in major_stats_keywords):
                picked.append(item)
        elif section == "Other":
            # holidays can matter, keep
            if "Holiday" in (item.title or ""):
                picked.append(item)

    picked.sort(key=lambda x: x.time_et or "99:99")
    if debug:
        print(f"[debug] FED calendar url: {url}")
        print(f"[debug] FED picked events = {len(picked)} for {target.isoformat()}")
    return picked


def next_key_events(within_days: int, from_date: date, debug: bool = False) -> List[EventItem]:
    # Look ahead and pick a few upcoming major events
    upcoming: List[Tuple[date, EventItem]] = []
    for d in range(1, within_days + 1):
        day = from_date + timedelta(days=d)
        for e in fetch_fed_events_for_date(day, debug=False):
            upcoming.append((day, e))
        for e in fetch_bea_events_for_date(day, debug=False):
            upcoming.append((day, e))

    # prioritize FED FOMC/Beige Book/speeches and BEA 8:30 releases
    def score(item: EventItem) -> int:
        t = (item.title or "").lower()
        if "fomc" in t or "beige book" in t or "minutes" in t:
            return 100
        if item.source == "BEA":
            return 80
        if item.source == "FED":
            return 60
        return 10

    upcoming.sort(key=lambda x: (x[0], -(score(x[1])), x[1].time_et or "99:99"))
    out: List[EventItem] = []
    seen = set()
    for day, e in upcoming:
        key = (day.isoformat(), e.source, e.title)
        if key in seen:
            continue
        seen.add(key)
        # embed date into details for display
        e2 = EventItem(
            title=e.title,
            time_et=e.time_et,
            source=e.source,
            details=f"{day.strftime('%a %b %d')} ET" + (f" · {e.details}" if e.details else "")
        )
        out.append(e2)
        if len(out) >= 3:
            break
    if debug:
        print(f"[debug] upcoming key events = {len(out)}")
    return out


def build_script(target: date, headlines: List[Headline], events: List[EventItem], upcoming: List[EventItem]) -> str:
    # Keep it ~1 minute, TTS-friendly
    d = datetime(target.year, target.month, target.day)
    weekday = d.strftime("%A")
    monthday = d.strftime("%B %d")

    top = headlines[:3]
    lines = []
    lines.append(f"Good morning — {weekday}, {monthday}. This is Macro Minute Brief.")
    if top:
        lines.append("Overnight headlines:")
        for h in top:
            lines.append(f"- {h.title}.")
    else:
        lines.append("Overnight was quiet on major macro headlines across our sources.")

    if events:
        lines.append("Today’s key events:")
        for e in events[:4]:
            when = f"{e.time_et} ET" if e.time_et else "Time TBA"
            lines.append(f"- {when}: {e.title}.")
    else:
        lines.append("Today’s top-tier macro calendar is light.")
        if upcoming:
            lines.append("Next on deck:")
            for e in upcoming:
                when = f"{e.time_et} ET" if e.time_et else "Time TBA"
                # e.details already contains date label
                date_lbl = e.details or ""
                lines.append(f"- {date_lbl}, {when}: {e.title}.")

    lines.append("That’s the setup — trade safe, and I’ll see you tomorrow.")
    return " ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-et", help="YYYY-MM-DD in US/Eastern (default: today ET)")
    ap.add_argument("--hours-back", type=int, default=18)
    ap.add_argument("--query", default="US stocks futures Treasury yields Fed CPI earnings",
                    help="Google News search query")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.date_et:
        target = datetime.strptime(args.date_et, "%Y-%m-%d").date()
    else:
        target = now_et().date()

    headlines = fetch_google_headlines(args.hours_back, args.query, debug=args.debug)

    events: List[EventItem] = []
    events += fetch_fed_events_for_date(target, debug=args.debug)
    events += fetch_bea_events_for_date(target, debug=args.debug)

    # de-dup events by (source,title,time)
    seen = set()
    uniq_events = []
    for e in events:
        k = (e.source, e.title, e.time_et)
        if k in seen:
            continue
        seen.add(k)
        uniq_events.append(e)
    uniq_events.sort(key=lambda x: x.time_et or "99:99")
    events = uniq_events

    upcoming = []
    if not events:
        upcoming = next_key_events(7, target, debug=args.debug)

    script = build_script(target, headlines, events, upcoming)

    payload = {
        "date_et": target.isoformat(),
        "generated_at_utc": datetime.now(TZ_UTC).isoformat(),
        "headlines": [h.__dict__ for h in headlines],
        "events": [e.__dict__ for e in events],
        "upcoming_if_empty": [e.__dict__ for e in upcoming],
        "script": script,
    }

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", f"brief_{target.isoformat()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Console output (same style you pasted)
    print("\n[Top Headlines]")
    if headlines:
        for h in headlines[:6]:
            print(f"- {h.title} ({h.source})")
    else:
        print("(none)")

    print("\n[Today Events]")
    if events:
        for e in events:
            when = f"{e.time_et} ET" if e.time_et else "Time TBA"
            print(f"- {when} — {e.title} [{e.source}]")
    else:
        print("(none)")
        if upcoming:
            print("\n[Next Key Events]")
            for e in upcoming:
                when = f"{e.time_et} ET" if e.time_et else "Time TBA"
                print(f"- {e.details} — {when} — {e.title} [{e.source}]")

    print("\n[Script]")
    print(script)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
