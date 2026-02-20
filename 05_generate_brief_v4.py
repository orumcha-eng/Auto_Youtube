# 05_generate_brief_v4.py
# -*- coding: utf-8 -*-

import argparse
import calendar
import html as ihtml
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple

import requests
import feedparser
from dateutil import parser as dtparser
from icalendar import Calendar
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MacroMinuteBrief/1.0"

# ---- Sources ----
GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_BUSINESS = "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"

FED_CAL_URL_TMPL = "https://www.federalreserve.gov/newsevents/{year}-{monthname}.htm"
BEA_ICS_URL = "https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"

# ---- Macro-ish scoring ----
MACRO_KEYWORDS = [
    "premarket", "pre-market", "futures", "treasury", "yield", "yields", "fed",
    "powell", "fomc", "minutes", "inflation", "cpi", "ppi", "jobs", "payroll",
    "unemployment", "rates", "rate", "dollar", "oil", "crude", "opec", "gold",
    "bond", "bonds", "stocks", "market", "earnings", "gdp", "pmi", "recession",
    "risk", "sanction", "tariff", "volatility", "vix", "spread", "credit"
]
MACRO_STOP = [
    "ces", "humanoid", "robot", "smart glasses", "tv", "gadget", "hands-on",
    "autonomous car", "gaming", "smartphone", "camera", "wearable"
]

# Source weight (publisher string from Google News title)
SOURCE_BOOST = {
    "Reuters": 4,
    "The Wall Street Journal": 3,
    "Financial Times": 3,
    "Bloomberg": 3,
    "Bloomberg.com": 3,
    "CNBC": 2,
    "MarketWatch": 2,
    "The Economist": 2,
    "The New York Times": 1,
    "Fortune": 1,
    "WSJ": 3,
    "FT": 3,
}

@dataclass
class Headline:
    title: str
    source: str
    link: str
    published_utc: Optional[str] = None
    score: int = 0

@dataclass
class EventItem:
    title: str
    time_et: Optional[str]   # "HH:MM" or None
    source: str              # FED / BEA / BLS
    date_et: str             # YYYY-MM-DD
    details: Optional[str] = None

def normalize_ws(s: str) -> str:
    if s is None:
        return ""
    s = ihtml.unescape(s)
    s = s.replace("\u00a0", " ")
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = s.replace("…", "...")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def http_get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    return r.text

def http_get_bytes(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    return r.content

def now_et() -> datetime:
    return datetime.now(tz=ET)

def parse_time_ampm_to_24h(s: str) -> Optional[str]:
    s = normalize_ws(s).lower()
    m = re.match(r"^(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.)$", s)
    if not m:
        return None
    hh = int(m.group(1)); mm = int(m.group(2)); ap = m.group(3)
    if ap == "p.m." and hh != 12:
        hh += 12
    if ap == "a.m." and hh == 12:
        hh = 0
    return f"{hh:02d}:{mm:02d}"

def html_to_lines(raw_html: str) -> List[str]:
    raw_html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "\n", raw_html)
    text = re.sub(r"(?s)<[^>]+>", "\n", raw_html)
    text = normalize_ws(text)
    # normalize_ws already collapses; split by some separators by injecting newlines
    # We'll re-run a conservative tag-strip approach:
    # (We already stripped tags -> now just one line; so instead we keep the earlier method)
    # For Fed page parsing, it's better to keep multiple lines. We'll do a second pass:
    text = ihtml.unescape(re.sub(r"(?s)<[^>]+>", "\n", re.sub(r"(?is)<(script|style).*?>.*?</\1>", "\n", raw_html)))
    lines = [normalize_ws(ln) for ln in text.splitlines()]
    return [ln for ln in lines if ln]

def source_weight(source: str) -> int:
    s = normalize_ws(source)
    return SOURCE_BOOST.get(s, 0)

def score_headline(title: str, source: str) -> int:
    t = normalize_ws(title).lower()

    if any(x in t for x in MACRO_STOP):
        return -5

    score = 0
    for k in MACRO_KEYWORDS:
        if k in t:
            score += 1

    # small penalties for very local/corporate admin stuff
    if any(x in t for x in ["headquarters", "layoffs", "restructuring"]):
        score -= 1

    # source boost
    score += source_weight(source)

    return score

def _parse_published_utc(s: Optional[str]) -> datetime:
    if not s:
        return datetime(1970, 1, 1, tzinfo=UTC)
    try:
        dt = dtparser.isoparse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return datetime(1970, 1, 1, tzinfo=UTC)

def fetch_google_headlines(hours_back: int, query: str, debug: bool=False) -> List[Headline]:
    cutoff = datetime.now(tz=UTC) - timedelta(hours=hours_back)

    def pull(url: str) -> List[Headline]:
        feed = feedparser.parse(url)
        out: List[Headline] = []
        for e in (feed.entries or [])[:100]:
            raw_title = normalize_ws(e.get("title") or "")
            link = normalize_ws(e.get("link") or "")
            if not raw_title or not link:
                continue

            # "Headline - Source"
            source = "Google News"
            title = raw_title
            if " - " in raw_title:
                a, b = raw_title.rsplit(" - ", 1)
                title, source = normalize_ws(a), normalize_ws(b)

            published_utc = None
            dt = None
            if e.get("published"):
                try:
                    dt = dtparser.parse(e["published"])
                except Exception:
                    dt = None

            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                dt_utc = dt.astimezone(UTC)
                if dt_utc < cutoff:
                    continue
                published_utc = dt_utc.isoformat()

            sc = score_headline(title, source)
            out.append(Headline(title=title, source=source, link=link, published_utc=published_utc, score=sc))
        return out

    url_q = GOOGLE_NEWS_SEARCH.format(q=requests.utils.quote(query))
    h = pull(url_q)
    if debug:
        print(f"[debug] google search url={url_q} count={len(h)}")

    if not h:
        h = pull(GOOGLE_NEWS_BUSINESS)
        if debug:
            print(f"[debug] fallback business feed count={len(h)}")

    # de-dup by title
    seen = set()
    uniq: List[Headline] = []
    for x in h:
        key = x.title.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)

    # sort: score desc, recency desc
    uniq.sort(key=lambda x: (x.score, _parse_published_utc(x.published_utc)), reverse=True)

    # primary pick: score >= 3 (macro + decent publisher)
    filtered = [x for x in uniq if x.score >= 3]

    # backfill: ensure at least 6 candidates so top-3 won't be too weak/short
    if len(filtered) < 6:
        for x in uniq:
            if x in filtered:
                continue
            if x.score < 0:
                continue
            filtered.append(x)
            if len(filtered) >= 6:
                break

    if not filtered:
        filtered = uniq[:8]

    return filtered[:8]

def fetch_ics_events(url: str, source_name: str, target: date) -> List[EventItem]:
    raw = http_get_bytes(url)
    cal = Calendar.from_ical(raw)
    out: List[EventItem] = []
    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue

        summary = normalize_ws(str(comp.get("summary", "") or ""))
        dtstart = comp.get("dtstart")
        if not summary or not dtstart:
            continue

        dtv = dtstart.dt
        time_et = None

        if isinstance(dtv, datetime):
            dt_et = dtv.astimezone(ET) if dtv.tzinfo else dtv.replace(tzinfo=ET)
            if dt_et.date() != target:
                continue
            time_et = dt_et.strftime("%H:%M")
        else:
            if dtv != target:
                continue

        out.append(EventItem(
            title=summary,
            time_et=time_et,
            source=source_name,
            date_et=target.isoformat(),
            details=None
        ))

    out.sort(key=lambda x: x.time_et or "99:99")
    return out

def fetch_fed_events_for_date(target: date, debug: bool=False) -> List[EventItem]:
    monthname = calendar.month_name[target.month].lower()
    url = FED_CAL_URL_TMPL.format(year=target.year, monthname=monthname)
    raw_html = http_get(url)
    lines = html_to_lines(raw_html)

    sections = {"Speeches", "Testimony", "FOMC Meetings", "Beige Book", "Statistical Releases", "Other"}
    ignore = {
        "Time:", "Release Date(s):", "Calendar", "Time", "Release Date(s)",
        "Please enable JavaScript if it is disabled in your browser or access the information through the links provided below."
    }
    day_re = re.compile(r"^\d{1,2}$")

    current = None
    buffer: List[str] = []
    events: List[Tuple[str, int, List[str]]] = []

    for ln in lines:
        if ln in sections:
            current = ln
            buffer = []
            continue
        if current is None:
            continue
        if ln in ignore:
            continue
        if ln.lower() in {"skip to main content", "previous", "next"}:
            continue

        if day_re.match(ln):
            if buffer:
                events.append((current, int(ln), buffer[:]))
            buffer = []
            continue

        buffer.append(ln)

    picked: List[EventItem] = []
    for section, day, buf in events:
        if day != target.day:
            continue

        t24 = None
        cleaned: List[str] = []
        for x in buf:
            x = normalize_ws(x)
            if not x or x in ignore or x == "To Be Announced":
                continue
            if t24 is None:
                maybe = parse_time_ampm_to_24h(x)
                if maybe:
                    t24 = maybe
                    continue
            cleaned.append(x)

        if not cleaned and not t24:
            continue

        title = cleaned[0] if cleaned else f"{section} event"
        details = " · ".join(cleaned[1:]) if len(cleaned) > 1 else None

        major_sections = {"Speeches", "Testimony", "FOMC Meetings", "Beige Book", "Other"}
        major_stats_keywords = ["Industrial Production", "Consumer Credit", "Financial Accounts", "Z.1", "SLOOS", "Senior Loan Officer"]

        keep = False
        if section in major_sections:
            keep = True
        elif section == "Statistical Releases":
            t = title.lower()
            if any(k.lower() in t for k in [k.lower() for k in major_stats_keywords]):
                keep = True

        if keep:
            picked.append(EventItem(
                title=title,
                time_et=t24,
                source="FED",
                date_et=target.isoformat(),
                details=details
            ))

    picked.sort(key=lambda x: x.time_et or "99:99")
    if debug:
        print(f"[debug] FED url={url} picked={len(picked)}")
    return picked

def next_key_events(from_date: date, lookahead_days: int = 3) -> List[EventItem]:
    out: List[EventItem] = []
    for d in range(1, lookahead_days + 1):
        day = from_date + timedelta(days=d)
        out += fetch_fed_events_for_date(day, debug=False)
        out += fetch_ics_events(BEA_ICS_URL, "BEA", day)
        out += fetch_ics_events(BLS_ICS_URL, "BLS", day)

    # de-dup
    seen = set()
    uniq: List[EventItem] = []
    for e in out:
        k = (e.date_et, e.source, e.title, e.time_et)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)

    def prio(e: EventItem) -> int:
        t = e.title.lower()
        if "fomc" in t or "minutes" in t or "beige book" in t:
            return 100
        if e.source == "BLS":
            return 90
        if e.source == "BEA":
            return 80
        if e.source == "FED":
            return 60
        return 10

    uniq.sort(key=lambda e: (e.date_et, -prio(e), e.time_et or "99:99"))
    return uniq[:5]

def shorten_title(s: str, max_len: int = 88) -> str:
    s = normalize_ws(s)
    s = re.sub(r"\([^)]*\)", "", s).strip()
    if len(s) <= max_len:
        return s
    cut = s[:max_len].rsplit(" ", 1)[0]
    return cut + "..."

def build_script(target: date,
                 headlines: List[Headline],
                 today_events: List[EventItem],
                 next_events: List[EventItem],
                 min_words: int,
                 max_words: int) -> str:
    weekday = datetime(target.year, target.month, target.day).strftime("%A")
    monthday = datetime(target.year, target.month, target.day).strftime("%B %d")

    top = headlines[:3]
    titles_concat = " ".join([h.title for h in top]).lower()

    lines: List[str] = []
    lines.append(f"Good morning — {weekday}, {monthday}. This is Macro Minute Brief.")

    if top:
        lines.append("Overnight headlines:")
        for h in top:
            lines.append(f"- {shorten_title(h.title)}.")
    else:
        lines.append("Overnight was quiet on major macro headlines across our sources.")

    # So-what line (simple heuristic)
    if any(k in titles_concat for k in ["oil", "crude", "venezuela", "sanction", "middle east", "geopolit"]):
        lines.append("So what: geopolitics is in focus — watch crude, defense names, and any flight-to-quality into Treasuries.")
    else:
        lines.append("So what: watch futures, Treasury yields, and the U.S. dollar for the market’s risk tone into the open.")

    if today_events:
        lines.append("Today’s key events (Eastern Time):")
        for e in today_events[:4]:
            when = f"{e.time_et} ET" if e.time_et else "Time TBA"
            lines.append(f"- {when}: {shorten_title(e.title, 92)}.")
    else:
        lines.append("Today’s top-tier macro calendar is light.")
        if next_events:
            lines.append("Next key events:")
            for e in next_events[:3]:
                dlabel = datetime.fromisoformat(e.date_et).strftime("%a %b %d")
                when = f"{e.time_et} ET" if e.time_et else "Time TBA"
                lines.append(f"- {dlabel}: {when} — {shorten_title(e.title, 92)}.")

    # volatility hint if 08:30 exists
    if any((ev.time_et == "08:30") for ev in today_events):
        lines.append("Heads up: 8:30 a.m. ET is the main volatility window today.")

    lines.append("That’s the setup — trade safe, and I’ll see you tomorrow.")

    script = normalize_ws(" ".join(lines))

    # If too short, add neutral fillers (TTS-friendly)
    fillers = [
        "If you’re trading the open, size down into data and let the first reaction settle.",
        "Keep an eye on rate-sensitive sectors and mega-cap leadership for confirmation.",
        "If yields move fast, expect a quick read-through into equities and FX.",
    ]
    wi = 0
    while len(script.split()) < min_words and wi < len(fillers):
        script = normalize_ws(script + " " + fillers[wi])
        wi += 1

    # Hard cap
    words = script.split()
    if len(words) > max_words:
        script = " ".join(words[:max_words]) + "..."

    return script

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-et", help="Target date in ET (YYYY-MM-DD). Default: today ET")
    ap.add_argument("--hours-back", type=int, default=18)
    ap.add_argument("--query", default="premarket futures Treasury yields dollar Fed CPI inflation jobs economic calendar")
    ap.add_argument("--min-words", type=int, default=135)
    ap.add_argument("--max-words", type=int, default=155)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    target = dtparser.isoparse(args.date_et).date() if args.date_et else now_et().date()

    if args.debug:
        print(f"[debug] args.date_et={args.date_et} -> target={target.isoformat()}")

    headlines = fetch_google_headlines(args.hours_back, args.query, debug=args.debug)

    today_events: List[EventItem] = []
    today_events += fetch_fed_events_for_date(target, debug=args.debug)
    today_events += fetch_ics_events(BEA_ICS_URL, "BEA", target)
    today_events += fetch_ics_events(BLS_ICS_URL, "BLS", target)

    # de-dup today events
    seen = set()
    uniq_today = []
    for e in today_events:
        k = (e.source, e.title, e.time_et)
        if k in seen:
            continue
        seen.add(k)
        uniq_today.append(e)
    uniq_today.sort(key=lambda x: x.time_et or "99:99")
    today_events = uniq_today

    next_events = next_key_events(target, lookahead_days=3) if not today_events else []

    script = build_script(
        target, headlines, today_events, next_events,
        min_words=args.min_words, max_words=args.max_words
    )

    if args.debug:
        print("[debug] word_count:", len(script.split()))

    payload = {
        "brand": "Macro Minute Brief",
        "date_et": target.isoformat(),
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "query": args.query,
        "headlines": [asdict(h) for h in headlines],
        "today_events": [asdict(e) for e in today_events],
        "next_events_if_today_empty": [asdict(e) for e in next_events],
        "script": script,
    }

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join("out", f"brief_{target.isoformat()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[Top Headlines]")
    if headlines:
        for h in headlines[:6]:
            print(f"- {h.title} ({h.source})")
    else:
        print("(none)")

    print("\n[Today Events]")
    if today_events:
        for e in today_events:
            when = f"{e.time_et} ET" if e.time_et else "Time TBA"
            print(f"- {when} — {e.title} [{e.source}]")
    else:
        print("(none)")

    if not today_events:
        print("\n[Next Key Events]")
        for e in next_events[:3]:
            dlabel = datetime.fromisoformat(e.date_et).strftime("%a %b %d")
            when = f"{e.time_et} ET" if e.time_et else "Time TBA"
            print(f"- {dlabel} — {when} — {e.title} [{e.source}]")

    print("\n[Script]")
    print(script)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
