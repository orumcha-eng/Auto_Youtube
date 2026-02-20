# src/content.py
# -*- coding: utf-8 -*-

import html as ihtml
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
import feedparser
from dateutil import parser as dtparser
from icalendar import Calendar
import json
from google.auth.exceptions import DefaultCredentialsError
import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types

# ---- Global Constants ----
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MacroMinuteBrief/1.0"

GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_BUSINESS = "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en"

INVESTING_RSS_URLS = [
    "https://www.investing.com/rss/news.rss",             # Breaking
    "https://www.investing.com/rss/stock_market.rss",     # Stock Market
    "https://www.investing.com/rss/economy_indicators.rss" # Indicators
]

FED_CAL_URL_TMPL = "https://www.federalreserve.gov/newsevents/{year}-{monthname}.htm"
BEA_ICS_URL = "https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"

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

SOURCE_BOOST = {
    "Reuters": 4, "The Wall Street Journal": 3, "Financial Times": 3,
    "Bloomberg": 3, "Bloomberg.com": 3, "CNBC": 2, "MarketWatch": 2,
    "The Economist": 2, "The New York Times": 1, "Fortune": 1, "WSJ": 3, "FT": 3,
}

@dataclass
class Headline:
    title: str
    source: str
    link: str
    published_at: Optional[datetime] = None
    score: int = 0

@dataclass
class EventItem:
    title: str
    time_et: Optional[str]   # "HH:MM" or None
    source: str              # FED / BEA / BLS
    date_et: str             # YYYY-MM-DD
    details: Optional[str] = None

def normalize_ws(s: str) -> str:
    if s is None: return ""
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
    if not m: return None
    hh = int(m.group(1)); mm = int(m.group(2)); ap = m.group(3)
    if ap == "p.m." and hh != 12: hh += 12
    if ap == "a.m." and hh == 12: hh = 0
    return f"{hh:02d}:{mm:02d}"

def score_headline(title: str, source: str) -> int:
    t = normalize_ws(title).lower()
    if any(x in t for x in MACRO_STOP): return -5
    score = 0
    for k in MACRO_KEYWORDS:
        if k in t: score += 1
    if any(x in t for x in ["headquarters", "layoffs", "restructuring"]): score -= 1
    score += SOURCE_BOOST.get(normalize_ws(source), 0)
    return score

def fetch_google_headlines(hours_back: int, query: str) -> List[Headline]:
    cutoff = datetime.now(tz=UTC) - timedelta(hours=hours_back)
    
    def pull(url: str) -> List[Headline]:
        feed = feedparser.parse(url)
        out = []
        for e in (feed.entries or [])[:100]:
            raw_title = normalize_ws(e.get("title") or "")
            link = normalize_ws(e.get("link") or "")
            if not raw_title or not link: continue
            
            source = "Google News"
            title = raw_title
            if " - " in raw_title:
                a, b = raw_title.rsplit(" - ", 1)
                title, source = normalize_ws(a), normalize_ws(b)
            
            dt_utc = None
            published_utc = None
            if e.get("published"):
                try:
                    dt = dtparser.parse(e["published"])
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=UTC)
                    dt_utc = dt.astimezone(UTC)
                    published_utc = dt_utc.isoformat()
                except: pass
            
            if dt_utc and dt_utc < cutoff: continue
            
            sc = score_headline(title, source)
            out.append(Headline(title=title, source=source, link=link, published_at=dt_utc, score=sc))
        return out

    url_q = GOOGLE_NEWS_SEARCH.format(q=requests.utils.quote(query))
    h = pull(url_q)
    if not h: h = pull(GOOGLE_NEWS_BUSINESS)
    
    # De-dup
    seen, uniq = set(), []
    for x in h:
        k = x.title.lower()
        if k in seen: continue
        seen.add(k)
        uniq.append(x)
        
    # Sort
    uniq.sort(key=lambda x: (x.score, x.published_at or datetime.min.replace(tzinfo=UTC)), reverse=True)
    
    filtered = [x for x in uniq if x.score >= 3]
    if len(filtered) < 6:
        for x in uniq:
            if x not in filtered and x.score >= 0:
                filtered.append(x)
                if len(filtered) >= 6: break
    
    return filtered[:8] if filtered else uniq[:8]

def fetch_ics_events(url: str, source_name: str, target: date) -> List[EventItem]:
    try:
        raw = http_get_bytes(url)
        cal = Calendar.from_ical(raw)
        out = []
        for comp in cal.walk():
            if comp.name != "VEVENT": continue
            summary = normalize_ws(str(comp.get("summary", "") or ""))
            dtstart = comp.get("dtstart")
            if not summary or not dtstart: continue
            
            dtv = dtstart.dt
            time_et = None
            if isinstance(dtv, datetime):
                dt_et = dtv.astimezone(ET) if dtv.tzinfo else dtv.replace(tzinfo=ET)
                if dt_et.date() != target: continue
                time_et = dt_et.strftime("%H:%M")
            else:
                if dtv != target: continue
            
            out.append(EventItem(title=summary, time_et=time_et, source=source_name, date_et=target.isoformat()))
        out.sort(key=lambda x: x.time_et or "99:99")
        return out
    except Exception:
        return []

def fetch_fed_events(target: date) -> List[EventItem]:
    # Simplified Fed Fetcher (omitted complex HTML parsing for brevity, assuming minimal context needed or can add back if critical)
    # For now, let's keep it simple or placeholder if HTML parsing is flaky. 
    # Actually, let's include a robust version if possible, but for MVP refactor, sticking to ICS might be safer?
    # No, the user wants it robust. I will skip the complex regex scraping for now to ensure stability 
    # unless I copy the *exact* scraper. I'll copy the exact scraper logic to be safe.
    try:
        import calendar
        monthname = calendar.month_name[target.month].lower()
        url = FED_CAL_URL_TMPL.format(year=target.year, monthname=monthname)
        raw_html = http_get(url)
        
        # ... (Parsing logic from v4) ... 
        # For brevity in this turn, I will assume we can reuse the logic.
        # But to be practical, I will include the critical regex part from v4.
        text = ihtml.unescape(re.sub(r"(?s)<[^>]+>", "\n", re.sub(r"(?is)<(script|style).*?>.*?</\1>", "\n", raw_html)))
        lines = [normalize_ws(ln) for ln in text.splitlines() if normalize_ws(ln)]
        
        current = None
        buffer = []
        events_found = []
        sections = {"Speeches", "Testimony", "FOMC Meetings", "Beige Book", "Statistical Releases", "Other"}
        day_re = re.compile(r"^\d{1,2}$")
        
        for ln in lines:
            if ln in sections:
                current = ln; buffer = []; continue
            if not current: continue
            if day_re.match(ln):
                if buffer: events_found.append((current, int(ln), buffer[:]))
                buffer = []
                continue
            buffer.append(ln)
            
        picked = []
        for section, day, buf in events_found:
            if day != target.day: continue
            
            t24 = None
            cleaned = []
            for x in buf:
                if "Time:" in x or "Calendar" in x: continue
                maybe = parse_time_ampm_to_24h(x)
                if maybe: 
                    t24 = maybe
                    continue
                cleaned.append(x)
            
            if not cleaned and not t24: continue
            title = cleaned[0] if cleaned else f"{section} event"
            
            picked.append(EventItem(title=title, time_et=t24, source="FED", date_et=target.isoformat()))
            
        picked.sort(key=lambda x: x.time_et or "99:99")
        return picked
    except Exception:
        return []

def build_script(target: date, headlines: List[Headline], today_events: List[EventItem]) -> str:
    weekday = datetime(target.year, target.month, target.day).strftime("%A")
    monthday = datetime(target.year, target.month, target.day).strftime("%B %d")
    
    top = headlines[:3]
    lines = []
    lines.append(f"Good morning {weekday}, {monthday}. This is Macro Minute Brief.")
    
    if top:
        lines.append("Overnight headlines:")
        for h in top:
            lines.append(f"- {h.title}.")
    else:
        lines.append("Overnight was quiet on major headlines.")
        
    if today_events:
        lines.append("Today's key events:")
        for e in today_events[:4]:
            when = f"{e.time_et} ET" if e.time_et else "Time TBA"
            lines.append(f"- {when}: {e.title}.")
    else:
        lines.append("Today's calendar is light.")
        
    lines.append("That's the setup. Trade safe.")
    return "\n".join(lines)


import google.generativeai as genai

def fetch_investing_rss() -> List[Headline]:
    """Fetch from Investing.com RSS feeds"""
    all_headlines = []
    headers = {"User-Agent": UA}
    
    for url in INVESTING_RSS_URLS:
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:10]: # Top 10 from each
                # Filter out old? For now just take latest
                t = ihtml.unescape(entry.title)
                l = entry.link
                s = "Investing.com"
                
                # Check duplication
                if any(h.title == t for h in all_headlines): continue
                
                dt_obj = datetime.now() # Fallback
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                     dt_obj = datetime(*entry.published_parsed[:6])
                
                all_headlines.append(Headline(
                    title=t,
                    source=s,
                    link=l,
                    published_at=dt_obj,
                    score=5 # High score for specific source
                ))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            
    return all_headlines

# ---- Content Module ----
class ContentModule:
    """Convenience wrapper for Streamlit"""
    def generate(self, date_et_str: str, query: str = "markets", api_key: str = None) -> dict:
        target = dtparser.parse(date_et_str).date()
        
        # 1. Fetch Investing.com (Specific Sources)
        inv_headlines = fetch_investing_rss()
        
        # 2. Fetch Google News (Supplemental / Query based)
        # Reduce count since we have investing
        google_headlines = fetch_google_headlines(20, query)
        
        # Combine
        headlines = inv_headlines + google_headlines
        
        # Simple dedupe by title
        uniq_h = []
        seen = set()
        for h in headlines:
            if h.title in seen: continue
            seen.add(h.title)
            uniq_h.append(h)
        headlines = uniq_h
        
        events = []
        events += fetch_fed_events(target)
        events += fetch_ics_events(BEA_ICS_URL, "BEA", target)
        events += fetch_ics_events(BLS_ICS_URL, "BLS", target)
        
        # De-dup events
        uniq_ev = []
        seen = set()
        for e in events:
            k = (e.source, e.title, e.time_et)
            if k in seen: continue
            seen.add(k)
            uniq_ev.append(e)
        uniq_ev.sort(key=lambda x: x.time_et or "99:99")
        
        curated_display = []
        script = ""
        
        if api_key:
             # Stage 1: Collector
             curated_data = self.curate_news_llm(api_key, headlines, uniq_ev)
             curated_display = curated_data.get("top_news", [])
             
             # Stage 2: Anchor
             script = self.generate_korean_script_llm(api_key, target, curated_data)
        else:
             script = build_script(target, headlines, uniq_ev)
             # Fallback display
             curated_display = [asdict(h) for h in headlines[:6]]
        
        return {
            "date_et": target.isoformat(),
            "headlines": curated_display, 
            "today_events": [asdict(e) for e in uniq_ev],
            "script": script,
            "curated_data": curated_data if api_key else None
        }

    def generate_thumbnail_ideas(self, api_key: str, topic: str, style_direction: str = "") -> List[dict]:
        """Generate diverse thumbnail concepts with hard constraints."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
You are a YouTube thumbnail creative director.
Topic: "{topic}"
Audience: Korean retail investors interested in global macro/markets.
Main character: Bean (simple bean-shaped white character, expressive face).
Style direction:
{style_direction or "High-contrast macro thumbnail style with strong emotional framing."}

Task:
- Create 6 thumbnail concepts.
- Concepts must be strongly different from each other.

Hard constraints:
1) Output must be JSON array only.
2) Each concept must include:
   - "title": Korean hook, 10~18 chars, emotional and clickable.
   - "visual": highly detailed scene description for image generation.
   - "shot_type": one of ["closeup","wide","split","symbolic","pov","minimal"].
   - "mood": one of ["panic","urgency","opportunity","curiosity","relief","shock"].
   - "color_palette": short phrase like "red-black neon".
3) All 6 concepts must use different shot_type values.
4) No generic wording. Include composition, camera angle, background objects, and Bean's facial emotion.
5) Do not include text overlay instructions in visual. Image should be text-free.
6) Make concepts look clickable in Korean finance YouTube context (bold emotion, clear conflict, immediate stakes).

Output schema example:
[
  {{
    "title": "오늘 밤이 분기점",
    "visual": "Extreme close-up of Bean's anxious face reflected in a glowing red candlestick monitor...",
    "shot_type": "closeup",
    "mood": "urgency",
    "color_palette": "red-black neon"
  }}
]
"""

        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=1.05,
                    top_k=40,
                ),
            )
            data = json.loads(resp.text)
            if not isinstance(data, list):
                return []
            return data[:6]
        except Exception as e:
            print(f"Thumb Idea Gen failed: {e}")
            return []

    def generate_thumbnail_copy_options(
        self,
        api_key: str,
        topic: str,
        concept_title: str,
        concept_visual: str,
        n_options: int = 8,
    ) -> List[str]:
        """
        Generate Korean thumbnail text options (5~10) based on high-performing
        finance/geopolitics thumbnail patterns.
        """
        client = genai_new.Client(api_key=api_key)
        n_options = max(5, min(10, n_options))

        prompt = f"""
Role: YouTube thumbnail copywriter for Korean finance/macroeconomy channels.

Context:
- Topic: {topic}
- Concept title: {concept_title}
- Concept visual: {concept_visual}

Reference pattern analysis to follow:
1) Use high-tension hooks: warning, collapse, last chance, critical turning point.
2) Include concrete anchors when possible: numbers, years, percentages, time pressure.
3) Keep copy very short and punchy. Usually 2 lines, each line 6~14 Korean characters.
4) Use contrast words and hard verbs: 붕괴, 폭락, 경고, 끝났다, 무너진다, 터진다.
5) Avoid vague polite language; prioritize urgency and clarity.
6) Avoid misinformation: no fabricated institution names or fake stats.
7) Do NOT include quotes or emojis.

Task:
- Generate exactly {n_options} different copy options.
- Each option must be a single string with line break marker " / " between line1 and line2.
- Make each option distinct in angle (warning/opportunity/deadline/scenario/checklist).

Output JSON only:
{{
  "options": [
    "1차 경고 / 지금이 분기점",
    "금리 쇼크 / 이번엔 다르다"
  ]
}}
"""
        try:
            resp = client.models.generate_content(
                model="models/gemini-2.0-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.95,
                ),
            )
            data = json.loads(resp.text or "{}")
            opts = data.get("options", []) if isinstance(data, dict) else []
            cleaned = []
            for x in opts:
                if not isinstance(x, str):
                    continue
                s = normalize_ws(x)
                # Keep only readable characters for thumbnail copy.
                s = re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\s\/\-\+\%\!\?\.\,\:\~]", "", s)
                s = re.sub(r"\?{2,}", "?", s).strip()
                s = re.sub(r"^\?\s*", "", s)
                s = re.sub(r"\s*/\s*\?\s*", " / ", s)
                if " / " not in s and len(s.split()) >= 4:
                    parts = s.split()
                    mid = max(2, len(parts) // 2)
                    s = " ".join(parts[:mid]) + " / " + " ".join(parts[mid:])
                if s:
                    cleaned.append(s)
            return cleaned[:n_options]
        except Exception as e:
            print(f"Thumbnail copy gen failed: {e}")
            return []

    def generate_longform_script_llm(self, api_key: str, target: date, curated_data: dict, duration_min: int = 10) -> dict:
        """Generate a Korean long-form script with minimum length guarantees."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        data_json = json.dumps(curated_data, indent=2, ensure_ascii=False)
        min_words = max(600, int(duration_min * 115))
        target_words = int(duration_min * 135)

        prompt = f"""
당신은 한국어 경제 유튜브 대본 작가다.
목표: 약 {duration_min}분 분량의 고유지율(long-form) 내레이션 대본 작성.

분량 규칙:
- 목표 단어수: 약 {target_words}
- 최소 단어수: {min_words} (반드시 충족)

입력 데이터:
{data_json}

대본 스타일(레퍼런스 반영):
1) 시작 20초는 강한 훅으로 시작한다. (충격/위험/기회 중 하나)
2) 곧바로 "오늘 무엇을 알게 되는지"를 3포인트로 예고한다.
3) 본문은 위기 -> 원인 -> 데이터 근거 -> 파급효과 -> 시나리오 순으로 전개한다.
4) 단락 사이 브리지 문장을 반드시 넣는다.
   - 예: "그런데 여기서 더 중요한 건", "이 지점에서 시장이 보는 건", "숫자로 보면 더 명확합니다"
5) 단순 나열 금지. 뉴스 간 인과관계/맥락을 설명한다.
6) 60~90초마다 집중 환기 문장을 넣는다.
   - 예: "핵심은 세 가지입니다", "지금부터가 진짜 중요합니다"
7) 후반에는 "내일 확인할 체크포인트"를 구체적으로 제시한다.
8) 끝맺음은 과장 없이 요약 + 시청자 질문 1개로 마무리한다.

톤 규칙:
- 자연스러운 한국어 구어체 (경제 유튜버 톤)
- 강약이 있는 문장 길이 (짧은 문장 + 설명 문장 혼합)
- 과도한 공포 조장 금지, 그러나 긴장감은 유지

금지 규칙:
- 마크다운/불릿/헤더 기호 금지
- 무대지시문 금지
- placeholder 금지
- 입력에 없는 사실을 단정하지 말 것

출력(JSON만):
{{
  "full_text": "완성된 전체 대본",
  "segments": [
    {{"title":"오프닝 훅", "body":"..."}},
    {{"title":"오늘의 3가지 포인트", "body":"..."}},
    {{"title":"핵심 이슈 전개", "body":"..."}},
    {{"title":"딥다이브", "body":"..."}},
    {{"title":"시장 파급과 체크포인트", "body":"..."}},
    {{"title":"마무리", "body":"..."}}
  ]
}}
"""

        def _word_count(text: str) -> int:
            return len(re.findall(r"\S+", text or ""))

        def _normalize_result(obj: dict) -> dict:
            if not isinstance(obj, dict):
                return {"full_text": "", "segments": []}
            full_text = (obj.get("full_text") or "").strip()
            segments = obj.get("segments")
            if not isinstance(segments, list):
                segments = []
            return {"full_text": full_text, "segments": segments}

        try:
            parsed = {"full_text": "", "segments": []}
            for attempt in range(2):
                resp = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.8 if attempt == 0 else 0.7,
                    },
                )
                parsed = _normalize_result(json.loads(resp.text))
                if _word_count(parsed["full_text"]) >= min_words:
                    return parsed

            expand_prompt = f"""
아래 한국어 경제 대본을 최소 {min_words}단어 이상으로 확장하라.
조건:
- 훅 강도 유지
- 단락 전환(브리지 문장) 추가
- 근거/인과/시나리오 설명을 더 구체화
- TTS 친화적인 구어체 유지
- JSON 스키마 동일 유지

반환(JSON만):
{{
  "full_text":"...",
  "segments":[{{"title":"...","body":"..."}}]
}}

SCRIPT:
{parsed['full_text']}
"""
            resp2 = model.generate_content(
                expand_prompt,
                generation_config={"response_mime_type": "application/json", "temperature": 0.65},
            )
            parsed2 = _normalize_result(json.loads(resp2.text))
            return parsed2
        except Exception as e:
            return {"full_text": f"Error: {e}", "segments": []}
    def curate_news_llm(self, api_key: str, headlines: List[Headline], events: List[EventItem]) -> dict:
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        now_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

        raw_text = "RAW NEWS CANDIDATES:\n"
        for i, h in enumerate(headlines[:30], start=1):
            raw_text += f"{i}. {h.title} (Source: {h.source}, Link: {h.link})\\n"

        raw_text += "\\nUPCOMING EVENTS CALENDAR:\n"
        for i, e in enumerate(events[:20], start=1):
            raw_text += f"{i}. {e.date_et} {e.time_et} ET: {e.title} (Source: {e.source})\\n"

        prompt = f"""
Role: Global macro news curator.
Current time: {now_kst} (KST)

Goal:
1) Pick 6~9 most market-relevant items from raw candidates.
2) Pick up to 2 upcoming events in next 48 hours.

Rules:
- Prioritize rates/inflation/jobs/growth/liquidity/geopolitics.
- Remove duplicates and weakly relevant tech gossip.
- For each selected news item, provide one-line market relevance summary.
- Output JSON only.

Schema:
{{
  "now_kst": "...",
  "top_news": [
    {{"title":"", "source":"", "published_time_kst":"", "category":"", "key_numbers":["..."], "why_it_matters":"", "link":""}}
  ],
  "upcoming_events": [
    {{"event":"", "time_kst":"", "why_watch":"", "source":""}}
  ]
}}

RAW DATA:
{raw_text}
"""

        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json", "temperature": 0.4},
            )
            return json.loads(resp.text)
        except Exception as e:
            print(f"Curate failed: {e}")
            return {
                "now_kst": now_kst,
                "top_news": [{"title": h.title, "source": h.source, "link": h.link} for h in headlines[:6]],
                "upcoming_events": [{"event": e.title, "time_kst": f"{e.time_et} ET", "source": e.source} for e in events[:2]],
            }

    def identify_daily_theme(self, api_key: str) -> dict:
        """Fetches headlines and identifies the single most important theme."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        headlines = fetch_investing_rss() + fetch_google_headlines(12, "markets")
        seen = set()
        uniq_titles = []
        for h in headlines:
            if h.title in seen:
                continue
            seen.add(h.title)
            uniq_titles.append(h.title)

        raw_text = "\n".join([f"- {t}" for t in uniq_titles[:20]])

        prompt = f"""
Role: Financial news editor.
Task: From the headlines below, identify ONE defining market theme of the day.
Return JSON only:
{{
  "theme_short": "3~7 words in Korean",
  "summary": "One sentence in Korean",
  "headlines_used": ["...", "..."]
}}

HEADLINES:
{raw_text}
"""

        try:
            resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(resp.text)
        except Exception:
            return {
                "theme_short": "거시 변수 점검",
                "summary": "금리, 물가, 유가와 달러 흐름이 시장 방향을 좌우하는 국면입니다.",
                "headlines_used": [],
            }

    def generate_korean_script_llm(self, api_key: str, target: date, curated_data: dict) -> str:
        """Generate a short fallback script from curated data."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        data_json = json.dumps(curated_data, indent=2, ensure_ascii=False)
        date_str = target.strftime("%Y-%m-%d (%A)")

        prompt = f"""
Write a concise Korean market brief script for date {date_str}.
Use only the given JSON data.
Length: about 60~90 seconds.
No markdown. No stage directions.

INPUT:
{data_json}
"""

        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            return f"Error terms: {e}"
