# CODEX Handoff (Auto_Youtube)

## 새 PC에서 Codex에게 이렇게 말하세요
`CODEX_HANDOFF.md 읽고, 현재 상태 그대로 이어서 작업해줘.`

## 프로젝트 개요
- 목적: 경제/시장 뉴스 기반 롱폼 영상 자동 제작 (썸네일 -> 대본 -> 씬 -> TTS -> 렌더 -> 업로드 문구)
- 메인 앱: `app_v3.py` (Streamlit)
- 최근 기준 커밋: `379257e`

## 빠른 실행
1. 저장소 클론 후 최신 pull
2. 가상환경/의존성 설치
3. `.env` 설정
4. `streamlit run app_v3.py`

## 필수 환경변수(.env)
- `GOOGLE_API_KEY=...`
- `GOOGLE_API_KEY_2=...` (보조키, 429 시 자동 전환)
- `LUMA_API_KEY=...` (선택)

중요:
- 변수명은 반드시 `GOOGLE_API_KEY_2` (언더스코어 포함)
- `.env`는 BOM 없는 일반 텍스트(UTF-8 또는 ASCII) 권장

## 현재 핵심 동작(안정화 포인트)
- Step 네비게이션: 1~6 북마크 이동 지원
- Step 4 TTS:
  - 문장 단위 생성
  - 스타일 지시 비활성화(빈 스타일)
  - `Keep one voice only` 기본 ON (보이스 혼합 방지)
  - AI Studio 429 시 키 로테이션: k1 -> k2
  - 세그먼트별 Provider/Voice/Gender/Error 확인 가능
- Step 5 렌더:
  - 롱폼 16:9 기준
  - 누락 씬 skip 금지(타임라인 보존)
  - 이미지 zoom-in 모션 적용
  - 렌더 진행률 표시
- 길이 가드:
  - 목표 분량 대비 대본/오디오가 너무 짧으면 경고/차단

## 자주 헷갈리는 포인트
- `Prepared 60 sentence segments`는 고정값 아님
  - 대본 문장 수에 따라 변함
- 영상 길이는 렌더 문제가 아니라 `merged audio` 길이를 따라감
- AI Studio 에러가 `per_day limit: 0`이면 대기해도 해결 안 됨
  - 보조키로 넘어가거나 Cloud TTS 사용 필요

## 최근 이슈와 상태
1. TTS 보이스가 중간에 바뀌는 문제
- 원인: fallback 엔진 섞임
- 대응: strict voice 옵션 + provider 표시

2. 영상/자막 싱크 밀림
- 대응: 씬 길이 계산 및 누락 씬 skip 제거로 개선

3. 이미지/영상/TTS API 429
- 대응: 기본키+보조키 자동 로테이션 반영

## 구버전 경고 관련
- `src/content.py`는 일부 `google.generativeai`(legacy)를 아직 사용
- 기능상 당장 치명적 이슈는 아니나, 경고 제거 원하면 `google.genai`로 전면 마이그레이션 필요

## 작업 원칙(현재 사용자 선호)
- 새 기능 추가 최소화
- 안정성 우선, 회귀 버그 방지 우선
- 동작 검증(실행/결과 확인) 후 반영

## 추천 점검 루틴
1. Step 2 대본 단어수 확인
2. Step 4 세그먼트 생성 후 실패/엔진 혼합 확인
3. Merge audio 길이 확인(목표 분 대비)
4. Step 5 렌더 후 실제 길이/싱크 확인

