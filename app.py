"""
장애 대응 도우미 - 웹 화면

지금까지 만든 네 단계를 한 화면에 묶는다.
  로그 적재 -> 집계 -> 이상 탐지 -> 사건 묶기 -> LLM 요약 -> 런북 추천

실행:
  streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import altair as alt
import pandas as pd
import streamlit as st

import detect
import summarize
import runbook

DEFAULT_LOG = PROJECT_DIR / "data" / "logs.jsonl"
RUNBOOK_DIR = PROJECT_DIR / "runbook"


# ---------------------------------------------------------------- 캐시 계층

@st.cache_data(show_spinner=False)
def cached_load(path_or_bytes, key: str) -> pd.DataFrame:
    """로그를 DataFrame으로 읽는다. key가 같으면 다시 읽지 않는다.

    올린 파일(bytes)도 detect.load_logs 로 읽는다. pd.read_json 을 쓰면 ts 가 문자열로 남아
    이후 집계가 깨지므로, 시각 변환과 is_error 계산이 들어 있는 같은 함수를 재사용한다.
    """
    if isinstance(path_or_bytes, bytes):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            f.write(path_or_bytes)
            f.flush()
            return detect.load_logs(Path(f.name))
    return detect.load_logs(Path(path_or_bytes))


@st.cache_data(show_spinner=False)
def cached_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return detect.aggregate(df)


@st.cache_resource(show_spinner=False)
def cached_index():
    """런북 색인은 한 번만 만들고 모든 재실행이 공유한다."""
    chunks = runbook.load_chunks(RUNBOOK_DIR)
    return runbook.RunbookIndex(chunks), len(chunks)


# ---------------------------------------------------------------- 사이드바

st.set_page_config(page_title="장애 대응 도우미", layout="wide")
st.title("장애 대응 도우미")
st.caption("합성 로그 기반 데모 · 실제 운영 데이터 미사용")

with st.sidebar:
    st.header("설정")

    uploaded = st.file_uploader("로그 파일 (.jsonl)", type=["jsonl"])

    method = st.radio(
        "탐지 방식",
        ["IsolationForest", "에러율 규칙"],
        help="규칙은 에러 없는 성능 저하를 놓칩니다.",
    )

    contamination = st.slider(
        "contamination (이상 비율 가정)",
        min_value=0.01, max_value=0.20,
        value=detect.CONTAMINATION, step=0.01,
        disabled=(method == "에러율 규칙"),
    )

    min_duration = st.slider(
        "최소 지속 시간 (분)",
        min_value=1, max_value=10,
        value=summarize.MIN_DURATION_MIN, step=1,
    )

    st.divider()
    st.caption(f"LLM 백엔드: `{summarize.LLM_BACKEND}`")


# ---------------------------------------------------------------- 데이터 적재

REQUIRED_COLUMNS = {"ts", "service", "level", "status", "latency_ms", "message"}

if uploaded is not None:
    raw = uploaded.getvalue()
    try:
        df = cached_load(raw, key=uploaded.name + str(len(raw)))
    except Exception:
        st.error("로그 형식을 읽지 못했습니다. 한 줄에 JSON 하나씩 있는 .jsonl 파일인지 확인하세요.")
        st.stop()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"필요한 항목이 없습니다: {', '.join(sorted(missing))}")
        st.stop()
    source = uploaded.name
elif DEFAULT_LOG.exists():
    df = cached_load(str(DEFAULT_LOG), key=str(DEFAULT_LOG.stat().st_mtime))
    source = DEFAULT_LOG.name
else:
    st.warning("로그 파일을 올려주세요. data/logs.jsonl 도 없습니다.")
    st.stop()

agg = cached_aggregate(df)

if method == "에러율 규칙":
    pred = detect.detect_by_rule(agg)
else:
    pred = detect.detect_by_model(agg, contamination=contamination)

incidents = [
    inc for inc in summarize.group_incidents(agg, pred)
    if inc.duration_min >= min_duration
]


# ---------------------------------------------------------------- 지표

c1, c2, c3, c4 = st.columns(4)
c1.metric("로그", f"{len(df):,}줄")
c2.metric("집계 구간", f"{len(agg):,}행")
c3.metric("이상 분", f"{int(pred.sum()):,}분")
c4.metric(f"{min_duration}분 이상 사건", f"{len(incidents)}건")

st.caption(f"출처: {source} · 서비스 {df['service'].nunique()}개 · "
           f"{df['ts'].min():%Y-%m-%d %H:%M} ~ {df['ts'].max():%H:%M}")


# ---------------------------------------------------------------- 타임라인

st.subheader("서비스별 추이")
metric_name = st.selectbox("지표", ["error_rate", "latency_p95", "latency_mean", "req_count"])

chart = alt.Chart(agg).mark_line().encode(
    # utchoursminutes: 브라우저 시간대로 바꾸지 말고 로그에 적힌 UTC 그대로 그려라
    x=alt.X("utchoursminutes(ts):T", title="시각 (UTC)"),
    y=alt.Y(f"{metric_name}:Q", title=metric_name),
    color=alt.Color("service:N", title="서비스"),
).properties(height=280)
st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------- 사건 상세

st.subheader("탐지된 사건")

if not incidents:
    st.info("조건에 맞는 사건이 없습니다. 최소 지속 시간을 줄여보세요.")
    st.stop()

table = pd.DataFrame([{
    "서비스": i.service,
    "시작": i.start.strftime("%H:%M"),
    "종료": i.end.strftime("%H:%M"),
    "지속(분)": i.duration_min,
    "평균 에러율": round(float(i.rows["error_rate"].mean()), 3),
    "평균 p95(ms)": int(i.rows["latency_p95"].mean()),
} for i in incidents])
st.dataframe(table, width="stretch", hide_index=True)

labels = [f"{i.service}  {i.start:%H:%M}~{i.end:%H:%M}  ({i.duration_min}분)"
          for i in incidents]
choice = st.selectbox("상세히 볼 사건", range(len(incidents)),
                      format_func=lambda n: labels[n])
inc = incidents[choice]

left, right = st.columns([3, 2])

context_logs = summarize.pick_context_logs(df, inc)

with left:
    st.markdown("#### LLM 요약")
    if st.button("요약 생성", type="primary"):
        logs = context_logs
        prompt, numbered = summarize.build_prompt(inc, logs)
        answer = summarize.call_llm(prompt)
        check = summarize.verify_citations(answer, numbered)

        if check["ok"]:
            st.success(f"인용 검증 통과 · {check['cited_count']}개 인용 전부 실재")
        elif check["cited_count"] == 0:
            st.error("인용 없음 · 근거를 제시하지 않았습니다")
        else:
            st.error(f"존재하지 않는 로그 인용: {', '.join(check['invalid'])}")

        st.markdown(answer.replace("## ", "##### "))

        with st.expander(f"근거 로그 {len(numbered)}줄"):
            st.dataframe(logs, width="stretch", hide_index=True)
        with st.expander("실제로 보낸 프롬프트"):
            st.code(prompt, language="text")
    else:
        st.caption("버튼을 누르면 이 사건의 로그를 LLM에 보냅니다.")

with right:
    st.markdown("#### 관련 런북")
    index, n_chunks = cached_index()
    query = runbook.build_query(
        service=inc.service,
        messages=context_logs["message"].tolist(),
        error_rate=float(inc.rows["error_rate"].mean()),
        latency_p95=float(inc.rows["latency_p95"].mean()),
    )
    st.caption(f"질의: {query}")
    for hit in index.search(query, k=runbook.TOP_K):
        st.markdown(f"**{hit['score']:.3f}**  `{hit['doc']}` / {hit['section']}")
        with st.expander("내용 보기"):
            st.text(hit["text"])
    st.caption(f"색인: 문서 {len({c['doc'] for c in index.chunks})}개 · 조각 {n_chunks}개")
