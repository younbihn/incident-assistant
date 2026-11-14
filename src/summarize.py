"""
장애 구간 요약

이상 탐지가 찾은 구간의 로그를 추려 LLM에게 넘기고,
"무엇이 언제 왜 났는지"를 사람 말로 요약받는다.

핵심 설계:
  - LLM 호출을 추상화해 Mock과 실제 API를 바꿔 끼울 수 있게 한다
  - 요약문이 반드시 근거 로그 번호를 인용하게 만든다
  - 인용한 번호가 실제로 존재하는지 자동 검증한다 (환각 차단)
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from detect import load_logs, aggregate, detect_by_model, DATA_DIR

MAX_CONTEXT_LOGS = 25          # 프롬프트에 넣을 로그 최대 줄 수
GAP_TOLERANCE_MIN = 2          # 이 분 이내로 끊기면 같은 사건으로 본다
MIN_DURATION_MIN = 3           # 이보다 짧은 구간은 잡음으로 보고 건너뛴다
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock")   # mock | anthropic | openai


# ─────────────────────────────────────────────
# 1. 탐지 구간을 사건 단위로 묶기
# ─────────────────────────────────────────────

@dataclass
class Incident:
    """연속된 이상 구간 하나."""
    service: str
    start: pd.Timestamp
    end: pd.Timestamp
    rows: pd.DataFrame = field(repr=False)

    @property
    def duration_min(self) -> int:
        return int((self.end - self.start).total_seconds() // 60) + 1


def group_incidents(agg: pd.DataFrame, pred: pd.Series) -> list[Incident]:
    """이상으로 표시된 분들을 서비스별로 묶어 연속 구간으로 만든다."""
    flagged = agg[pred].sort_values(["service", "ts"])
    incidents = []

    for service, grp in flagged.groupby("service"):
        grp = grp.sort_values("ts")
        # 직전 행과 GAP_TOLERANCE_MIN분 넘게 벌어지면 새 사건으로 끊는다
        gap = grp["ts"].diff() > pd.Timedelta(minutes=GAP_TOLERANCE_MIN)
        block_id = gap.cumsum()

        for _, block in grp.groupby(block_id):
            incidents.append(Incident(
                service=service,
                start=block["ts"].min(),
                end=block["ts"].max(),
                rows=block,
            ))

    return sorted(incidents, key=lambda i: i.start)


# ─────────────────────────────────────────────
# 2. 프롬프트 만들기
# ─────────────────────────────────────────────

def pick_context_logs(df: pd.DataFrame, inc: Incident) -> pd.DataFrame:
    """구간에 속한 로그 중 대표적인 것만 추린다."""
    window = df[
        (df["service"] == inc.service)
        & (df["ts"] >= inc.start)
        & (df["ts"] <= inc.end + pd.Timedelta(minutes=1))
    ].copy()

    errors = window[window["status"] >= 500]
    slow = window.nlargest(min(10, len(window)), "latency_ms")
    picked = pd.concat([errors.head(15), slow]).drop_duplicates()
    return picked.sort_values("ts").head(MAX_CONTEXT_LOGS).reset_index(drop=True)


def build_prompt(inc: Incident, logs: pd.DataFrame) -> tuple[str, dict]:
    """프롬프트와 '인용 가능한 로그 목록'을 함께 만든다."""
    numbered = {}
    lines = []
    for i, row in logs.iterrows():
        tag = f"L{i + 1}"
        numbered[tag] = row.to_dict()
        lines.append(
            f"[{tag}] {row['ts']} {row['service']} "
            f"status={row['status']} latency={row['latency_ms']}ms "
            f"{row['message']}"
        )

    stats = inc.rows[["req_count", "error_rate", "latency_mean", "latency_p95"]].mean()

    prompt = f"""당신은 공공 시스템 운영 담당자를 돕는 장애 분석 보조 도구입니다.

[탐지된 구간]
서비스: {inc.service}
시간: {inc.start} ~ {inc.end} ({inc.duration_min}분)
분당 평균 요청 수: {stats['req_count']:.1f}
분당 평균 에러율: {stats['error_rate']:.1%}
평균 응답시간: {stats['latency_mean']:.0f}ms
p95 응답시간: {stats['latency_p95']:.0f}ms

[로그]
{chr(10).join(lines)}

[작성 규칙]
1. 무슨 일이 언제 일어났는지 2~3문장으로 요약하세요.
2. 가능한 원인을 최대 2개까지 제시하고, 각각에 확신도(높음/보통/낮음)를 붙이세요.
3. 모든 주장에는 근거 로그 번호를 [L1] 형식으로 반드시 인용하세요.
4. 위 로그에 없는 내용은 절대 추측해서 쓰지 마세요.
5. 아래 형식을 그대로 지키세요.

## 요약
(2~3문장)

## 추정 원인
1. (원인) — 확신도: (높음/보통/낮음) — 근거: [L?] [L?]
2. ...

## 다음 확인 사항
- (무엇을 더 봐야 하는지 1~2개)
"""
    return prompt, numbered


# ─────────────────────────────────────────────
# 3. LLM 호출 (Mock / 실제 교체 가능)
# ─────────────────────────────────────────────

def call_mock(prompt: str) -> str:
    """API 키 없이 파이프라인을 끝까지 돌려보기 위한 가짜 응답."""
    tags = re.findall(r"\[(L\d+)\]", prompt)
    sample = tags[:3] if tags else ["L1"]
    ref = " ".join(f"[{t}]" for t in sample)
    return f"""## 요약
탐지 구간에서 오류 응답과 응답시간 증가가 함께 관측되었습니다. {ref}

## 추정 원인
1. 자원 고갈로 인한 처리 지연 — 확신도: 보통 — 근거: {ref}
2. 외부 연계 지연 — 확신도: 낮음 — 근거: [{sample[0]}]

## 다음 확인 사항
- 해당 시간대 커넥션 풀 사용률
- 디스크 여유 공간
"""


def call_anthropic(prompt: str) -> str:
    """실제 호출. ANTHROPIC_API_KEY 필요."""
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = resp.usage
    print(f"  토큰: 입력 {usage.input_tokens} / 출력 {usage.output_tokens}")
    return resp.content[0].text


BACKENDS = {"mock": call_mock, "anthropic": call_anthropic}


def call_llm(prompt: str) -> str:
    fn = BACKENDS.get(LLM_BACKEND)
    if fn is None:
        raise ValueError(f"알 수 없는 백엔드: {LLM_BACKEND}")
    return fn(prompt)


# ─────────────────────────────────────────────
# 4. 인용 검증 (환각 차단)
# ─────────────────────────────────────────────

def verify_citations(answer: str, numbered: dict) -> dict:
    """요약문이 인용한 로그 번호가 실제로 존재하는지 확인한다."""
    cited = re.findall(r"\[(L\d+)\]", answer)
    unique = sorted(set(cited), key=lambda t: int(t[1:]))
    valid = [t for t in unique if t in numbered]
    invalid = [t for t in unique if t not in numbered]
    return {
        "cited_count": len(unique),
        "valid": valid,
        "invalid": invalid,
        "ok": len(invalid) == 0 and len(unique) > 0,
    }


def report(inc: Incident, answer: str, check: dict) -> None:
    print("=" * 70)
    print(f"{inc.service}  {inc.start:%H:%M} ~ {inc.end:%H:%M}  ({inc.duration_min}분)")
    print("=" * 70)
    print(answer.strip())
    print("-" * 70)
    if check["ok"]:
        print(f"✓ 인용 검증 통과 — {check['cited_count']}개 인용, 전부 실재")
    elif check["cited_count"] == 0:
        print("✗ 인용 없음 — 근거를 제시하지 않았습니다")
    else:
        print(f"✗ 존재하지 않는 로그 인용: {', '.join(check['invalid'])}")
    print()


if __name__ == "__main__":
    df = load_logs(DATA_DIR / "logs.jsonl")
    agg = aggregate(df)
    pred = detect_by_model(agg)

    incidents = group_incidents(agg, pred)
    targets = [i for i in incidents if i.duration_min >= MIN_DURATION_MIN]
    print(f"탐지 구간 {len(incidents)}개 중 {MIN_DURATION_MIN}분 이상 {len(targets)}건을 요약합니다 "
          f"(나머지는 짧은 잡음, 백엔드: {LLM_BACKEND})\n")

    passed = 0
    for inc in targets:
        logs = pick_context_logs(df, inc)
        prompt, numbered = build_prompt(inc, logs)
        answer = call_llm(prompt)
        check = verify_citations(answer, numbered)
        report(inc, answer, check)
        passed += int(check["ok"])

    print(f"인용 검증 통과: {passed}건")
