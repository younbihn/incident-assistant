"""
로그 이상 탐지

1분 × 서비스 단위로 로그를 집계하고, 두 가지 방법으로 이상 구간을 찾는다.
  (1) 규칙 기반: 에러율이 임계값을 넘으면 이상
  (2) IsolationForest: 여러 지표를 함께 보고 "흔하지 않은" 구간을 찾음
마지막에 labels.json 정답과 대조해 두 방법의 성적을 매긴다.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = Path("data")
ERROR_RATE_THRESHOLD = 0.30     # 규칙 기반: 분당 에러율 30% 초과
CONTAMINATION = 0.08            # IsolationForest: 전체의 몇 %가 이상인지 가정

FEATURES = ["req_count", "error_rate", "latency_mean", "latency_p95"]


def load_logs(path: Path) -> pd.DataFrame:
    """JSONL 로그를 DataFrame으로 읽는다."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df["is_error"] = df["status"] >= 500
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """1분 × 서비스 단위로 묶어 지표를 만든다."""
    g = df.groupby([pd.Grouper(key="ts", freq="1min"), "service"])
    agg = g.agg(
        req_count=("status", "size"),
        error_count=("is_error", "sum"),
        latency_mean=("latency_ms", "mean"),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
    ).reset_index()
    agg["error_rate"] = agg["error_count"] / agg["req_count"]
    return agg.dropna()


def detect_by_rule(agg: pd.DataFrame) -> pd.Series:
    """규칙 기반: 에러율이 임계값을 넘으면 이상."""
    return agg["error_rate"] > ERROR_RATE_THRESHOLD


def detect_by_model(agg: pd.DataFrame, contamination: float = CONTAMINATION) -> pd.Series:
    """IsolationForest: 여러 지표를 함께 보고 드문 구간을 찾는다.

    contamination 을 인자로 받아 화면에서 슬라이더로 조절할 수 있게 한다.
    생략하면 모듈 상수를 그대로 쓰므로 기존 호출부는 바꿀 필요가 없다.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(agg[FEATURES])   # -1 이상, 1 정상
    return pd.Series(pred == -1, index=agg.index)


def mark_truth(agg: pd.DataFrame, labels: list) -> pd.Series:
    """정답 라벨을 보고 각 행이 실제 장애 구간인지 표시한다."""
    truth = pd.Series(False, index=agg.index)
    for lab in labels:
        start = pd.to_datetime(lab["start"])
        end = pd.to_datetime(lab["end"])
        hit = (agg["service"] == lab["service"]) & (agg["ts"] >= start) & (agg["ts"] < end)
        truth = truth | hit
    return truth


def score(name: str, pred: pd.Series, truth: pd.Series) -> dict:
    """정밀도와 재현율을 계산한다."""
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"name": name, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def print_report(results: list) -> None:
    print(f"\n{'방법':<22}{'TP':>5}{'FP':>5}{'FN':>5}{'정밀도':>10}{'재현율':>10}{'F1':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<22}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
              f"{r['precision']:>10.2f}{r['recall']:>10.2f}{r['f1']:>8.2f}")


def print_missed(agg, pred, truth, labels, name):
    """놓친 장애가 어떤 유형인지 보여준다."""
    missed = agg[~pred & truth]
    if missed.empty:
        print(f"\n[{name}] 놓친 구간 없음")
        return
    print(f"\n[{name}] 놓친 구간 {len(missed)}건")
    for svc, cnt in missed["service"].value_counts().items():
        kind = next((l["type"] for l in labels if l["service"] == svc), "?")
        print(f"  {svc:<16} {cnt:>3}분  ({kind})")


if __name__ == "__main__":
    df = load_logs(DATA_DIR / "logs.jsonl")
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))

    agg = aggregate(df)
    print(f"집계 결과: {len(agg)}행 (1분 × 서비스)")

    truth = mark_truth(agg, labels)
    print(f"실제 장애 구간: {int(truth.sum())}행")

    rule_pred = detect_by_rule(agg)
    model_pred = detect_by_model(agg)

    results = [
        score("규칙 기반(에러율)", rule_pred, truth),
        score("IsolationForest", model_pred, truth),
    ]
    print_report(results)
    print_missed(agg, rule_pred, truth, labels, "규칙 기반")
    print_missed(agg, model_pred, truth, labels, "IsolationForest")
