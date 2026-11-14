"""
"분당 요청이 N건 미만이면 판단하지 않는다"를 규칙에 넣으면 오탐이 줄어드는지 시험한다.
사용법: python src/min_requests.py
"""

import json
from pathlib import Path

import pandas as pd

from detect import ERROR_RATE_THRESHOLD, DATA_DIR, aggregate, load_logs, mark_truth, score
from evaluate_seeds import SEEDS, build


def rule_with_min_requests(agg, n):
    """에러율 기준 규칙에 최소 요청 수 조건을 더한다."""
    return (agg["error_rate"] > ERROR_RATE_THRESHOLD) & (agg["req_count"] >= n)


def main():
    df = load_logs(Path(DATA_DIR) / "logs.jsonl")
    labels = json.loads((Path(DATA_DIR) / "labels.json").read_text(encoding="utf-8"))
    agg42 = aggregate(df)
    truth42 = mark_truth(agg42, labels)
    others = [build(seed) for seed in SEEDS]       # (agg, labels, truth)

    rows = []
    for n in (1, 3, 4, 5):                          # 1이면 조건이 없는 것과 같다
        s42 = score("r", rule_with_min_requests(agg42, n), truth42)
        avg = pd.DataFrame([score("r", rule_with_min_requests(a, n), t) for a, _, t in others])
        rows.append({"최소 요청 수": n, "오탐(42)": s42["fp"], "놓침(42)": s42["fn"],
                     "F1(42)": round(s42["f1"], 2),
                     "F1(5시드 평균)": round(avg["f1"].mean(), 2),
                     "재현율(5시드 평균)": round(avg["recall"].mean(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
