"""
오탐(정상인데 이상이라고 한 분)이 어떤 모양인지 들여다본다.
사용법: python src/false_positives.py
"""

import json
from pathlib import Path

import pandas as pd

from detect import DATA_DIR, aggregate, detect_by_model, detect_by_rule, load_logs, mark_truth


def main():
    df = load_logs(Path(DATA_DIR) / "logs.jsonl")
    labels = json.loads((Path(DATA_DIR) / "labels.json").read_text(encoding="utf-8"))
    agg = aggregate(df)
    truth = mark_truth(agg, labels)

    normal = agg[~truth]
    print("[정상 구간의 평소 값]  평균 / 최대")
    for col in ("req_count", "error_rate", "latency_mean", "latency_p95"):
        print(f"  {col:<13} {normal[col].mean():>9.2f} / {normal[col].max():>9.2f}")

    for name, pred in (("규칙 기반", detect_by_rule(agg)), ("IsolationForest", detect_by_model(agg))):
        fp = agg[pred & ~truth].copy()
        print(f"\n[{name}] 오탐 {len(fp)}분")
        if fp.empty:
            continue
        print("  서비스별:", fp["service"].value_counts().to_dict())
        # 어떤 지표가 평소에서 가장 멀리 벗어났는지: (값 - 평균) / 표준편차
        feats = ["req_count", "error_rate", "latency_mean", "latency_p95"]
        z = (fp[feats] - normal[feats].mean()) / normal[feats].std()
        fp["가장 튄 지표"] = z.abs().idxmax(axis=1)
        fp["벗어난 정도(σ)"] = z.abs().max(axis=1).round(1)
        print("  가장 튄 지표:", fp["가장 튄 지표"].value_counts().to_dict())
        show = fp.sort_values("벗어난 정도(σ)", ascending=False).head(4).copy()
        show["ts"] = show["ts"].dt.strftime("%H:%M")
        cols = ["ts", "service", "req_count", "error_rate", "latency_p95", "가장 튄 지표", "벗어난 정도(σ)"]
        print(show[cols].round(2).to_string(index=False))

        # 오탐이 장애 구간 바로 옆(앞뒤 2분)에 붙어 있는지
        near = 0
        for _, r in fp.iterrows():
            for lab in labels:
                if lab["service"] != r["service"]:
                    continue
                s, e = pd.to_datetime(lab["start"]), pd.to_datetime(lab["end"])
                if s - pd.Timedelta(minutes=2) <= r["ts"] < e + pd.Timedelta(minutes=2):
                    near += 1
                    break
        print(f"  장애 구간 앞뒤 2분 이내: {near}분 / 그 밖의 순수 잡음: {len(fp) - near}분")


if __name__ == "__main__":
    main()
