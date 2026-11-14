"""
여러 시드에서 탐지 성능을 재서 한 시드에 우연히 맞춘 것이 아님을 확인한다.

시드마다 장애 4건의 위치와 길이가 달라지므로, 시드 5개면 장애 케이스 20건이 된다.
사용법: python src/evaluate_seeds.py
"""

import pandas as pd

from detect import aggregate, detect_by_model, detect_by_rule, mark_truth, score
from generate_logs import generate, random_incidents
from summarize import MIN_DURATION_MIN, group_incidents

SEEDS = [101, 102, 103, 104, 105]


def build(seed):
    """파일을 거치지 않고 메모리에서 로그를 만들어 집계까지 한다."""
    events, labels = generate(seed, random_incidents(seed))
    df = pd.DataFrame(events)
    df["ts"] = pd.to_datetime(df["ts"])
    df["is_error"] = df["status"] >= 500
    agg = aggregate(df)
    return agg, labels, mark_truth(agg, labels)


def case_detected(agg, pred, label):
    """장애 케이스 하나에서 이상으로 표시된 분이 한 분이라도 있으면 잡았다고 본다."""
    start, end = pd.to_datetime(label["start"]), pd.to_datetime(label["end"])
    inside = (agg["service"] == label["service"]) & (agg["ts"] >= start) & (agg["ts"] < end)
    return bool(pred[inside].any())


def main():
    rows, case_rule, case_model, cases = [], 0, 0, 0
    for seed in SEEDS:
        agg, labels, truth = build(seed)
        rule = detect_by_rule(agg)
        model = detect_by_model(agg)
        for name, pred in (("규칙", rule), ("IForest", model)):
            r = score(name, pred, truth)
            r["seed"] = seed
            rows.append(r)
        for lab in labels:
            cases += 1
            case_rule += case_detected(agg, rule, lab)
            case_model += case_detected(agg, model, lab)

    df = pd.DataFrame(rows)
    print("\n[시드별 결과]")
    print(df.pivot(index="seed", columns="name",
                   values=["precision", "recall", "f1"]).round(2).to_string())

    print("\n[5개 시드 평균]")
    print(df.groupby("name")[["precision", "recall", "f1"]].mean().round(2).to_string())
    print("\n[최악의 시드 (F1 기준)]")
    print(df.loc[df.groupby("name")["f1"].idxmin()][["name", "seed", "precision", "recall", "f1"]]
          .round(2).to_string(index=False))

    print(f"\n[장애 케이스 단위] 총 {cases}건 중 한 분이라도 잡은 케이스")
    print(f"  규칙 {case_rule}건 / IForest {case_model}건")

    print("\n[사건 단위: 화면에 실제로 보이는 사건 목록 (IForest, 3분 미만 제외)]")
    inc_rows = []
    for seed in SEEDS:
        agg, labels, _ = build(seed)
        incs = [i for i in group_incidents(agg, detect_by_model(agg)) if i.duration_min >= MIN_DURATION_MIN]
        real = 0
        for inc in incs:
            for lab in labels:
                if lab["service"] == inc.service and inc.start < pd.to_datetime(lab["end"]) and inc.end >= pd.to_datetime(lab["start"]):
                    real += 1
                    break
        inc_rows.append({"seed": seed, "보인 사건": len(incs), "진짜": real, "가짜": len(incs) - real, "정답 장애": len(labels)})
    print(pd.DataFrame(inc_rows).to_string(index=False))

    print("\n[임계값을 바꿔 가며: 규칙 기반 에러율 임계값 (5개 시드 평균)]")
    sweep = []
    for th in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        vals = []
        for seed in SEEDS:
            agg, _, truth = build(seed)
            vals.append(score("r", agg["error_rate"] > th, truth))
        v = pd.DataFrame(vals)[["precision", "recall", "f1"]].mean()
        sweep.append({"임계값": th, **v.round(2).to_dict()})
    print(pd.DataFrame(sweep).to_string(index=False))

    print("\n[임계값을 바꿔 가며: IForest contamination (5개 시드 평균)]")
    sweep = []
    for c in (0.04, 0.08, 0.12, 0.20):
        vals = []
        for seed in SEEDS:
            agg, _, truth = build(seed)
            vals.append(score("m", detect_by_model(agg, c), truth))
        v = pd.DataFrame(vals)[["precision", "recall", "f1"]].mean()
        sweep.append({"contamination": c, **v.round(2).to_dict()})
    print(pd.DataFrame(sweep).to_string(index=False))


if __name__ == "__main__":
    main()
