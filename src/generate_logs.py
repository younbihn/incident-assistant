import json, random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
TOTAL_MINUTES = 360
EVENTS_PER_MINUTE = 20
START_AT = datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc)
SERVICES = ["auth-api", "payment-api", "batch-worker", "file-gateway"]
OUT_DIR = Path("data")

INCIDENTS = [
    {"type": "connection_pool_exhausted", "service": "payment-api", "start_min": 75, "duration_min": 20},
    {"type": "disk_full", "service": "batch-worker", "start_min": 190, "duration_min": 35},
    {"type": "upstream_timeout", "service": "file-gateway", "start_min": 290, "duration_min": 15},
    {"type": "slow_degradation", "service": "auth-api", "start_min": 240, "duration_min": 25},
]

def make_normal_event(ts, service):
    latency = max(5, int(random.gauss(120, 40)))
    status = random.choices([200, 400, 500], weights=[97, 2, 1], k=1)[0]
    level = "INFO" if status < 400 else "ERROR"
    return {"ts": ts.isoformat(), "service": service, "level": level,
            "status": status, "latency_ms": latency, "message": "request completed"}

def make_incident_event(ts, incident):
    kind, service = incident["type"], incident["service"]
    base = {"ts": ts.isoformat(), "service": service, "level": "ERROR"}
    if kind == "connection_pool_exhausted":
        failed = random.random() < 0.7
        return {**base, "level": "ERROR" if failed else "WARN",
                "status": 500 if failed else 200,
                "latency_ms": int(random.gauss(4200, 900)),
                "message": "connection pool exhausted (active=50, idle=0, wait=30s)"}
    if kind == "disk_full":
        return {**base, "status": 500, "latency_ms": int(random.gauss(90, 25)),
                "message": "write failed: No space left on device (/data 100% used)"}
    if kind == "slow_degradation":
        return {**base, "level": "WARN", "status": 200,
                "latency_ms": int(random.gauss(1800, 300)),
                "message": "slow response detected"}
    if kind == "upstream_timeout":
        return {**base, "status": 504, "latency_ms": int(random.gauss(10050, 120)),
                "message": "upstream did not respond within 10000ms"}
    raise ValueError(f"알 수 없는 장애 유형: {kind}")

def random_incidents(seed):
    """시드마다 장애 위치와 길이를 바꾼다. 서로 시간이 겹치지 않게 뽑는다."""
    rng = random.Random(seed * 1000 + 7)
    spans = {"connection_pool_exhausted": (15, 40), "disk_full": (15, 40),
             "upstream_timeout": (10, 30), "slow_degradation": (15, 40)}
    homes = {"connection_pool_exhausted": "payment-api", "disk_full": "batch-worker",
             "upstream_timeout": "file-gateway", "slow_degradation": "auth-api"}
    while True:
        picked = []
        for kind, (lo, hi) in spans.items():
            dur = rng.randint(lo, hi)
            picked.append({"type": kind, "service": homes[kind],
                           "start_min": rng.randint(10, TOTAL_MINUTES - dur - 10),
                           "duration_min": dur})
        ordered = sorted(picked, key=lambda i: i["start_min"])
        gaps_ok = all(a["start_min"] + a["duration_min"] + 5 <= b["start_min"]
                      for a, b in zip(ordered, ordered[1:]))
        if gaps_ok:
            return ordered


def generate(seed=SEED, incidents=None):
    """seed 와 incidents 를 생략하면 기존과 똑같은 데이터가 나온다."""
    incidents = INCIDENTS if incidents is None else incidents
    random.seed(seed)
    events = []
    for minute in range(TOTAL_MINUTES):
        now = START_AT + timedelta(minutes=minute)
        active = None
        for inc in incidents:
            if inc["start_min"] <= minute < inc["start_min"] + inc["duration_min"]:
                active = inc; break
        count = max(1, int(random.gauss(EVENTS_PER_MINUTE, 4)))
        for _ in range(count):
            ts = now + timedelta(seconds=random.randint(0, 59))
            service = random.choice(SERVICES)
            if active and service == active["service"]:
                events.append(make_incident_event(ts, active))
            else:
                events.append(make_normal_event(ts, service))
    events.sort(key=lambda e: e["ts"])
    labels = [{"type": i["type"], "service": i["service"],
               "start": (START_AT + timedelta(minutes=i["start_min"])).isoformat(),
               "end": (START_AT + timedelta(minutes=i["start_min"]+i["duration_min"])).isoformat()}
              for i in incidents]
    return events, labels

def save(events, labels, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    with open(out_dir/"logs.jsonl", "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    with open(out_dir/"labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"로그 {len(events)}건 → {out_dir/'logs.jsonl'}")
    print(f"정답 라벨 {len(labels)}건 → {out_dir/'labels.json'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        e, l = generate(); save(e, l)          # 기본: 시드 42, 고정된 장애 위치
    else:
        seed = int(sys.argv[1])                 # 예: python src/generate_logs.py 7 data_s7
        out = sys.argv[2] if len(sys.argv) > 2 else f"data_s{seed}"
        e, l = generate(seed, random_incidents(seed)); save(e, l, out)
