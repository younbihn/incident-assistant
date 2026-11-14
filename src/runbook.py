"""
런북 검색 (RAG의 R 부분)

장애 상황을 질의로 만들어 대응 매뉴얼에서 관련 절차를 찾아온다.

한국어 문서를 다루므로 단어 단위가 아니라 글자 n-gram으로 쪼갠다.
형태소 분석기 없이도 "고갈"과 "고갈된"을 비슷하게 취급할 수 있다.
"""

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RUNBOOK_DIR = Path("runbook")
CHUNK_MIN_CHARS = 40          # 이보다 짧은 조각은 버린다
TOP_K = 3


# ─────────────────────────────────────────────
# 1. 문서 읽고 조각내기
# ─────────────────────────────────────────────

def load_chunks(dir_path: Path) -> list[dict]:
    """런북을 섹션(## 제목) 단위로 쪼갠다."""
    chunks = []
    for path in sorted(dir_path.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()

        # '## '로 시작하는 줄을 기준으로 자른다
        parts = re.split(r"\n(?=## )", text)
        for part in parts:
            body = part.strip()
            if len(body) < CHUNK_MIN_CHARS:
                continue
            heading = body.splitlines()[0].lstrip("# ").strip()
            chunks.append({
                "doc": path.stem,
                "title": title,
                "section": heading,
                "text": f"{title} {body}",   # 문서 제목을 섞어 맥락을 보강
            })
    return chunks


# ─────────────────────────────────────────────
# 2. 색인과 검색
# ─────────────────────────────────────────────

class RunbookIndex:
    def __init__(self, chunks: list[dict], **vectorizer_kw):
        self.chunks = chunks
        # char_wb = 단어 경계를 지키며 글자 n-gram을 만든다. 한국어에 유효
        options = {"analyzer": "char_wb", "ngram_range": (2, 4), "min_df": 1}
        options.update(vectorizer_kw)
        self.vectorizer = TfidfVectorizer(**options)
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in chunks)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in ranked]


# ─────────────────────────────────────────────
# 3. 장애 상황을 질의로 변환
# ─────────────────────────────────────────────

def build_query(service: str, messages: list[str], error_rate: float,
                latency_p95: float) -> str:
    """탐지 결과를 검색용 문장으로 만든다."""
    uniq = []
    for m in messages:
        if m not in uniq:
            uniq.append(m)
    symptom = []
    if error_rate > 0.3:
        symptom.append("오류가 급증한다")
    if latency_p95 > 1000:
        symptom.append("응답시간이 매우 느리다")
    if error_rate <= 0.05 and latency_p95 > 1000:
        symptom.append("오류는 없는데 느려진다")
    return f"{service} {' '.join(symptom)} {' '.join(uniq[:3])}"


# ─────────────────────────────────────────────
# 4. 평가
# ─────────────────────────────────────────────

EVAL_SET = [
    ("결제 쪽이 엄청 느려지면서 가끔 터지는데 DB 연결 문제인가", "RB-001-connection-pool"),
    ("세션이 안 닫혀서 생기는 문제 어떻게 보나", "RB-001-connection-pool"),
    ("파일 저장이 안 되는데 읽기는 잘 됨", "RB-002-disk-full"),
    ("로그 지웠는데 용량이 그대로다", "RB-002-disk-full"),
    ("타 기관 붙는 구간에서 10초씩 걸리다 실패함", "RB-003-upstream-timeout"),
    ("재시도를 계속하면 상대방에 부담이 되나", "RB-003-upstream-timeout"),
    ("에러는 안 나는데 사용자들이 느리다고 함", "RB-004-slow-response"),
    ("배포 이후로 반응이 굼뜨다", "RB-004-slow-response"),
    ("윗선에 언제 알려야 하는지 기준이 있나", "RB-005-escalation"),
    ("새벽 작업이 안 끝났을 때 다시 돌려도 되나", "RB-006-batch-failure"),
]


def evaluate(index: RunbookIndex, ks=(1, 3)) -> dict:
    """Recall@k를 계산한다. 정답 문서가 상위 k개 안에 들어오면 성공."""
    result = {}
    for k in ks:
        hit = 0
        misses = []
        for query, gold in EVAL_SET:
            got = [r["doc"] for r in index.search(query, k=k)]
            if gold in got:
                hit += 1
            else:
                misses.append((query, gold, got[0]))
        result[k] = {"recall": hit / len(EVAL_SET), "hit": hit,
                     "total": len(EVAL_SET), "misses": misses}
    return result


def compare_tokenizers(chunks: list[dict]) -> None:
    """단어 기준과 글자 n-gram을 같은 평가셋으로 비교한다."""
    settings = [
        ("단어 기준(기본 토크나이저)",
         {"analyzer": "word", "ngram_range": (1, 1), "min_df": 1}),
        ("글자 n-gram(char_wb 2~4)",
         {"analyzer": "char_wb", "ngram_range": (2, 4), "min_df": 1}),
    ]
    print(f"{'방식':<30}{'Recall@1':>10}{'Recall@3':>10}{'어휘수':>9}")
    print("-" * 60)
    for name, kw in settings:
        idx = RunbookIndex(chunks, **kw)
        res = evaluate(idx)
        print(f"{name:<30}{res[1]['recall']:>10.2f}{res[3]['recall']:>10.2f}"
              f"{len(idx.vectorizer.vocabulary_):>9}")


if __name__ == "__main__":
    chunks = load_chunks(RUNBOOK_DIR)
    docs = len({c["doc"] for c in chunks})
    print(f"런북 {docs}개 문서 → {len(chunks)}개 조각\n")

    index = RunbookIndex(chunks)

    print("검색 예시: 'connection pool exhausted 가 보이고 응답시간이 매우 느리다'")
    for r in index.search("connection pool exhausted 가 보이고 응답시간이 매우 느리다"):
        print(f"  {r['score']:.3f}  {r['doc']} / {r['section']}")

    print("\n토크나이저 비교")
    compare_tokenizers(chunks)

    print("\n기본 설정에서 @1이 틀린 질의")
    for q, gold, got in evaluate(index)[1]["misses"]:
        print(f"  {q[:30]:<32} 정답 {gold[:6]}  1위 {got[:6]}")
