"""Golden-set eval harness. Usage: uv run python -m eval.harness <pipeline> [-k 5] [--limit N]

A pipeline is any module in pipelines/ exposing run(question) -> {"answer": str, "sources": [str]}.
"""
import argparse
import importlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def keyword_recall(answer: str, expected: list[str]) -> float:
    a = answer.lower()
    return sum(kw.lower() in a for kw in expected) / len(expected)


def hit_at_k(sources: list[str], gold: list[str], k: int) -> float | None:
    if not sources:
        return None  # pipeline does no retrieval
    top = [Path(s).name for s in sources[:k]]
    return float(any(Path(g).name in top for g in gold))


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))]


def summarize(rows: list[dict]) -> dict:
    def agg(rs):
        hits = [r["hit"] for r in rs if r["hit"] is not None]
        return {
            "n": len(rs),
            "answer_acc": statistics.mean(r["recall"] == 1.0 for r in rs),
            "keyword_recall": statistics.mean(r["recall"] for r in rs),
            "hit@k": statistics.mean(hits) if hits else None,
        }
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    lat = [r["latency_s"] for r in rows]
    return {"all": agg(rows), "by_type": {t: agg(rs) for t, rs in by_type.items()},
            "latency_p50_s": percentile(lat, 0.5), "latency_p95_s": percentile(lat, 0.95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--golden", default=ROOT / "eval/golden.jsonl", type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    pipe = importlib.import_module(f"pipelines.{args.pipeline}")
    golden = [json.loads(l) for l in args.golden.read_text().splitlines() if l.strip()][:args.limit]

    rows = []
    for g in golden:
        t0 = time.perf_counter()
        out = pipe.run(g["question"])
        rows.append({"id": g["id"], "type": g["type"], "latency_s": time.perf_counter() - t0,
                     "recall": keyword_recall(out["answer"], g["expected"]),
                     "hit": hit_at_k(out.get("sources", []), g["sources"], args.k),
                     "answer": out["answer"], "sources": out.get("sources", [])})
        print(f'{g["id"]}  recall={rows[-1]["recall"]:.2f}  hit={rows[-1]["hit"]}  {rows[-1]["latency_s"]:.1f}s')

    summary = summarize(rows)
    out_path = ROOT / "results" / f"{args.pipeline}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"pipeline": args.pipeline, "k": args.k, "summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
