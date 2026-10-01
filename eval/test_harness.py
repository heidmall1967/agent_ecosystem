"""Self-check for scoring logic. Run: uv run python -m eval.test_harness"""
from eval.harness import hit_at_k, keyword_recall, percentile, summarize

assert keyword_recall("Reciprocal Rank Fusion merges ranks", ["reciprocal rank fusion"]) == 1.0
assert keyword_recall("context relevance only", ["context relevance", "faithfulness"]) == 0.5
assert hit_at_k([], ["a.md"], 5) is None
assert hit_at_k(["corpus/x.md", "corpus/a.md"], ["a.md"], 5) == 1.0
assert hit_at_k(["x.md", "a.md"], ["a.md"], 1) == 0.0
assert percentile([1, 2, 3, 4, 5], 0.5) == 3 and percentile([1, 2, 3, 4, 5], 0.95) == 5

s = summarize([
    {"type": "concept", "recall": 1.0, "hit": None, "latency_s": 1.0},
    {"type": "identifier", "recall": 0.5, "hit": None, "latency_s": 3.0},
])
assert s["all"]["answer_acc"] == 0.5 and s["all"]["hit@k"] is None
assert s["by_type"]["concept"]["answer_acc"] == 1.0
print("ok")
