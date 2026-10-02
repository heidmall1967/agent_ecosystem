"""Self-check for scoring logic. Run: uv run python -m eval.test_harness"""
import json
from eval.harness import hit_at_k, keyword_recall, percentile, summarize

assert keyword_recall("Reciprocal Rank Fusion merges ranks", ["reciprocal rank fusion"]) == 1.0
assert keyword_recall("context relevance only", ["context relevance", "faithfulness"]) == 0.5
assert hit_at_k([], ["a.md"], 5) is None
assert hit_at_k(["corpus/x.md", "corpus/mcp/a.md"], ["mcp/a.md"], 5) == 1.0
assert hit_at_k(["/abs/corpus/mcp/server/index.mdx"], ["mcp/server/index.mdx"], 5) == 1.0
assert hit_at_k(["corpus/mcp/client/index.mdx"], ["mcp/server/index.mdx"], 5) == 0.0  # same name, other dir
assert hit_at_k(["corpus/xmcp/a.md"], ["mcp/a.md"], 5) == 0.0
assert hit_at_k(["x.md", "a.md"], ["a.md"], 1) == 0.0
assert percentile([1, 2, 3, 4, 5], 0.5) == 3 and percentile([1, 2, 3, 4, 5], 0.95) == 5

s = summarize([
    {"type": "concept", "recall": 1.0, "hit": None, "latency_s": 1.0},
    {"type": "identifier", "recall": 0.5, "hit": None, "latency_s": 3.0},
])
assert s["all"]["answer_acc"] == 0.5 and s["all"]["hit@k"] is None
assert s["by_type"]["concept"]["answer_acc"] == 1.0
print("ok")

# golden-set validator
import tempfile
from pathlib import Path
from eval.check_golden import check

with tempfile.TemporaryDirectory() as d:
    (Path(d) / "mcp").mkdir()
    (Path(d) / "mcp/a.mdx").write_text("Servers MUST implement server/discover.")
    (Path(d) / "mcp/b.mdx").write_text("Tools are listed with tools/list.")
    row = lambda **kw: json.dumps({"id": "q1", "type": "identifier", "question": "?", "expected": ["server/discover"],
                                   "sources": ["mcp/a.mdx"]} | kw)
    e, w, t = check([row()], Path(d))
    assert not e and not w and t["identifier"] == 1
    e, _, _ = check([row(), row()], Path(d))
    assert any("duplicate" in x for x in e)
    e, _, _ = check([row(sources=["mcp/nope.mdx"])], Path(d))
    assert any("not found" in x for x in e)
    e, _, _ = check([row(type="multi-hop")], Path(d))
    assert any(">= 2 sources" in x for x in e)
    _, w, _ = check([row(expected=["tools/list"])], Path(d))
    assert any("not in its sources" in x for x in w)
    e, _, _ = check(["{bad"], Path(d))
    assert any("bad JSON" in x for x in e)
print("ok golden checker")
