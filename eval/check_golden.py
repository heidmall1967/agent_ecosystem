"""Validate the golden set. Usage: uv run python -m eval.check_golden [path]

Errors (exit 1): bad JSON, missing fields, duplicate ids, unknown type, missing source files,
multi-hop with < 2 sources. Warnings: expected keyword not found in any listed source
(question may be ungrounded), keyword so common it would match wrong answers, type balance.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
TYPES = {"identifier", "concept", "multi-hop"}
TARGET = {"identifier": 13, "concept": 20, "multi-hop": 3}
TOTAL = 40


def check(lines: list[str], corpus: Path) -> tuple[list[str], list[str], Counter]:
    errors, warns, types, ids = [], [], Counter(), set()
    docs = {p.relative_to(corpus).as_posix(): p.read_text(errors="ignore").lower()
            for p in corpus.rglob("*") if p.suffix in {".md", ".mdx", ".txt", ".py"}}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            g = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: bad JSON ({e.msg})")
            continue
        missing = {"id", "type", "question", "expected", "sources"} - g.keys()
        if missing:
            errors.append(f"line {n}: missing {sorted(missing)}")
            continue
        qid = g["id"]
        if qid in ids:
            errors.append(f"{qid}: duplicate id")
        ids.add(qid)
        if g["type"] not in TYPES:
            errors.append(f"{qid}: type {g['type']!r} not in {sorted(TYPES)}")
        types[g["type"]] += 1
        if not g["expected"] or not all(isinstance(k, str) and k.strip() for k in g["expected"]):
            errors.append(f"{qid}: expected must be a non-empty list of non-empty strings")
            continue
        bad = [s for s in g["sources"] if s not in docs]
        if bad or not g["sources"]:
            errors.append(f"{qid}: sources not found under corpus/: {bad or '(empty)'}")
            continue
        if g["type"] == "multi-hop" and len(g["sources"]) < 2:
            errors.append(f"{qid}: multi-hop needs >= 2 sources")
        for kw in g["expected"]:
            k = kw.lower()
            if not any(k in docs[s] for s in g["sources"]):
                warns.append(f"{qid}: keyword {kw!r} not in its sources (paraphrase, or wrong source?)")
            share = sum(k in d for d in docs.values()) / len(docs)
            if share > 0.5:
                warns.append(f"{qid}: keyword {kw!r} appears in {share:.0%} of docs; a wrong answer could match it")
    return errors, warns, types


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval/golden.jsonl"
    errors, warns, types = check(path.read_text().splitlines(), CORPUS)
    for e in errors:
        print("ERROR ", e)
    for w in warns:
        print("warn  ", w)
    total = sum(types.values())
    balance = "  ".join(f"{t}: {types[t]}/{TARGET[t]}+" for t in TARGET)
    print(f"\n{total}/{TOTAL} questions   {balance}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
