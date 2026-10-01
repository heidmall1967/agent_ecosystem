# full_path

Learn-by-implementing AI engineering path. Plan: [blueprint](ai_ecosystem_learning_paths_and_stitched_architecture_blueprint.md).

```
corpus/      documents the system answers over (put your own docs here)
eval/        golden.jsonl + harness: every level is measured here
pipelines/   one module per pipeline version: run(question) -> {answer, sources}
notes/adr/   one ADR per level (copy 0000-template.md)
results/     eval outputs (gitignored)
```

```bash
uv run python -m eval.test_harness          # scoring self-check
uv run python -m eval.harness baseline      # run a pipeline over the golden set
LLM_MODEL=qwen2.5-fast:latest uv run python -m eval.harness baseline --limit 2
```

Golden set row: `{"id", "type": "identifier|concept|multi-hop", "question", "expected": [keywords], "sources": [file names]}`.
Answer counts as correct when every expected keyword appears (case-insensitive). `hit@k` is
whether a gold source file is among the first k retrieved sources (null for no-retrieval pipelines).

## Running itinerary (PDF)

Week-by-week plan with checklists, assignments, quizzes (answer key at the end) and architect questions.
Update progress in `itinerary/plan.toml` (status, hours, quiz_score, `[x] ` on tasks), then rebuild:

```bash
uv run python itinerary/build.py   # -> itinerary/itinerary.pdf, pulls latest results/ and notes/adr/
```
