"""Command line for the gold-set evals.

    python -m evals.cli fetch
    python -m evals.cli all --embedding hash --reranker lexical
    python -m evals.cli all --embedding sentence-transformers --reranker cross-encoder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.harness import (
    environment,
    evaluate,
    fetch,
    load_manifest,
    missing_pdfs,
    probe_llm,
    run_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download gold PDFs from the source URLs")
    sub.add_parser("check", help="fail if a label anchor is missing or a checksum differs")
    all_cmd = sub.add_parser("all", help="retrieval, extraction, eligibility, faithfulness, latency")
    all_cmd.add_argument("--embedding", default="hash")
    all_cmd.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    all_cmd.add_argument("--reranker", default="lexical")
    all_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    all_cmd.add_argument("--out", default="evals/results/measured.json")
    all_cmd.add_argument("--skip-cases", action="store_true")
    args = parser.parse_args()

    if args.command == "fetch":
        paths = fetch()
        print(f"fetched {len(paths)} PDFs into {paths[0].parent}")
        return
    if args.command == "check":
        _check()
        print("gold labels resolve inside the parsed PDFs")
        return
    if args.command == "all":
        from bidready.config import Settings

        settings = Settings.from_env()
        # The scored run must use the providers named on the command line, not a stale env.
        settings = _with_providers(settings, args)
        scored = evaluate(
            embedding_provider=args.embedding,
            embedding_model=args.embedding_model,
            reranker_name=args.reranker,
            reranker_model=args.reranker_model,
        )
        cases = None if args.skip_cases else run_cases(settings)
        payload = {
            "environment": environment(),
            "llm_prompt_comparison": probe_llm(),
            **scored,
            "cases": cases,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(out)
        _print_summary(payload)


def _check() -> None:
    from bidready.rag import HashEmbedder, LexicalReranker
    from evals.harness import index_tender, pdf_dir

    manifest = load_manifest()
    missing = missing_pdfs(manifest)
    if missing:
        raise SystemExit("missing PDFs: " + ", ".join(missing))
    folder = pdf_dir(manifest)
    embedder = HashEmbedder()
    reranker = LexicalReranker()
    for tender in manifest["tenders"]:
        _, chunks, _ = index_tender(folder / tender["filename"], embedder, reranker)
        for item in list(tender["requirements"]) + list(tender["retrieval_queries"]):
            phrases = item["match_all"]
            if not any(all(phrase.lower() in chunk.text.lower() for phrase in phrases) for chunk in chunks):
                raise SystemExit(f"anchor missing after parse: {tender['id']} {item['id']} {phrases}")


def _with_providers(settings, args):
    from dataclasses import replace

    return replace(
        settings,
        embedding_provider=args.embedding,
        embedding_model=args.embedding_model,
        reranker=args.reranker,
        reranker_model=args.reranker_model,
    )


def _print_summary(payload: dict) -> None:
    retrieval = payload["retrieval"]
    for mode, scores in retrieval.items():
        print(
            f"retrieval {mode}: recall@5={scores['recall@5']:.3f} "
            f"recall@10={scores['recall@10']:.3f} mrr={scores['mrr']:.3f} n={scores['queries']}"
        )
    extraction = payload["extraction"]
    print(
        f"extraction precision={extraction['precision']} recall={extraction['recall']} "
        f"matched={extraction['matched']}/{extraction['gold']} predicted={extraction['predicted']}"
    )
    eligibility = payload["eligibility"]
    print(f"eligibility accuracy={eligibility['accuracy']} n={eligibility['n']}")
    cases = payload.get("cases") or {}
    faith = cases.get("faithfulness")
    if faith:
        print(f"faithfulness {faith['supported']}/{faith['claims']} = {faith['rate']}")


if __name__ == "__main__":
    main()
