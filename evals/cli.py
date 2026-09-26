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
    llm_eligibility,
    load_manifest,
    missing_pdfs,
    probe_llm,
    run_cases,
    score_nli,
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
    local_cmd = sub.add_parser("local", help="score retrieval and the pipeline with a local model")
    local_cmd.add_argument("--llm-model", default="qwen2.5:3b")
    local_cmd.add_argument("--prompts", default="v1,v2")
    local_cmd.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    local_cmd.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    local_cmd.add_argument("--out", default="evals/results/local.json")
    local_cmd.add_argument("--tenders", default="", help="comma-separated tender ids; empty means all")
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
        return
    if args.command == "local":
        _local(args)


def _local(args) -> None:
    from dataclasses import replace

    from bidready.config import Settings
    from bidready.llm import build_llm
    from bidready.rag import build_embedder, build_reranker

    settings = Settings.from_env()
    prompts = [item.strip() for item in args.prompts.split(",") if item.strip()]
    settings = replace(
        settings,
        llm_provider="ollama",
        llm_model=args.llm_model,
        embedding_provider="sentence-transformers",
        embedding_model=args.embedding_model,
        reranker="cross-encoder",
        reranker_model=args.reranker_model,
        heuristic_fallback=False,
        prompt_version=prompts[0],
    )
    tender_ids = [item.strip() for item in args.tenders.split(",") if item.strip()] or None
    print("retrieval with", args.embedding_model, "and", args.reranker_model, flush=True)
    scored = evaluate(
        embedding_provider="sentence-transformers",
        embedding_model=args.embedding_model,
        reranker_name="cross-encoder",
        reranker_model=args.reranker_model,
    )
    if tender_ids:
        wanted = set(tender_ids)
        scored["tenders"] = [row for row in scored["tenders"] if row["id"] in wanted]
    by_prompt = {}
    eligibility = None

    def dump(partial: bool) -> None:
        payload = {
            "environment": environment(),
            "partial": partial,
            "llm_provider": "ollama",
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "tender_ids": tender_ids,
            "note": (
                "Retrieval, extraction, eligibility and faithfulness below are from the local model. "
                "The heuristic_* fields are the cue extractor and the rule checker on this same label "
                "set, not the earlier mock-baseline file."
            ),
            "retrieval": scored["retrieval"],
            "tenders": scored["tenders"],
            "heuristic_extraction": scored["prompt_policy"],
            "heuristic_eligibility": {
                key: scored["eligibility"][key] for key in ("n", "correct", "accuracy", "confusion")
            },
            "prompts": by_prompt,
            "llm_eligibility": None
            if eligibility is None
            else {key: eligibility[key] for key in ("n", "correct", "accuracy", "confusion", "judge", "rows")},
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out = Path(args.out)
    dump(True)
    for prompt in prompts:
        print(f"pipeline prompt {prompt}", flush=True)
        prompt_settings = replace(settings, prompt_version=prompt)
        by_prompt[prompt] = run_cases(prompt_settings, tender_ids)
        by_prompt[prompt]["semantic_faithfulness"] = score_nli(by_prompt[prompt])
        dump(True)
        extraction = by_prompt[prompt].get("extraction") or {}
        print(
            f"prompt {prompt} done matched={extraction.get('matched')}/{extraction.get('gold')}",
            flush=True,
        )
    print("gold eligibility with the llm", flush=True)
    embedder = build_embedder("sentence-transformers", args.embedding_model)
    reranker = build_reranker("cross-encoder", args.reranker_model)
    eligibility = llm_eligibility(build_llm(settings), embedder, reranker, tender_ids)
    dump(False)
    print(out)
    for prompt, cases in by_prompt.items():
        extraction = cases.get("extraction") or {}
        faith = cases.get("faithfulness") or {}
        semantic = cases.get("semantic_faithfulness") or {}
        print(
            f"prompt {prompt}: extraction matched={extraction.get('matched')}/{extraction.get('gold')} "
            f"precision={extraction.get('precision')} recall={extraction.get('recall')} "
            f"span {faith.get('supported')}/{faith.get('claims')} "
            f"nli {semantic.get('entailment')}/{semantic.get('n')}"
        )
    print(f"llm eligibility {eligibility.get('correct')}/{eligibility.get('n')}")


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
