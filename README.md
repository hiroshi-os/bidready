# bidready

Indian MSMEs lose government tenders because a pack runs 100 to 300 pages across scanned PDFs, corrigenda and annexures. Working out eligibility, and which certificate answers which clause, takes days. Consultants charge for that reading.

bidready takes a tender pack plus a company's own documents (GST certificate, turnover statements, past work orders, certifications) and writes four things:

1. A cited go / no-go eligibility report.
2. A compliance matrix that maps each extracted requirement to the company evidence that satisfies it, or marks it `MISSING`.
3. Risks and key deadlines (EMD, bid due date, pre-bid meeting, liquidated damages).
4. A draft pre-bid query letter.

Every claim cites a clause and a page. The draft is for a person to review. It is not legal advice and it is not a bid recommendation.

This repository is a public resume project. The company files under `src/bidready/synthetic/` are fictional.

## Architecture

```mermaid
flowchart LR
  subgraph ui [Web]
    Browser[Jinja UI and JSON API]
  end
  subgraph app [bidready]
    API[FastAPI]
    Parse[PyMuPDF plus Tesseract OCR]
    Index[BM25 plus FAISS]
    Graph[LangGraph agents]
    Rules[Eligibility rules and verifier]
  end
  Browser --> API
  API --> Parse
  Parse --> Store[(Document store)]
  Parse --> Index
  Index --> Graph
  Graph --> Rules
  Rules --> DB[(Postgres or sqlite)]
  API --> Notify[Webhook and email]
  TL[tenderlens adapter] -.-> API
```

The tenderlens adapter is optional. bidready runs with `TENDERLENS_BASE_URL` unset. The sibling crawler at `hiroshi-os/tenderlens` was only a README when this MVP was written, so the adapter speaks a small JSON contract and is not required for any scored path.

## Agent graph

```mermaid
flowchart TD
  planner[planner] --> extractor[requirement extractor]
  extractor --> eligibility[eligibility checker]
  eligibility --> risk[risk and deadline agent]
  risk --> drafter[pre-bid letter drafter]
  drafter --> verifier[verifier]
  verifier -->|uncited quote, retry the failing stage| extractor
  verifier -->|uncited quote| eligibility
  verifier -->|uncited quote| risk
  verifier -->|uncited quote| drafter
  verifier -->|passed, or stripped after retries| done[end]
```

State is a LangGraph `TypedDict`. Each node appends a trace event (`node`, milliseconds, detail). Retries are a dict on the state. A stage may be sent back at most twice. If one stage exceeds that, or the total retries exceed 6, the verifier deletes the unsupported claims and ends. The graph recursion limit is 50.

On the mock plumbing run the verifier passed on the first pass (964 of 964 claims). On the local-model run every stored quote was also a span (143/143 on prompt v1, 145/145 on prompt v2), because a quote that fails the substring test is dropped before it is stored. The retry edge is covered by `tests/test_pipeline.py`, which injects one fabricated quote and checks that the second pass cites a real line.

Mock mode does not call a model. It runs the same graph and applies the cue rules written in `src/bidready/prompts.py`. `openai` and `ollama` call a model for extraction, eligibility, risks and a question note, then fall back to those rules if the call fails or returns nothing. That fallback is recorded in the trace. The LLM path was not scored in the numbers below.

## Data model

```mermaid
erDiagram
  cases ||--o{ documents : has
  cases ||--o{ chunks : has
  cases ||--o{ runs : has
  documents ||--o{ chunks : split_into
  runs ||--o{ requirements : extracts
  runs ||--o{ notifications : emits
  requirements ||--o{ evidence_links : judged_by
```

| Table | What it stores |
| --- | --- |
| `cases` | One tender review. Status, synthetic `profile_id`, go / no-go. |
| `documents` | Tender or company file, storage URI, page count, OCR page count. |
| `chunks` | Layout chunk with page range, clause, section heading, text. |
| `runs` | Provider, model, embedding, reranker, prompt version, latency, token counts, full report JSON, trace. |
| `requirements` | Extracted clause, kind, quote, page, chunk id. |
| `evidence_links` | Decision (`met`, `not_met`, `missing`, `unclear`), obligation, rationale, evidence quote. |
| `notifications` | Webhook and email rows. `stubbed` when no URL or SMTP host is set. |

sqlite is the default for a laptop and for tests. Postgres is the database in `docker-compose.yml`.

## Design

### Parsing and chunking

Text PDFs are read with PyMuPDF `get_text("dict")` so each visual line stays a line. A page with fewer than 40 alphanumeric characters is rendered at 200 dpi and passed to Tesseract. On this gold set that fired for 8 pages of the 50-page SGGSCC pack and for none of the other nine PDFs.

Headings (`Clause 2. Earnest Money Deposit`, `NOTICE INVITING TENDER`) become the chunk's section and are not glued onto the next requirement. All-caps body text is common in these packs, so a line that contains `should`, `shall`, `rupees` or `lakh` stays in the body. A wrapped amount such as `(GROSS) OF 88 LAKHS` is joined to the line above. A new `BIDDER SHOULD` or `EMD` line is not.

Chunks flush on a page change and around 1000 characters (hard stop 1800). Each chunk keeps `page_start`, `page_end`, a clause number when the line starts with one, and the section heading.

### Retrieval

Four modes share one index:

| Mode | What it does |
| --- | --- |
| `vector` | FAISS `IndexFlatIP` on L2-normalised embeddings (cosine). |
| `bm25` | `rank_bm25` Okapi over word tokens. |
| `hybrid` | Reciprocal rank fusion of the two lists, `k = 60`. |
| `hybrid_rerank` | Fusion pool of up to 30 chunks, then a reranker cuts it to k. |

The default embedder is `hash`: a 384-dimensional feature hash of tokens. It is lexical. It is not a semantic model. It is the default so Docker and CI do not download weights. `EMBEDDING_PROVIDER=sentence-transformers` uses `all-MiniLM-L6-v2` after `pip install -e ".[local]"`. That provider was not installed for the measured run.

The default reranker is character-trigram Jaccard. `RERANKER=cross-encoder` loads `cross-encoder/ms-marco-MiniLM-L-6-v2` from the same local extra. That model is the one in the local-model retrieval table below.

Tradeoff. In-process FAISS avoids a second server. pgvector would be the better index once many tenants share one Postgres. With hash embeddings, fusion beat the vector index and the lexical reranker lowered MRR. With MiniLM embeddings, the ms-marco MiniLM cross-encoder raised MRR above fusion. Both comparisons are in the results.

### Eligibility and the verifier

The checker does not read a hidden profile JSON at decision time. It retrieves company chunks and reads the amount, headcount or certificate sentence it cites.

| Signal in the tender sentence | Decision rule |
| --- | --- |
| Turnover, solvency, net worth | Compare the cited evidence amount with the rupee threshold next to the keyword. A percent of estimated cost is `unclear` when no estimated cost was parsed. |
| Similar works | Parse alternatives such as "3 works of Rs. X" or "THREE SIMILAR WORKS ... 70 LAKHS". |
| GST, PAN, PSARA, ISO | `met` only when a company sentence contains the marker. |
| Blacklist, loss years | `met` when the cited sentence is the negative declaration. |
| EMD, bid security | A submission item. `missing` unless a company sentence cites an instrument. `missing` here is `CONDITIONAL`, not `NO-GO`. |

Go / no-go, which is not a legal opinion: any `not_met` is `NO-GO`. A `missing` qualification is `NO-GO`. A `missing` submission item or any `unclear` row is `CONDITIONAL`. All `met` is `GO`.

The verifier is programmatic. A quote is supported when it has at least 20 characters after whitespace normalisation and is a substring of the cited chunk (exact or whitespace-normalised). Periods in `Rs.` and `Mr.` are not sentence breaks, because splitting there used to drop the amount. The verifier does not ask a model whether a citation "seems" right.

## Eval methodology

Labels live in `evals/gold/manifest.json`, dated 2026-09-25. Ten public tender PDFs. The PDFs are not in git. `python -m evals.cli fetch` downloads them into `data/gold_pdfs/` and checks the sha256 in the manifest.

The label set was read while the heuristic was being written. These scores are not an unbiased estimate on unseen tenders.

| id | pages | source |
| --- | --- | --- |
| barc-ced-2026 | 17 | https://barc.gov.in/tenders/tender265.pdf |
| cci-forensic-2024 | 34 | https://www.cci.gov.in/images/whatsnew/en/nit-dated-24102024-11729760843.pdf |
| epi-1366 | 40 | https://epi.gov.in/admin/image/tenders/1741783038_NIT-1366.pdf |
| epi-1367 | 39 | https://epi.gov.in/admin/image/tenders/1741689388_NIT1367.pdf |
| igidr-travel-2024 | 15 | http://www.igidr.ac.in/tender/2024/TD-09.pdf |
| iiml-website | 17 | https://www.iiml.ac.in/sites/default/files/upload/tender/1620136915IIML_Website.pdf |
| iimtrichy-stp-2024 | 24 | https://www.iimtrichy.ac.in/sites/default/files/upload/14Oct2024193737_2024101419373324SP204T_STPPlant_Final.pdf |
| iitpkd-ambulance-2024 | 13 | https://iitpkd.ac.in/sites/default/files/2024-02/TENDER_DOCUMENT_399.pdf |
| sggscc-civil-2024 | 50 | https://www.sggscc.ac.in/uploads/tenders/tenderDocument/02da0157d53780863da33b98f4c446f0.pdf |
| nhm-dnh-2025 | 25 | https://cdnbbsr.s3waas.gov.in/s371e09b16e21f7b6919bbfc43f6a5b2f0/uploads/2025/11/202511132120554845.pdf |

sha256 values are in the manifest. NHM's first page names the Mission Director, National Health Mission, UT of Dadra & Nagar Haveli, Daman & Diu.

Company profiles `sample-civil` and `sample-security` are synthetic. Both files say so in the first line. GSTINs are fake. Eligibility labels say which profile should be `met`, `not_met`, `missing` or `unclear`.

What is scored:

- Retrieval recall@5, recall@10 and MRR for `vector`, `hybrid` and `hybrid_rerank`. A chunk is relevant when it contains every phrase in the query's `match_all`. 28 queries.
- Requirement extraction. A prediction matches a gold clause when the predicted sentence contains every gold phrase. Micro precision and recall.
- Eligibility accuracy on the labelled clauses, against the two synthetic profiles. 39 labels. The local-model number calls the LLM with rule fallback off. The 39/39 figure is the rule checker.
- Citation faithfulness. Span faithfulness requires every stored tender quote, evidence quote, risk, deadline and letter quote to be a span of the cited chunk. Semantic faithfulness asks `cross-encoder/nli-MiniLM2-L6-H768` whether the span entails the claim (argmax of contradiction, entailment, neutral). Pairs whose two sides are the same string are reported separately, because a sentence entails itself.
- Latency is wall clock around `analyse()` for each tender with `sample-civil`. Token counts are `prompt_eval_count` and `eval_count` from Ollama. Cost was not computed from those counts.

The label file has 70 requirements. Three packs are exhaustive for bidder obligations to possess, hold, or submit: `igidr-travel-2024` (15), `iiml-website` (15), `iitpkd-ambulance-2024` (16). Product specs, portal navigation, and post-award administration are outside that definition. The other seven packs are still a sample (24 requirements). Precision on the exhaustive packs is the figure to read. Precision on all 70 still treats an unlabelled true requirement in a sample pack as a false positive. Eligibility labels were not added for the new requirements, so that set is still 39.

## Results

The local-model numbers lead. The mock table further down is the pipeline and plumbing baseline from 2026-09-25, on the original 33-clause label set.

### Local model, 2026-09-26

Scored run written 2026-09-26. It started 2026-09-25 22:13 UTC. Python 3.12.3, Linux 6.12.94+ x86_64, Intel Xeon, 4 CPUs, 15.64 GiB RAM, no swap. Tesseract 5.3.4. Ollama 0.34.4.

| piece | model |
| --- | --- |
| LLM | `qwen2.5:3b` (Q4_K_M), temperature 0, `num_predict` 900, `num_ctx` 4096 |
| embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| NLI | `cross-encoder/nli-MiniLM2-L6-H768` |

Rule fallback was off (`HEURISTIC_FALLBACK` false). The model reads at most 12 chunks per tender: hybrid retrieval with the cross-encoder, then obligation lines if the window is still short, in batches of 3. Risks and deadlines come from at most 8 retrieved chunks, and the prompt asks for at most 4 of each. Unrounded values are in `evals/results/local.json`.

`qwen2.5:7b-instruct` was not used for this suite. On its own, on 2026-09-26, a 23-token reply took 3.6 seconds of generation (6.39 tokens/second) and left 5.61 GiB available. During the 3B suite, available memory fell to about 2.3 GiB with MiniLM and the cross-encoder resident, so the 7B model was not run across the 10 tenders.

#### Retrieval (28 queries, same labels as the mock run)

| mode | recall@5 | recall@10 | MRR |
| --- | ---: | ---: | ---: |
| vector (MiniLM) | 0.775 | 0.823 | 0.676 |
| hybrid (MiniLM + BM25) | 0.842 | 0.919 | 0.810 |
| hybrid + ms-marco MiniLM cross-encoder | 0.877 | 0.937 | 0.946 |

The cross-encoder is the best of the three on all three metrics. On the hash-embedding plumbing run, the lexical reranker had lowered MRR.

#### Requirement extraction (`qwen2.5:3b`)

A prediction matches when it contains every gold phrase. The model only sees 12 chunks, and a quote that is not a span of a chunk is dropped.

| prompt | scope | predicted | matched | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v1 | all 70 | 63 | 24 / 70 | 0.381 | 0.343 | 0.361 |
| v1 | 3 exhaustive packs (46) | 26 | 15 / 46 | 0.577 | 0.326 | 0.417 |
| v2 | all 70 | 47 | 14 / 70 | 0.298 | 0.200 | 0.239 |
| v2 | 3 exhaustive packs (46) | 13 | 6 / 46 | 0.462 | 0.130 | 0.203 |

v1 is the stronger of the two prompts on this model. Recall stays near a third on the exhaustive packs: most labelled obligations were outside the 12-chunk window or the quote was not a verbatim span.

#### Eligibility (`qwen2.5:3b`, no rule fallback)

5 / 39. Accuracy 0.128. Every row was judged by the model (`source` `llm`).

| expected \ predicted | met | not_met | missing | unclear |
| --- | ---: | ---: | ---: | ---: |
| met (19) | 0 | 0 | 2 | 17 |
| not_met (7) | 0 | 0 | 0 | 7 |
| missing (10) | 0 | 2 | 3 | 5 |
| unclear (3) | 0 | 0 | 1 | 2 |

The five matches are three `missing` decisions (BARC solvency for `sample-security`, CCI EMD for `sample-civil`, IGIDR EMD for `sample-civil`) and two `unclear` decisions (EPI-1366 turnover for `sample-civil`, NHM turnover for `sample-security`). Those two `unclear` rows are the path that discards a `met` or `not_met` when the evidence quote is not a span. No labelled `met` or `not_met` was predicted correctly. The rule checker on the same 39 labels is still 39/39; that number is the rules, recorded in `local.json` as `heuristic_eligibility`.

#### Citation faithfulness

Span containment, after the pipeline drops quotes that are not spans: v1 143/143, v2 145/145. That rate is the filter.

Semantic support is the NLI label. Requirement rows store the quote as the requirement text, so those pairs are identical and the model calls them entailment (v1 63/63, v2 47/47). Excluding identical pairs:

| prompt | pairs | entailment | rate | evidence | risk | deadline |
| --- | ---: | ---: | ---: | --- | --- | --- |
| v1 | 33 | 7 | 0.212 | 3/17 | 0/10 | 4/6 |
| v2 | 45 | 10 | 0.222 | 1/11 | 0/19 | 9/15 |

A hand check of 16 pairs, on 2026-09-26, agreed with the NLI support decision on 13 and disagreed on 3. The three disagreements are forfeiture, re-tender exclusion, and damages equal to EMD: a reader treats them as penalties, and the NLI model labelled them neutral. The risk hypothesis is the fixed sentence "This tender sentence states a contractual risk or penalty." The 0 entailment counts on risks follow that wording. The 16 pairs are in `evals/results/local.json` under `hand_check`.

#### Latency and tokens

Wall clock around `analyse()` with `sample-civil`. Tokens are Ollama counts. Cost was not computed.

Prompt v1 total 3209.7 seconds, 97,783 prompt tokens, 25,682 completion tokens.

| tender | go / no-go | requirements | risks | deadlines | seconds | prompt / completion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| barc-ced-2026 | NO-GO | 11 | 2 | 2 | 458.2 | 13391 / 3520 |
| cci-forensic-2024 | NO-GO | 11 | 0 | 0 | 487.4 | 12562 / 3961 |
| epi-1366 | NO-GO | 1 | 0 | 0 | 157.3 | 6240 / 1284 |
| epi-1367 | CONDITIONAL | 0 | 3 | 0 | 110.3 | 5358 / 763 |
| igidr-travel-2024 | NO-GO | 11 | 0 | 0 | 445.8 | 12524 / 3741 |
| iiml-website | NO-GO | 11 | 2 | 2 | 470.9 | 13729 / 3856 |
| iimtrichy-stp-2024 | CONDITIONAL | 7 | 2 | 0 | 312.8 | 10071 / 2343 |
| iitpkd-ambulance-2024 | CONDITIONAL | 4 | 1 | 1 | 222.1 | 7459 / 1673 |
| sggscc-civil-2024 | CONDITIONAL | 6 | 0 | 0 | 390.4 | 9760 / 3401 |
| nhm-dnh-2025 | CONDITIONAL | 1 | 0 | 1 | 154.6 | 6689 / 1140 |

Prompt v2 total 2502.4 seconds, 88,665 prompt tokens, 20,301 completion tokens.

| tender | go / no-go | requirements | risks | deadlines | seconds | prompt / completion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| barc-ced-2026 | NO-GO | 6 | 1 | 2 | 282.4 | 10169 / 2253 |
| cci-forensic-2024 | NO-GO | 5 | 1 | 0 | 242.9 | 8650 / 1993 |
| epi-1366 | CONDITIONAL | 4 | 3 | 4 | 246.5 | 8417 / 1950 |
| epi-1367 | NO-GO | 6 | 4 | 0 | 287.4 | 9872 / 2320 |
| igidr-travel-2024 | CONDITIONAL | 5 | 0 | 2 | 243.2 | 8853 / 2012 |
| iiml-website | CONDITIONAL | 5 | 4 | 3 | 260.0 | 10091 / 2011 |
| iimtrichy-stp-2024 | NO-GO | 4 | 2 | 0 | 235.7 | 7957 / 1887 |
| iitpkd-ambulance-2024 | CONDITIONAL | 3 | 3 | 0 | 202.7 | 6805 / 1794 |
| sggscc-civil-2024 | CONDITIONAL | 3 | 1 | 3 | 208.8 | 7638 / 1667 |
| nhm-dnh-2025 | CONDITIONAL | 6 | 0 | 1 | 292.8 | 10213 / 2414 |

#### Prompt comparison on the real model

| setup | measured | precision (all 70) | recall (all 70) | exhaustive precision | exhaustive recall | eligibility |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen2.5:3b`, prompt v1 | yes, 2026-09-26, hardware above | 0.381 | 0.343 | 0.577 | 0.326 | 5/39, scored once, prompt fixed by the eligibility system prompt |
| `qwen2.5:3b`, prompt v2 | yes, same run | 0.298 | 0.200 | 0.462 | 0.130 | same 5/39 |
| `qwen2.5:7b-instruct` | timed only, 23 tokens, 6.39 tok/s, 5.61 GiB free; suite not run |  |  |  |  |  |
| OpenAI `gpt-4o-mini` | not run. `OPENAI_API_KEY` was unset |  |  |  |  |  |

Eligibility was one pass, shared by the two extraction prompts. It uses `ELIGIBILITY_SYSTEM`, not extract v1/v2.

### Plumbing baseline (mock), 2026-09-25

This is the pipeline with no model call. It is not an LLM quality number. LLM provider `mock`. Embeddings `feature-hash-384`. Reranker `lexical-trigram-jaccard`. Prompt text sha256 prefixes: v1 `20b6de85dd7e`, v2 `2b80157d0c34`. Gold set at that time: 33 selected clauses, 39 eligibility labels. File: `evals/results/mock-baseline.json` (same bytes as `evals/results/measured.json`).

#### Retrieval (28 queries)

| mode | recall@5 | recall@10 | MRR |
| --- | ---: | ---: | ---: |
| vector | 0.536 | 0.607 | 0.519 |
| hybrid | 0.680 | 0.883 | 0.685 |
| hybrid + lexical rerank | 0.608 | 0.854 | 0.525 |

#### Requirement extraction (cue policies, full document, 33 clauses)

| prompt | predicted sentences | labelled clauses matched | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 (35 cues) | 549 | 33 / 33 | 0.060 | 1.000 | 0.113 |
| v2 (20 cues, modal required) | 191 | 30 / 33 | 0.157 | 0.909 | 0.268 |

These precisions are on the 33-clause sample. They are not comparable to the 70-label local-model precisions above.

The same cue extractor, re-run on 2026-09-26 against the widened 70 labels while the local stack was scoring retrieval, matched 48/70 (v1, precision 0.087, recall 0.686) and 40/70 (v2, precision 0.209, recall 0.571). That is `heuristic_extraction` in `local.json`. It is still the cue list, not `qwen2.5:3b`.

#### Eligibility and span faithfulness

Rule checker 39/39 on the synthetic profiles. Confusion is the diagonal: `met` 19, `missing` 10, `not_met` 7, `unclear` 3. Span faithfulness on the full mock pipeline 964/964. Prompt tokens 0 and completion tokens 0. Cost was not measured.

| tender | go / no-go | requirements | risks | deadlines | seconds | claims supported |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| barc-ced-2026 | NO-GO | 67 | 24 | 3 | 0.200 | 124 / 124 |
| cci-forensic-2024 | CONDITIONAL | 38 | 13 | 0 | 0.170 | 70 / 70 |
| epi-1366 | NO-GO | 100 | 24 | 2 | 0.283 | 148 / 148 |
| epi-1367 | NO-GO | 95 | 24 | 2 | 0.272 | 140 / 140 |
| igidr-travel-2024 | CONDITIONAL | 31 | 24 | 0 | 0.124 | 70 / 70 |
| iiml-website | NO-GO | 55 | 22 | 1 | 0.180 | 95 / 95 |
| iimtrichy-stp-2024 | CONDITIONAL | 41 | 18 | 6 | 0.256 | 87 / 87 |
| iitpkd-ambulance-2024 | CONDITIONAL | 19 | 7 | 1 | 0.119 | 38 / 38 |
| sggscc-civil-2024 | NO-GO | 33 | 12 | 4 | 6.529 | 72 / 72 |
| nhm-dnh-2025 | NO-GO | 70 | 24 | 0 | 0.194 | 120 / 120 |

Total wall clock 8.327 seconds. SGGSCC is the slow one because 8 pages went through OCR. Risks are capped at 24 and deadlines at 16. Verifier `passed` was true on all ten.

## Limits

- This is not legal advice. A human still has to read the pack.
- OCR quality is not labelled. Only the SGGSCC pack needed OCR here (8 of 50 pages). A bad scan will cite the OCR text, including its errors.
- Company profiles are synthetic. There are no real bidders' documents in this repo.
- The gold set is 10 tenders, 70 requirements (46 of them in three exhaustive packs), and 39 eligibility labels. The labels were used while the heuristic was written. The 12-chunk window, the cap of 4 risks and 4 deadlines, and the truncated-JSON repair were added while this local run was being debugged. The scores are one pass on that development set.
- The local model sees 12 chunks. Extraction recall includes that limit.
- Span faithfulness of 143/143 and 145/145 is the substring filter. Semantic support is the non-identical NLI table. On risks, that NLI model labelled forfeiture sentences neutral; a reader disagreed on 3 of 16 hand-checked pairs.
- Two of the five eligibility matches are `unclear` after an evidence quote was dropped.
- No paid model was called. Token counts are recorded. Cost was not computed from them.
- `qwen2.5:7b-instruct` was timed (23 tokens, 6.39 tokens/second, 5.61 GiB still available) and was not run on the 10 tenders.
- Docker Compose was run on 2026-09-26. Docker 29.1.3 and Compose 2.40.3 were installed with apt. The first `docker compose up --build` failed while extracting the Postgres image: overlayfs refused a whiteout file (`operation not permitted` on `etc/alternatives/.wh.pager.1.gz`). A second attempt with `dockerd --storage-driver=vfs` built the app image and started both containers. Postgres 16.15 became healthy and the host could open `127.0.0.1:5432`. The app container exited with `psycopg.errors.ConnectionTimeout` to `db:5432`. A second container on the same bridge also timed out connecting to `db:5432`. dockerd logged `Deleting nftables IPv4 rules` with exit status 1. The app did not serve `/health`. The log excerpt is in `evals/results/docker-compose.txt`.
- GeM hosts (`fulfilment.gem.gov.in`, `bidplus.gem.gov.in`) did not return PDFs from this network (TLS error). They are not in the gold set.
- Webhook, SMTP and S3 are implemented and unconfigured by default. S3 needs `boto3`, which is not a required dependency. tenderlens is optional.

## Run it

Python 3.12, Tesseract, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env
uv run bidready
```

Open http://127.0.0.1:8000 , upload a tender PDF, and pick the synthetic profile `sample-civil` or `sample-security`. The case page shows the decision, the matrix, deadlines, risks, the letter and the agent trace. `GET /api/cases/{id}` returns the same report as JSON. `GET /health` shows the providers.

```bash
docker compose up --build
```

Compose is meant to start Postgres 16 and the app on port 8000 in mock / hash / lexical mode. On this VM that command did not reach a healthy app. See the limits and `evals/results/docker-compose.txt`.

```bash
# after the PDFs are fetched
python -m evals.cli fetch
python -m evals.cli check
python -m evals.cli all --embedding hash --reranker lexical --out evals/results/measured.json
python -m evals.cli local --llm-model qwen2.5:3b --prompts v1,v2 --out evals/results/local.json
```

`local` needs Ollama, the `local` extra (`sentence-transformers`), and `HEURISTIC_FALLBACK` is forced off by that command. The scored model in the results is `qwen2.5:3b`.

### Providers

| Variable | Values | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `mock`, `ollama`, `openai` | `mock` |
| `LLM_MODEL` | empty uses `qwen2.5:7b-instruct` for Ollama and `gpt-4o-mini` for OpenAI | empty |
| `EMBEDDING_PROVIDER` | `hash`, `sentence-transformers` | `hash` |
| `RERANKER` | `lexical`, `cross-encoder` | `lexical` |
| `PROMPT_VERSION` | `v1`, `v2` | `v1` |

Keys are read from the environment only. See `.env.example`. Nothing in the repo is a credential.

Local model mode is Ollama at `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`). The default model name is `qwen2.5:7b-instruct`. This VM scored `qwen2.5:3b` because that is the model that fit beside MiniLM and the cross-encoder. Install the local extra with `uv sync --extra local` before switching embeddings or the reranker off the hash / lexical defaults.

### Tests

GitHub Actions installs Tesseract, runs `uv sync --frozen --extra dev`, then `ruff check` and `pytest` with mock / hash / lexical. Tests use sqlite and do not download models.
