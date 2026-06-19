# Firecrawl Scrape Quality Evals

A structured eval framework for measuring `/scrape` extraction quality across real-world site types.

## Why this exists

Firecrawl needs a consistent way to answer: **did the scrape return correct content, and how do we know?**

This framework provides:

* Deterministic code checks: fast, free, catches clear failures
* LLM-as-judge: used only for ambiguous semantic quality cases
* Human calibration loop: Cohen's Kappa to catch judge bias
* Per-category metrics: different site types need different checks

## Architecture

```text
URL
→ Scrape
→ Code checks
→ If clearly pass/fail, use deterministic score
→ If ambiguous, call LLM judge
→ If score is borderline or high variance, send to human review queue
→ Compute final score
```

Final score:

```text
60% deterministic score + 40% LLM judge score
```

Missing required ground-truth keywords cap the score at 70 regardless of LLM opinion. The judge cannot override objective failures.

## Setup

```bash
pip install firecrawl-py
export FIRECRAWL_API_KEY=fc-your-key
export OPENROUTER_API_KEY=sk-or-your-key
python evals/scrape_quality_eval.py
```

## Results

Tested on 100 URLs across 7 categories.

| Category    | Avg Score | n  |
| ----------- | --------- | -- |
| spa         | 76.5      | 15 |
| research    | 75.3      | 10 |
| adversarial | 72.3      | 15 |
| docs        | 64.7      | 15 |
| news        | 63.3      | 15 |
| jobs        | 54.7      | 15 |
| ecommerce   | 51.6      | 15 |

Overall: **65.0/100 average, 95% success rate**

## Key findings

### 1. Paywall detection is weak

Only **8/100** paywalls were detected. Known paywall sites such as WSJ, Economist, and FT returned full content with no paywall flag, suggesting either Firecrawl bypasses these or the current detection misses JS-rendered paywall overlays.

### 2. Table and code block extraction are major structural gaps

Only **16/100** URLs preserved tables, and only **13/100** preserved code blocks, even on pages where these structures clearly exist, such as docs and pricing pages.

### 3. LLM judge calibration is non-trivial

I ran Cohen's Kappa calibration against 27 human-labeled URLs:

* Initial rubric + Llama 3.1 8B: Kappa = 0.195
  Judge over-scored and collapsed to “4” almost universally.
* Tightened rubric + same model: Kappa = 0.069
  Judge became more inconsistent.
* Tightened rubric + gpt-4o-mini: Kappa = 0.118
  Judge became systematically too harsh.

This shows that small rubric changes can shift judge bias direction rather than improve discrimination, and that a stronger model alone does not fix a miscalibrated rubric.

Going forward, I reverted to the simpler original rubric with gpt-4o-mini and recommend treating any LLM judge as untrusted until it reaches strong agreement against a real human-labeled set.

### 4. Hard failures and low-quality outputs should be tracked separately

Several failures, such as Amazon and Lever jobs, returned near-empty content rather than poor-quality content. These are scraper blocking/fetching issues, not normal extraction-quality issues, and should be tracked separately.

## Files

```text
evals/
├── ground_truth.json        # hand-labeled expected content per URL
├── human_labels.json        # human scores for Kappa calibration
├── checks/
│   ├── code_checks.py       # deterministic checks
│   └── llm_judge.py         # LLM judge + Kappa calculation
├── scrape_quality_eval.py   # main runner
├── eval_results.json        # full results output
└── human_review_queue.json  # ambiguous cases flagged for review
```

## Limitations / Next steps

* Ground truth currently covers about 25% of the 100 URLs; the rest score on universal and category heuristics only.
* No `/crawl`, `/search`, or `/map` eval modules yet, though the architecture is designed to extend to these.
* Jobs and ecommerce categories need deeper investigation because both score lowest and may reflect product gaps such as bot blocking or listing-vs-detail page confusion.
* No CI integration yet.
