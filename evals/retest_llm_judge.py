import os
import json
import sys
sys.path.insert(0, 'evals')
from firecrawl import FirecrawlApp
from checks.code_checks import run_code_checks, load_ground_truth
from checks.llm_judge import run_llm_judge, compute_cohens_kappa

fc_key = os.environ.get("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=fc_key)
gt_map = load_ground_truth("evals/ground_truth.json")

with open("evals/llm_invoked_urls.json") as f:
    human_labels = json.load(f)

llm_scores = []
human_scores = []
details = []

print(f"Re-testing {len(human_labels)} URLs with fixed rubric...\n")

for label in human_labels:
    url = label["url"]
    human_score = label["human_score"]
    if human_score is None:
        continue

    gt = gt_map.get(url, {})
    category = gt.get("category", "unknown")

    try:
        response = app.scrape_url(url, formats=["markdown"])
        markdown = getattr(response, "markdown", "") or ""
    except Exception as e:
        print(f"  SKIP {url}: {e}")
        continue

    checks = run_code_checks(markdown, url, gt_map, category=category)
    code_score = checks["code_score"]

    llm_result = run_llm_judge(markdown, url, gt, category, code_score)
    llm_raw = llm_result["llm_raw_avg"]
    llm_score_1to5 = round(llm_raw)
    llm_score_1to5 = max(1, min(5, llm_score_1to5))

    llm_scores.append(llm_score_1to5)
    human_scores.append(human_score)
    details.append((url, llm_score_1to5, human_score))

    print(f"  {url[:55]:<55} LLM={llm_score_1to5}  Human={human_score}")

kappa = compute_cohens_kappa(llm_scores, human_scores)
trusted = kappa >= 0.8

print(f"\n{'='*50}")
print(f"NEW Cohen's Kappa: {kappa}")
print(f"Trusted: {trusted}")
print(f"{'='*50}")
print(f"\nOLD Kappa was: 0.195")
print(f"NEW Kappa is:  {kappa}")
