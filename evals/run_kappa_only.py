import json
import sys
sys.path.insert(0, 'evals')
from checks.llm_judge import compute_cohens_kappa

with open("evals/eval_results.json") as f:
    data = json.load(f)

with open("evals/llm_invoked_urls.json") as f:
    human_labels = json.load(f)

results_map = {r["url"]: r for r in data["results"]}

llm_scores = []
human_scores = []

for label in human_labels:
    url = label["url"]
    human_score = label["human_score"]
    if human_score is None:
        continue

    result = results_map.get(url)
    if not result:
        continue

    # Convert llm_score (0-100) back to 1-5 scale
    llm_score_1to5 = round(result["llm_score"] / 20)
    llm_score_1to5 = max(1, min(5, llm_score_1to5))

    llm_scores.append(llm_score_1to5)
    human_scores.append(human_score)

print(f"Comparing {len(llm_scores)} URLs\n")
for i, label in enumerate([l for l in human_labels if l["human_score"] is not None]):
    print(f"  {label['url'][:60]:<60} LLM={llm_scores[i]}  Human={human_scores[i]}")

kappa = compute_cohens_kappa(llm_scores, human_scores)
trusted = kappa >= 0.8

print(f"\n{'='*50}")
print(f"Cohen's Kappa: {kappa}")
print(f"Trusted: {trusted}")
print(f"{'='*50}")

if not trusted:
    print("\nFINDING: LLM judge shows low agreement with human labels.")
    print("This suggests the judge is over-scoring outputs (score collapse to 80/100)")
    print("and needs prompt/rubric calibration before being trusted at scale.")
