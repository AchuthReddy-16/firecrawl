import os
import re
import json
import time
import math

#    Configurable model
MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

#   Rubric only — tight prompt 
RUBRIC = """
1 = Empty, garbled, only boilerplate
2 = Heavy noise, missing most fields
3 = Partial content, some noise (flag human review)
4 = Most content present, low noise, usable
5 = All fields present, clean, AI-ready
"""


#  Markdown preview — head + tail 
def build_markdown_preview(markdown: str, max_chars: int = 3000) -> str:
    if not markdown:
        return ""
    if len(markdown) <= max_chars:
        return markdown
    head = markdown[:1800]
    tail = markdown[-1200:]
    return head + "\n\n...[TRUNCATED]...\n\n" + tail


#  Tight prompt 
def build_prompt(markdown: str, url: str, gt: dict, category: str) -> str:
    must_contain = gt.get("must_contain", [])
    must_not_contain = gt.get("must_not_contain", [])
    preview = build_markdown_preview(markdown)

    return f"""You are evaluating web scraping quality for a {category} page.

URL: {url}
Must contain: {must_contain}
Must not contain: {must_not_contain}

Rubric:
{RUBRIC}

Rules:
- If any must_contain item is missing, score <= 3
- If must_not_contain item is present, reduce score
- If content is mostly boilerplate, score 1 or 2

Extracted markdown:
{preview}

Return only valid JSON:
{{
  "score": <1-5>,
  "reason": "<2 sentences max>",
  "needs_human_review": <true/false>,
  "confidence": "<low/medium/high>"
}}"""


# ── LLM call 
def call_llm(prompt: str) -> str:
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 300,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AchuthReddy-16/firecrawl",
            "X-Title": "Firecrawl Evals"
        }
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


#  JSON parsing — strict 
def parse_llm_response(text: str) -> dict:
    try:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(
                r"^```json|^```|```$", "",
                text,
                flags=re.MULTILINE
            ).strip()

        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            raise ValueError("No JSON found")

        result = json.loads(match.group())

        # Clamp score 1-5
        score = int(result.get("score", 3))
        score = max(1, min(5, score))

        confidence = result.get("confidence", "low")
        if confidence not in ["low", "medium", "high"]:
            confidence = "low"

        return {
            "score": score,
            "reason": str(result.get("reason", ""))[:300],
            "needs_human_review": bool(result.get("needs_human_review", True)),
            "confidence": confidence
        }

    except Exception:
        return {
            "score": 3,
            "reason": "Failed to parse LLM response",
            "needs_human_review": True,
            "confidence": "low"
        }


#  Single LLM call, multi only if borderline 
def run_llm_judge(
    markdown: str,
    url: str,
    gt: dict,
    category: str,
    code_score: float
) -> dict:
    """
    Single pass by default.
    3 runs only if score is borderline or confidence is low.
    Reduces cost and latency significantly.
    """
    prompt = build_prompt(markdown, url, gt, category)

    # Run once first
    try:
        raw = call_llm(prompt)
        first = parse_llm_response(raw)
    except Exception as e:
        return {
            "llm_score": 60.0,
            "llm_raw_avg": 3.0,
            "llm_scores_all_runs": [3],
            "llm_variance": 0,
            "llm_reason": f"LLM call failed: {e}",
            "needs_human_review": True,
            "confidence": "low"
        }

    score = first["score"]
    borderline = 2.5 <= score <= 3.5
    low_confidence = first["confidence"] == "low"
    disagrees_with_code = abs((score / 5 * 100) - code_score) > 30

    # Multi-run only when needed
    if borderline or low_confidence or disagrees_with_code:
        scores = [score]
        for _ in range(2):
            try:
                raw = call_llm(prompt)
                r = parse_llm_response(raw)
                scores.append(r["score"])
                time.sleep(0.5)
            except Exception:
                scores.append(3)
    else:
        scores = [score]

    avg_score = round(sum(scores) / len(scores), 2)
    variance = max(scores) - min(scores)
    needs_human = (
        first["needs_human_review"] or
        variance >= 2 or
        2.5 <= avg_score <= 3.5
    )

    llm_score_100 = round((avg_score / 5) * 100, 1)

    return {
        "llm_score": llm_score_100,
        "llm_raw_avg": avg_score,
        "llm_scores_all_runs": scores,
        "llm_variance": variance,
        "llm_reason": first["reason"],
        "needs_human_review": needs_human,
        "confidence": first["confidence"]
    }


# ── Final score combiner
def compute_final_score(
    code_score: float,
    llm_score: float,
    must_contain_pass: bool,
    must_contain_keywords: list
) -> float:
    """
    60% deterministic + 40% LLM.
    Cap at 70 if required keywords missing.
    LLM judge cannot override objective failures.
    """
    final = (0.6 * code_score) + (0.4 * llm_score)

    # Cap if ground truth keywords missing
    if must_contain_keywords and not must_contain_pass:
        final = min(final, 70.0)

    return round(final, 1)


# ── Cohen's Kappa 
def compute_cohens_kappa(
    llm_scores: list,
    human_scores: list
) -> float:
    """
    Measures agreement between LLM judge and human labels.
    > 0.8 = LLM is trustworthy
    < 0.8 = LLM may be biased or hallucinating

    Both lists must be same length, scores 1-5.
    """
    if len(llm_scores) != len(human_scores):
        raise ValueError("Score lists must be same length")

    n = len(llm_scores)
    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for a, b in zip(llm_scores, human_scores) if a == b) / n

    # Expected agreement
    categories = list(range(1, 6))
    pe = 0.0
    for c in categories:
        p_llm = llm_scores.count(c) / n
        p_human = human_scores.count(c) / n
        pe += p_llm * p_human

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return round(kappa, 3)


def run_kappa_check(human_labels_path: str, llm_results: list) -> dict:
    """
    Compare LLM scores against human labels.
    human_labels.json = [{url, human_score}]
    llm_results = [{url, llm_raw_avg}]
    """
    try:
        with open(human_labels_path) as f:
            human_data = json.load(f)
    except FileNotFoundError:
        return {
            "kappa": None,
            "status": "human_labels_not_ready",
            "message": "Fill evals/human_labels.json to enable Kappa check"
        }

    human_map = {h["url"]: h["human_score"] for h in human_data}
    llm_map = {r["url"]: round(r.get("llm_raw_avg", 3)) for r in llm_results}

    # Only compare URLs present in both
    common_urls = [u for u in human_map if u in llm_map]
    if len(common_urls) < 10:
        return {
            "kappa": None,
            "status": "insufficient_labels",
            "message": f"Need at least 10 labeled URLs, got {len(common_urls)}"
        }

    llm_scores = [llm_map[u] for u in common_urls]
    human_scores = [human_map[u] for u in common_urls]

    kappa = compute_cohens_kappa(llm_scores, human_scores)
    trusted = kappa >= 0.8

    return {
        "kappa": kappa,
        "status": "complete",
        "trusted": trusted,
        "n_compared": len(common_urls),
        "message": (
            "LLM judge is trustworthy" if trusted
            else "LLM judge may be biased — review human labels"
        )
    }
