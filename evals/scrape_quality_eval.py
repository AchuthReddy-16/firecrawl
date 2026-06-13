import os
import re
import json
import time
from dataclasses import dataclass
from firecrawl import FirecrawlApp

TEST_URLS = [
    {"url": "https://docs.python.org/3/library/json.html", "type": "docs"},
    {"url": "https://en.wikipedia.org/wiki/Large_language_model", "type": "wiki"},
    {"url": "https://react.dev/learn", "type": "spa"},
    {"url": "https://news.ycombinator.com", "type": "news"},
    {"url": "https://www.amazon.com/dp/B08N5WRWNW", "type": "ecommerce"},
    {"url": "https://arxiv.org/abs/2305.10601", "type": "research"},
    {"url": "https://jobs.lever.co/anthropic", "type": "jobs"},
]

@dataclass
class EvalResult:
    url: str
    site_type: str
    has_headings: bool
    has_tables: bool
    has_links: bool
    noise_ratio: float
    char_count: int
    success: bool
    latency_ms: float
    score: float

def score_markdown(markdown: str) -> dict:
    if not markdown or len(markdown) < 50:
        return {
            "has_headings": False,
            "has_tables": False,
            "has_links": False,
            "noise_ratio": 1.0,
            "char_count": len(markdown) if markdown else 0,
            "score": 0.0
        }

    lines = markdown.split("\n")
    total_lines = len(lines)

    heading_lines = [l for l in lines if re.match(r'^#{1,3} ', l)]
    has_headings = len(heading_lines) > 0

    table_lines = [l for l in lines if '|' in l and '---' in markdown]
    has_tables = len(table_lines) > 0

    links = re.findall(r'\[.+?\]\(https?://.+?\)', markdown)
    has_links = len(links) > 0

    noise_patterns = ['cookie', 'subscribe', 'sign in', 'log in',
                      'privacy policy', 'terms of service', 'all rights reserved']
    noise_lines = sum(1 for l in lines
                      if any(p in l.lower() for p in noise_patterns))
    noise_ratio = round(noise_lines / max(total_lines, 1), 2)

    score = 0
    if has_headings: score += 30
    if has_tables: score += 20
    if has_links: score += 20
    if noise_ratio < 0.1: score += 30
    elif noise_ratio < 0.2: score += 15

    return {
        "has_headings": has_headings,
        "has_tables": has_tables,
        "has_links": has_links,
        "noise_ratio": noise_ratio,
        "char_count": len(markdown),
        "score": score
    }

def run_evals():
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("Set FIRECRAWL_API_KEY environment variable")

    app = FirecrawlApp(api_key=api_key)
    results = []

    for test in TEST_URLS:
        print(f"\nEvaluating: {test['url']}")
        try:
            start = time.time()
            response = app.scrape_url(test["url"], formats=["markdown"])
            latency_ms = round((time.time() - start) * 1000, 2)

            markdown = response.markdown or ""
            metrics = score_markdown(markdown)

            result = EvalResult(
                url=test["url"],
                site_type=test["type"],
                latency_ms=latency_ms,
                success=True,
                **metrics
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            result = EvalResult(
                url=test["url"],
                site_type=test["type"],
                has_headings=False,
                has_tables=False,
                has_links=False,
                noise_ratio=1.0,
                char_count=0,
                success=False,
                latency_ms=0,
                score=0.0
            )

        results.append(result)
        print(f"  Score:    {result.score}/100")
        print(f"  Latency:  {result.latency_ms}ms")
        print(f"  Headings: {result.has_headings}")
        print(f"  Tables:   {result.has_tables}")
        print(f"  Links:    {result.has_links}")
        print(f"  Noise:    {result.noise_ratio}")

    print("\n===== EVAL SUMMARY =====")
    avg_score = sum(r.score for r in results) / len(results)
    success_rate = sum(1 for r in results if r.success) / len(results) * 100
    print(f"Average Score:  {avg_score:.1f}/100")
    print(f"Success Rate:   {success_rate:.0f}%")
    print(f"Heading Pass:   {sum(1 for r in results if r.has_headings)}/{len(results)}")
    print(f"Table Pass:     {sum(1 for r in results if r.has_tables)}/{len(results)}")
    print(f"Link Pass:      {sum(1 for r in results if r.has_links)}/{len(results)}")

    with open("evals/eval_results.json", "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print("\nResults saved to evals/eval_results.json")

if __name__ == "__main__":
    run_evals()
