import os
import json
import time
from dataclasses import dataclass, asdict
from firecrawl import FirecrawlApp

from checks.code_checks import run_code_checks, load_ground_truth
from checks.llm_judge import run_llm_judge, compute_final_score, run_kappa_check

os.makedirs("evals", exist_ok=True)

TEST_URLS = [
    # Docs (15)
    {"url": "https://docs.python.org/3/library/json.html",        "type": "docs"},
    {"url": "https://fastapi.tiangolo.com/tutorial/",              "type": "docs"},
    {"url": "https://docs.docker.com/get-started/",                "type": "docs"},
    {"url": "https://kubernetes.io/docs/concepts/overview/",       "type": "docs"},
    {"url": "https://pytorch.org/docs/stable/torch.html",          "type": "docs"},
    {"url": "https://redis.io/docs/latest/",                       "type": "docs"},
    {"url": "https://docs.aws.amazon.com/s3/index.html",           "type": "docs"},
    {"url": "https://developers.google.com/maps/documentation",    "type": "docs"},
    {"url": "https://stripe.com/docs/api",                         "type": "docs"},
    {"url": "https://docs.anthropic.com/en/docs/intro-to-claude",  "type": "docs"},
    {"url": "https://platform.openai.com/docs/introduction",       "type": "docs"},
    {"url": "https://docs.github.com/en/rest",                     "type": "docs"},
    {"url": "https://flask.palletsprojects.com/en/stable/",        "type": "docs"},
    {"url": "https://www.postgresql.org/docs/current/index.html",  "type": "docs"},
    {"url": "https://nginx.org/en/docs/",                          "type": "docs"},

    # Ecommerce (15)
    {"url": "https://www.amazon.com/dp/B08N5WRWNW",                "type": "ecommerce"},
    {"url": "https://www.walmart.com/ip/Apple-AirPods/969494903",  "type": "ecommerce"},
    {"url": "https://www.newegg.com/p/pl?d=gpu",                   "type": "ecommerce"},
    {"url": "https://www.ebay.com/sch/i.html?_nkw=macbook+pro",    "type": "ecommerce"},
    {"url": "https://www.target.com/c/electronics/-/N-5xtg6",      "type": "ecommerce"},
    {"url": "https://store.steampowered.com/app/1091500/",         "type": "ecommerce"},
    {"url": "https://www.apple.com/shop/buy-iphone",               "type": "ecommerce"},
    {"url": "https://www.bestbuy.com/site/laptops/",               "type": "ecommerce"},
    {"url": "https://www.etsy.com/c/jewelry",                      "type": "ecommerce"},
    {"url": "https://www.homedepot.com/b/Tools/N-5yc1vZc1qv",     "type": "ecommerce"},
    {"url": "https://www.costco.com/electronics.html",             "type": "ecommerce"},
    {"url": "https://shop.lululemon.com/c/womens-leggings",        "type": "ecommerce"},
    {"url": "https://www.bhphotovideo.com/c/browse/cameras/ci/9811","type": "ecommerce"},
    {"url": "https://www.chewy.com/b/food-332",                    "type": "ecommerce"},
    {"url": "https://www.zappos.com/c/sneakers",                   "type": "ecommerce"},

    # News (15)
    {"url": "https://techcrunch.com",                              "type": "news"},
    {"url": "https://arstechnica.com",                             "type": "news"},
    {"url": "https://www.theverge.com",                            "type": "news"},
    {"url": "https://news.ycombinator.com",                        "type": "news"},
    {"url": "https://venturebeat.com",                             "type": "news"},
    {"url": "https://www.wired.com",                               "type": "news"},
    {"url": "https://mashable.com",                                "type": "news"},
    {"url": "https://www.bbc.com/news/technology",                 "type": "news"},
    {"url": "https://www.reuters.com/technology/",                 "type": "news"},
    {"url": "https://thenextweb.com",                              "type": "news"},
    {"url": "https://www.engadget.com",                            "type": "news"},
    {"url": "https://gizmodo.com",                                 "type": "news"},
    {"url": "https://www.zdnet.com",                               "type": "news"},
    {"url": "https://9to5mac.com",                                 "type": "news"},
    {"url": "https://www.androidcentral.com",                      "type": "news"},

    # Jobs (15)
    {"url": "https://jobs.lever.co/anthropic",                     "type": "jobs"},
    {"url": "https://jobs.ashbyhq.com/firecrawl",                  "type": "jobs"},
    {"url": "https://www.ycombinator.com/jobs",                    "type": "jobs"},
    {"url": "https://wellfound.com/jobs",                          "type": "jobs"},
    {"url": "https://careers.google.com/jobs/results/",            "type": "jobs"},
    {"url": "https://www.metacareers.com/jobs/",                   "type": "jobs"},
    {"url": "https://jobs.netflix.com/search",                     "type": "jobs"},
    {"url": "https://www.amazon.jobs/en/search",                   "type": "jobs"},
    {"url": "https://boards.greenhouse.io/openai",                 "type": "jobs"},
    {"url": "https://www.linkedin.com/jobs/search/?keywords=ml+engineer", "type": "jobs"},
    {"url": "https://remoteok.com/remote-ml-jobs",                 "type": "jobs"},
    {"url": "https://weworkremotely.com/categories/remote-programming-jobs", "type": "jobs"},
    {"url": "https://jobs.github.com",                             "type": "jobs"},
    {"url": "https://www.indeed.com/jobs?q=machine+learning",      "type": "jobs"},
    {"url": "https://angel.co/jobs",                               "type": "jobs"},

    # Research (10)
    {"url": "https://arxiv.org/abs/2305.10601",                    "type": "research"},
    {"url": "https://arxiv.org/abs/2307.09288",                    "type": "research"},
    {"url": "https://arxiv.org/abs/2303.08774",                    "type": "research"},
    {"url": "https://en.wikipedia.org/wiki/Large_language_model",  "type": "research"},
    {"url": "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)", "type": "research"},
    {"url": "https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback", "type": "research"},
    {"url": "https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word", "type": "research"},
    {"url": "https://huggingface.co/papers",                       "type": "research"},
    {"url": "https://distill.pub",                                 "type": "research"},
    {"url": "https://lilianweng.github.io/posts/2023-06-23-agent/","type": "research"},

    # SPA (15)
    {"url": "https://react.dev/learn",                             "type": "spa"},
    {"url": "https://nextjs.org/docs",                             "type": "spa"},
    {"url": "https://vercel.com/docs",                             "type": "spa"},
    {"url": "https://linear.app",                                  "type": "spa"},
    {"url": "https://www.notion.so/product",                       "type": "spa"},
    {"url": "https://tailwindcss.com/docs",                        "type": "spa"},
    {"url": "https://ui.shadcn.com/docs",                          "type": "spa"},
    {"url": "https://www.figma.com/community",                     "type": "spa"},
    {"url": "https://supabase.com/docs",                           "type": "spa"},
    {"url": "https://planetscale.com/docs",                        "type": "spa"},
    {"url": "https://railway.app/",                                "type": "spa"},
    {"url": "https://fly.io/docs/",                                "type": "spa"},
    {"url": "https://deno.com/deploy/docs",                        "type": "spa"},
    {"url": "https://svelte.dev/docs",                             "type": "spa"},
    {"url": "https://vuejs.org/guide/introduction",                "type": "spa"},

    # Adversarial (15)
    {"url": "https://www.wsj.com",                                 "type": "adversarial"},
    {"url": "https://www.nytimes.com",                             "type": "adversarial"},
    {"url": "https://medium.com",                                  "type": "adversarial"},
    {"url": "https://www.wired.com",                               "type": "adversarial"},
    {"url": "https://www.bloomberg.com/technology",                "type": "adversarial"},
    {"url": "https://www.ft.com",                                  "type": "adversarial"},
    {"url": "https://www.economist.com",                           "type": "adversarial"},
    {"url": "https://substack.com",                                "type": "adversarial"},
    {"url": "https://www.patreon.com",                             "type": "adversarial"},
    {"url": "https://www.cloudflare.com",                          "type": "adversarial"},
    {"url": "https://www.forbes.com/technology/",                  "type": "adversarial"},
    {"url": "https://www.businessinsider.com/tech",                "type": "adversarial"},
    {"url": "https://hackernoon.com",                              "type": "adversarial"},
    {"url": "https://dev.to",                                      "type": "adversarial"},
    {"url": "https://lobste.rs",                                   "type": "adversarial"},
]


def percentile(values: list, p: float) -> float:
    if not values:
        return 0
    values = sorted(values)
    idx = min(int((len(values) - 1) * p), len(values) - 1)
    return values[idx]


@dataclass
class EvalResult:
    url: str
    category: str
    success: bool
    latency_ms: float
    char_count: int
    has_headings: bool
    has_tables: bool
    has_links: bool
    has_code_blocks: bool
    has_price: bool
    has_date: bool
    paywall_detected: bool
    noise_ratio: float
    must_contain_pass: bool
    must_not_contain_pass: bool
    code_score: float
    llm_invoked: bool
    llm_score: float
    llm_reason: str
    llm_variance: float
    llm_confidence: str
    final_score: float
    needs_human_review: bool
    fail_reason: str


def run_evals():
    fc_key = os.environ.get("FIRECRAWL_API_KEY")
    if not fc_key:
        raise ValueError("Set FIRECRAWL_API_KEY")

    app = FirecrawlApp(api_key=fc_key)
    gt_map = load_ground_truth("evals/ground_truth.json")

    results = []
    human_review_queue = []
    category_scores = {}

    print(f"Running evals on {len(TEST_URLS)} URLs...\n")
    print("=" * 60)

    for test in TEST_URLS:
        url = test["url"]
        gt = gt_map.get(url, {})
        category = gt.get("category", test["type"])
        llm_invoked = False

        print(f"\n[{category.upper()}] {url}")

        start = time.time()
        try:
            response = app.scrape_url(url, formats=["markdown"])
            latency_ms = round((time.time() - start) * 1000, 2)

            if isinstance(response, dict):
                markdown = (
                    response.get("markdown") or
                    response.get("data", {}).get("markdown", "")
                )
            else:
                markdown = getattr(response, "markdown", "") or ""

            success = True

        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 2)
            print(f"  SCRAPE FAILED: {e}")
            markdown = ""
            success = False

        if not success:
            checks = {
                "category": category,
                "char_count": 0,
                "has_headings": False,
                "has_tables": False,
                "has_links": False,
                "has_code_blocks": False,
                "has_price": False,
                "has_date": False,
                "paywall_detected": False,
                "noise_ratio": 1.0,
                "must_contain_pass": False,
                "must_contain_missing": [],
                "must_not_contain_pass": True,
                "must_not_contain_found": [],
                "code_score": 0.0,
                "needs_llm": False,
                "fail_reason": "scrape_failed"
            }
        else:
            checks = run_code_checks(markdown, url, gt_map, category=category)

        code_score = checks["code_score"]

        print(f"  Code Score:  {code_score}/100")
        print(f"  Noise:       {checks['noise_ratio']}")
        print(f"  GT Pass:     {checks['must_contain_pass']}")
        print(f"  Needs LLM:   {checks['needs_llm']}")

        llm_result = {
            "llm_score": code_score,
            "llm_raw_avg": code_score / 20,
            "llm_scores_all_runs": [],
            "llm_variance": 0,
            "llm_reason": "LLM not invoked — deterministic score not borderline",
            "needs_human_review": False,
            "confidence": "high"
        }

        if success and checks["needs_llm"]:
            llm_invoked = True
            print(f"  Calling LLM judge...")
            llm_result = run_llm_judge(
                markdown, url, gt, category, code_score
            )
            print(f"  LLM Score:   {llm_result['llm_score']}/100")
            print(f"  Confidence:  {llm_result['confidence']}")

        final_score = compute_final_score(
            code_score=code_score,
            llm_score=llm_result["llm_score"],
            must_contain_pass=checks["must_contain_pass"],
            must_contain_keywords=gt.get("must_contain", [])
        )
        print(f"  Final Score: {final_score}/100")

        llm_raw = llm_result.get("llm_raw_avg", 0)
        llm_variance = llm_result.get("llm_variance", 0)

        needs_human_review = (
            (llm_invoked and 2.5 <= llm_raw <= 3.5) or
            (llm_invoked and llm_variance >= 2)
        )

        result = EvalResult(
            url=url,
            category=category,
            success=success,
            latency_ms=latency_ms,
            char_count=checks["char_count"],
            has_headings=checks["has_headings"],
            has_tables=checks["has_tables"],
            has_links=checks["has_links"],
            has_code_blocks=checks["has_code_blocks"],
            has_price=checks["has_price"],
            has_date=checks["has_date"],
            paywall_detected=checks["paywall_detected"],
            noise_ratio=checks["noise_ratio"],
            must_contain_pass=checks["must_contain_pass"],
            must_not_contain_pass=checks["must_not_contain_pass"],
            code_score=code_score,
            llm_invoked=llm_invoked,
            llm_score=llm_result["llm_score"],
            llm_reason=llm_result["llm_reason"],
            llm_variance=llm_result["llm_variance"],
            llm_confidence=llm_result["confidence"],
            final_score=final_score,
            needs_human_review=needs_human_review,
            fail_reason=checks.get("fail_reason") or ""
        )

        results.append(result)

        if needs_human_review:
            human_review_queue.append({
                "url": url,
                "category": category,
                "final_score": final_score,
                "llm_reason": llm_result["llm_reason"],
                "llm_variance": llm_result["llm_variance"],
                "fail_reason": checks.get("fail_reason") or ""
            })

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(final_score)

    total = len(results)
    successful = [r for r in results if r.success]

    avg_score_all = sum(r.final_score for r in results) / total
    avg_score_success = (
        sum(r.final_score for r in successful) / len(successful)
        if successful else 0
    )
    success_rate = len(successful) / total * 100
    llm_rate = sum(1 for r in results if r.llm_invoked) / total * 100

    latencies = sorted([r.latency_ms for r in results])
    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)

    print("\n" + "=" * 60)
    print("FIRECRAWL SCRAPE EVAL REPORT")
    print("=" * 60)
    print(f"Total URLs:            {total}")
    print(f"Success Rate:          {success_rate:.0f}%")
    print(f"Avg Score (all):       {avg_score_all:.1f}/100")
    print(f"Avg Score (success):   {avg_score_success:.1f}/100")
    print(f"LLM Invocation Rate:   {llm_rate:.0f}%")
    print(f"Human Review Count:    {len(human_review_queue)}")
    print(f"Latency p50:           {p50:.0f}ms")
    print(f"Latency p95:           {p95:.0f}ms")

    print("\nPer Category:")
    for cat, scores in sorted(category_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat:<15} {avg:.1f}/100  (n={len(scores)})")

    print("\nCheck Pass Rates:")
    print(f"  Headings:     {sum(1 for r in results if r.has_headings)}/{total}")
    print(f"  Tables:       {sum(1 for r in results if r.has_tables)}/{total}")
    print(f"  Links:        {sum(1 for r in results if r.has_links)}/{total}")
    print(f"  Code Blocks:  {sum(1 for r in results if r.has_code_blocks)}/{total}")
    print(f"  Price:        {sum(1 for r in results if r.has_price)}/{total}")
    print(f"  GT Pass:      {sum(1 for r in results if r.must_contain_pass)}/{total}")
    print(f"  Paywall Det:  {sum(1 for r in results if r.paywall_detected)}/{total}")

    print("\nCohen's Kappa Check:")
    if os.path.exists("evals/human_labels.json"):
        kappa = run_kappa_check(
            "evals/human_labels.json",
            [asdict(r) for r in results]
        )
    else:
        kappa = {"status": "skipped", "message": "No human labels found"}

    print(f"  Status:  {kappa['status']}")
    print(f"  Message: {kappa['message']}")
    if kappa.get("kappa"):
        print(f"  Kappa:   {kappa['kappa']}")
        print(f"  Trusted: {kappa['trusted']}")

    with open("evals/eval_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_urls": total,
                "success_rate": success_rate,
                "avg_score_all": avg_score_all,
                "avg_score_successful": avg_score_success,
                "llm_invocation_rate": llm_rate,
                "human_review_count": len(human_review_queue),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "per_category": {
                    cat: round(sum(s) / len(s), 1)
                    for cat, s in category_scores.items()
                }
            },
            "results": [asdict(r) for r in results]
        }, f, indent=2)

    with open("evals/human_review_queue.json", "w") as f:
        json.dump(human_review_queue, f, indent=2)

    print("\nSaved → evals/eval_results.json")
    print(f"Saved → evals/human_review_queue.json ({len(human_review_queue)} items)")


if __name__ == "__main__":
    run_evals()
