import re
import json


def load_ground_truth(path="evals/ground_truth.json"):
    with open(path) as f:
        data = json.load(f)
    gt_map = {}
    for category, items in data.items():
        for item in items:
            gt_map[item["url"]] = {**item, "category": category}
    return gt_map


def check_headings(markdown: str) -> bool:
    return bool(re.search(r'^#{1,3} ', markdown, re.MULTILINE))


def check_tables(markdown: str) -> bool:
    return bool(re.search(
        r'^\s*\|.+\|\s*\n\s*\|[\s:\-|]+\|\s*$',
        markdown,
        re.MULTILINE
    ))


def check_links(markdown: str) -> bool:
    return bool(re.findall(r'\[[^\]]+\]\((https?://|/)[^)]+\)', markdown))


def check_code_blocks(markdown: str) -> bool:
    return '```' in markdown


def check_noise(markdown: str) -> float:
    lines = markdown.split("\n")
    noise_patterns = [
        'cookie policy', 'accept cookies', 'subscribe now',
        'sign up for newsletter', 'all rights reserved',
        'privacy policy', 'terms of service', 'advertisement'
    ]
    noise_lines = sum(
        1 for l in lines
        if any(p in l.lower() for p in noise_patterns)
    )
    return round(noise_lines / max(len(lines), 1), 2)


def check_price(markdown: str) -> bool:
    return bool(re.search(r'\$[\d,]+(\.\d{2})?', markdown))


def check_date(markdown: str) -> bool:
    patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4}\b'
    ]
    return any(bool(re.search(p, markdown, re.IGNORECASE)) for p in patterns)


def check_paywall(markdown: str) -> bool:
    paywall_signals = [
        'subscribe to read', 'subscribe to continue',
        'already a subscriber', 'sign in to read',
        'create an account to read', 'premium content'
    ]
    md_lower = markdown.lower()
    return any(p in md_lower for p in paywall_signals)


def check_must_contain(markdown: str, keywords: list) -> tuple:
    if not keywords:
        return True, []
    md_lower = markdown.lower()
    missing = [k for k in keywords if k.lower() not in md_lower]
    return len(missing) == 0, missing


def check_must_not_contain(markdown: str, keywords: list) -> tuple:
    if not keywords:
        return True, []
    md_lower = markdown.lower()
    found = [k for k in keywords if k.lower() in md_lower]
    return len(found) == 0, found


def score_category(
    category: str,
    has_headings: bool,
    has_tables: bool,
    has_links: bool,
    has_code_blocks: bool,
    has_price: bool,
    has_date: bool,
    markdown: str,
    gt: dict,
    must_contain_keywords: list,
    must_contain_pass: bool,
    paywall_detected: bool
) -> int:
    score = 0

    if category == "docs":
        if has_headings:        score += 15
        if has_code_blocks:     score += 15
        if has_tables:          score += 10

    elif category == "ecommerce":
        if has_price:           score += 20
        if has_headings:        score += 10
        if has_tables:          score += 10

    elif category == "news":
        if has_headings:        score += 15
        if has_date:            score += 15
        if len(markdown) > 500: score += 10

    elif category == "jobs":
        if has_headings:        score += 20
        if len(markdown) > 300: score += 10
        if must_contain_keywords and must_contain_pass:
            score += 10

    elif category == "research":
        if has_headings:        score += 15
        if has_tables:          score += 15
        if len(markdown) > 500: score += 10

    elif category == "spa":
        if has_headings:        score += 15
        if has_code_blocks:     score += 15
        if len(markdown) > 300: score += 10

    elif category == "adversarial":
        expected_paywall = gt.get("expected_paywall", False)
        if expected_paywall and paywall_detected:
            score += 40
        elif expected_paywall and not paywall_detected:
            if len(markdown) > 500:
                score += 20
        elif not expected_paywall:
            if has_headings:        score += 15
            if has_links:           score += 15
            if len(markdown) > 300: score += 10

    else:
        # Unknown category — fixes 40-everywhere bug
        if has_headings:        score += 15
        if has_code_blocks:     score += 10
        if has_tables:          score += 10
        if len(markdown) > 500: score += 5

    return score


def run_code_checks(
    markdown: str,
    url: str,
    gt_map: dict,
    category: str = None
) -> dict:
    gt = gt_map.get(url, {})
    category = category or gt.get("category", "unknown")

    must_contain_keywords = gt.get("must_contain", [])
    must_not_contain_keywords = gt.get("must_not_contain", [])

    if not markdown or len(markdown) < 100:
        return {
            "category": category,
            "char_count": len(markdown) if markdown else 0,
            "has_headings": False,
            "has_tables": False,
            "has_links": False,
            "has_code_blocks": False,
            "has_price": False,
            "has_date": False,
            "paywall_detected": False,
            "noise_ratio": 1.0,
            "must_contain_pass": False,
            "must_contain_missing": must_contain_keywords,
            "must_not_contain_pass": True,
            "must_not_contain_found": [],
            "code_score": 0.0,
            "needs_llm": False,
            "fail_reason": "empty_or_too_short"
        }

    has_headings     = check_headings(markdown)
    has_tables       = check_tables(markdown)
    has_links        = check_links(markdown)
    has_code_blocks  = check_code_blocks(markdown)
    noise_ratio      = check_noise(markdown)
    has_price        = check_price(markdown)
    has_date         = check_date(markdown)
    paywall_detected = check_paywall(markdown)

    must_contain_pass, missing = check_must_contain(
        markdown, must_contain_keywords
    )
    must_not_contain_pass, found = check_must_not_contain(
        markdown, must_not_contain_keywords
    )

    # Universal score — 40 pts
    score = 0
    if len(markdown) >= 200:    score += 10
    if has_links:               score += 10
    if noise_ratio < 0.05:      score += 15
    elif noise_ratio < 0.10:    score += 8
    if must_not_contain_pass:   score += 5

    # Category score — 40 pts
    score += score_category(
        category=category,
        has_headings=has_headings,
        has_tables=has_tables,
        has_links=has_links,
        has_code_blocks=has_code_blocks,
        has_price=has_price,
        has_date=has_date,
        markdown=markdown,
        gt=gt,
        must_contain_keywords=must_contain_keywords,
        must_contain_pass=must_contain_pass,
        paywall_detected=paywall_detected
    )

    # Ground truth score — 20 pts, only if keywords defined
    if must_contain_keywords and must_contain_pass:
        score += 20

    score = min(score, 100)

    # Route to LLM only for semantic-heavy categories when ambiguous
    needs_llm = (
    (55 <= score <= 75) or
    (must_contain_keywords and not must_contain_pass and score > 30) or
    (noise_ratio >= 0.15)
) and category in ["ecommerce", "news", "research", "spa"]

    return {
        "category": category,
        "char_count": len(markdown),
        "has_headings": has_headings,
        "has_tables": has_tables,
        "has_links": has_links,
        "has_code_blocks": has_code_blocks,
        "has_price": has_price,
        "has_date": has_date,
        "paywall_detected": paywall_detected,
        "noise_ratio": noise_ratio,
        "must_contain_pass": must_contain_pass,
        "must_contain_missing": missing,
        "must_not_contain_pass": must_not_contain_pass,
        "must_not_contain_found": found,
        "code_score": float(score),
        "needs_llm": needs_llm,
        "fail_reason": None
    }
