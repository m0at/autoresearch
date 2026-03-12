import html
import re
import zlib

_stats = {"bytes_before": 0, "bytes_after": 0, "docs_discarded": 0, "paragraphs_stripped": 0}

# --- Stage 1: normalization ---
_NONPRINT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_LONG_URL = re.compile(r"https?://\S{60,}")
_MULTI_NL = re.compile(r"\n{3,}")

# --- Stage 2: header/footer ---
_NAV_CHARS = re.compile(r"[|>»]")
_FOOTER_KEYWORDS = {
    "©", "copyright", "all rights reserved", "terms", "privacy",
    "contact us", "sitemap", "block this user", "sign in to add a comment",
    "reset your password", "pings are currently closed",
    "both comments and pings are currently closed",
}
# Literal first-paragraph CMS nav prefixes to strip from doc start
_NAV_PREFIX_RE = re.compile(
    r"^(Post navigation|More Listening|Keep up with \w+|Explore More"
    r"|You May Also Like|About This Class|Share\s*\nOverview"
    r"|We'?re sorry but this (website|page|app) doesn'?t work properly without JavaScript[^\n]*"
    r"\s*\nPlease enable it to continue\.?)\s*\n+",
    re.IGNORECASE | re.MULTILINE,
)

# --- Stage 3: boilerplate paragraphs ---
_COOKIE_KEYWORDS = {"cookie", "consent", "privacy", "accept", "decline", "gdpr"}
_NEWS_KEYWORDS = {"subscribe", "newsletter", "inbox", "unsubscribe", "sign up"}
_SOCIAL_KEYWORDS = {"facebook", "twitter", "linkedin", "pinterest", "reddit", "instagram", "share", "tweet"}
_AD_RE = re.compile(
    r"\b(advertisement|sponsored|affiliate link|amazon associate|paid partnership|this post contains)\b",
    re.IGNORECASE,
)
# CMS comment-system tail paragraphs (exact / near-exact)
_CMS_TAIL_RE = re.compile(
    r"\b(Please sign in to add a comment|Registration is free, and takes less than a minute"
    r"|Click here to reset your password|Sign in to get notified via email when new comments"
    r"|kata lagi|FINE PRINT|RSS 2\.0 feed|pings are currently closed"
    r"|Block this user|Are you sure you want to block this user)\b",
    re.IGNORECASE,
)
# Forum user-status badge lines
_FORUM_BADGE_RE = re.compile(r"^(Well-Known Member|Senior Member|Junior Member|New Member|Banned)\s*$", re.MULTILINE)
# E-commerce metadata
_ECOM_RE = re.compile(
    r"^\s*SKU:\s*\n|\$\d+\.\d{2}\s*\n\s*(Unavailable|In Stock|Out of Stock)\s*\n\s*per item",
    re.IGNORECASE | re.MULTILINE,
)


def _stage1_normalize(text):
    text = html.unescape(text)
    text = _MULTI_NL.sub("\n\n", text)
    text = _LONG_URL.sub("", text)
    text = _NONPRINT.sub("", text)
    return text


def _stage2_strip_nav_footer(text):
    # Strip known CMS nav prefixes from doc start
    text = _NAV_PREFIX_RE.sub("", text, count=1)

    # Strip nav header: pipe/arrow-heavy first block
    head = text[:500]
    head_lines = head.splitlines()
    if len(head_lines) > 5:
        avg_len = sum(len(l) for l in head_lines) / len(head_lines)
        nav_chars = len(_NAV_CHARS.findall(head))
        if avg_len < 40 and nav_chars / max(len(head), 1) > 0.10:
            paragraphs = text.split("\n\n")
            for i, p in enumerate(paragraphs[1:], 1):
                if len(p.strip()) > 100:
                    text = "\n\n".join(paragraphs[i:])
                    break

    # Strip footer: last 10% of doc
    cutoff = max(0, len(text) - len(text) // 10)
    tail = text[cutoff:].lower()
    hits = sum(1 for kw in _FOOTER_KEYWORDS if kw in tail)
    if hits >= 2:
        paragraphs = text.split("\n\n")
        keep = len(paragraphs)
        for i in range(len(paragraphs) - 1, -1, -1):
            chunk = "\n\n".join(paragraphs[i:]).lower()
            if sum(1 for kw in _FOOTER_KEYWORDS if kw in chunk) >= 2:
                keep = i
            else:
                break
        text = "\n\n".join(paragraphs[:keep])

    return text


def _stage3_strip_boilerplate(paragraphs):
    kept = []
    stripped = 0
    for para in paragraphs:
        low = para.lower()
        wc = len(para.split())

        # Cookie/consent
        if wc < 200 and sum(1 for kw in _COOKIE_KEYWORDS if kw in low) >= 3:
            stripped += 1; continue

        # Newsletter
        if wc < 100 and sum(1 for kw in _NEWS_KEYWORDS if kw in low) >= 2:
            stripped += 1; continue

        # Social share line
        social_hit = any(
            sum(1 for kw in _SOCIAL_KEYWORDS if kw in line.lower()) >= 3
            for line in para.splitlines()
        )
        if social_hit:
            stripped += 1; continue

        # Ad disclosure
        if _AD_RE.search(para):
            stripped += 1; continue

        # CMS comment tails
        if _CMS_TAIL_RE.search(para):
            stripped += 1; continue

        # Forum user badge (whole paragraph is just badge text)
        cleaned = _FORUM_BADGE_RE.sub("", para).strip()
        if not cleaned:
            stripped += 1; continue

        # E-commerce SKU/price metadata
        if _ECOM_RE.search(para):
            stripped += 1; continue

        kept.append(para)

    return kept, stripped


def _is_keyword_spam(text):
    """Detect SEO keyword-list docs: many short lines, no sentence-ending punctuation."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 8:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    if avg_len > 60:
        return False
    sentence_lines = sum(1 for l in lines if re.search(r"[.!?]\s*$", l))
    return sentence_lines / len(lines) < 0.10


def _compression_ratio(text):
    encoded = text.encode("utf-8")
    if not encoded:
        return 1.0
    return len(zlib.compress(encoded, level=6)) / len(encoded)


def strip_document(text: str):
    if len(text) < 50:
        _stats["docs_discarded"] += 1
        return None

    _stats["bytes_before"] += len(text.encode("utf-8"))

    text = _stage1_normalize(text)
    text = _stage2_strip_nav_footer(text)

    paragraphs = text.split("\n\n")
    paragraphs, n_stripped = _stage3_strip_boilerplate(paragraphs)
    _stats["paragraphs_stripped"] += n_stripped
    text = "\n\n".join(p for p in paragraphs if p.strip())

    # Stage 4: document-level discard
    words = text.split()
    if len(words) < 100:
        _stats["docs_discarded"] += 1
        return None
    if _is_keyword_spam(text):
        _stats["docs_discarded"] += 1
        return None
    if _compression_ratio(text) < 0.12:
        _stats["docs_discarded"] += 1
        return None
    if len(set(w.lower() for w in words)) / len(words) < 0.10:
        _stats["docs_discarded"] += 1
        return None

    _stats["bytes_after"] += len(text.encode("utf-8"))
    return text


def stripping_stats() -> dict:
    s = dict(_stats)
    bb, ba = s["bytes_before"], s["bytes_after"]
    s["bytes_removed"] = bb - ba
    s["reduction_pct"] = round(100.0 * (bb - ba) / bb, 2) if bb else 0.0
    return s
