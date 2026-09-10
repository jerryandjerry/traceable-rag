"""Deterministic keyword extraction for intent and GraphRAG retrieval."""

import re


def extract_keywords_simple(query: str) -> tuple[list[str], list[str]]:
    """Return meaningful query tokens in both keyword categories."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    words = query_clean.split()

    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    return keywords, keywords


def extract_keywords_advanced(query: str) -> tuple[list[str], list[str]]:
    """Partition query tokens into concept-level and detail-level keywords."""
    high_level_indicators = {
        'strategy', 'policy', 'plan', 'approach', 'method', 'process', 'system',
        'framework', 'architecture', 'design', 'model', 'concept', 'theory',
        'principle', 'guideline', 'standard', 'protocol', 'procedure'
    }
    
    low_level_indicators = {
        'specific', 'detail', 'example', 'instance', 'case', 'step', 'action',
        'task', 'operation', 'function', 'feature', 'component', 'element',
        'parameter', 'value', 'setting', 'configuration', 'implementation'
    }
    
    all_keywords = extract_keywords_simple(query)[0]

    hl_keywords = []
    ll_keywords = []
    
    for keyword in all_keywords:
        if keyword in high_level_indicators:
            hl_keywords.append(keyword)
        elif keyword in low_level_indicators:
            ll_keywords.append(keyword)
        else:
            ll_keywords.append(keyword)

    # Graph relationship search still needs a concept query when no indicator matched.
    if not hl_keywords and ll_keywords:
        hl_keywords = ll_keywords[:2]
        ll_keywords = ll_keywords[2:]

    return hl_keywords, ll_keywords
