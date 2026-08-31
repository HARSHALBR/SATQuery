#!/usr/bin/env python3
"""
RS-InternVL: Generation Evaluation & Semantic Metrics (Step 6).

Implements rigorous generation quality and semantic evaluation tools:
- Deterministic text normalization
- Robust binary answer extraction (YES / NO / UNKNOWN)
- Garbage & repetition detection
- Exact match calculation
- Structured record formatting and aggregate metrics computation
"""

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    r"""
    Deterministic text normalization:
    - Lowercases text
    - Strips leading/trailing whitespace
    - Collapses repeated whitespace
    - Strips harmless punctuation (.,!?;:'"()[]{}~`^#$*-_/\)
    """
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove punctuation except alphanumeric and space
    # Replace punctuation characters with space or empty
    chars = []
    for c in text:
        if c in string.punctuation or c in "“”‘’«»—–…":
            chars.append(" ")
        else:
            chars.append(c)
    normalized = "".join(chars)
    # Collapse multiple whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_binary_answer(text: str) -> str:
    """
    Robust binary answer extraction classifying text into YES, NO, or UNKNOWN.
    
    Prevents classifying garbage as YES merely because "yes" appears randomly
    in an unstructured or repetitive string.
    """
    if not text:
        return "UNKNOWN"

    norm = normalize_text(text)
    words = norm.split()
    if not words:
        return "UNKNOWN"

    first_word = words[0]

    # Explicit leading yes / no signals
    if first_word in ("yes", "true", "present", "confirmed", "detected"):
        # Check if immediately followed by negative counter-signal
        if len(words) > 1 and words[1] in ("not", "no", "false"):
            return "UNKNOWN"
        return "YES"

    if first_word in ("no", "false", "not", "none", "absent", "undetected"):
        return "NO"

    # Pattern match for common phrasing: "is present", "is observed", "is detected"
    if "is present" in norm or "are present" in norm or "observed" in norm or "detected" in norm:
        if "not present" in norm or "not observed" in norm or "not detected" in norm or "no " in norm:
            return "NO"
        return "YES"

    if "not present" in norm or "no " in norm or "is not" in norm:
        return "NO"

    return "UNKNOWN"


def detect_repetition(text: str) -> Tuple[bool, float]:
    """
    Detects severe token, word, or character n-gram repetition.
    
    Returns:
        (is_repetitive, repetition_score) where repetition_score in [0.0, 1.0]
    """
    if not text or len(text.strip()) == 0:
        return False, 0.0

    words = normalize_text(text).split()
    if not words:
        return False, 0.0

    if len(words) >= 4:
        # Word-level uniqueness ratio
        word_counts = Counter(words)
        unique_words = len(word_counts)
        total_words = len(words)
        uniqueness_ratio = unique_words / total_words
        most_common_freq = word_counts.most_common(1)[0][1] / total_words

        # If more than 50% of words are identical or uniqueness is very low
        if uniqueness_ratio < 0.4 or most_common_freq > 0.5:
            return True, round(1.0 - uniqueness_ratio, 4)

        # Bigram repetition check
        if len(words) >= 6:
            bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
            bigram_counts = Counter(bigrams)
            most_common_bg = bigram_counts.most_common(1)[0][1]
            if most_common_bg >= 3 and (most_common_bg * 2 / len(words)) > 0.5:
                return True, round(most_common_bg * 2 / len(words), 4)

    # Character-level 3-gram and 4-gram repetition check (for non-spaced repetitive artifacts)
    raw_clean = re.sub(r"\s+", "", text.lower())
    if len(raw_clean) >= 12:
        for n in (3, 4, 5):
            ngrams = [raw_clean[i : i + n] for i in range(len(raw_clean) - n + 1)]
            if ngrams:
                counts = Counter(ngrams)
                most_freq_count = counts.most_common(1)[0][1]
                coverage = (most_freq_count * n) / len(raw_clean)
                if coverage > 0.5 and most_freq_count >= 3:
                    return True, round(min(1.0, coverage), 4)

    return False, 0.0


def is_garbage_generation(text: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation of generation validity:
    - Empty generation
    - Repetitive text (token loops, word loops)
    - Unintelligible token fragments / abnormal Unicode noise
    """
    if text is None or len(str(text).strip()) == 0:
        return {
            "is_valid": False,
            "is_garbage": True,
            "is_empty": True,
            "is_repetitive": False,
            "repetition_ratio": 0.0,
            "quality": "empty",
            "reason": "Generated text is empty or whitespace-only",
        }

    clean_text = str(text).strip()

    # 1. Check repetition
    is_rep, rep_ratio = detect_repetition(clean_text)
    if is_rep:
        return {
            "is_valid": False,
            "is_garbage": True,
            "is_empty": False,
            "is_repetitive": True,
            "repetition_ratio": rep_ratio,
            "quality": "repetitive",
            "reason": f"Severe token/word repetition detected (ratio: {rep_ratio})",
        }

    # 2. Check for abnormal character corruption / non-Latin junk mixture
    # If text has a high density of symbols or mixed unreadable scripts
    non_ascii_symbols = re.findall(r"[^\w\s.,!?'\"-]", clean_text)
    if len(clean_text) > 10 and len(non_ascii_symbols) / len(clean_text) > 0.35:
        return {
            "is_valid": False,
            "is_garbage": True,
            "is_empty": False,
            "is_repetitive": False,
            "repetition_ratio": rep_ratio,
            "quality": "garbage",
            "reason": "Abnormal character distribution / corrupted token artifacts",
        }

    # 3. Check for typical code / syntax noise fragments (e.g., "$db本来", "DbSet本来", ".WinForms")
    noise_patterns = [
        r"\$db",
        r"DbSet",
        r"WinForms",
        r"///",
        r"<pair",
        r"_modify",
    ]
    for pat in noise_patterns:
        if re.search(pat, clean_text):
            return {
                "is_valid": False,
                "is_garbage": True,
                "is_empty": False,
                "is_repetitive": False,
                "repetition_ratio": rep_ratio,
                "quality": "garbage",
                "reason": f"Corrupted token noise pattern matched: '{pat}'",
            }

    return {
        "is_valid": True,
        "is_garbage": False,
        "is_empty": False,
        "is_repetitive": False,
        "repetition_ratio": rep_ratio,
        "quality": "valid",
        "reason": "Valid natural language output",
    }


def compute_exact_match(target: str, generated: str) -> bool:
    """Check normalized exact match between target and generated answer."""
    norm_t = normalize_text(target)
    norm_g = normalize_text(generated)
    return bool(norm_t and norm_g and norm_t == norm_g)


def evaluate_sample(
    query: str,
    target: str,
    generated: str,
    patch_id: str = "",
    task_type: str = "binary:presence",
) -> Dict[str, Any]:
    """
    Evaluate an individual generated sample against ground truth target.
    
    Adheres strictly to the required Section 11 schema.
    """
    norm_target = normalize_text(target)
    norm_gen = normalize_text(generated)

    bin_target = extract_binary_answer(target)
    bin_pred = extract_binary_answer(generated)

    exact_match = compute_exact_match(target, generated)
    quality_info = is_garbage_generation(generated)

    # For binary presence tasks, prediction is correct if binary classification matches
    # and generated text is valid (not garbage).
    is_valid_gen = quality_info["is_valid"]
    quality_str = quality_info["quality"]

    return {
        "patch_id": patch_id,
        "query": query,
        "target": target,
        "generated": generated,
        "normalized_target": norm_target,
        "normalized_generated": norm_gen,
        "binary_target": bin_target,
        "binary_prediction": bin_pred,
        "binary_match": (bin_target != "UNKNOWN" and bin_target == bin_pred),
        "exact_match": exact_match,
        "generation_valid": is_valid_gen,
        "generation_quality": quality_str,
        "task_type": task_type,
    }


def compute_aggregate_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics over a collection of evaluated sample records."""
    total = len(eval_records)
    if total == 0:
        return {
            "total_samples": 0,
            "exact_match_accuracy": 0.0,
            "binary_accuracy": 0.0,
            "binary_unknown_rate": 0.0,
            "generation_validity_rate": 0.0,
            "garbage_rate": 0.0,
            "repetition_rate": 0.0,
            "empty_rate": 0.0,
        }

    exact_matches = sum(1 for r in eval_records if r.get("exact_match", False))
    valid_gens = sum(1 for r in eval_records if r.get("generation_valid", False))
    garbage_gens = sum(1 for r in eval_records if r.get("generation_quality") in ("garbage", "repetitive", "empty"))
    rep_gens = sum(1 for r in eval_records if r.get("generation_quality") == "repetitive")
    empty_gens = sum(1 for r in eval_records if r.get("generation_quality") == "empty")

    binary_evaluable = [r for r in eval_records if r.get("binary_target") in ("YES", "NO")]
    if binary_evaluable:
        binary_correct = sum(
            1 for r in binary_evaluable if r.get("binary_match", False) and r.get("generation_valid", False)
        )
        binary_unknown = sum(1 for r in binary_evaluable if r.get("binary_prediction") == "UNKNOWN")
        binary_acc = binary_correct / len(binary_evaluable)
        binary_unk_rate = binary_unknown / len(binary_evaluable)
    else:
        binary_acc = 0.0
        binary_unk_rate = 0.0

    return {
        "total_samples": total,
        "exact_match_accuracy": round(exact_matches / total, 4),
        "binary_accuracy": round(binary_acc, 4),
        "binary_unknown_rate": round(binary_unk_rate, 4),
        "generation_validity_rate": round(valid_gens / total, 4),
        "garbage_rate": round(garbage_gens / total, 4),
        "repetition_rate": round(rep_gens / total, 4),
        "empty_rate": round(empty_gens / total, 4),
    }
