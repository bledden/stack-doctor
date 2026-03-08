"""
Stack Doctor — Filter raw scraped triples to high-quality training data.

Takes raw_triples.json and applies filtering rules to produce
filtered_triples.json with only high-quality bug-fix triples.

Filters:
  1. Bug-only: keep PRs that are actual bug fixes
  2. Diff size: keep diffs between 5-500 lines
  3. Classification confidence: re-score failure family, require >= 2 keyword hits
  4. Content quality: require code blocks, linked issues, or error patterns
  5. Model name extraction: detect model names from PR body and diff
  6. Deduplication: remove duplicate pr_numbers

Usage:
    python data/filter_triples.py
    python data/filter_triples.py --input data/raw_triples.json --output data/filtered_triples.json
"""

import json
import re
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Failure family keyword lists (mirrored from scrape_issues.py for scoring)
# ---------------------------------------------------------------------------

FAILURE_FAMILY_KEYWORDS = {
    "arch_guard": [
        "cuda arch", "compute capability", "sm_120", "sm_121",
        "unsupported architecture", "arch check", "gpu architecture",
        "__cuda_arch__", "sm_90", "sm_89", "arch guard",
    ],
    "backend_whitelist": [
        "whitelist gpu", "marlin supported", "not supported gpu",
        "gpu not in", "supported_gpus", "whitelist", "allow list",
        "supported gpu list",
    ],
    "runtime_loader": [
        "runtime path", "libcuda", "cuda_home", "ld_library_path",
        "shared object", "dlopen", "rocm_path", "libcudart",
        "library not found", "cannot open shared",
    ],
    "backend_selector": [
        "backend selector", "flash_attn", "xformers",
        "attention backend", "cutlass dispatch", "kernel selection",
        "backend fallback", "dispatch",
    ],
    "model_config": [
        "config mismatch", "num_expert", "shape mismatch",
        "rope scaling", "model config", "config.json",
        "num_heads", "hidden_size", "intermediate_size",
    ],
    "weight_layout": [
        "weight mapping", "weight layout", "gate_proj",
        "weight convert", "tensor mapping", "checkpoint load",
        "weight name", "state_dict", "weight key",
        "weight", "mapping", "layout", "convert",
        "up_proj", "qkv", "safetensors",
    ],
    "memory_oom": [
        "cuda out of memory", "oom", "kv_cache",
        "memory fragmentation", "max_model_len", "gpu memory",
        "torch.cuda.outofmemoryerror", "out of memory",
        "vram", "memory allocation",
    ],
    "quantization_error": [
        "fp8 error", "quantization mismatch", "calibration",
        "precision loss", "nf4", "int4 error", "awq error",
        "gptq error", "fp8", "int8", "quantize",
    ],
    "distributed_comm": [
        "nccl error", "tensor parallel", "all_reduce", "rdma",
        "pipeline parallel hang", "nccl timeout",
        "collective operation", "nccl", "all_gather",
    ],
    "driver_compat": [
        "driver version", "cudnn mismatch", "cuda toolkit",
        "nvcc error", "driver too old", "cuda version mismatch",
        "libcudnn", "driver compat", "toolkit version",
        "cuda compat", "driver", "toolkit", "nvcc",
        "ptx", "cuda_home", "cuda version",
    ],
}

# Model names to detect
MODEL_NAMES = [
    "llama", "deepseek", "qwen", "mistral", "falcon", "gemma",
    "phi", "mixtral", "moe", "nemotron", "yi", "baichuan",
    "chatglm", "internlm", "codellama", "starcoder", "opt",
    "bloom", "mpt", "dolly", "vicuna", "wizardlm",
]

# Error patterns that indicate real bugs
ERROR_PATTERNS = [
    r"(?i)error[:\s]",
    r"(?i)traceback",
    r"(?i)exception",
    r"(?i)failed",
    r"(?i)segfault",
    r"(?i)core dump",
    r"(?i)assert.*fail",
    r"(?i)cuda error",
    r"(?i)runtime error",
    r"(?i)not supported",
    r"(?i)mismatch",
    r"(?i)invalid",
    r"(?i)cannot",
    r"(?i)unable to",
]

# Labels indicating bug fixes
BUG_LABELS = {"bug", "bugfix", "fix", "bug-fix", "hotfix", "regression"}

# Title prefixes/keywords that indicate bugs (case-insensitive)
BUG_TITLE_INDICATORS = [
    "[bugfix]", "[bug]", "[fix]", "fix:", "fix ",
    "bugfix:", "hotfix:", "patch:",
]

# Title keywords that indicate NON-bugs (only exclude if no bug indicator)
NON_BUG_TITLE_KEYWORDS = [
    "refactor", "ci/", "docs", "test:", "perf:", "chore",
    "feat:", "feature", "ci:", "doc:", "style:", "build:",
]


def is_bug_pr(triple: dict) -> bool:
    """Filter 1: Check if a PR is a bug fix."""
    labels = {l.lower().strip() for l in triple.get("pr_labels", [])}
    title = (triple.get("pr_title") or "").lower()

    # Check labels
    if labels & BUG_LABELS:
        return True

    # Check title for bug indicators
    has_bug_indicator = any(ind in title for ind in BUG_TITLE_INDICATORS)
    if has_bug_indicator:
        return True

    # If title has non-bug keywords and no bug indicator, reject
    has_non_bug = any(kw in title for kw in NON_BUG_TITLE_KEYWORDS)
    if has_non_bug:
        return False

    # Default: keep it (PRs without clear category may still be bug fixes)
    return True


def is_valid_diff_size(triple: dict) -> bool:
    """Filter 2: Check if diff is between 5 and 500 lines."""
    diff_lines = triple.get("diff_lines", 0)
    return 5 <= diff_lines <= 500


def score_failure_family(text: str) -> dict[str, int]:
    """Score text against all failure families. Returns {family: hit_count}."""
    text_lower = text.lower()
    scores = {}
    for family, keywords in FAILURE_FAMILY_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            if kw in text_lower:
                hits += 1
        scores[family] = hits
    return scores


def reclassify_family(triple: dict) -> tuple[str | None, int]:
    """Filter 3: Re-classify with stricter scoring. Returns (family, score)."""
    pr_title = triple.get("pr_title", "") or ""
    pr_body = triple.get("pr_body", "") or ""
    diff = triple.get("diff", "") or ""

    # Combine issue text too
    issue_text = ""
    for issue in triple.get("linked_issues", []):
        issue_text += " " + (issue.get("title", "") or "")
        issue_text += " " + (issue.get("body", "") or "")

    combined = f"{pr_title} {pr_body} {issue_text} {diff}"
    scores = score_failure_family(combined)

    best_family = max(scores, key=scores.get)
    best_score = scores[best_family]

    if best_score < 2:
        return None, best_score

    return best_family, best_score


def has_content_quality(triple: dict) -> bool:
    """Filter 4: Check content quality indicators."""
    pr_body = triple.get("pr_body", "") or ""
    diff = triple.get("diff", "") or ""

    # Check for code blocks in PR body
    if "```" in pr_body:
        return True

    # Check for linked issues
    if triple.get("linked_issues"):
        return True

    # Check for error patterns in diff
    for pattern in ERROR_PATTERNS:
        if re.search(pattern, diff):
            return True

    return False


def detect_model_names(triple: dict) -> list[str]:
    """Filter 5: Extract model names from PR body and diff."""
    pr_title = triple.get("pr_title", "") or ""
    pr_body = triple.get("pr_body", "") or ""
    diff = triple.get("diff", "") or ""

    combined = f"{pr_title} {pr_body} {diff}".lower()
    detected = []
    for model in MODEL_NAMES:
        if model in combined:
            detected.append(model)
    return sorted(set(detected))


def filter_triples(triples: list[dict]) -> list[dict]:
    """Apply all filters and return high-quality triples."""
    stats = {
        "input": len(triples),
        "removed_not_bug": 0,
        "removed_diff_size": 0,
        "removed_low_confidence": 0,
        "removed_low_quality": 0,
        "removed_duplicate": 0,
        "output": 0,
    }

    results = []
    seen_pr_numbers = set()

    for triple in triples:
        # Filter 6: Deduplication (check first to avoid wasted work)
        pr_key = (triple.get("repo", ""), triple.get("pr_number"))
        if pr_key in seen_pr_numbers:
            stats["removed_duplicate"] += 1
            continue
        seen_pr_numbers.add(pr_key)

        # Filter 1: Bug-only
        if not is_bug_pr(triple):
            stats["removed_not_bug"] += 1
            continue

        # Filter 2: Diff size
        if not is_valid_diff_size(triple):
            stats["removed_diff_size"] += 1
            continue

        # Filter 3: Classification confidence
        family, score = reclassify_family(triple)
        if family is None:
            stats["removed_low_confidence"] += 1
            continue

        # Filter 4: Content quality
        if not has_content_quality(triple):
            stats["removed_low_quality"] += 1
            continue

        # Filter 5: Model name extraction (enrichment, not a filter)
        detected_models = detect_model_names(triple)

        # Build filtered triple with updated fields
        filtered = dict(triple)
        filtered["failure_family"] = family
        filtered["family_confidence_score"] = score
        filtered["detected_models"] = detected_models
        filtered["detected_model"] = detected_models[0] if detected_models else None

        results.append(filtered)

    stats["output"] = len(results)
    return results, stats


def print_summary(results: list[dict], stats: dict) -> None:
    """Print a human-readable summary of the filtering."""
    print("=" * 60)
    print("STACK DOCTOR — TRIPLE FILTERING SUMMARY")
    print("=" * 60)
    print(f"Input triples:              {stats['input']}")
    print(f"Removed (not bug fix):      {stats['removed_not_bug']}")
    print(f"Removed (diff size):        {stats['removed_diff_size']}")
    print(f"Removed (low confidence):   {stats['removed_low_confidence']}")
    print(f"Removed (low quality):      {stats['removed_low_quality']}")
    print(f"Removed (duplicate):        {stats['removed_duplicate']}")
    print(f"Output triples:             {stats['output']}")
    print()

    # Family distribution
    family_counts = {}
    for t in results:
        f = t.get("failure_family", "unknown")
        family_counts[f] = family_counts.get(f, 0) + 1
    print("Failure family distribution:")
    for f, c in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {f:25s} {c:4d}")
    print()

    # Repo distribution
    repo_counts = {}
    for t in results:
        r = t.get("repo", "unknown")
        repo_counts[r] = repo_counts.get(r, 0) + 1
    print("Repo distribution:")
    for r, c in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"  {r:40s} {c:4d}")
    print()

    # Model detection stats
    with_model = sum(1 for t in results if t.get("detected_model"))
    model_counts = {}
    for t in results:
        for m in t.get("detected_models", []):
            model_counts[m] = model_counts.get(m, 0) + 1
    print(f"Triples with detected model: {with_model}/{len(results)}")
    if model_counts:
        print("Detected models:")
        for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
            print(f"  {m:20s} {c:4d}")
    print()

    # Confidence score distribution
    if results:
        scores = [t.get("family_confidence_score", 0) for t in results]
        avg_score = sum(scores) / len(scores)
        print(f"Avg confidence score:       {avg_score:.1f}")
        print(f"Min confidence score:       {min(scores)}")
        print(f"Max confidence score:       {max(scores)}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter raw triples to high-quality training data"
    )
    parser.add_argument(
        "--input",
        default="data/raw_triples.json",
        help="Input raw triples file (default: data/raw_triples.json)",
    )
    parser.add_argument(
        "--output",
        default="data/filtered_triples.json",
        help="Output filtered triples file (default: data/filtered_triples.json)",
    )
    parser.add_argument(
        "--min-diff",
        type=int,
        default=5,
        help="Minimum diff lines (default: 5)",
    )
    parser.add_argument(
        "--max-diff",
        type=int,
        default=500,
        help="Maximum diff lines (default: 500)",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=2,
        help="Minimum keyword hits for family classification (default: 2)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run scrape_issues.py first to generate raw_triples.json")
        return

    with open(input_path) as f:
        triples = json.load(f)

    print(f"Loaded {len(triples)} raw triples from {input_path}")
    print()

    results, stats = filter_triples(triples)

    # Parity balancer: cap over-represented families to improve balance
    from collections import Counter
    family_counts = Counter(t["failure_family"] for t in results)
    if family_counts:
        median_count = sorted(family_counts.values())[len(family_counts) // 2]
        cap = max(median_count * 2, 15)  # cap at 2x median or 15, whichever is higher
        balanced = []
        family_taken = Counter()
        # Shuffle to avoid always taking the same ones
        import random
        random.shuffle(results)
        for t in results:
            fam = t["failure_family"]
            if family_taken[fam] < cap:
                balanced.append(t)
                family_taken[fam] += 1
        if len(balanced) < len(results):
            removed = len(results) - len(balanced)
            stats["removed_parity"] = removed
            print(f"Parity balancer: capped at {cap}/family, removed {removed} over-represented triples")
            results = balanced

    print_summary(results, stats)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} filtered triples to {output_path}")


if __name__ == "__main__":
    main()
