"""
Stack Doctor — GitHub Issue/PR Scraper

Scrapes resolved issues with linked merged PRs from inference framework repos
to extract (issue, PR, diff) triples for training data.

Targets: vLLM, SGLang, FlashInfer, CUTLASS, TensorRT-LLM, Flash-Attention, Triton

Usage:
    python data/scrape_issues.py
    python data/scrape_issues.py --limit 50 --output data/raw_triples.json
"""

import json
import subprocess
import sys
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Target repos and search queries per failure family
# ---------------------------------------------------------------------------

REPOS = [
    "vllm-project/vllm",
    "sgl-project/sglang",
    "flashinfer-ai/flashinfer",
    "NVIDIA/cutlass",
    "NVIDIA/TensorRT-LLM",
    "Dao-AILab/flash-attention",
    "triton-inference-server/server",
]

# Search queries mapped to failure families
FAILURE_QUERIES = {
    "arch_guard": [
        "cuda arch",
        "compute capability",
        "sm_120",
        "sm_121",
        "unsupported architecture",
        "arch check",
        "gpu architecture",
    ],
    "backend_whitelist": [
        "whitelist gpu",
        "marlin supported",
        "not supported gpu",
        "gpu not in",
        "supported_gpus",
    ],
    "runtime_loader": [
        "runtime path",
        "libcuda",
        "cuda_home",
        "ld_library_path",
        "shared object",
        "dlopen",
        "rocm_path",
    ],
    "backend_selector": [
        "backend selector",
        "flash_attn",
        "xformers",
        "attention backend",
        "cutlass dispatch",
        "kernel selection",
    ],
    "model_config": [
        "config mismatch",
        "num_expert",
        "shape mismatch",
        "rope scaling",
        "model config",
        "config.json",
    ],
    "weight_layout": [
        "weight mapping",
        "weight layout",
        "gate_proj",
        "weight convert",
        "tensor mapping",
        "checkpoint load",
    ],
    "memory_oom": [
        "CUDA out of memory",
        "OOM",
        "kv_cache",
        "memory fragmentation",
        "max_model_len",
        "gpu memory",
        "torch.cuda.OutOfMemoryError",
    ],
    "quantization_error": [
        "fp8 error",
        "quantization mismatch",
        "calibration",
        "precision loss",
        "nf4",
        "int4 error",
        "awq error",
        "gptq error",
    ],
    "distributed_comm": [
        "NCCL error",
        "tensor parallel",
        "all_reduce",
        "RDMA",
        "pipeline parallel hang",
        "nccl timeout",
        "collective operation",
    ],
    "driver_compat": [
        "driver version",
        "cudnn mismatch",
        "cuda toolkit",
        "nvcc error",
        "driver too old",
        "cuda version mismatch",
        "libcudnn",
    ],
}

# Generic bugfix queries that may catch issues across all families
GENERIC_QUERIES = [
    "kernel launch failed",
    "CUDA error",
    "RuntimeError",
    "GPU not supported",
    "inference crash",
    "model loading error",
]


def run_gh(args: list[str], timeout: int = 30) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def search_merged_prs(repo: str, query: str, limit: int = 30) -> list[dict]:
    """Search for merged PRs in a repo matching a query."""
    raw = run_gh([
        "search", "prs",
        "--repo", repo,
        "--state", "closed",
        "--merged",
        "--limit", str(limit),
        "--json", "number,title,body,closedAt,labels,url",
        "--", query,
    ])
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def get_pr_details(repo: str, pr_number: int) -> dict | None:
    """Get full PR details including files changed."""
    raw = run_gh([
        "pr", "view", str(pr_number),
        "--repo", repo,
        "--json", "number,title,body,closedAt,labels,files,additions,deletions,url",
    ])
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def get_pr_diff(repo: str, pr_number: int) -> str:
    """Get the code diff for a PR."""
    return run_gh(["pr", "diff", str(pr_number), "--repo", repo], timeout=60)


def get_issue_details(repo: str, issue_number: int) -> dict | None:
    """Get issue details."""
    raw = run_gh([
        "issue", "view", str(issue_number),
        "--repo", repo,
        "--json", "number,title,body,labels,closedAt,url",
    ])
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_linked_issues(pr_body: str) -> list[int]:
    """Extract issue numbers from PR body (Closes #N, Fixes #N patterns)."""
    import re
    pattern = r"(?:closes?|fixes?|resolves?)\s+#(\d+)"
    matches = re.findall(pattern, pr_body, re.IGNORECASE)
    return [int(m) for m in matches]


def classify_failure_family(title: str, body: str, diff: str) -> str | None:
    """Classify a PR into a failure family based on keywords."""
    text = f"{title} {body} {diff}".lower()

    scores = {}
    for family, queries in FAILURE_QUERIES.items():
        score = 0
        for q in queries:
            # Count keyword hits
            for keyword in q.split():
                if keyword in text:
                    score += 1
        scores[family] = score

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    return best


def scrape_repo(repo: str, limit_per_query: int = 10) -> list[dict]:
    """Scrape a single repo for issue→PR→diff triples."""
    seen_prs = set()
    triples = []

    # Search with failure-family-specific queries
    all_queries = []
    for family, queries in FAILURE_QUERIES.items():
        for q in queries:
            all_queries.append((q, family))
    for q in GENERIC_QUERIES:
        all_queries.append((q, None))

    for query, hint_family in all_queries:
        prs = search_merged_prs(repo, query, limit=limit_per_query)

        for pr in prs:
            pr_num = pr.get("number")
            if not pr_num or pr_num in seen_prs:
                continue
            seen_prs.add(pr_num)

            pr_body = pr.get("body", "") or ""
            pr_title = pr.get("title", "") or ""

            # Get the diff
            diff = get_pr_diff(repo, pr_num)
            if not diff:
                continue

            # Classify into failure family
            family = hint_family or classify_failure_family(pr_title, pr_body, diff)

            # Extract linked issues
            linked_issue_nums = extract_linked_issues(pr_body)
            linked_issues = []
            for issue_num in linked_issue_nums[:3]:  # cap at 3
                issue = get_issue_details(repo, issue_num)
                if issue:
                    linked_issues.append(issue)
                time.sleep(0.2)  # rate limit

            triple = {
                "repo": repo,
                "pr_number": pr_num,
                "pr_title": pr_title,
                "pr_body": pr_body,
                "pr_url": pr.get("url", ""),
                "pr_closed_at": pr.get("closedAt", ""),
                "pr_labels": [l.get("name", "") for l in pr.get("labels", [])],
                "diff": diff,
                "diff_lines": len(diff.splitlines()),
                "linked_issues": linked_issues,
                "failure_family": family,
            }
            triples.append(triple)
            print(f"  [{repo}] PR #{pr_num}: {pr_title[:60]}... → {family or 'unclassified'}")

            time.sleep(0.3)  # rate limit

    return triples


def main():
    parser = argparse.ArgumentParser(description="Scrape inference framework repos for training data")
    parser.add_argument("--repos", nargs="*", default=None, help="Specific repos to scrape (default: all)")
    parser.add_argument("--limit", type=int, default=10, help="Max PRs per query per repo")
    parser.add_argument("--output", default="data/raw_triples.json", help="Output file path")
    parser.add_argument("--families", nargs="*", default=None, help="Only scrape specific failure families")
    args = parser.parse_args()

    repos = args.repos or REPOS

    if args.families:
        global FAILURE_QUERIES
        FAILURE_QUERIES = {k: v for k, v in FAILURE_QUERIES.items() if k in args.families}

    print(f"Scraping {len(repos)} repos with {len(FAILURE_QUERIES)} failure families")
    print(f"Queries per family: {sum(len(v) for v in FAILURE_QUERIES.values())} + {len(GENERIC_QUERIES)} generic")
    print()

    all_triples = []
    for repo in repos:
        print(f"\n{'='*60}")
        print(f"Scraping {repo}...")
        print(f"{'='*60}")
        triples = scrape_repo(repo, limit_per_query=args.limit)
        all_triples.extend(triples)
        print(f"  → {len(triples)} triples from {repo}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"Total triples: {len(all_triples)}")

    family_counts = {}
    for t in all_triples:
        f = t.get("failure_family") or "unclassified"
        family_counts[f] = family_counts.get(f, 0) + 1
    for f, c in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {f}: {c}")

    repo_counts = {}
    for t in all_triples:
        r = t["repo"]
        repo_counts[r] = repo_counts.get(r, 0) + 1
    print()
    for r, c in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_triples, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
