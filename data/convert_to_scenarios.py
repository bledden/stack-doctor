"""
Stack Doctor — Convert raw triples to training scenarios.

Takes the scraped (issue, PR, diff) triples and converts them into
Stack Doctor Scenario format for training.

The specialist opinions layer is generated synthetically since real
GitHub issues don't have 4 specialists with one liar.

Usage:
    python data/convert_to_scenarios.py
    python data/convert_to_scenarios.py --input data/filtered_triples.json --output data/generated_scenarios.json
"""

import json
import random
import re
import argparse
from pathlib import Path


# Maps failure families to root causes and fixes
ROOT_CAUSE_TO_FIX = {
    "arch_guard": "relax_arch_check",
    "backend_whitelist": "add_whitelist_entry",
    "runtime_loader": "fix_runtime_path",
    "backend_selector": "switch_backend",
    "model_config": "update_model_config",
    "weight_layout": "fix_weight_mapping",
    "memory_oom": "tune_memory_config",
    "quantization_error": "fix_quantization",
    "distributed_comm": "fix_comm_config",
    "driver_compat": "update_driver_config",
}

SPECIALISTS = ["runtime", "dispatch", "kernel", "loader"]

# Hardware strings we can assign based on context clues
HARDWARE_MAP = {
    "sm_121": "NVIDIA SM121 (DGX Spark)",
    "sm_120": "NVIDIA SM120 (GeForce RTX 5090)",
    "sm_90": "NVIDIA H100",
    "sm_89": "NVIDIA RTX 4090",
    "mi300": "AMD MI300X",
    "mi355": "AMD MI355X",
    "b200": "NVIDIA B200",
    "h100": "NVIDIA H100",
    "a100": "NVIDIA A100",
    "4090": "NVIDIA RTX 4090",
    "5090": "NVIDIA SM120 (GeForce RTX 5090)",
    "blackwell": "NVIDIA B200",
    "hopper": "NVIDIA H100",
}

BACKEND_MAP = {
    "vllm": "vLLM 0.8.x",
    "sglang": "SGLang 0.5.x",
    "flashinfer": "FlashInfer 0.4",
    "tensorrt": "TensorRT-LLM 0.18",
    "cutlass": "CUTLASS 3.x",
    "flash_attn": "Flash-Attention 2.x",
    "flash-attention": "Flash-Attention 2.x",
    "triton": "Triton Inference Server",
}

# Specialist opinion templates per failure family
# Each family has correct and incorrect opinion templates
OPINION_TEMPLATES = {
    "arch_guard": {
        "correct": [
            "The architecture check is blocking kernel dispatch. The GPU architecture is not in the supported set despite being compatible at the instruction level.",
            "This is a capability check issue. The GPU supports the needed instructions but fails the hardcoded arch whitelist.",
            "The dispatch table maps arch to kernel. This architecture has no entry. Adding it to the arch check should fix it.",
        ],
        "incorrect": [
            "CUDA runtime loaded successfully. No runtime issues detected.",
            "Model weights loaded correctly. Weight layout is standard.",
            "Possible kernel incompatibility — this GPU may lack required instructions.",
            "The backend configuration looks correct. This might be a driver issue.",
        ],
    },
    "backend_whitelist": {
        "correct": [
            "The quantization backend has a hardcoded GPU whitelist that doesn't include this hardware. The GPU is capable but not listed.",
            "GPU name not found in the supported GPU list. This is a whitelist gap, not a hardware limitation.",
            "The kernel availability check is failing because the GPU isn't in the allow list, despite supporting the needed operations.",
        ],
        "incorrect": [
            "The architecture check passed. This isn't an arch issue.",
            "Runtime libraries are loaded correctly. No path issues.",
            "This could be a weight format incompatibility with the quantization scheme.",
            "The model config looks correct for this backend.",
        ],
    },
    "runtime_loader": {
        "correct": [
            "The runtime library path is incorrect. The shared objects can't be found at the expected location.",
            "CUDA/ROCm runtime loading is failing. The library search path needs to be updated.",
            "The loader is searching for libraries in the wrong directory. Path configuration needs fixing.",
        ],
        "incorrect": [
            "GPU architecture is supported. This isn't a capability issue.",
            "The backend selection logic looks correct.",
            "Model weights and config are properly formatted.",
            "The kernel compilation succeeded. This isn't a kernel issue.",
        ],
    },
    "backend_selector": {
        "correct": [
            "The wrong attention/compute backend is being selected. The dispatch logic is choosing an incompatible or suboptimal path.",
            "Backend selection is incorrect for this hardware/model combination. Need to switch to a compatible backend.",
            "The auto-selection logic is picking the wrong kernel path. Manual backend override should fix this.",
        ],
        "incorrect": [
            "All runtime libraries are loaded correctly.",
            "The GPU architecture is fully supported.",
            "Weight layout matches expected format.",
            "The model config is valid for this backend.",
        ],
    },
    "model_config": {
        "correct": [
            "The model configuration doesn't match the checkpoint. There's a mismatch in model parameters.",
            "Config values don't align with the actual checkpoint structure. The config needs updating.",
            "Shape mismatch during model loading — the config specifies different dimensions than the checkpoint contains.",
        ],
        "incorrect": [
            "Backend selection is appropriate for this model.",
            "Runtime and driver are functioning correctly.",
            "The GPU has sufficient capability for this model.",
            "Weight conversion completed without errors.",
        ],
    },
    "weight_layout": {
        "correct": [
            "Weight tensors are mapped incorrectly. The layout doesn't match what the backend expects.",
            "There's a weight mapping issue — projections are assigned to the wrong layers.",
            "The weight conversion process produced incorrect tensor assignments. Layout needs fixing.",
        ],
        "incorrect": [
            "Model config is valid and matches the checkpoint.",
            "The attention backend is correctly selected.",
            "GPU architecture check passed without issues.",
            "Runtime libraries are all accessible.",
        ],
    },
    "memory_oom": {
        "correct": [
            "GPU memory is exhausted. The KV cache or batch size exceeds available VRAM.",
            "Out of memory during inference. Memory configuration needs tuning — reduce max_model_len or batch size.",
            "The memory allocator is running out of space. Chunked prefill or paged attention config should help.",
        ],
        "incorrect": [
            "The model loaded successfully. This isn't a memory issue.",
            "GPU architecture is fine. The hardware supports this model.",
            "Backend selection is correct for this workload.",
            "Weight layout matches expected format. No conversion issues.",
        ],
    },
    "quantization_error": {
        "correct": [
            "The quantization format is incompatible with this backend or hardware. Precision handling is broken.",
            "FP8/INT4 quantization is producing incorrect results. The calibration or format conversion has a bug.",
            "Quantized weights are being loaded with the wrong scale factors or format assumptions.",
        ],
        "incorrect": [
            "Memory usage is within bounds. Not an OOM issue.",
            "The model config is valid for quantized inference.",
            "Runtime libraries loaded without issues.",
            "GPU architecture supports the needed quantization instructions.",
        ],
    },
    "distributed_comm": {
        "correct": [
            "NCCL collective operations are failing. Communication between GPUs is broken or timing out.",
            "Tensor parallel communication is hanging. The distributed config needs fixing.",
            "The all-reduce operation is failing across ranks. NCCL topology or network config is wrong.",
        ],
        "incorrect": [
            "Each GPU individually runs fine. The kernel is correct.",
            "Model config is valid. This isn't a config mismatch.",
            "Memory is sufficient on each rank.",
            "The quantization format is compatible across all devices.",
        ],
    },
    "driver_compat": {
        "correct": [
            "CUDA toolkit version doesn't match the installed driver. Version incompatibility is causing failures.",
            "The driver is too old for this CUDA version. Need to update the driver or downgrade CUDA.",
            "cuDNN version mismatch — the installed version isn't compatible with the framework build.",
        ],
        "incorrect": [
            "The GPU architecture is supported. This isn't a capability check issue.",
            "Model weights loaded correctly. No format issues.",
            "Backend selection logic is working as expected.",
            "Memory is sufficient. Not an OOM problem.",
        ],
    },
}


def detect_hardware(text: str) -> str:
    """Detect hardware from text context."""
    text_lower = text.lower()
    for key, hw in HARDWARE_MAP.items():
        if key in text_lower:
            return hw
    return random.choice(["NVIDIA H100", "NVIDIA A100", "NVIDIA RTX 4090"])


def detect_backend(repo: str, text: str) -> str:
    """Detect backend from repo and text context."""
    text_lower = text.lower()
    # Check repo first
    for key, backend in BACKEND_MAP.items():
        if key in repo.lower():
            return backend
    # Then check text
    for key, backend in BACKEND_MAP.items():
        if key in text_lower:
            return backend
    return "vLLM 0.8.x"


def extract_error_log(issue_body: str, pr_body: str) -> str:
    """Extract error/log text from issue and PR bodies."""
    combined = f"{issue_body}\n{pr_body}"

    # Look for code blocks that contain errors/logs
    code_blocks = re.findall(r"```(?:\w*\n)?(.*?)```", combined, re.DOTALL)

    log_lines = []
    for block in code_blocks:
        lines = block.strip().splitlines()
        for line in lines:
            if any(kw in line.lower() for kw in [
                "error", "fail", "traceback", "exception", "warning",
                "cuda", "runtime", "kernel", "arch", "not supported",
                "mismatch", "invalid", "cannot", "unable",
            ]):
                log_lines.append(line.strip())

    if log_lines:
        return "\n".join(log_lines[:10])

    # Fallback: use first code block
    if code_blocks:
        return code_blocks[0].strip()[:500]

    return "Error details not extracted from issue"


def extract_code_snippet(diff: str) -> str:
    """Extract the relevant code snippet from a diff."""
    # Get the removed lines (the buggy code)
    lines = diff.splitlines()
    snippet_lines = []
    current_file = ""

    for line in lines:
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[2].lstrip("a/")
        elif line.startswith("-") and not line.startswith("---"):
            snippet_lines.append(f"# {current_file}")
            snippet_lines.append(line[1:])  # remove the leading -
        elif line.startswith("+") and not line.startswith("+++"):
            pass  # skip added lines for the snippet (that's the fix)

    if snippet_lines:
        # Deduplicate file headers
        seen = set()
        result = []
        for line in snippet_lines:
            if line.startswith("# ") and line in seen:
                continue
            seen.add(line)
            result.append(line)
        return "\n".join(result[:15])

    return "# Code snippet not extracted from diff"


def extract_fix_diff(diff: str) -> str:
    """Extract the fix (added lines) from a diff."""
    lines = diff.splitlines()
    fix_lines = []
    current_file = ""

    for line in lines:
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[2].lstrip("a/")
        elif line.startswith("+") and not line.startswith("+++"):
            if not fix_lines or fix_lines[-1] != f"# {current_file}":
                fix_lines.append(f"# {current_file}")
            fix_lines.append(line[1:])

    return "\n".join(fix_lines[:20]) if fix_lines else ""


def generate_specialist_opinions(family: str) -> dict:
    """Generate 4 specialist opinions with at least one wrong."""
    templates = OPINION_TEMPLATES.get(family, OPINION_TEMPLATES["arch_guard"])
    correct_opinions = templates["correct"]
    incorrect_opinions = templates["incorrect"]

    # Decide how many are correct (2-3 out of 4)
    n_correct = random.choice([2, 2, 3])
    n_incorrect = 4 - n_correct

    # Shuffle specialists
    specs = SPECIALISTS.copy()
    random.shuffle(specs)

    opinions = {}
    for i, spec in enumerate(specs):
        if i < n_correct:
            opinion_text = random.choice(correct_opinions)
            confidence = round(random.uniform(0.75, 0.95), 2)
            is_correct = True
        else:
            opinion_text = random.choice(incorrect_opinions)
            confidence = round(random.uniform(0.55, 0.90), 2)
            is_correct = False

        opinions[spec] = {
            "opinion": opinion_text,
            "confidence": confidence,
            "is_correct": is_correct,
        }

    return opinions


def build_incident_ticket(pr_title: str, issue_title: str, hardware: str, backend: str, model: str = "") -> str:
    """Build an incident ticket string from PR/issue info.

    If a linked issue title exists, use it directly (it already reads like a
    bug report).  Otherwise, clean up the PR title to sound like an incident
    report rather than a commit message.
    """
    if issue_title:
        source = issue_title.strip()
    else:
        source = pr_title.strip()
        # Strip conventional-commit / tag prefixes like [Bugfix], [Bug], [Fix], fix:, etc.
        source = re.sub(r"^\[(?:Bugfix|Bug|Fix|Kernel|Model|Misc|Core|CI|RFC|Misc)\]\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"^(?:fix|bugfix|hotfix|patch)[:\s]+", "", source, flags=re.IGNORECASE)
        # Capitalise first letter after stripping
        if source:
            source = source[0].upper() + source[1:]

    model_str = f" Model: {model}." if model else ""
    return f"INCIDENT: {source} | Hardware: {hardware} | Backend: {backend}.{model_str}"


def convert_triple(triple: dict, idx: int) -> dict | None:
    """Convert a raw triple to a Stack Doctor scenario."""
    family = triple.get("failure_family")
    if not family or family not in ROOT_CAUSE_TO_FIX:
        return None

    pr_body = triple.get("pr_body", "") or ""
    pr_title = triple.get("pr_title", "") or ""
    diff = triple.get("diff", "") or ""
    repo = triple.get("repo", "")

    # Get issue info if available
    issue_title = ""
    issue_body = ""
    if triple.get("linked_issues"):
        issue = triple["linked_issues"][0]
        issue_title = issue.get("title", "")
        issue_body = issue.get("body", "") or ""

    hardware = detect_hardware(f"{pr_title} {pr_body} {issue_body}")
    backend = detect_backend(repo, f"{pr_title} {pr_body} {issue_body}")

    # Use detected_model from filter_triples.py if available
    detected_model = triple.get("detected_model")
    if detected_model:
        model_name = detected_model.capitalize()
    else:
        model_name = "Unknown Model"

    scenario = {
        "id": f"scraped_{family}_{idx:03d}",
        "root_cause": family,
        "correct_fix": ROOT_CAUSE_TO_FIX[family],
        "incident_ticket": build_incident_ticket(pr_title, issue_title, hardware, backend, model=model_name if model_name != "Unknown Model" else ""),
        "hardware": hardware,
        "model_name": model_name,
        "backend": backend,
        "initial_log": extract_error_log(issue_body, pr_body),
        "initial_snippet": extract_code_snippet(diff),
        "specialist_opinions": generate_specialist_opinions(family),
        "correct_patch": extract_fix_diff(diff),
        # Metadata
        "source_repo": repo,
        "source_pr": triple.get("pr_number"),
        "source_pr_url": triple.get("pr_url", ""),
        "source_issues": [i.get("number") for i in triple.get("linked_issues", [])],
    }

    return scenario


def main():
    parser = argparse.ArgumentParser(description="Convert raw triples to Stack Doctor scenarios")
    parser.add_argument("--input", default="data/filtered_triples.json", help="Input filtered triples file")
    parser.add_argument("--output", default="data/generated_scenarios.json", help="Output scenarios file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run scrape_issues.py first.")
        sys.exit(1)

    with open(input_path) as f:
        triples = json.load(f)

    print(f"Loaded {len(triples)} raw triples")

    scenarios = []
    for i, triple in enumerate(triples):
        scenario = convert_triple(triple, i)
        if scenario:
            scenarios.append(scenario)

    print(f"Converted {len(scenarios)} scenarios")

    # Summary by family
    family_counts = {}
    for s in scenarios:
        f = s["root_cause"]
        family_counts[f] = family_counts.get(f, 0) + 1
    for f, c in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {f}: {c}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    import sys
    main()
