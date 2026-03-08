"""
Oracle, heuristic, and random baselines for Stack Doctor.

Used to validate the reward function: random < heuristic < oracle must hold.
"""

from __future__ import annotations

import json
import random

from .scenarios import (
    ROOT_CAUSE_TO_FIX,
    ROOT_CAUSES,
    FIXES,
    SPECIALISTS,
    Scenario,
    SCENARIOS,
    TRAIN_SCENARIOS,
    EVAL_SCENARIOS,
)


def oracle_policy(scenario: Scenario) -> list[dict]:
    """Perfect policy: submit correct answer in 1 step."""
    return [
        {
            "type": "submit",
            "root_cause": scenario.root_cause,
            "fix": scenario.correct_fix,
            "justification": f"Root cause is {scenario.root_cause}, applying the correct fix.",
        }
    ]


def heuristic_policy(scenario: Scenario) -> list[dict]:
    """
    Reasonable heuristic: inspect logs, ask the highest-confidence specialist,
    then submit based on clues.

    Uses keyword matching on specialist opinions and logs to guess root cause.
    """
    actions = []

    # Step 1: inspect logs
    actions.append({"type": "inspect", "target": "logs"})

    # Step 2: ask the highest-confidence specialist
    best_spec = max(
        scenario.specialist_opinions.items(),
        key=lambda kv: kv[1].confidence,
    )
    actions.append({"type": "ask_specialist", "specialist": best_spec[0]})

    # Step 3: heuristic root-cause guess from keywords
    combined_text = (
        scenario.incident_ticket
        + " " + scenario.initial_log
        + " " + best_spec[1].opinion
    ).lower()

    guess = _keyword_guess(combined_text)

    # Step 4: apply fix
    actions.append({"type": "apply_fix", "fix": ROOT_CAUSE_TO_FIX[guess]})

    # Step 5: submit
    actions.append({
        "type": "submit",
        "root_cause": guess,
        "fix": ROOT_CAUSE_TO_FIX[guess],
        "justification": f"Keyword analysis of logs and specialist opinions points to {guess}.",
    })

    return actions


def random_policy(scenario: Scenario) -> list[dict]:
    """Random policy: random actions, random submit."""
    actions = []
    n_steps = random.randint(1, 5)

    for _ in range(n_steps - 1):
        choice = random.choice(["inspect", "ask_specialist"])
        if choice == "inspect":
            actions.append({
                "type": "inspect",
                "target": random.choice(["logs", "config", "snippet", "metrics"]),
            })
        else:
            actions.append({
                "type": "ask_specialist",
                "specialist": random.choice(SPECIALISTS),
            })

    # Final: random submit
    rc = random.choice(ROOT_CAUSES)
    actions.append({
        "type": "submit",
        "root_cause": rc,
        "fix": ROOT_CAUSE_TO_FIX[rc],
    })

    return actions


def _keyword_guess(text: str) -> str:
    """Guess root cause from keyword presence in text."""
    scores = {rc: 0 for rc in ROOT_CAUSES}

    # arch_guard keywords
    for kw in ["arch", "architecture", "sm_12", "sm_120", "sm_121", "supported_arch", "capability", "is_supported"]:
        if kw in text:
            scores["arch_guard"] += 1

    # backend_whitelist keywords
    for kw in ["whitelist", "supported_gpu", "not in", "marlin", "awq", "gpu name"]:
        if kw in text:
            scores["backend_whitelist"] += 1

    # runtime_loader keywords
    for kw in ["runtime", "libcuda", "ld_library", "cuda_home", "symlink", "shared object", "rocm_path", "hipError"]:
        if kw in text:
            scores["runtime_loader"] += 1

    # backend_selector keywords
    for kw in ["backend", "selector", "xformers", "flash_attn", "latency", "slow", "e4m3fn", "fp8 format"]:
        if kw in text:
            scores["backend_selector"] += 1

    # model_config keywords
    for kw in ["config", "num_expert", "shape mismatch", "rope", "checkpoint", "config.json"]:
        if kw in text:
            scores["model_config"] += 1

    # weight_layout keywords
    for kw in ["weight", "mapping", "swap", "gate_proj", "up_proj", "convert", "layout", "qkv"]:
        if kw in text:
            scores["weight_layout"] += 1

    # memory_oom keywords
    for kw in ["out of memory", "oom", "kv_cache", "memory", "max_model_len", "batch size", "vram"]:
        if kw in text:
            scores["memory_oom"] += 1

    # quantization_error keywords
    for kw in ["quantiz", "fp8", "int4", "nf4", "calibrat", "precision", "scale factor", "gptq"]:
        if kw in text:
            scores["quantization_error"] += 1

    # distributed_comm keywords
    for kw in ["nccl", "tensor parallel", "all_reduce", "rdma", "pipeline parallel", "collective", "rank"]:
        if kw in text:
            scores["distributed_comm"] += 1

    # driver_compat keywords
    for kw in ["driver", "cudnn", "toolkit", "nvcc", "cuda version", "driver version", "libcudnn"]:
        if kw in text:
            scores["driver_compat"] += 1

    return max(scores, key=scores.get)


def evaluate_policy(policy_fn, scenarios: list[Scenario], n_runs: int = 1) -> dict:
    """
    Run a policy across scenarios and compute metrics.

    Returns dict with:
      - rc_accuracy: fraction of correct root cause submissions
      - fix_accuracy: fraction of correct fix submissions
      - avg_steps: average steps to resolution
      - avg_reward: average cumulative reward
    """
    from .stack_doctor_environment import StackDoctorEnvironment
    from models import StackDoctorAction

    total_rc_correct = 0
    total_fix_correct = 0
    total_steps = 0
    total_reward = 0.0
    total_episodes = 0

    for _ in range(n_runs):
        for scenario in scenarios:
            env = StackDoctorEnvironment()
            env.reset(scenario_id=scenario.id)

            actions = policy_fn(scenario)
            cumulative = 0.0
            steps = 0

            for action_dict in actions:
                obs = env.step(StackDoctorAction(message=json.dumps(action_dict)))
                cumulative += obs.reward
                steps += 1
                if obs.done:
                    break

            # Check if submit happened
            last_action = actions[-1] if actions else {}
            if last_action.get("type") == "submit":
                if last_action["root_cause"] == scenario.root_cause:
                    total_rc_correct += 1
                if last_action["fix"] == scenario.correct_fix:
                    total_fix_correct += 1

            total_steps += steps
            total_reward += cumulative
            total_episodes += 1

    return {
        "rc_accuracy": total_rc_correct / total_episodes if total_episodes else 0,
        "fix_accuracy": total_fix_correct / total_episodes if total_episodes else 0,
        "avg_steps": total_steps / total_episodes if total_episodes else 0,
        "avg_reward": total_reward / total_episodes if total_episodes else 0,
        "n_episodes": total_episodes,
    }
