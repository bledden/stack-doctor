---
title: Stack Doctor Environment Server
emoji: 🩺
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Stack Doctor

An OpenEnv RL environment where an overseer LLM diagnoses sick inference stacks. The agent probes subsystems, reconciles conflicting specialist-agent reports (some of which are wrong), and selects the minimal correct fix — all within a 6-step budget.

Inspired by real SM12x enablement bugs across vLLM, FlashInfer, SGLang, CUTLASS, and Flash-Attention.

**Track**: Statement 3.1 — World Modeling / Professional Tasks
**Sub-theme**: Fleet AI — Scalable Oversight Agents ($10K)

**Live Environment**: https://bledden-stack-doctor.hf.space
**GitHub**: https://github.com/bledden/stack-doctor

## What Makes This Different

1. **Conflicting specialists** — 4 specialist agents provide opinions per incident, but at least one is always wrong. The agent must reconcile conflicting reports rather than blindly trusting any single source.
2. **MCP tool interface** — Agents discover investigation tools (`read_log()`, `check_config()`, `query_specialist()`, etc.) via the Model Context Protocol. Same environment supports both MCP agent interaction and WebSocket RL training.
3. **Justification-rewarded decisions** — The agent must explain its diagnosis, not just submit an answer. A justification bonus rewards explainable decision-making.
4. **35 realistic scenarios** — Based on real GPU enablement bugs (SM121/SM120, MI355X, B200) across production inference frameworks.

## Quick Start

### MCP Agent Interaction

Agents discover and call tools directly:

```
read_log()          → System and application logs
check_config()      → Configuration files
view_code()         → Source code snippets
run_diagnostic()    → Performance metrics
query_specialist()  → Expert analysis (beware: some lie)
apply_fix()         → Apply a remediation (once per episode)
submit_diagnosis()  → Final diagnosis with justification
```

### WebSocket Training API

```python
from client import StackDoctorEnv, StackDoctorAction
import json

env = StackDoctorEnv(base_url="https://bledden-stack-doctor.hf.space")
env.connect()

result = env.reset()
print(result.observation.incident_ticket)
print(result.observation.specialist_opinions)

# Investigate
result = env.step(StackDoctorAction(message=json.dumps(
    {"type": "inspect", "target": "logs"}
)))

# Submit diagnosis with justification
result = env.step(StackDoctorAction(message=json.dumps({
    "type": "submit",
    "root_cause": "arch_guard",
    "fix": "relax_arch_check",
    "justification": "Logs show sm_121 rejected by hardcoded arch check despite SM90 ISA compatibility"
})))
print(f"Reward: {result.reward}, Done: {result.done}")

env.close()
```

## Environment Design

### Root Causes (6) and Fixes (6)

| Root Cause | Fix | Real-World Motif |
|-----------|-----|-----------------|
| `arch_guard` | `relax_arch_check` | FlashInfer SM121 capability checks |
| `backend_whitelist` | `add_whitelist_entry` | vLLM Marlin SM121+ whitelist gaps |
| `runtime_loader` | `fix_runtime_path` | SGLang CUDA 13 runtime issues |
| `backend_selector` | `switch_backend` | CUTLASS dispatch mistakes |
| `model_config` | `update_model_config` | Model config mismatches on new hardware |
| `weight_layout` | `fix_weight_mapping` | Weight layout problems across backends |

### Specialists (4)

`runtime`, `dispatch`, `kernel`, `loader` — at least one gives wrong advice per scenario. The agent cannot learn "always trust specialist X" and must evaluate evidence on each case.

### Reward Function

| Event | Reward |
|-------|--------|
| `inspect` or `ask_specialist` | -0.25 |
| Correct `apply_fix` | +3 |
| Wrong `apply_fix` | -2 |
| Correct `submit` (per field) | +8 |
| Wrong `submit` (per field) | -4 |
| Justification provided | +1 bonus |
| Solved in ≤4 steps | +2 bonus |
| Invalid action | -2 |

### Baselines

| Policy | RC Accuracy | Fix Accuracy | Avg Steps | Avg Reward |
|--------|:-:|:-:|:-:|:-:|
| Oracle | 100% | 100% | 1.0 | 19.0 |
| Heuristic | 88% | 88% | 4.0 | 16.6 |
| Random | 7% | 7% | 3.0 | -6.8 |

### Scenarios

35 scenarios across 6 root-cause families:
- 24 training scenarios (diverse hardware, models, backends)
- 11 held-out evaluation scenarios
- Hardware: H100, B200, MI355X, MI300X, RTX 5090, SM121 (DGX Spark)
- Models: DeepSeek-V3, Llama-4-Maverick, Qwen3-235B, Mistral-Large-2, and more
- Backends: vLLM, SGLang, TensorRT-LLM, FlashInfer, Triton

## Fleet AI: Specialist Oversight

The core mechanic targeting Fleet AI's $10K sub-theme: the agent acts as a **scalable oversight agent** that reconciles conflicting specialist reports. This mirrors real incident response where on-call engineers receive contradictory signals from monitoring systems, runbooks, and team members — and must synthesize them under time pressure to choose the correct remediation.

## Training

GRPO (Group Relative Policy Optimization) via Unsloth + TRL on Qwen3-1.7B with 4 reward signals:

1. **Valid JSON** — can the output be parsed as an action plan?
2. **Environment reward** — cumulative reward from executing the plan against Stack Doctor
3. **Efficiency** — bonus for shorter plans that still submit correctly
4. **Justification** — bonus for including an explanation of the diagnosis

Training on H100 via Northflank. 300 GRPO steps over 24 training scenarios (80 repeats = 1920 episodes).

## Development

```bash
# Local server
cd stack_doctor && PYTHONPATH=. uvicorn server.app:app --port 8000

# Run baselines
PYTHONPATH=. python3 -c "
from server.baselines import oracle_policy, heuristic_policy, random_policy, evaluate_policy
from server.scenarios import TRAIN_SCENARIOS
for name, fn in [('Oracle', oracle_policy), ('Heuristic', heuristic_policy), ('Random', random_policy)]:
    r = evaluate_policy(fn, TRAIN_SCENARIOS)
    print(f'{name}: reward={r[\"avg_reward\"]:.1f} rc_acc={r[\"rc_accuracy\"]:.0%} fix_acc={r[\"fix_accuracy\"]:.0%}')
"

# Deploy to HF Spaces
openenv push
```
