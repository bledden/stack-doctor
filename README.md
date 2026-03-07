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

## Quick Start

```python
from stack_doctor import StackDoctorEnv, StackDoctorAction
import json

env = StackDoctorEnv(base_url="https://bledden-stack-doctor.hf.space")
env.connect()

# Start a new incident
result = env.reset()
print(result.observation.incident_ticket)
print(result.observation.specialist_opinions)

# Investigate
result = env.step(StackDoctorAction(message=json.dumps(
    {"type": "inspect", "target": "logs"}
)))
print(result.observation.output)

# Submit diagnosis
result = env.step(StackDoctorAction(message=json.dumps(
    {"type": "submit", "root_cause": "arch_guard", "fix": "relax_arch_check"}
)))
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

`runtime`, `dispatch`, `kernel`, `loader` — at least one gives wrong advice per scenario.

### Action Space (JSON)

```json
{"type":"inspect","target":"logs|config|snippet|metrics"}
{"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}
{"type":"apply_fix","fix":"<one of 6 fixes>"}
{"type":"submit","root_cause":"<one of 6>","fix":"<one of 6>"}
```

### Reward Function

| Event | Reward |
|-------|--------|
| `inspect` or `ask_specialist` | -0.25 |
| Correct `apply_fix` | +3 |
| Wrong `apply_fix` | -2 |
| Correct `submit` (per field) | +8 |
| Wrong `submit` (per field) | -4 |
| Solved in ≤4 steps | +2 bonus |
| Invalid action | -2 |

### Baselines

| Policy | RC Accuracy | Fix Accuracy | Avg Steps | Avg Reward |
|--------|:-:|:-:|:-:|:-:|
| Oracle | 100% | 100% | 1.0 | 18.0 |
| Heuristic | 100% | 100% | 4.0 | 20.5 |
| Random | 18% | 18% | 3.2 | -4.1 |

## Fleet AI: Specialist Oversight

The core mechanic that targets Fleet AI's $10K sub-theme: the agent must act as a **scalable oversight agent** that reconciles conflicting specialist reports. Specialists have per-scenario reliability — the agent cannot learn "always trust specialist X" and must evaluate evidence on each case.

## Training

Uses Unsloth + TRL GRPO with 3 reward signals:
1. **Valid JSON** — can the output be parsed as an action plan?
2. **Environment reward** — cumulative reward from executing the plan
3. **Efficiency** — bonus for shorter plans that still submit correctly

## Development

```bash
# Local server
cd stack_doctor && PYTHONPATH=. uvicorn server.app:app --port 8000

# Run baselines
PYTHONPATH=. python3 -c "from server.baselines import *; ..."

# Deploy to HF Spaces
openenv push --repo-id bledden/stack-doctor
```
