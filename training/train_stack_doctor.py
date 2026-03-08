"""
Stack Doctor — GRPO Training Script

Train an LLM to diagnose inference-stack incidents using Group Relative
Policy Optimization (GRPO) with Unsloth + TRL.

The model generates a JSON action plan, which gets executed against the
Stack Doctor environment. Reward = cumulative episode reward.

Fleet AI sub-theme: the agent must reconcile conflicting specialist reports
(some specialists lie) to identify the correct root cause and fix.

Usage (Colab with GPU):
    !pip install unsloth trl openenv-core
    !python train_stack_doctor.py
"""

import json
import os
import sys
import random
import re
import types

import weave

# ---------------------------------------------------------------------------
# 1. Environment setup — add server to path for direct import
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from server.stack_doctor_environment import StackDoctorEnvironment
from server.scenarios import (
    SCENARIOS, TRAIN_SCENARIOS, EVAL_SCENARIOS,
    HANDCRAFTED_TRAIN, SCRAPED_TRAIN_SCENARIOS,
)
from models import StackDoctorAction, StackDoctorObservation

# ---------------------------------------------------------------------------
# 2. Build the system prompt and dataset
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Stack Doctor, an expert AI agent that diagnoses inference-stack incidents.

You receive an incident ticket with hardware/model/backend context, log excerpts, code snippets, and specialist opinions. Some specialists may be wrong — you must reconcile conflicting reports.

You must output a JSON array of actions to investigate and then submit your diagnosis. Available actions:
  {"type":"inspect","target":"logs|config|snippet|metrics"}
  {"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}
  {"type":"apply_fix","fix":"relax_arch_check|add_whitelist_entry|fix_runtime_path|switch_backend|update_model_config|fix_weight_mapping|tune_memory_config|fix_quantization|fix_comm_config|update_driver_config"}
  {"type":"submit","root_cause":"arch_guard|backend_whitelist|runtime_loader|backend_selector|model_config|weight_layout|memory_oom|quantization_error|distributed_comm|driver_compat","fix":"relax_arch_check|add_whitelist_entry|fix_runtime_path|switch_backend|update_model_config|fix_weight_mapping|tune_memory_config|fix_quantization|fix_comm_config|update_driver_config","justification":"short explanation of your reasoning"}

Rules:
- You have 6 steps max. Each inspect/ask costs -0.25. Wrong fix costs -2. Wrong submit costs -4 per field.
- Correct submit: +8 per correct field. Efficiency bonus +2 if solved in ≤4 steps.
- Justification bonus: +1 if you include a justification (≥10 chars) explaining your reasoning.
- apply_fix can only be used once per episode.
- submit MUST be your final action.
- Minimize investigation steps — be decisive.
- Always include a justification explaining what evidence led to your diagnosis.

Output ONLY a JSON array, e.g.:
[{"type":"inspect","target":"logs"},{"type":"submit","root_cause":"arch_guard","fix":"relax_arch_check","justification":"Logs show sm_121 rejected by arch check despite being SM90-compatible"}]"""


def format_scenario_prompt(scenario):
    """Convert a scenario's initial observation into a user prompt."""
    specialist_text = ""
    for name, op in scenario.specialist_opinions.items():
        specialist_text += f"\n  {name} (confidence {op.confidence:.2f}): {op.opinion}"

    return f"""INCIDENT TICKET:
{scenario.incident_ticket}

HARDWARE: {scenario.hardware}
MODEL: {scenario.model_name}
BACKEND: {scenario.backend}

LOG EXCERPT:
{scenario.initial_log}

CODE SNIPPET:
{scenario.initial_snippet}

SPECIALIST OPINIONS:{specialist_text}

Provide your action plan as a JSON array. End with a submit action."""


def build_dataset(scenarios, n_repeats=50):
    """Build training dataset: each scenario repeated n times."""
    data = []
    for _ in range(n_repeats):
        for sc in scenarios:
            data.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_scenario_prompt(sc)},
                ],
                "scenario_id": sc.id,
            })
    random.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# 3. Reward functions
# ---------------------------------------------------------------------------

@weave.op()
def extract_actions(text):
    """Extract JSON action array from model output."""
    text = text.strip()
    # Strip Qwen3.5 thinking blocks before parsing
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try to find JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            actions = json.loads(text[start:end + 1])
            if isinstance(actions, list):
                return actions
        except json.JSONDecodeError:
            pass
    # Try parsing the whole thing
    try:
        actions = json.loads(text)
        if isinstance(actions, list):
            return actions
        return [actions]  # single action
    except json.JSONDecodeError:
        return None


@weave.op()
def valid_json_reward(completions, **kwargs):
    """Reward for producing valid JSON action array."""
    scores = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        actions = extract_actions(response)
        if actions is None:
            scores.append(-3.0)
        elif not any(a.get("type") == "submit" for a in actions):
            scores.append(-1.0)  # no submit = useless
        else:
            scores.append(1.0)
    return scores


@weave.op()
def environment_reward(completions, **kwargs):
    """Execute action plan against Stack Doctor and return episode reward."""
    scores = []
    scenario_ids = kwargs.get("scenario_id", [None] * len(completions))

    for i, completion in enumerate(completions):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        actions = extract_actions(response)

        if actions is None:
            scores.append(-5.0)
            continue

        sid = scenario_ids[i] if i < len(scenario_ids) else None
        env = StackDoctorEnvironment()
        env.reset(scenario_id=sid)

        cumulative = 0.0
        for action_dict in actions:
            if not isinstance(action_dict, dict):
                cumulative -= 2.0
                continue
            try:
                obs = env.step(StackDoctorAction(message=json.dumps(action_dict)))
                cumulative += obs.reward
                if obs.done:
                    break
            except Exception:
                cumulative -= 2.0
                break

        scores.append(cumulative)

    return scores


@weave.op()
def efficiency_reward(completions, **kwargs):
    """Bonus for shorter action plans that still submit."""
    scores = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        actions = extract_actions(response)
        if actions is None:
            scores.append(0.0)
        elif len(actions) <= 2 and any(a.get("type") == "submit" for a in actions):
            scores.append(2.0)  # very efficient
        elif len(actions) <= 4 and any(a.get("type") == "submit" for a in actions):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


@weave.op()
def justification_reward(completions, **kwargs):
    """Reward for including a justification grounded in actual evidence."""
    EVIDENCE_KEYWORDS = {
        "log", "config", "metric", "snippet", "specialist",
        "runtime", "dispatch", "kernel", "loader",
    }
    scores = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        actions = extract_actions(response)
        if actions is None:
            scores.append(-0.5)
            continue
        submit_actions = [a for a in actions if a.get("type") == "submit"]
        if not submit_actions:
            scores.append(-0.5)
            continue
        justification = submit_actions[-1].get("justification", "").strip()
        if len(justification) < 10:
            scores.append(-0.5)
        elif any(kw in justification.lower() for kw in EVIDENCE_KEYWORDS):
            scores.append(1.0)
        else:
            scores.append(0.5)
    return scores


# Family clusters: root causes that are conceptually close
_FAMILY_CLUSTERS = [
    {"arch_guard", "backend_whitelist"},          # GPU rejected by check
    {"runtime_loader", "driver_compat"},          # system library issues
    {"model_config", "weight_layout"},            # model loading issues
    {"backend_selector", "quantization_error"},   # backend/precision issues
    {"memory_oom", "distributed_comm"},           # resource issues
]

# Build a lookup: root_cause -> cluster index for O(1) access
_CAUSE_TO_CLUSTER = {}
for _idx, _cluster in enumerate(_FAMILY_CLUSTERS):
    for _cause in _cluster:
        _CAUSE_TO_CLUSTER[_cause] = _idx


@weave.op()
def partial_credit_reward(completions, **kwargs):
    """Partial credit when root_cause is wrong but in the same family cluster."""
    scores = []
    scenario_ids = kwargs.get("scenario_id", [None] * len(completions))

    for i, completion in enumerate(completions):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        actions = extract_actions(response)
        if actions is None:
            scores.append(0.0)
            continue

        submit_actions = [a for a in actions if a.get("type") == "submit"]
        if not submit_actions:
            scores.append(0.0)
            continue

        predicted = submit_actions[-1].get("root_cause", "")

        # Look up the ground-truth root cause from the scenario
        sid = scenario_ids[i] if i < len(scenario_ids) else None
        true_cause = None
        if sid is not None:
            for sc in SCENARIOS:
                if sc.id == sid:
                    true_cause = sc.root_cause
                    break

        if true_cause is None:
            scores.append(0.0)
            continue

        if predicted == true_cause:
            scores.append(1.0)
        elif (_CAUSE_TO_CLUSTER.get(predicted) is not None
              and _CAUSE_TO_CLUSTER.get(predicted) == _CAUSE_TO_CLUSTER.get(true_cause)):
            scores.append(0.3)
        else:
            scores.append(0.0)
    return scores


@weave.op()
def investigation_quality_reward(completions, **kwargs):
    """Reward models that investigate before diagnosing, penalize blind guessing."""
    scores = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        actions = extract_actions(response)
        if actions is None:
            scores.append(0.0)
            continue

        # Count investigation steps (inspect or ask_specialist) before the first submit
        investigation_steps = 0
        for a in actions:
            if not isinstance(a, dict):
                continue
            if a.get("type") == "submit":
                break
            if a.get("type") in ("inspect", "ask_specialist"):
                investigation_steps += 1

        has_submit = any(
            isinstance(a, dict) and a.get("type") == "submit" for a in actions
        )

        if not has_submit:
            scores.append(0.0)
        elif investigation_steps == 0:
            scores.append(-1.0)   # blind guess — penalize
        elif investigation_steps <= 2:
            scores.append(0.5)    # reasonable investigation
        else:
            scores.append(0.0)    # over-investigation — neutral
    return scores


# ---------------------------------------------------------------------------
# 4. Qwen3.5 VL text-only patch
# ---------------------------------------------------------------------------

def patch_qwen35_text_only(model):
    """Monkey-patch Qwen3.5 VL to avoid 3D position encoding crash on text-only batches.

    The VL model's compute_3d_position_ids stores rope_deltas from prior forward
    passes. When GRPO batches multiple generations (num_generations > 1), the
    stale rope_deltas have a different batch dimension, causing a shape mismatch:
        RuntimeError: tensor a (N) must match tensor b (0) at non-singleton dimension 1

    For text-only training we compute simple 1D positions expanded to 3D (all three
    RoPE dimensions identical) with zero rope_deltas. This is correct since text has
    no spatial/temporal vision tokens.
    """
    import torch

    # Find the object that has compute_3d_position_ids — check the model itself
    # first, then navigate through PEFT/LoRA/Unsloth wrappers.
    candidates = [model]
    for attr_chain in [
        ("base_model",),
        ("base_model", "model"),
        ("model",),
        ("model", "model"),
    ]:
        obj = model
        for attr in attr_chain:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                break
        candidates.append(obj)

    target = None
    for obj in candidates:
        if hasattr(obj, "compute_3d_position_ids"):
            target = obj
            break

    if target is not None:
        def _text_only_position_ids(self, input_ids=None, inputs_embeds=None,
                                     attention_mask=None, past_key_values=None, **kwargs):
            """Compute 3D position IDs for text-only inputs (no vision tokens).

            Returns position_ids of shape (3, batch_size, seq_len) where all 3
            RoPE dimensions are identical 1D positions, and sets rope_deltas to
            zeros so subsequent calls with different batch sizes don't crash.
            """
            past_len = 0 if past_key_values is None else past_key_values.get_seq_length()

            if inputs_embeds is not None:
                batch_size, seq_length = inputs_embeds.shape[:2]
                device = inputs_embeds.device
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
            else:
                self.rope_deltas = None
                return None

            if attention_mask is not None:
                # Cumulative sum gives proper positions respecting padding
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                # Expand to 3 dimensions: (3, batch_size, seq_len)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = torch.arange(past_len, past_len + seq_length, device=device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)

            # Zero deltas — text has no vision spatial offsets
            self.rope_deltas = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
            return position_ids

        target.compute_3d_position_ids = types.MethodType(_text_only_position_ids, target)
        print(f"Patched {type(target).__name__}.compute_3d_position_ids for text-only GRPO")
    else:
        print("WARNING: Could not find compute_3d_position_ids to patch")


# ---------------------------------------------------------------------------
# 5. Training (Unsloth + TRL GRPO)
# ---------------------------------------------------------------------------

def main():
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    import torch

    # Initialize Weave observability
    weave.init("grpo-training")

    max_seq_length = 8192
    lora_rank = 16

    # Load model — Qwen2.5-1.5B (small model to demonstrate learning curve)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        load_in_4bit=True,
        max_seq_length=max_seq_length,
    )

    # For VL models the tokenizer wraps a text tokenizer — unwrap for encode() calls
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Build dataset — hand-crafted scenarios get more repeats (high quality),
    # scraped scenarios get fewer repeats (lower quality, more numerous).
    train_data_handcrafted = build_dataset(HANDCRAFTED_TRAIN, n_repeats=30)
    train_data_scraped = build_dataset(SCRAPED_TRAIN_SCENARIOS, n_repeats=5)
    train_data = train_data_handcrafted + train_data_scraped
    random.shuffle(train_data)
    dataset = Dataset.from_list(train_data)

    # Compute prompt length from longest scenario
    prompt_lengths = []
    for sc in TRAIN_SCENARIOS:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_scenario_prompt(sc)},
        ]
        p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompt_lengths.append(len(text_tokenizer.encode(p)))
    max_prompt_length = max(prompt_lengths) + 10
    # Qwen2.5-1.5B doesn't use thinking mode, so 1024 tokens is plenty for JSON output.
    max_completion_length = max(256, min(1024, max_seq_length - max_prompt_length))

    print(f"Prompt length: ~{max_prompt_length} tokens")
    print(f"Completion budget: ~{max_completion_length} tokens")
    print(f"Dataset size: {len(dataset)} episodes")
    print(f"Train scenarios: {len(TRAIN_SCENARIOS)}, Eval scenarios: {len(EVAL_SCENARIOS)}")

    # GRPO config — short run with small model to demonstrate learning curve.
    # 50 steps, save every 10 so we get 5 points on the reward chart.
    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=50,
        save_steps=10,
        report_to="none",
        output_dir="outputs",
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            valid_json_reward,
            environment_reward,
            efficiency_reward,
            justification_reward,
            partial_credit_reward,
            investigation_quality_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Fresh start — small model, short run to demonstrate learning curve
    print("Starting GRPO training (Qwen2.5-1.5B, 50 steps)...")
    trainer.train()

    # Save
    model.save_pretrained("stack_doctor_lora_1.5b")
    tokenizer.save_pretrained("stack_doctor_lora_1.5b")
    print("Training complete. LoRA saved to stack_doctor_lora_1.5b/")


if __name__ == "__main__":
    main()
