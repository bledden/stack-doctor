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

# ---------------------------------------------------------------------------
# 1. Environment setup — add server to path for direct import
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from server.stack_doctor_environment import StackDoctorEnvironment
from server.scenarios import SCENARIOS, TRAIN_SCENARIOS, EVAL_SCENARIOS
from models import StackDoctorAction, StackDoctorObservation

# ---------------------------------------------------------------------------
# 2. Build the system prompt and dataset
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Stack Doctor, an expert AI agent that diagnoses inference-stack incidents.

You receive an incident ticket with hardware/model/backend context, log excerpts, code snippets, and specialist opinions. Some specialists may be wrong — you must reconcile conflicting reports.

You must output a JSON array of actions to investigate and then submit your diagnosis. Available actions:
  {"type":"inspect","target":"logs|config|snippet|metrics"}
  {"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}
  {"type":"apply_fix","fix":"relax_arch_check|add_whitelist_entry|fix_runtime_path|switch_backend|update_model_config|fix_weight_mapping"}
  {"type":"submit","root_cause":"arch_guard|backend_whitelist|runtime_loader|backend_selector|model_config|weight_layout","fix":"relax_arch_check|add_whitelist_entry|fix_runtime_path|switch_backend|update_model_config|fix_weight_mapping","justification":"short explanation of your reasoning"}

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

def extract_actions(text):
    """Extract JSON action array from model output."""
    text = text.strip()
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


def justification_reward(completions, **kwargs):
    """Reward for including a justification in the submit action."""
    scores = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        actions = extract_actions(response)
        if actions is None:
            scores.append(0.0)
            continue
        submit_actions = [a for a in actions if a.get("type") == "submit"]
        if not submit_actions:
            scores.append(0.0)
            continue
        justification = submit_actions[-1].get("justification", "")
        if len(justification.strip()) >= 10:
            scores.append(1.0)
        else:
            scores.append(-0.5)
    return scores


# ---------------------------------------------------------------------------
# 4. Training (Unsloth + TRL GRPO)
# ---------------------------------------------------------------------------

def main():
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    import torch

    max_seq_length = 4096
    lora_rank = 8

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B",
        load_in_4bit=True,
        max_seq_length=max_seq_length,
    )

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

    # Build dataset
    train_data = build_dataset(TRAIN_SCENARIOS, n_repeats=80)
    dataset = Dataset.from_list(train_data)

    # Compute prompt length from longest scenario
    prompt_lengths = []
    for sc in TRAIN_SCENARIOS:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_scenario_prompt(sc)},
        ]
        p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompt_lengths.append(len(tokenizer.encode(p)))
    max_prompt_length = max(prompt_lengths) + 10
    max_completion_length = max_seq_length - max_prompt_length

    print(f"Prompt length: ~{max_prompt_length} tokens")
    print(f"Completion budget: ~{max_completion_length} tokens")
    print(f"Dataset size: {len(dataset)} episodes")
    print(f"Train scenarios: {len(TRAIN_SCENARIOS)}, Eval scenarios: {len(EVAL_SCENARIOS)}")

    # GRPO config
    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=2e-4,
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
        max_steps=300,
        save_steps=50,
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
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Resume from checkpoint if available
    import glob
    checkpoints = sorted(glob.glob("outputs/checkpoint-*"), key=os.path.getmtime)
    if checkpoints:
        resume_from = checkpoints[-1]
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        print("Starting GRPO training from scratch...")
        trainer.train()

    # Save
    model.save_pretrained("stack_doctor_lora")
    tokenizer.save_pretrained("stack_doctor_lora")
    print("Training complete. LoRA saved to stack_doctor_lora/")


if __name__ == "__main__":
    main()
