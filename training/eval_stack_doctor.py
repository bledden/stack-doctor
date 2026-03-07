"""
Stack Doctor — Evaluation Script

Produces the 4 metrics for judges:
1. Root-cause accuracy
2. Fix-family accuracy
3. Average steps to resolution
4. Mean reward before vs after RL

Can evaluate any model (base or fine-tuned) against held-out eval scenarios.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from server.stack_doctor_environment import StackDoctorEnvironment
from server.scenarios import EVAL_SCENARIOS
from models import StackDoctorAction
from training.train_stack_doctor import (
    SYSTEM_PROMPT,
    format_scenario_prompt,
    extract_actions,
)


def evaluate_model(model, tokenizer, scenarios, label="Model"):
    """Run model against scenarios and compute metrics."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    total_rc_correct = 0
    total_fix_correct = 0
    total_justified = 0
    total_steps = 0
    total_reward = 0.0
    n = 0

    for sc in scenarios:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_scenario_prompt(sc)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        actions = extract_actions(response)
        if actions is None:
            total_reward -= 5.0
            n += 1
            continue

        env = StackDoctorEnvironment()
        env.reset(scenario_id=sc.id)

        cum_reward = 0.0
        steps = 0
        last_submit = None

        for action_dict in actions:
            if not isinstance(action_dict, dict):
                continue
            try:
                obs = env.step(StackDoctorAction(message=json.dumps(action_dict)))
                cum_reward += obs.reward
                steps += 1
                if action_dict.get("type") == "submit":
                    last_submit = action_dict
                if obs.done:
                    break
            except Exception:
                break

        if last_submit:
            if last_submit.get("root_cause") == sc.root_cause:
                total_rc_correct += 1
            if last_submit.get("fix") == sc.correct_fix:
                total_fix_correct += 1
            if len(last_submit.get("justification", "").strip()) >= 10:
                total_justified += 1

        total_steps += steps
        total_reward += cum_reward
        n += 1

        has_j = "J" if last_submit and len(last_submit.get("justification", "").strip()) >= 10 else "-"
        print(f"  {sc.id}: rc={'OK' if last_submit and last_submit.get('root_cause')==sc.root_cause else 'FAIL'} "
              f"fix={'OK' if last_submit and last_submit.get('fix')==sc.correct_fix else 'FAIL'} "
              f"just={has_j} steps={steps} reward={cum_reward:.1f}")

    print(f"\n{'='*50}")
    print(f"{label} Results ({n} episodes):")
    print(f"  Root-cause accuracy:  {total_rc_correct/n:.1%}")
    print(f"  Fix accuracy:         {total_fix_correct/n:.1%}")
    print(f"  Justification rate:   {total_justified/n:.1%}")
    print(f"  Avg steps:            {total_steps/n:.1f}")
    print(f"  Avg reward:           {total_reward/n:.1f}")
    print(f"{'='*50}")

    return {
        "rc_accuracy": total_rc_correct / n,
        "fix_accuracy": total_fix_correct / n,
        "justification_rate": total_justified / n,
        "avg_steps": total_steps / n,
        "avg_reward": total_reward / n,
    }


def main():
    from unsloth import FastLanguageModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-1.7B", help="Model name or path")
    parser.add_argument("--lora", default=None, help="Path to LoRA adapter")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        load_in_4bit=True,
        max_seq_length=2048,
    )

    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)

    print(f"Evaluating {args.model}" + (f" + {args.lora}" if args.lora else ""))
    print(f"Eval scenarios: {len(EVAL_SCENARIOS)}")
    print()

    evaluate_model(model, tokenizer, EVAL_SCENARIOS, label=args.model)


if __name__ == "__main__":
    main()
