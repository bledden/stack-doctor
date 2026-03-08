"""Run a real model episode against the live Stack Doctor environment.

Usage:
    .venv-infer/bin/python run_live_model.py [--server ws://localhost:8000/ws] [--lora path/to/lora]

This loads Qwen2.5-1.5B-Instruct via MLX, connects to the environment
over WebSocket, and runs a full diagnostic episode with real model inference.
"""

import asyncio
import json
import re
import sys
import time
import argparse

import websockets
from mlx_lm import load, generate


# ── Config ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Stack Doctor, an expert AI agent that diagnoses inference-stack incidents.
You receive an incident ticket with hardware/model/backend context, log excerpts, and specialist opinions.
Some specialists may be wrong. Output a JSON array of actions:
  {"type":"inspect","target":"logs|config|snippet|metrics"}
  {"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}
  {"type":"apply_fix","fix":"<fix_name>"}
  {"type":"submit","root_cause":"<cause>","fix":"<fix>","justification":"<why>"}"""


def format_prompt(obs):
    """Format the environment observation into a user prompt."""
    ops = obs.get("specialist_opinions", {})
    ops_str = "\n".join(
        f"  {name}: {o['opinion']} (confidence: {o['confidence']})"
        for name, o in ops.items()
    )
    return f"""INCIDENT: {obs.get('incident_ticket', '')}
Hardware: {obs.get('hardware', '')} | Model: {obs.get('model_name', '')} | Backend: {obs.get('backend', '')}
LOG:
{obs.get('log_excerpt', '')}
SPECIALISTS:
{ops_str}

Investigate and submit your diagnosis as a JSON action array."""


def extract_actions(text):
    """Extract JSON action array from model output."""
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            actions = json.loads(text[start : end + 1])
            if isinstance(actions, list):
                return [a for a in actions if isinstance(a, dict)] or None
        except json.JSONDecodeError:
            pass
    # Try whole string as JSON
    try:
        actions = json.loads(text)
        if isinstance(actions, list):
            return [a for a in actions if isinstance(a, dict)] or None
        if isinstance(actions, dict):
            return [actions]
    except json.JSONDecodeError:
        pass
    return None


async def run_episode(ws_url, model, tokenizer, max_actions=6):
    """Run one full episode: model ↔ environment."""
    cumulative_reward = 0.0
    step_num = 0

    async with websockets.connect(ws_url) as ws:
        # ── Reset ──
        await ws.send(json.dumps({"type": "reset", "data": {}}))
        resp = json.loads(await ws.recv())
        obs = resp["data"]["observation"]
        print(f"\n{'='*60}")
        print(f"SCENARIO: {obs['backend']} on {obs['hardware']}")
        print(f"MODEL: {obs['model_name']}")
        print(f"TICKET: {obs['incident_ticket'][:120]}...")
        print(f"{'='*60}\n")

        # Build conversation for multi-turn
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_prompt(obs)},
        ]

        done = False
        while not done and step_num < max_actions:
            # ── Model generates ──
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            t0 = time.time()
            output = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=512, verbose=False
            )
            gen_time = time.time() - t0

            print(f"[MODEL] ({gen_time:.1f}s) {output[:300]}")
            if len(output) > 300:
                print(f"  ... ({len(output)} chars total)")

            # ── Parse actions ──
            actions = extract_actions(output)
            if actions is None:
                print(f"  ⚠ Could not parse actions. Sending raw as single submit.")
                actions = [{"type": "submit", "root_cause": "unknown", "fix": "switch_backend", "justification": output[:200]}]

            # ── Execute each action against environment ──
            for action in actions:
                step_num += 1
                action_str = json.dumps(action)
                print(f"\n  STEP {step_num}: {action.get('type', '?')} ", end="")
                if action.get("type") == "inspect":
                    print(f"→ {action.get('target', '?')}")
                elif action.get("type") == "ask_specialist":
                    print(f"→ {action.get('specialist', '?')}")
                elif action.get("type") == "apply_fix":
                    print(f"→ {action.get('fix', '?')}")
                elif action.get("type") == "submit":
                    print(f"→ rc={action.get('root_cause', '?')} fix={action.get('fix', '?')}")
                else:
                    print()

                await ws.send(json.dumps({"type": "step", "data": {"message": action_str}}))
                step_resp = json.loads(await ws.recv())

                if step_resp.get("type") == "error":
                    print(f"  ✗ Error: {step_resp['data'].get('message', 'unknown')}")
                    continue

                step_data = step_resp["data"]
                reward = step_data.get("reward", 0)
                done = step_data.get("done", False)
                step_obs = step_data.get("observation", {})
                cumulative_reward += reward

                env_output = step_obs.get("output", "")
                print(f"  → reward: {reward:+.2f} (cumulative: {cumulative_reward:+.2f})")
                if env_output:
                    # Print first 200 chars of environment output
                    for line in env_output.strip().split("\n")[:6]:
                        print(f"    {line}")

                if done:
                    break

                # Add environment response to conversation for next turn
                conversation.append({"role": "assistant", "content": output})
                conversation.append({"role": "user", "content": f"Environment response:\n{env_output}\n\nContinue investigating or submit your diagnosis."})

            if done:
                break

    print(f"\n{'='*60}")
    print(f"EPISODE COMPLETE — {step_num} steps, total reward: {cumulative_reward:+.2f}")
    print(f"{'='*60}\n")
    return cumulative_reward


def main():
    parser = argparse.ArgumentParser(description="Run live model inference against Stack Doctor")
    parser.add_argument("--server", default="ws://localhost:8000/ws", help="WebSocket URL")
    parser.add_argument("--lora", default=None, help="Path to LoRA adapter weights")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    print("Loading Qwen2.5-1.5B-Instruct (4-bit MLX)...")
    t0 = time.time()
    if args.lora:
        model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit", adapter_path=args.lora)
        print(f"  Model + LoRA loaded in {time.time()-t0:.1f}s")
    else:
        model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
        print(f"  Base model loaded in {time.time()-t0:.1f}s")

    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f"\n>>> Episode {ep+1}/{args.episodes}")
        asyncio.run(run_episode(args.server, model, tokenizer))


if __name__ == "__main__":
    main()
