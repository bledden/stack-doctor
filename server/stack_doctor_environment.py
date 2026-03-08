"""
Stack Doctor Environment.

An overseer LLM diagnoses sick inference stacks by probing subsystems,
reconciling conflicting specialist-agent reports, and selecting the
minimal correct fix.

Inspired by real SM12x enablement bugs across vLLM, FlashInfer, SGLang,
CUTLASS, and Flash-Attention.
"""

from __future__ import annotations

import json
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import StackDoctorAction, StackDoctorObservation
from .scenarios import (
    ROOT_CAUSE_TO_FIX,
    FIX_TO_ROOT_CAUSE,
    ROOT_CAUSES,
    FIXES,
    SPECIALISTS,
    Scenario,
    SpecialistOpinion,
    get_scenario,
    randomize_specialist_opinions,
)

MAX_STEPS = 6

INSPECT_TARGETS = {"logs", "config", "snippet", "metrics"}
VALID_FIXES = set(FIXES)
VALID_ROOT_CAUSES = set(ROOT_CAUSES)


class EpisodeState:
    """Internal mutable episode state (not exposed to agent)."""

    def __init__(
        self,
        scenario: Scenario,
        specialist_opinions: dict[str, SpecialistOpinion] | None = None,
    ):
        self.scenario = scenario
        # Per-episode randomized specialist opinions (falls back to scenario defaults)
        self.specialist_opinions = specialist_opinions or scenario.specialist_opinions
        self.step_count = 0
        self.fix_applied = False
        self.fix_was_correct: bool | None = None
        self.done = False
        self.cumulative_reward = 0.0
        self.actions_taken: list[dict] = []


class StackDoctorEnvironment(Environment):
    """
    Stack Doctor: incident-response RL environment for
    inference-stack diagnosis.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: EpisodeState | None = None

    def reset(self, seed=None, episode_id=None, **kwargs) -> StackDoctorObservation:
        scenario_id = kwargs.get("scenario_id")
        split = kwargs.get("split", "train")
        scenario = get_scenario(scenario_id, split=split)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        randomized_opinions = randomize_specialist_opinions(scenario)
        self._episode = EpisodeState(scenario, specialist_opinions=randomized_opinions)

        specialist_obs = {}
        for name, op in randomized_opinions.items():
            specialist_obs[name] = {
                "opinion": op.opinion,
                "confidence": op.confidence,
            }

        return StackDoctorObservation(
            output=(
                "STACK DOCTOR — New incident assigned.\n"
                "Diagnose the root cause, optionally apply a fix, then submit your diagnosis.\n"
                "You have 6 steps. Use them wisely.\n\n"
                "Available actions (send as JSON):\n"
                '  {"type":"inspect","target":"logs|config|snippet|metrics"}\n'
                '  {"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}\n'
                '  {"type":"apply_fix","fix":"relax_arch_check|add_whitelist_entry|fix_runtime_path|switch_backend|update_model_config|fix_weight_mapping|tune_memory_config|fix_quantization|fix_comm_config|update_driver_config"}\n'
                '  {"type":"submit","root_cause":"...","fix":"...","justification":"reason for diagnosis"}\n'
            ),
            incident_ticket=scenario.incident_ticket,
            hardware=scenario.hardware,
            model_name=scenario.model_name,
            backend=scenario.backend,
            log_excerpt=scenario.initial_log,
            code_snippet=scenario.initial_snippet,
            specialist_opinions=specialist_obs,
            steps_remaining=MAX_STEPS,
            fix_used=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: StackDoctorAction, **kwargs) -> StackDoctorObservation:
        ep = self._episode
        if ep is None or ep.done:
            return self._terminal_obs("Episode is over. Call reset() to start a new incident.", 0.0)

        self._state.step_count += 1
        ep.step_count += 1

        try:
            parsed = json.loads(action.message)
        except (json.JSONDecodeError, TypeError):
            return self._handle_invalid(ep, f"Invalid JSON: {action.message[:200]}")

        action_type = parsed.get("type")

        if action_type == "inspect":
            return self._handle_inspect(ep, parsed)
        elif action_type == "ask_specialist":
            return self._handle_ask_specialist(ep, parsed)
        elif action_type == "apply_fix":
            return self._handle_apply_fix(ep, parsed)
        elif action_type == "submit":
            return self._handle_submit(ep, parsed)
        else:
            return self._handle_invalid(ep, f"Unknown action type: {action_type}")

    @property
    def state(self) -> State:
        return self._state

    def _handle_inspect(self, ep: EpisodeState, parsed: dict) -> StackDoctorObservation:
        target = parsed.get("target")
        if target not in INSPECT_TARGETS:
            return self._handle_invalid(ep, f"Invalid inspect target: {target}. Use: {INSPECT_TARGETS}")

        reward = -0.25
        ep.cumulative_reward += reward
        ep.actions_taken.append({"type": "inspect", "target": target})

        ir = ep.scenario.inspect_results
        result_map = {"logs": ir.logs, "config": ir.config, "snippet": ir.snippet, "metrics": ir.metrics}

        return self._step_obs(ep, output=f"[INSPECT {target.upper()}]\n{result_map[target]}", reward=reward)

    def _handle_ask_specialist(self, ep: EpisodeState, parsed: dict) -> StackDoctorObservation:
        specialist = parsed.get("specialist")
        if specialist not in SPECIALISTS:
            return self._handle_invalid(ep, f"Invalid specialist: {specialist}. Use: {SPECIALISTS}")

        reward = -0.25
        ep.cumulative_reward += reward
        ep.actions_taken.append({"type": "ask_specialist", "specialist": specialist})

        followup = ep.scenario.specialist_followups.get(specialist, "No additional information.")
        return self._step_obs(ep, output=f"[SPECIALIST: {specialist.upper()}]\n{followup}", reward=reward)

    def _handle_apply_fix(self, ep: EpisodeState, parsed: dict) -> StackDoctorObservation:
        if ep.fix_applied:
            return self._handle_invalid(ep, "apply_fix already used this episode. You can only apply one fix.")

        fix = parsed.get("fix")
        if fix not in VALID_FIXES:
            return self._handle_invalid(ep, f"Invalid fix: {fix}. Use one of: {sorted(VALID_FIXES)}")

        ep.fix_applied = True
        is_correct = fix == ep.scenario.correct_fix
        ep.fix_was_correct = is_correct

        reward = 3.0 if is_correct else -2.0
        ep.cumulative_reward += reward
        ep.actions_taken.append({"type": "apply_fix", "fix": fix, "correct": is_correct})

        if is_correct:
            output = f"[FIX APPLIED: {fix}] Fix applied successfully. Systems recovering. Now submit your diagnosis."
        else:
            output = f"[FIX APPLIED: {fix}] Fix applied but the issue persists. Consider your diagnosis carefully."

        return self._step_obs(ep, output=output, reward=reward)

    def _handle_submit(self, ep: EpisodeState, parsed: dict) -> StackDoctorObservation:
        root_cause = parsed.get("root_cause")
        fix = parsed.get("fix")
        justification = parsed.get("justification", "")

        if root_cause not in VALID_ROOT_CAUSES:
            return self._handle_invalid(ep, f"Invalid root_cause: {root_cause}. Use one of: {sorted(VALID_ROOT_CAUSES)}")
        if fix not in VALID_FIXES:
            return self._handle_invalid(ep, f"Invalid fix: {fix}. Use one of: {sorted(VALID_FIXES)}")

        ep.done = True
        correct_rc = ep.scenario.root_cause
        correct_fix = ep.scenario.correct_fix
        rc_correct = root_cause == correct_rc
        fix_correct = fix == correct_fix
        has_justification = len(justification.strip()) >= 10

        reward = 0.0
        reward += 8.0 if rc_correct else -4.0
        reward += 8.0 if fix_correct else -4.0
        if (rc_correct and fix_correct) and ep.step_count <= 4:
            reward += 2.0
        if has_justification:
            reward += 1.0

        ep.cumulative_reward += reward
        ep.actions_taken.append({
            "type": "submit", "root_cause": root_cause, "fix": fix,
            "justification": justification,
            "rc_correct": rc_correct, "fix_correct": fix_correct,
            "has_justification": has_justification,
        })

        output_lines = ["[DIAGNOSIS SUBMITTED]"]
        output_lines.append(f"  Root cause: {root_cause} — {'CORRECT' if rc_correct else 'WRONG (was: ' + correct_rc + ')'}")
        output_lines.append(f"  Fix: {fix} — {'CORRECT' if fix_correct else 'WRONG (was: ' + correct_fix + ')'}")
        if has_justification:
            output_lines.append(f"  Justification: {justification.strip()}")
            output_lines.append("  JUSTIFICATION BONUS: +1")
        else:
            output_lines.append("  No justification provided (missed +1 bonus)")
        output_lines.append(f"  Steps used: {ep.step_count}/{MAX_STEPS}")
        if rc_correct and fix_correct and ep.step_count <= 4:
            output_lines.append("  EFFICIENCY BONUS: +2 (solved in <= 4 steps)")
        output_lines.append(f"  Episode reward: {ep.cumulative_reward:.2f}")

        return self._terminal_obs("\n".join(output_lines), reward)

    def _handle_invalid(self, ep: EpisodeState, msg: str) -> StackDoctorObservation:
        reward = -2.0
        ep.cumulative_reward += reward
        ep.actions_taken.append({"type": "invalid", "message": msg})

        if ep.step_count >= MAX_STEPS:
            ep.done = True
            return self._terminal_obs(f"[INVALID ACTION] {msg}\n[EPISODE OVER] Max steps reached. Auto-fail.", reward)

        return self._step_obs(ep, output=f"[INVALID ACTION] {msg}", reward=reward)

    def _step_obs(self, ep: EpisodeState, output: str, reward: float) -> StackDoctorObservation:
        remaining = MAX_STEPS - ep.step_count
        if remaining <= 0 and not ep.done:
            ep.done = True
            reward -= 4.0
            ep.cumulative_reward += -4.0
            output += "\n\n[EPISODE OVER] Max steps reached without submission. Auto-fail. Reward: -4"

        return StackDoctorObservation(
            output=output, incident_ticket=ep.scenario.incident_ticket,
            hardware=ep.scenario.hardware, model_name=ep.scenario.model_name,
            backend=ep.scenario.backend, log_excerpt="", code_snippet="",
            specialist_opinions={}, steps_remaining=remaining, fix_used=ep.fix_applied,
            done=ep.done, reward=reward,
            metadata={"cumulative_reward": ep.cumulative_reward, "step": ep.step_count, "scenario_id": ep.scenario.id},
        )

    def _terminal_obs(self, output: str, reward: float) -> StackDoctorObservation:
        ep = self._episode
        return StackDoctorObservation(
            output=output, incident_ticket=ep.scenario.incident_ticket if ep else "",
            hardware=ep.scenario.hardware if ep else "", model_name=ep.scenario.model_name if ep else "",
            backend=ep.scenario.backend if ep else "", log_excerpt="", code_snippet="",
            specialist_opinions={}, steps_remaining=0, fix_used=ep.fix_applied if ep else False,
            done=True, reward=reward,
            metadata={"cumulative_reward": ep.cumulative_reward if ep else 0.0, "step": ep.step_count if ep else 0, "scenario_id": ep.scenario.id if ep else ""},
        )
