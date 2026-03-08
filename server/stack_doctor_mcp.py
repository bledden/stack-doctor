"""
Stack Doctor MCP Environment.

Wraps the core Stack Doctor environment with MCP tools that agents
can discover and invoke. This is the agent-facing interface —
agents call tools like read_log(), query_specialist(), submit_diagnosis()
instead of constructing JSON action strings.

The training (WebSocket) API still works through _step_impl().
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

from models import StackDoctorAction, StackDoctorObservation
from .scenarios import (
    ROOT_CAUSE_TO_FIX,
    FIX_TO_ROOT_CAUSE,
    ROOT_CAUSES,
    FIXES,
    SPECIALISTS,
    Scenario,
    get_scenario,
)

MAX_STEPS = 6
VALID_FIXES = set(FIXES)
VALID_ROOT_CAUSES = set(ROOT_CAUSES)


class StackDoctorMCPEnvironment(MCPEnvironment):
    """
    Stack Doctor with MCP tool interface for agent interaction.

    Agents discover available tools (read_log, check_config, view_code,
    run_diagnostic, query_specialist, apply_fix, submit_diagnosis) and
    call them to investigate incidents and submit diagnoses.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        mcp = FastMCP("stack_doctor")
        self._state_obj = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Scenario | None = None
        self._step_count = 0
        self._fix_applied = False
        self._fix_was_correct: bool | None = None
        self._done = False
        self._cumulative_reward = 0.0
        self._actions_taken: list[dict] = []

        env = self  # capture for closures

        @mcp.tool()
        def read_log() -> str:
            """Read system and application logs for the current incident.
            Returns log output from the affected inference stack including
            error messages, warnings, and system state information.
            Costs 1 step (-0.25 reward)."""
            return env._do_inspect("logs")

        @mcp.tool()
        def check_config() -> str:
            """Check configuration files for the current incident.
            Returns relevant configuration parameters including GPU settings,
            backend configuration, model parameters, and environment variables.
            Costs 1 step (-0.25 reward)."""
            return env._do_inspect("config")

        @mcp.tool()
        def view_code() -> str:
            """View relevant source code snippets for the current incident.
            Returns code from the affected component showing the likely
            location of the bug or misconfiguration.
            Costs 1 step (-0.25 reward)."""
            return env._do_inspect("snippet")

        @mcp.tool()
        def run_diagnostic() -> str:
            """Run performance diagnostics and metrics collection.
            Returns metrics like latency, throughput, GPU utilization,
            error rates, and memory usage for the affected system.
            Costs 1 step (-0.25 reward)."""
            return env._do_inspect("metrics")

        @mcp.tool()
        def query_specialist(specialist: str) -> str:
            """Ask a specialist for their analysis of the incident.
            Specialists: 'runtime', 'dispatch', 'kernel', 'loader'.
            WARNING: At least one specialist gives wrong advice per incident.
            Cross-verify specialist opinions before trusting them.
            Costs 1 step (-0.25 reward)."""
            return env._do_ask_specialist(specialist)

        @mcp.tool()
        def apply_fix(fix: str) -> str:
            """Apply a fix to the system. Can only be used ONCE per incident.
            Available fixes: 'relax_arch_check', 'add_whitelist_entry',
            'fix_runtime_path', 'switch_backend', 'update_model_config',
            'fix_weight_mapping', 'tune_memory_config', 'fix_quantization',
            'fix_comm_config', 'update_driver_config'.
            Correct fix: +3 reward. Wrong fix: -2 reward."""
            return env._do_apply_fix(fix)

        @mcp.tool()
        def submit_diagnosis(root_cause: str, fix: str, justification: str = "") -> str:
            """Submit your final diagnosis. This ends the episode.
            Root causes: 'arch_guard', 'backend_whitelist', 'runtime_loader',
            'backend_selector', 'model_config', 'weight_layout',
            'memory_oom', 'quantization_error', 'distributed_comm', 'driver_compat'.
            Fixes: 'relax_arch_check', 'add_whitelist_entry', 'fix_runtime_path',
            'switch_backend', 'update_model_config', 'fix_weight_mapping',
            'tune_memory_config', 'fix_quantization', 'fix_comm_config', 'update_driver_config'.
            justification: A short sentence explaining WHY you chose this root cause
            and fix based on the evidence you gathered. Bonus +1 if provided.
            Correct root_cause: +8. Wrong: -4. Correct fix: +8. Wrong: -4.
            Bonus +2 if solved in 4 or fewer steps. Bonus +1 for justification."""
            return env._do_submit(root_cause, fix, justification)

        super().__init__(mcp)

    # ------------------------------------------------------------------
    # MCP tool implementations
    # ------------------------------------------------------------------

    def _check_episode(self) -> str | None:
        """Return error message if episode is not active."""
        if self._scenario is None:
            return "No active incident. Call reset() first."
        if self._done:
            return "Episode is over. Call reset() to start a new incident."
        if self._step_count >= MAX_STEPS:
            self._done = True
            return "Max steps reached. Episode over."
        return None

    def _record_step(self, reward: float, action: dict) -> None:
        self._step_count += 1
        self._state_obj.step_count = self._step_count
        self._cumulative_reward += reward
        self._actions_taken.append(action)

    def _do_inspect(self, target: str) -> str:
        err = self._check_episode()
        if err:
            return err

        ir = self._scenario.inspect_results
        result_map = {
            "logs": ir.logs,
            "config": ir.config,
            "snippet": ir.snippet,
            "metrics": ir.metrics,
        }

        self._record_step(-0.25, {"type": "inspect", "target": target})

        remaining = MAX_STEPS - self._step_count
        return (
            f"[INSPECT {target.upper()}]\n"
            f"{result_map[target]}\n\n"
            f"[Steps remaining: {remaining} | Reward: -0.25 | Cumulative: {self._cumulative_reward:.2f}]"
        )

    def _do_ask_specialist(self, specialist: str) -> str:
        err = self._check_episode()
        if err:
            return err

        if specialist not in SPECIALISTS:
            self._record_step(-2.0, {"type": "invalid", "message": f"Unknown specialist: {specialist}"})
            return f"Invalid specialist '{specialist}'. Available: {SPECIALISTS}. Penalty: -2.0"

        followup = self._scenario.specialist_followups.get(specialist, "No additional information.")
        self._record_step(-0.25, {"type": "ask_specialist", "specialist": specialist})

        remaining = MAX_STEPS - self._step_count
        return (
            f"[SPECIALIST: {specialist.upper()}]\n"
            f"{followup}\n\n"
            f"[Steps remaining: {remaining} | Reward: -0.25 | Cumulative: {self._cumulative_reward:.2f}]"
        )

    def _do_apply_fix(self, fix: str) -> str:
        err = self._check_episode()
        if err:
            return err

        if self._fix_applied:
            self._record_step(-2.0, {"type": "invalid", "message": "Fix already applied"})
            return "You already applied a fix this episode. Only one fix allowed. Penalty: -2.0"

        if fix not in VALID_FIXES:
            self._record_step(-2.0, {"type": "invalid", "message": f"Invalid fix: {fix}"})
            return f"Invalid fix '{fix}'. Available: {sorted(VALID_FIXES)}. Penalty: -2.0"

        self._fix_applied = True
        is_correct = fix == self._scenario.correct_fix
        self._fix_was_correct = is_correct
        reward = 3.0 if is_correct else -2.0
        self._record_step(reward, {"type": "apply_fix", "fix": fix, "correct": is_correct})

        remaining = MAX_STEPS - self._step_count
        if is_correct:
            return (
                f"[FIX APPLIED: {fix}] Fix applied successfully. Systems recovering.\n"
                f"Now submit your diagnosis with submit_diagnosis().\n\n"
                f"[Steps remaining: {remaining} | Reward: +3.0 | Cumulative: {self._cumulative_reward:.2f}]"
            )
        else:
            return (
                f"[FIX APPLIED: {fix}] Fix applied but the issue persists.\n"
                f"Consider your diagnosis carefully.\n\n"
                f"[Steps remaining: {remaining} | Reward: -2.0 | Cumulative: {self._cumulative_reward:.2f}]"
            )

    def _do_submit(self, root_cause: str, fix: str, justification: str = "") -> str:
        err = self._check_episode()
        if err:
            return err

        if root_cause not in VALID_ROOT_CAUSES:
            self._record_step(-2.0, {"type": "invalid", "message": f"Invalid root_cause: {root_cause}"})
            return f"Invalid root_cause '{root_cause}'. Available: {sorted(VALID_ROOT_CAUSES)}. Penalty: -2.0"

        if fix not in VALID_FIXES:
            self._record_step(-2.0, {"type": "invalid", "message": f"Invalid fix: {fix}"})
            return f"Invalid fix '{fix}'. Available: {sorted(VALID_FIXES)}. Penalty: -2.0"

        self._done = True
        rc_correct = root_cause == self._scenario.root_cause
        fix_correct = fix == self._scenario.correct_fix
        has_justification = len(justification.strip()) >= 10

        reward = 0.0
        reward += 8.0 if rc_correct else -4.0
        reward += 8.0 if fix_correct else -4.0
        if rc_correct and fix_correct and self._step_count + 1 <= 4:
            reward += 2.0
        if has_justification:
            reward += 1.0

        self._record_step(reward, {
            "type": "submit", "root_cause": root_cause, "fix": fix,
            "justification": justification,
            "rc_correct": rc_correct, "fix_correct": fix_correct,
            "has_justification": has_justification,
        })

        lines = ["[DIAGNOSIS SUBMITTED]"]
        lines.append(f"  Root cause: {root_cause} — {'CORRECT' if rc_correct else 'WRONG (was: ' + self._scenario.root_cause + ')'}")
        lines.append(f"  Fix: {fix} — {'CORRECT' if fix_correct else 'WRONG (was: ' + self._scenario.correct_fix + ')'}")
        if has_justification:
            lines.append(f"  Justification: {justification.strip()}")
            lines.append("  JUSTIFICATION BONUS: +1")
        else:
            lines.append("  No justification provided (missed +1 bonus)")
        lines.append(f"  Steps used: {self._step_count}/{MAX_STEPS}")
        if rc_correct and fix_correct and self._step_count <= 4:
            lines.append("  EFFICIENCY BONUS: +2 (solved in <= 4 steps)")
        lines.append(f"  Episode reward: {self._cumulative_reward:.2f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # OpenEnv Environment interface (for training / WebSocket API)
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, **kwargs) -> StackDoctorObservation:
        scenario_id = kwargs.get("scenario_id")
        split = kwargs.get("split", "train")
        self._scenario = get_scenario(scenario_id, split=split)

        self._state_obj = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._step_count = 0
        self._fix_applied = False
        self._fix_was_correct = None
        self._done = False
        self._cumulative_reward = 0.0
        self._actions_taken = []

        specialist_obs = {}
        for name, op in self._scenario.specialist_opinions.items():
            specialist_obs[name] = {
                "opinion": op.opinion,
                "confidence": op.confidence,
            }

        return StackDoctorObservation(
            output=(
                "STACK DOCTOR — New incident assigned.\n"
                "Investigate using the available tools: read_log(), check_config(), "
                "view_code(), run_diagnostic(), query_specialist(name).\n"
                "When ready, apply_fix(fix) and/or submit_diagnosis(root_cause, fix).\n"
                "You have 6 steps. At least one specialist is WRONG — cross-verify.\n"
            ),
            incident_ticket=self._scenario.incident_ticket,
            hardware=self._scenario.hardware,
            model_name=self._scenario.model_name,
            backend=self._scenario.backend,
            log_excerpt=self._scenario.initial_log,
            code_snippet=self._scenario.initial_snippet,
            specialist_opinions=specialist_obs,
            steps_remaining=MAX_STEPS,
            fix_used=False,
            done=False,
            reward=0.0,
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (JSON action strings for training)."""
        if not isinstance(action, StackDoctorAction):
            return self._make_obs("Invalid action type.", -2.0)

        try:
            parsed = json.loads(action.message)
        except (json.JSONDecodeError, TypeError):
            return self._make_obs(f"Invalid JSON: {action.message[:200]}", -2.0)

        action_type = parsed.get("type")

        if action_type == "inspect":
            result = self._do_inspect(parsed.get("target", "logs"))
        elif action_type == "ask_specialist":
            result = self._do_ask_specialist(parsed.get("specialist", ""))
        elif action_type == "apply_fix":
            result = self._do_apply_fix(parsed.get("fix", ""))
        elif action_type == "submit":
            result = self._do_submit(parsed.get("root_cause", ""), parsed.get("fix", ""), parsed.get("justification", ""))
        else:
            self._record_step(-2.0, {"type": "invalid", "message": f"Unknown: {action_type}"})
            result = f"Unknown action type: {action_type}. Penalty: -2.0"

        # Extract last reward from actions
        last_reward = 0.0
        if self._actions_taken:
            last = self._actions_taken[-1]
            if last.get("type") == "submit":
                # Calculate submit reward
                rc_c = last.get("rc_correct", False)
                fx_c = last.get("fix_correct", False)
                last_reward = (8.0 if rc_c else -4.0) + (8.0 if fx_c else -4.0)
                if rc_c and fx_c and self._step_count <= 4:
                    last_reward += 2.0
                if last.get("has_justification", False):
                    last_reward += 1.0
            elif last.get("type") == "apply_fix":
                last_reward = 3.0 if last.get("correct") else -2.0
            elif last.get("type") == "invalid":
                last_reward = -2.0
            else:
                last_reward = -0.25

        return self._make_obs(result, last_reward)

    def _make_obs(self, output: str, reward: float) -> StackDoctorObservation:
        remaining = MAX_STEPS - self._step_count
        return StackDoctorObservation(
            output=output,
            incident_ticket=self._scenario.incident_ticket if self._scenario else "",
            hardware=self._scenario.hardware if self._scenario else "",
            model_name=self._scenario.model_name if self._scenario else "",
            backend=self._scenario.backend if self._scenario else "",
            log_excerpt="",
            code_snippet="",
            specialist_opinions={},
            steps_remaining=remaining,
            fix_used=self._fix_applied,
            done=self._done,
            reward=reward,
            metadata={
                "cumulative_reward": self._cumulative_reward,
                "step": self._step_count,
                "scenario_id": self._scenario.id if self._scenario else "",
            },
        )

    @property
    def state(self) -> State:
        return self._state_obj
