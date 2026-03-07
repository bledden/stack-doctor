"""Stack Doctor Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import StackDoctorAction, StackDoctorObservation


class StackDoctorEnv(EnvClient[StackDoctorAction, StackDoctorObservation, State]):
    """
    Client for the Stack Doctor Environment.

    Example:
        >>> env = StackDoctorEnv(base_url="http://localhost:8000")
        >>> env.connect()
        >>> result = env.reset()
        >>> print(result.observation.incident_ticket)
        >>> result = env.step(StackDoctorAction(message='{"type":"inspect","target":"logs"}'))
        >>> print(result.observation.output)
        >>> env.close()
    """

    def _step_payload(self, action: StackDoctorAction) -> Dict:
        return {"message": action.message}

    def _parse_result(self, payload: Dict) -> StepResult[StackDoctorObservation]:
        obs_data = payload.get("observation", {})
        observation = StackDoctorObservation(
            output=obs_data.get("output", ""),
            incident_ticket=obs_data.get("incident_ticket", ""),
            hardware=obs_data.get("hardware", ""),
            model_name=obs_data.get("model_name", ""),
            backend=obs_data.get("backend", ""),
            log_excerpt=obs_data.get("log_excerpt", ""),
            code_snippet=obs_data.get("code_snippet", ""),
            specialist_opinions=obs_data.get("specialist_opinions", {}),
            steps_remaining=obs_data.get("steps_remaining", 0),
            fix_used=obs_data.get("fix_used", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
