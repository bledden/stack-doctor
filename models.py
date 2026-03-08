"""
Data models for the Stack Doctor Environment.

An overseer LLM diagnoses sick inference stacks by probing subsystems,
reconciling conflicting specialist-agent reports, and selecting the
minimal correct fix.
"""

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class StackDoctorAction(Action):
    """Agent action — a JSON message selecting one of 4 action types."""

    message: str = Field(
        ...,
        description=(
            'JSON action. One of:\n'
            '  {"type":"inspect","target":"logs|config|snippet|metrics"}\n'
            '  {"type":"ask_specialist","specialist":"runtime|dispatch|kernel|loader"}\n'
            '  {"type":"apply_fix","fix":"relax_arch_check|add_whitelist_entry|fix_runtime_path|switch_backend|update_model_config|fix_weight_mapping|tune_memory_config|fix_quantization|fix_comm_config|update_driver_config"}\n'
            '  {"type":"submit","root_cause":"...","fix":"...","justification":"..."}'
        ),
    )


class StackDoctorObservation(Observation):
    """What the agent sees after each action."""

    output: str = Field(default="", description="Natural-language feedback")
    incident_ticket: str = Field(default="", description="The incident description")
    hardware: str = Field(default="", description="Hardware identifier")
    model_name: str = Field(default="", description="Model being served")
    backend: str = Field(default="", description="Inference backend in use")
    log_excerpt: str = Field(default="", description="Log snippet")
    code_snippet: str = Field(default="", description="Config or code snippet")
    specialist_opinions: dict = Field(default_factory=dict, description="Specialist name -> {opinion, confidence}")
    steps_remaining: int = Field(default=6, description="Steps left in episode")
    fix_used: bool = Field(default=False, description="Whether apply_fix has been used")
