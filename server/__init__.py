"""Stack Doctor environment server components."""

from .stack_doctor_environment import StackDoctorEnvironment

__all__ = ["StackDoctorEnvironment"]


def get_mcp_environment():
    """Lazy import of MCP environment (requires fastapi/uvicorn)."""
    from .stack_doctor_mcp import StackDoctorMCPEnvironment
    return StackDoctorMCPEnvironment
