"""
FastAPI application for the Stack Doctor Environment.

Exposes both:
  - WebSocket API (reset/step/state) for RL training
  - MCP API (tools/list, tools/call) for agent interaction

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from models import StackDoctorAction, StackDoctorObservation
from .stack_doctor_mcp import StackDoctorMCPEnvironment

app = create_app(
    StackDoctorMCPEnvironment,
    StackDoctorAction,
    StackDoctorObservation,
    env_name="stack_doctor",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
