"""Terminal-first assistant using github-copilot-sdk only.

This CLI follows the SDK getting-started flow directly:
1) Start CopilotClient
2) Create one session
3) Attach event handler (streaming optional)
4) Send prompts with send_and_wait
5) Stop client

No MCP servers and no custom tools are configured in this CLI.
The assistant is expected to rely on workspace instructions and skills.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from copilot import CopilotClient
from copilot.generated.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionErrorData,
)
from copilot.session import PermissionHandler


@dataclass
class CliState:
    stream: bool
    json_output: bool
    model: str
    skill_dirs: list[str]


@dataclass
class TurnState:
    deltas: list[str]
    messages: list[str]
    errors: list[str]


class CliRunner:
    """Holds one Copilot session and routes events for each turn."""

    def __init__(self, session: Any, stream: bool):
        self.session = session
        self.stream = stream
        self._active_turn: Optional[TurnState] = None

    def on_event(self, event: Any) -> None:
        if self._active_turn is None:
            return

        match event.data:
            case AssistantMessageDeltaData() as data:
                delta = data.delta_content or ""
                if delta:
                    self._active_turn.deltas.append(delta)
                    if self.stream:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
            case AssistantMessageData() as data:
                self._active_turn.messages.append(data.content)
            case SessionErrorData() as data:
                self._active_turn.errors.append(f"{data.error_type}: {data.message}")

    async def ask(self, query: str) -> dict[str, Any]:
        self._active_turn = TurnState(deltas=[], messages=[], errors=[])
        try:
            await self.session.send_and_wait(query)
        finally:
            turn = self._active_turn
            self._active_turn = None

        if self.stream and turn and turn.deltas:
            print()

        if turn is None:
            return {"result": "", "errors": ["turn state missing"]}

        text = "\n".join(turn.messages).strip()
        if not text:
            text = "".join(turn.deltas).strip()

        return {
            "result": text,
            "errors": turn.errors,
        }


def _build_client_kwargs() -> dict[str, Any]:
    """Build client kwargs using token env when available."""
    kwargs: dict[str, Any] = {}
    gh_token = os.getenv("GITHUB_TOKEN")
    if gh_token:
        kwargs["github_token"] = gh_token
    return kwargs


def _build_provider_config() -> Optional[dict[str, Any]]:
    """Optional Azure OpenAI provider config from environment variables."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        return None

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    provider: dict[str, Any] = {
        "type": "azure",
        "base_url": endpoint,
        "azure": {
            "api_version": api_version,
            "deployment": deployment,
        },
    }

    if api_key:
        provider["api_key"] = api_key

    return provider


def _validate_auth_inputs() -> None:
    """Fail fast with actionable guidance when auth is missing."""
    has_gh_token = bool(os.getenv("GITHUB_TOKEN"))
    has_azure_provider = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
    if has_gh_token or has_azure_provider:
        return

    raise RuntimeError(
        "No authentication configured. Set GITHUB_TOKEN for Copilot auth "
        "or set AZURE_OPENAI_ENDPOINT (and optionally AZURE_OPENAI_API_KEY) "
        "for Azure provider mode."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copilot-skills-cli",
        description="CLI assistant built directly on github-copilot-sdk.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model name passed to create_session.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="One-shot query. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream Azure responses token-by-token (azure mode only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    parser.add_argument(
        "--skill-dir",
        action="append",
        default=[],
        help="Additional skill directory to load (repeatable).",
    )
    return parser


def _print_human_result(payload: dict[str, Any]) -> None:
    errors = payload.get("errors") or []
    if errors:
        print("Errors:")
        for err in errors:
            print(f"- {err}")

    result = payload.get("result", "")
    if result:
        print(result)


def _print_interactive_help() -> None:
    print("\nCommands:")
    print("  /stream on|off             Toggle streaming")
    print("  /json on|off               Toggle JSON output")
    print("  /help                      Show this help")
    print("  /exit                      Quit")


def _handle_command(raw: str, state: CliState) -> bool:
    parts = raw.strip().split()
    cmd = parts[0].lower()

    if cmd == "/help":
        _print_interactive_help()
        return True

    if cmd == "/exit":
        raise EOFError

    if cmd == "/stream" and len(parts) == 2 and parts[1] in {"on", "off"}:
        state.stream = parts[1] == "on"
        print(f"stream set to {state.stream}")
        return True

    if cmd == "/json" and len(parts) == 2 and parts[1] in {"on", "off"}:
        state.json_output = parts[1] == "on"
        print(f"json output set to {state.json_output}")
        return True

    print("unknown command. type /help")
    return True


async def _interactive_loop(state: CliState) -> int:
    print("Copilot Skills CLI")
    print(f"model={state.model} stream={state.stream} json={state.json_output}")
    _print_interactive_help()

    _validate_auth_inputs()
    client = CopilotClient(**_build_client_kwargs())
    await client.start()

    session_kwargs: dict[str, Any] = {
        "on_permission_request": PermissionHandler.approve_all,
        "model": state.model,
        "streaming": state.stream,
    }

    if state.skill_dirs:
        session_kwargs["skill_directories"] = state.skill_dirs

    # Keep system prompt minimal so workspace skills drive behavior.
    session_kwargs["system_message"] = {
        "content": "Follow workspace instructions and use available skills for task execution."
    }

    provider = _build_provider_config()
    if provider is not None:
        session_kwargs["provider"] = provider

    session = await client.create_session(**session_kwargs)
    runner = CliRunner(session=session, stream=state.stream)
    session.on(runner.on_event)

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nYou> ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return 0

            text = user_input.strip()
            if not text:
                continue

            if text.startswith("/"):
                try:
                    _handle_command(text, state)
                    runner.stream = state.stream
                except EOFError:
                    print("bye")
                    return 0
                continue

            if not state.stream:
                print("Assistant>")

            payload = await runner.ask(text)

            if state.json_output:
                print(json.dumps(payload, indent=2, ensure_ascii=True))
            elif not state.stream:
                _print_human_result(payload)
    finally:
        await session.disconnect()
        await client.stop()


async def _main_async(args: argparse.Namespace) -> int:
    state = CliState(
        stream=args.stream,
        json_output=args.json,
        model=args.model,
        skill_dirs=args.skill_dir,
    )

    if not args.query:
        return await _interactive_loop(state)

    _validate_auth_inputs()
    client = CopilotClient(**_build_client_kwargs())
    await client.start()

    session_kwargs: dict[str, Any] = {
        "on_permission_request": PermissionHandler.approve_all,
        "model": state.model,
        "streaming": state.stream,
        "system_message": {
            "content": "Follow workspace instructions and use available skills for task execution."
        },
    }
    if state.skill_dirs:
        session_kwargs["skill_directories"] = state.skill_dirs

    provider = _build_provider_config()
    if provider is not None:
        session_kwargs["provider"] = provider

    session = await client.create_session(**session_kwargs)
    runner = CliRunner(session=session, stream=state.stream)
    session.on(runner.on_event)

    try:
        payload = await runner.ask(args.query)
        if state.json_output:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        elif not state.stream:
            _print_human_result(payload)
        return 0
    finally:
        await session.disconnect()
        await client.stop()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_main_async(args))
    except RuntimeError as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nbye")
        return 130
    except Exception as ex:
        print(f"Error: unexpected failure: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
