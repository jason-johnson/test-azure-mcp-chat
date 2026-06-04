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
import atexit
import hashlib
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

from pathlib import Path

from copilot import CopilotClient
from copilot.generated.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    PermissionCompletedData,
    SessionErrorData,
    ToolExecutionCompleteData,
    ToolExecutionStartData,
)
from copilot.session import PermissionHandler
import copilot.generated.rpc as rpc

# Skills bundled with this CLI — auto-loaded unless overridden.
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_SKILL_DIRS = [str(_REPO_ROOT / "skills")]

# Shared interactive input state.
_INTERACTIVE_PROMPT_SESSION: Any = None
_INPUT_LOCK: Optional[asyncio.Lock] = None

# Shared prompt suggestions for interactive input completion.
_PROMPT_HINTS = [
    "/help",
    "/exit",
    "/stream on",
    "/stream off",
    "/json on",
    "/json off",
    "/suggest",
    "/suggest subscriptions",
    "/suggest web apps",
    "list my subscriptions",
    "list resource groups",
    "list web apps",
    "show my current account",
]

_LOG = logging.getLogger("copilot_skills_cli")
_TELEMETRY_ENABLED = False


def _command_fingerprint(cmd: str) -> str:
    """Return a stable, non-reversible fingerprint for a command string."""
    digest = hashlib.sha256(cmd.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"len={len(cmd)} sha={digest}"


def _log_event(event_name: str, **fields: Any) -> None:
    """Emit safe structured diagnostics to logging sinks.

    This writes JSON payloads so they can be queried in App Insights when
    telemetry is enabled.
    """
    safe_fields: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            safe_fields[key] = value[:300]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe_fields[key] = value
        else:
            safe_fields[key] = str(value)[:300]

    _LOG.info("event=%s data=%s", event_name, json.dumps(safe_fields, ensure_ascii=True))


def _init_telemetry() -> None:
    """Initialize optional Azure Monitor telemetry.

    Enabled only when APPLICATIONINSIGHTS_CONNECTION_STRING is present.
    """
    global _TELEMETRY_ENABLED

    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=conn)
        logging.getLogger().setLevel(logging.INFO)
        _TELEMETRY_ENABLED = True
        _log_event("telemetry_enabled")
    except Exception as ex:
        print(f"Warning: telemetry initialization failed: {ex}", file=sys.stderr)
        _TELEMETRY_ENABLED = False


def _setup_readline_history() -> None:
    """Enable interactive line editing and persistent history on Unix."""
    try:
        import readline  # type: ignore
    except Exception:
        return

    history_path = os.path.join(os.path.expanduser("~"), ".copilot-skills-cli-history")

    try:
        if os.path.exists(history_path):
            readline.read_history_file(history_path)
    except Exception:
        pass

    readline.set_history_length(1000)

    def _completer(text: str, state: int):
        buffer = readline.get_line_buffer() if hasattr(readline, "get_line_buffer") else ""
        options = [item for item in _PROMPT_HINTS if item.startswith(buffer)]
        if state < len(options):
            return options[state]
        return None

    try:
        readline.set_completer_delims(" \t\n")
        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    def _save_history() -> None:
        try:
            readline.write_history_file(history_path)
        except Exception:
            pass

    atexit.register(_save_history)


def _get_input_lock() -> asyncio.Lock:
    global _INPUT_LOCK
    if _INPUT_LOCK is None:
        _INPUT_LOCK = asyncio.Lock()
    return _INPUT_LOCK


def _build_prompt_session() -> Any:
    """Build a prompt_toolkit session when available.

    Returns None if prompt_toolkit is unavailable so the caller can fall back
    to plain input().
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
    except Exception:
        return None

    history_path = os.path.join(os.path.expanduser("~"), ".copilot-skills-cli-history")
    key_bindings = KeyBindings()

    @key_bindings.add("c-space")
    def _trigger_completion(event) -> None:
        event.app.current_buffer.start_completion(select_first=False)

    completer = FuzzyCompleter(
        WordCompleter(_PROMPT_HINTS, ignore_case=True, sentence=True)
    )

    return PromptSession(
        history=FileHistory(history_path),
        auto_suggest=AutoSuggestFromHistory(),
        completer=completer,
        key_bindings=key_bindings,
        complete_while_typing=False,
    )


async def _read_user_input(prompt_session: Any, prompt_text: str) -> str:
    """Read user input using prompt_toolkit when present, else plain input."""
    async with _get_input_lock():
        if prompt_session is not None:
            return await prompt_session.prompt_async(prompt_text)
        return await asyncio.to_thread(input, prompt_text)


# Shell commands that the user has approved for the current session.
_SESSION_ALLOWED_COMMANDS: set[str] = set()


def _decision_approve_once():
    cls = getattr(rpc, "PermissionDecisionApproveOnce", None)
    if cls is not None:
        return cls()

    cls = getattr(rpc, "PermissionDecisionApproved", None)
    if cls is not None:
        return cls()

    raise RuntimeError("No compatible approve decision class found in copilot.generated.rpc")


def _decision_approve_for_session_command(cmd: str):
    approve_for_session = getattr(rpc, "PermissionDecisionApproveForSession", None)
    approval_commands = getattr(rpc, "PermissionDecisionApproveForSessionApprovalCommands", None)

    if approve_for_session is not None and approval_commands is not None:
        return approve_for_session(
            approval=approval_commands(command_identifiers=[cmd])
        )

    # Fallback for SDKs without explicit "approve for session command" types.
    return _decision_approve_once()


def _decision_deny():
    denied = getattr(rpc, "PermissionDecisionDeniedInteractivelyByUser", None)
    if denied is not None:
        return denied()

    reject = getattr(rpc, "PermissionDecisionReject", None)
    if reject is not None:
        return reject(feedback="Denied by user")

    cancelled = getattr(rpc, "PermissionDecisionCancelled", None)
    if cancelled is not None:
        return cancelled(reason="Denied by user")

    raise RuntimeError("No compatible deny decision class found in copilot.generated.rpc")


def _make_permission_handler():
    """Return a permissive fallback permission handler.

    Shell approval is handled in on_pre_tool_use so we get exactly one prompt
    in the terminal. This fallback prevents the runtime from introducing a
    second implicit ask/deny step.
    """

    def handler(request, invocation):
        return _decision_approve_once()

    return handler


async def _prompt_for_command(cmd: str) -> str:
    """Prompt the user once for a shell command and return y/a/n."""
    _log_event("permission_prompt", command_fingerprint=_command_fingerprint(cmd))
    print(f"\n[Permission required] The agent wants to run:\n  {cmd}")
    print("  [y] Approve once   [a] Approve for this session   [n] Deny")
    try:
        answer = await _read_user_input(
            _INTERACTIVE_PROMPT_SESSION,
            "  Your choice [y/a/n]: ",
        )
    except (EOFError, KeyboardInterrupt):
        _log_event("permission_decision", decision="n", reason="interrupt")
        return "n"
    decision = answer.strip().lower()
    _log_event("permission_decision", decision=decision)
    return decision


def _build_session_hooks() -> dict[str, Any]:
    """Return hooks that keep tool permission control in one place.

    The CLI's custom on_permission_request handler is the only approval gate.
    We explicitly allow tool execution in pre-tool hooks so the runtime does not
    apply a second implicit ask/deny step for tools such as `bash`.
    """

    async def on_pre_tool_use(input_data: dict[str, Any], invocation: dict[str, Any]):
        tool_name = str(input_data.get("toolName", ""))
        _log_event("tool_pre_use", tool_name=tool_name)
        if tool_name not in {"shell", "bash"}:
            return {"permissionDecision": "allow"}

        tool_args = input_data.get("toolArgs") or {}
        cmd = ""
        if isinstance(tool_args, dict):
            cmd = (
                str(tool_args.get("command") or "")
                or str(tool_args.get("full_command_text") or "")
                or str(tool_args.get("input") or "")
            )
        if not cmd:
            cmd = json.dumps(tool_args, ensure_ascii=True)

        if cmd in _SESSION_ALLOWED_COMMANDS:
            _log_event("permission_auto_allow", command_fingerprint=_command_fingerprint(cmd), scope="session")
            return {"permissionDecision": "allow"}

        answer = await _prompt_for_command(cmd)
        if answer == "a":
            _SESSION_ALLOWED_COMMANDS.add(cmd)
            _log_event("permission_allow", command_fingerprint=_command_fingerprint(cmd), scope="session")
            return {"permissionDecision": "allow"}
        if answer == "y":
            _log_event("permission_allow", command_fingerprint=_command_fingerprint(cmd), scope="once")
            return {"permissionDecision": "allow"}
        print("  Command denied.")
        _log_event("permission_deny", command_fingerprint=_command_fingerprint(cmd))
        return {
            "permissionDecision": "deny",
            "permissionDecisionReason": "Denied by user",
        }

    async def on_post_tool_use_failure(input_data: dict[str, Any], invocation: dict[str, Any]):
        _log_event(
            "tool_failure_hook",
            tool_name=input_data.get("toolName"),
            error=input_data.get("error"),
        )
        return None

    return {
        "on_pre_tool_use": on_pre_tool_use,
        "on_post_tool_use_failure": on_post_tool_use_failure,
    }


@dataclass
class CliState:
    stream: bool
    json_output: bool
    model: Optional[str]
    skill_dirs: list[str]


@dataclass
class TurnState:
    deltas: list[str]
    messages: list[str]
    errors: list[str]
    tool_calls: dict[str, str]


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
            case ToolExecutionStartData() as data:
                self._active_turn.tool_calls[data.tool_call_id] = data.tool_name
            case ToolExecutionCompleteData() as data:
                if data.success:
                    _log_event("tool_complete", tool_call_id=data.tool_call_id, success=True)
                    return

                tool_name = self._active_turn.tool_calls.get(data.tool_call_id, "tool")
                error_bits: list[str] = [f"{tool_name} failed"]

                if data.error and getattr(data.error, "message", None):
                    error_bits.append(str(data.error.message))

                result = getattr(data, "result", None)
                if result is not None:
                    detailed = getattr(result, "detailed_content", None)
                    content = getattr(result, "content", None)
                    extra = detailed or content
                    if extra:
                        compact = " ".join(str(extra).split())
                        error_bits.append(compact[:400])

                self._active_turn.errors.append(": ".join(error_bits))
                _log_event(
                    "tool_complete",
                    tool_call_id=data.tool_call_id,
                    success=False,
                    tool_name=tool_name,
                    error=": ".join(error_bits),
                )
            case PermissionCompletedData() as data:
                result = getattr(data, "result", None)
                result_name = type(result).__name__ if result is not None else "UnknownPermissionResult"
                _log_event("permission_completed", result=result_name)
                if "Denied" in result_name or "Cancelled" in result_name:
                    self._active_turn.errors.append(f"permission result: {result_name}")
            case SessionErrorData() as data:
                self._active_turn.errors.append(f"{data.error_type}: {data.message}")
                _log_event("session_error", error_type=data.error_type, message=data.message)

    async def ask(self, query: str) -> dict[str, Any]:
        self._active_turn = TurnState(deltas=[], messages=[], errors=[], tool_calls={})
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


def _normalize_auth_env() -> None:
    """Normalize token env names so SDK auto-discovery works consistently."""
    token = (
        os.getenv("COPILOT_GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("GITHUB_TOKEN")
    )
    if token and not os.getenv("COPILOT_GITHUB_TOKEN"):
        os.environ["COPILOT_GITHUB_TOKEN"] = token


def _build_runtime_env() -> dict[str, str]:
    """Forward key environment variables to the Copilot runtime process.

    This keeps shell tools (notably `az`) in the same auth/context as the
    current terminal session.
    """
    env_keys = [
        "PATH",
        "HOME",
        "SHELL",
        "USER",
        "LANG",
        "TERM",
        "AZURE_CONFIG_DIR",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_SUBSCRIPTION_ID",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ]

    env: dict[str, str] = {}
    for key in env_keys:
        val = os.getenv(key)
        if val:
            env[key] = val

    # If caller didn't set AZURE_CONFIG_DIR, default to the current user's profile.
    env.setdefault("AZURE_CONFIG_DIR", os.path.expanduser("~/.azure"))
    return env


def _create_copilot_client() -> CopilotClient:
    """Create CopilotClient in a way that works across SDK versions.

    Some SDK builds support CopilotClient(env=...), others do not.
    """
    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(CopilotClient)
        if "env" in sig.parameters:
            kwargs["env"] = _build_runtime_env()
    except Exception:
        # If signature introspection fails, fall back to default constructor.
        kwargs = {}

    try:
        return CopilotClient(**kwargs)
    except TypeError:
        # Backward/forward compatibility fallback for mismatched constructor args.
        return CopilotClient()


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
    has_gh_token = bool(
        os.getenv("COPILOT_GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("GITHUB_TOKEN")
    )
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
        default=None,
        help="Optional model name passed to create_session (for example, gpt-4.1).",
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
        help="Additional skill directory to load (repeatable). The bundled skills/ directory is always included.",
    )
    parser.add_argument(
        "--no-default-skills",
        action="store_true",
        help="Skip loading the bundled skills/ directory.",
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
    print("  /suggest [topic]           Ask AI for likely next prompts")
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

    if cmd == "/suggest":
        # Handled asynchronously in the interactive loop.
        return False

    print("unknown command. type /help")
    return True


async def _handle_suggest_command(
    raw: str,
    runner: "CliRunner",
    json_output: bool,
    stream: bool,
) -> None:
    """Generate suggested next prompts using the current model session."""
    topic = raw[len("/suggest"):].strip()
    topic_clause = f"Focus on: {topic}." if topic else "Focus on Azure operations in this workspace."

    suggest_prompt = (
        "Generate exactly 6 concise next prompts the user could ask in this CLI. "
        "Each suggestion must be one line, actionable, and specific. "
        "Do not run tools or shell commands. Do not ask for permission. "
        "Return plain text as a numbered list only. "
        f"{topic_clause}"
    )

    payload = await runner.ask(suggest_prompt)
    if json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    elif not stream:
        _print_human_result(payload)


async def _interactive_loop(state: CliState) -> int:
    global _INTERACTIVE_PROMPT_SESSION

    print("Copilot Skills CLI")
    selected_model = state.model if state.model else "default"
    print(f"model={selected_model} stream={state.stream} json={state.json_output}")
    if _TELEMETRY_ENABLED:
        print("logging=on (Application Insights)")
    else:
        print("logging=off")
    _print_interactive_help()

    _validate_auth_inputs()
    _normalize_auth_env()
    client = _create_copilot_client()
    _log_event("session_start", mode="interactive", telemetry_enabled=_TELEMETRY_ENABLED)
    await client.start()

    session_kwargs: dict[str, Any] = {
        "hooks": _build_session_hooks(),
        "on_permission_request": _make_permission_handler(),
        "streaming": state.stream,
        "working_directory": os.getcwd(),
        "system_message": {
            "content": "Follow workspace instructions and use available skills for task execution."
        },
    }
    if state.model:
        session_kwargs["model"] = state.model

    all_skill_dirs = list(state.skill_dirs)
    for d in _DEFAULT_SKILL_DIRS:
        if d not in all_skill_dirs:
            all_skill_dirs.append(d)
    if all_skill_dirs:
        session_kwargs["skill_directories"] = all_skill_dirs

    provider = _build_provider_config()
    if provider is not None:
        session_kwargs["provider"] = provider

    try:
        session = await client.create_session(**session_kwargs)
        _log_event("session_created", mode="interactive")
    except Exception as ex:
        await client.stop()
        _log_event("session_create_failed", mode="interactive", error=ex)
        raise RuntimeError(
            f"Failed to create session: {ex}. "
            "If this is a model availability issue, retry without --model "
            "or specify an available model with --model."
        )

    runner = CliRunner(session=session, stream=state.stream)
    session.on(runner.on_event)
    _INTERACTIVE_PROMPT_SESSION = _build_prompt_session()

    try:
        while True:
            try:
                user_input = await _read_user_input(_INTERACTIVE_PROMPT_SESSION, "\nYou> ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return 0

            text = user_input.strip()
            if not text:
                continue

            if text.startswith("/"):
                if text.startswith("/suggest"):
                    if not state.stream:
                        print("Assistant>")
                    try:
                        await _handle_suggest_command(
                            raw=text,
                            runner=runner,
                            json_output=state.json_output,
                            stream=state.stream,
                        )
                    except Exception as ex:
                        print(f"Error: {ex}", file=sys.stderr)
                    continue

                try:
                    _handle_command(text, state)
                    runner.stream = state.stream
                except EOFError:
                    print("bye")
                    return 0
                continue

            if not state.stream:
                print("Assistant>")

            try:
                payload = await runner.ask(text)
                _log_event("turn_complete", mode="interactive", error_count=len(payload.get("errors") or []))
            except Exception as ex:
                # Keep interactive mode alive on per-turn failures.
                print(f"Error: {ex}", file=sys.stderr)
                _log_event("turn_failed", mode="interactive", error=ex)
                continue

            if state.json_output:
                print(json.dumps(payload, indent=2, ensure_ascii=True))
            elif not state.stream:
                _print_human_result(payload)
    finally:
        _INTERACTIVE_PROMPT_SESSION = None
        _log_event("session_end", mode="interactive")
        await session.disconnect()
        await client.stop()


async def _main_async(args: argparse.Namespace) -> int:
    default_skills = [] if args.no_default_skills else _DEFAULT_SKILL_DIRS
    extra_skills = [d for d in args.skill_dir if d not in default_skills]
    state = CliState(
        stream=args.stream,
        json_output=args.json,
        model=args.model,
        skill_dirs=extra_skills,
    )

    if not args.query:
        return await _interactive_loop(state)

    _validate_auth_inputs()
    _normalize_auth_env()
    client = _create_copilot_client()
    _log_event("session_start", mode="oneshot", telemetry_enabled=_TELEMETRY_ENABLED)
    await client.start()

    session_kwargs: dict[str, Any] = {
        "hooks": _build_session_hooks(),
        "on_permission_request": _make_permission_handler(),
        "streaming": state.stream,
        "working_directory": os.getcwd(),
        "system_message": {
            "content": "Follow workspace instructions and use available skills for task execution."
        },
    }
    if state.model:
        session_kwargs["model"] = state.model

    all_skill_dirs = list(state.skill_dirs)
    for d in _DEFAULT_SKILL_DIRS:
        if d not in all_skill_dirs:
            all_skill_dirs.append(d)
    if all_skill_dirs:
        session_kwargs["skill_directories"] = all_skill_dirs

    provider = _build_provider_config()
    if provider is not None:
        session_kwargs["provider"] = provider

    session = await client.create_session(**session_kwargs)
    _log_event("session_created", mode="oneshot")
    runner = CliRunner(session=session, stream=state.stream)
    session.on(runner.on_event)

    try:
        payload = await runner.ask(args.query)
        _log_event("turn_complete", mode="oneshot", error_count=len(payload.get("errors") or []))
        if state.json_output:
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        elif not state.stream:
            _print_human_result(payload)
        return 0
    finally:
        _log_event("session_end", mode="oneshot")
        await session.disconnect()
        await client.stop()


def main() -> int:
    _init_telemetry()
    _setup_readline_history()
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
