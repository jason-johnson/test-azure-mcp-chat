# Copilot Skills CLI

A terminal-first assistant built with GitHub Copilot SDK.

This project is now CLI-only. It uses:
- `cli.py` as the main entrypoint
- `skills/` for behavior guidance (currently `skills/azure-cli/SKILL.md`)
- interactive approval prompts before shell command execution

The assistant is designed for Azure operational workflows using `az` CLI, with user confirmation before command execution.

## Project Layout

- `cli.py` - main application
- `requirements.txt` - Python dependencies
- `pyproject.toml` - project metadata
- `skills/azure-cli/SKILL.md` - Azure CLI skill instructions

## Requirements

- Python 3.11+
- GitHub Copilot SDK runtime support in your environment
- Azure CLI (`az`) installed and authenticated if you want Azure command execution

## Installation

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## Authentication

Set one of these for Copilot auth:

- `COPILOT_GITHUB_TOKEN` (preferred)
- `GH_TOKEN`
- `GITHUB_TOKEN`

Example:

```bash
export GH_TOKEN=your_token_here
```

Alternative provider mode (optional):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY` (or other supported auth in your environment)
- optional `AZURE_OPENAI_DEPLOYMENT_NAME`

## Run

Start interactive mode:

```bash
python3 cli.py
```

Run one-shot mode:

```bash
python3 cli.py --query "list my subscriptions"
```

## Interactive Commands

Inside the CLI:

- `/help` - show command help
- `/exit` - quit
- `/stream on|off` - toggle streaming output
- `/json on|off` - toggle JSON output mode
- `/suggest [topic]` - ask the model for suggested next prompts

## Permission Model

This CLI asks for approval before shell command execution.

When a command is requested, you will see:

- `y` - approve once
- `a` - approve for this session
- `n` - deny

If denied, the assistant explains and continues without exiting.

## Input UX

The CLI supports enhanced terminal input (when `prompt_toolkit` is available):

- command history recall (up/down)
- tab completion for slash commands and common starters
- history-based autosuggestions
- `Ctrl-Space` to trigger completion

History is persisted in:

- `~/.copilot-skills-cli-history`

## Skills

Skills are loaded from `skills/` by default.

Current skill:

- `skills/azure-cli/SKILL.md`

You can add more skill directories at runtime:

```bash
python3 cli.py --skill-dir /path/to/skills
```

Disable default bundled skills:

```bash
python3 cli.py --no-default-skills
```

## Useful Examples

```bash
python3 cli.py --query "list my subscriptions"
python3 cli.py --query "show all web apps in my subscription"
python3 cli.py --stream
python3 cli.py --model gpt-4.1
```

## Troubleshooting

### "No authentication configured"

Set one of:

```bash
export COPILOT_GITHUB_TOKEN=...
# or
export GH_TOKEN=...
# or
export GITHUB_TOKEN=...
```

### Model not available

Retry without `--model`, or pass a model available to your account.

### Azure commands fail

Verify Azure login in the same terminal:

```bash
az account show -o table
```

If needed:

```bash
az login
```

## Notes

- The skill guides behavior; actual command execution is still handled by the runtime shell tool path.
- Permission handling and SDK compatibility safeguards are implemented in `cli.py`.
