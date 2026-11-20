# PACE PIC Validation

Codebase for validating the PACE L2 Particulate Inorganic Carbon (PIC) product
against in-situ ground truth measurements.

## Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

## Daily script sync (macOS)

This Mac runs a `launchd` job every day at 00:00 local time. It stages changes
to script and notebook files (`.py`, `.ipynb`, `.sh`, `.R`, `.r`, `.js`, `.ts`,
and `.sql`), creates a dated commit when needed, and pushes the current branch
to `origin`. Other file types are not included. Failed commits or pushes are
recorded in `~/Library/Logs/pace-pic-validation-daily-push.log`; pushes that
cannot be completed are retried on the next scheduled run.

To check pending script changes without publishing them, run
`scripts/daily_git_push.sh --dry-run`. To disable the schedule, run:

```sh
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.astrotheus.pace-pic-validation.daily-push.plist"
```
