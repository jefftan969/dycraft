"""dycraft.cli — the launcher: one agent run per invocation.

    python -m dycraft.cli --backend claude --layer animation --prompt brief.md
    python -m dycraft.cli --backend codex  --layer object    --prompt "<prompt>"

`--backend` picks the agent CLI (`claude`, `codex`, or an account selector
such as `claude2`) and `--layer` the agent (`animation`, `scene`, `object`,
`plain`). `cli.py` is the launch sequence and `backends/` the agent CLIs;
README.md in this directory describes both.
"""
