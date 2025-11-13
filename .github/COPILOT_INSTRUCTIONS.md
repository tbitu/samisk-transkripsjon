Purpose
-------
This file documents a repository-level instruction for humans and automated helpers (editors, CI, assistants): when running anything related to Python for this project, ALWAYS use the project's virtual environment (the `.venv` directory at the project root).

Why
---
The project depends on packages that are installed into the project's virtual environment. Running Python commands outside the venv can cause ModuleNotFoundError, wrong package versions, or unpredictable behavior. This file exists so contributors and automated tools have an explicit, discoverable rule to follow.

Rule (short)
------------
Always activate the project's virtual environment before running Python-related commands:

```bash
# from project root
python -m venv .venv    # create if missing
source .venv/bin/activate
# now run Python commands, e.g.:
python -m uvicorn app.main:app --reload
python -m pip install -r requirements.txt
pytest
```

How to ask the assistant (recommended)
--------------------------------------
When you request that an assistant or a human "run" or "execute" Python commands from this repository, include one of these explicit instructions in the same message:

- "Activate the project's virtual environment (.venv) first and then run: <command>"
- or: "Run using the venv's Python: .venv/bin/python -m <module>"

Examples:

- "Please activate .venv and run: python -m uvicorn app.main:app --reload"
- "Run the tests using .venv/bin/python -m pytest tests/test_api.py"

Notes about assistants and automation
------------------------------------
- This file is informational and intended for humans, CI, and tools. It cannot change the internal configuration or system messages of hosted assistants.
- If you want a script or CI job to *enforce* venv activation automatically, prefer wrappers that explicitly source `.venv/bin/activate` or call `.venv/bin/python` directly.

Repository helpers
------------------
- `run.sh` in the project root already activates `.venv` before starting the server. Prefer using that script or invoking the venv's Python directly.

If you want me to add an automated wrapper, a git hook, or update CI to enforce this, tell me which one and I'll implement it.