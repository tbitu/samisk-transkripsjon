from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `import app` works when pytest runs
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
