from __future__ import annotations

import sys
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parent
SRC_ROOT = FRAMEWORK_ROOT / "src"
ERASURE_ROOT = FRAMEWORK_ROOT.parent / "ERASURE-main"
for import_root in (SRC_ROOT, ERASURE_ROOT):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kt_unlearn.runners.experiment import main


if __name__ == "__main__":
    main()
