"""Entry point: python -m scripts.run_measure [docs_dir]"""

import json
import sys
from pathlib import Path

from src.measure import main

if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    print(json.dumps(main(out_dir), indent=2))
