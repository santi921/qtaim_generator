"""full-runner-engine: full-runner-parsl-alcf with the in-process charge engine.

Takes every full-runner-parsl-alcf argument. Charges (hirshfeld, adch, cm5,
becke), fuzzy density/spin and fuzzy_bond come from core/charge_engine.py;
conversion, QTAIM, "other" and the ORCA parse run as usual. Prevalidation and
restarts treat a folder as done only when its timings.json records a
charge_engine run, so pointing it at folders Multiwfn already completed (with
--restart) recomputes just the engine routines and merges them into generator/.
"""

import sys
from typing import List, Optional

from qtaim_gen.source.scripts.full_runner_parsl_alcf import main as _main


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--charge_engine" not in argv:
        argv.append("--charge_engine")
    return _main(argv)


if __name__ == "__main__":
    sys.exit(main())
