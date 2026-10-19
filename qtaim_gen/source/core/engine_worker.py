"""The engine steps of gbw_analysis for one job folder, in a process of their own.

gbw_analysis launches this once per folder (omol._run_engines): parsl's local runner
runs folders as threads of one process, where the engines shared the GIL and the
process-wide BLAS limits. Runs the QTAIM engine, then the charge and surface engines,
as omol._run_qtaim_engine / _run_charge_engine / _run_surface_engine, logging to the
folder's gbw_analysis.log. Exit status 0 when every step requested succeeded,
STEP_FAILED when one reported a failure (already in the log).
"""

import argparse
import sys

STEP_FAILED = 3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder")
    p.add_argument("--qtaim", action="store_true", help="QTAIM engine (core/cp_search.py)")
    p.add_argument("--charge", action="store_true", help="charge and surface engines")
    p.add_argument("--n_threads", type=int, default=1)
    p.add_argument("--full_set", type=int, default=0)
    p.add_argument("--restart", action="store_true")
    args = p.parse_args(argv)

    from qtaim_gen.source.core.omol import (
        _run_charge_engine,
        _run_qtaim_engine,
        _run_surface_engine,
        setup_logger,
    )

    logger = setup_logger(args.folder)
    ok = True
    if args.qtaim:
        ok = _run_qtaim_engine(args.folder, n_threads=args.n_threads, logger=logger, restart=args.restart) and ok
    if args.charge:
        ok = _run_charge_engine(args.folder, n_threads=args.n_threads, logger=logger, restart=args.restart,
                                full_set=args.full_set) and ok
        ok = _run_surface_engine(args.folder, n_threads=args.n_threads, logger=logger, restart=args.restart) and ok
    return 0 if ok else STEP_FAILED


if __name__ == "__main__":
    sys.exit(main())
