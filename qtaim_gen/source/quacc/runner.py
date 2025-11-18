from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from shlex import split
from subprocess import PIPE, CompletedProcess, run
from typing import TYPE_CHECKING, ClassVar, Final

from quacc.runners._base import BaseRunner
from quacc.runners.generic import GenericRunner
from typing import Optional, Dict, Any, List
from logging import getLogger
from pathlib import Path
from shutil import move, rmtree
from typing import TYPE_CHECKING

from monty.shutil import gzip_dir

if TYPE_CHECKING:
    from ase.atoms import Atoms

LOGGER = getLogger(__name__)

# from quacc.runners.prep import calc_cleanup, terminate, calc_setup

# from ase.atoms import Atoms
# from quacc.types import Filenames, SourceDirectory
from quacc import JobFailure, get_settings


def calc_cleanup(
    atoms: Atoms | None, tmpdir: Path | str, job_results_dir: Path | str
) -> None:
    """
    Perform cleanup operations for a calculation, including gzipping files, copying
    files back to the original directory, and removing the tmpdir.

    Parameters
    ----------
    atoms
        The Atoms object after the calculation. Must have a calculator
        attached. If None, no modifications to the calculator's directory will be made.
    tmpdir
        The path to the tmpdir, where the calculation will be run. It will be
        deleted after the calculation is complete.
    job_results_dir
        The path to the job_results_dir, where the files will ultimately be
        stored.

    Returns
    -------
    None
    """
    job_results_dir, tmpdir = Path(job_results_dir), Path(tmpdir)
    settings = get_settings()

    # Safety check
    if "tmp-" not in tmpdir.name:
        msg = f"{tmpdir} does not appear to be a tmpdir... exiting for safety!"
        raise ValueError(msg)

    # Update the calculator's directory
    if atoms is not None:
        atoms.calc.directory = job_results_dir

    # Gzip files in tmpdir
    if settings.GZIP_FILES:
        gzip_dir(tmpdir)

    # Move files from tmpdir to job_results_dir
    if settings.CREATE_UNIQUE_DIR:
        move(tmpdir, job_results_dir)
    else:
        for file_name in os.listdir(tmpdir):
            if ".json" not in file_name:
                move(tmpdir / file_name, job_results_dir / file_name)
        rmtree(tmpdir)
    LOGGER.info(f"Calculation results stored at {job_results_dir}")

    # Remove symlink to tmpdir
    if os.name != "nt" and settings.SCRATCH_DIR:
        symlink_path = settings.RESULTS_DIR / f"symlink-{tmpdir.name}"
        symlink_path.unlink(missing_ok=True)


class GeneratorRunner(BaseRunner):
    """
    A class to run generic (IO) commands in a subprocess. Inherits from BaseRunner, which handles setup and cleanup of the calculation.
    """

    filepaths: ClassVar[dict] = {
        "fd_out": None,
        "fd_err": None,
    }

    def __init__(
        self,
        command: str,
        folder: str,
        multiwfn_cmd: Optional[str] = None,
        orca_2mkl_cmd: Optional[str] = None,
        n_threads: int = 3,
        parse_only: bool = False,
        restart: bool = False,
        clean: bool = False,
        debug: bool = False,
        overrun_running: bool = False,
        preprocess_compressed: bool = False,
        overwrite: bool = False,
        separate: bool = True,
        orca_6: bool = True,
        full_set: int = 0,
        move_results: bool = True,
        dry_run: bool = False,
        environment: Optional[Dict[str, str]] = None,
        copy_files: Optional[Dict[str, str]] = None,
        move_results_to_folder: Optional[str] = None,
    ) -> None:
        """
        Initialize the `GeneratorRunner` with the command and optional copy files and environment variables.

        Parameters
        ----------
        command
            The command to run in the subprocess.
        copy_files
            Files to copy to the runtime directory.
        environment
            Environment variables to set for the subprocess. If None, the current environment is used.

        Returns
        -------
        None
        """
        # Initialize BaseRunner first so it can set up any infrastructure
        # (tmpdir, logging, etc.) without accidentally overwriting attributes
        # we set on this subclass.
        super().__init__()

        # Store provided parameters on the instance. Do this after calling
        # the base-class initializer so BaseRunner doesn't override them.
        self.command: Final[list[str]] = split(command)
        self.folder = folder
        self.multiwfn_cmd = multiwfn_cmd
        self.orca_2mkl_cmd = orca_2mkl_cmd
        self.parse_only = parse_only
        self.restart = restart
        self.clean = clean
        self.debug = debug
        self.overrun_running = overrun_running
        self.preprocess_compressed = preprocess_compressed
        self.n_threads = n_threads
        self.overwrite = overwrite
        self.separate = separate
        self.orca_6 = orca_6
        self.full_set = full_set
        self.move_results = move_results
        self.dry_run = dry_run
        self.move_results_to_folder = move_results_to_folder

        self.environment = environment
        """
        print(self.command)
        print("folder:", self.folder)
        print("multiwfn_cmd:", self.multiwfn_cmd)
        print("orca_2mkl_cmd:", self.orca_2mkl_cmd)
        print("n_threads:", self.n_threads)
        print("parse_only:", self.parse_only)
        print("restart:", self.restart)
        print("clean:", self.clean)
        print("debug:", self.debug)
        print("overrun_running:", self.overrun_running)
        print("preprocess_compressed:", self.preprocess_compressed)
        print("overwrite:", self.overwrite)
        print("separate:", self.separate)
        print("orca_6:", self.orca_6)
        print("full_set:", self.full_set)
        print("move_results:", self.move_results)
        print("dry_run:", self.dry_run)
        print("move_results_to_folder:", self.move_results_to_folder)
        """

        results_folder = Path(self.folder) / "generator/"
        if self.move_results:
            copy_files = {
                self.folder: [
                    "input.gbw",
                    "input.gbw",
                    "input.in",
                    "gbw_analysis.log",
                ],
                results_folder: [
                    "timing.json",
                    "bond.json",
                    "fuzzy_full.json",
                    "qtaim.json",
                    "other.json",
                    "charge.json",
                ],
            }
        else:
            copy_files = {
                self.folder: [
                    "input.gbw",
                    "input.gbw",
                    "input.in",
                    "gbw_analysis.log",
                    "timing.json",
                    "bond.json",
                    "fuzzy_full.json",
                    "qtaim.json",
                    "other.json",
                    "charge.json",
                ]
            }

        self.setup(copy_files=copy_files)

    def run_cmd(self) -> CompletedProcess:
        """
        Run a command in a subprocess.

        Returns
        -------
        CompletedProcess
            The result of the subprocess execution.
        """
        with ExitStack() as stack:
            files = {
                name: stack.enter_context(Path(self.tmpdir, path).open("w"))
                for name, path in self.filepaths.items()
                if path is not None
            }

            command_post = [
                self.command[0],
                "--run_root",
                str(self.tmpdir),
                "--multiwfn_cmd",
                str(self.multiwfn_cmd),
                "--orca_2mkl_cmd",
                str(self.orca_2mkl_cmd),
                "--n_threads",
                str(self.n_threads),
                "--run_root",
                str(self.folder),  # might need to update this to quacc it up
                "--full_set",
                str(int(self.full_set)),
            ]
            if self.preprocess_compressed:
                command_post.append("--preprocess_compressed")

            if self.restart:
                command_post.append("--restart")

            if self.clean:
                command_post.append("--clean")

            if self.parse_only:
                command_post.append("--parse_only")

            if self.debug:
                command_post.append("--debug")

            if self.overrun_running:
                command_post.append("--overrun_running")

            if self.overwrite:
                command_post.append("--overwrite")

            if self.move_results:
                command_post.append("--move_results")

            if self.dry_run:
                command_post.append("--dry-run")

            # if self.move_results_to_folder is not None:
            command_post.extend(["--move_results_to_folder", str(self.job_results_dir)])

            print(command_post)
            # print as str
            # print("Command to run:", " ".join(command_post))
            cmd_results = run(
                command_post,
                cwd=self.tmpdir,
                shell=False,
                check=True,
                env=self.environment if self.environment is not None else os.environ,
                stdout=files.get("fd_out", PIPE),
                stderr=files.get("fd_err", PIPE),
                text=True,
            )

        self.cleanup()
        return cmd_results

    def cleanup(self) -> None:
        """
        Perform cleanup operations on the runtime directory.

        Returns
        -------
        None
        """
        calc_cleanup(self.atoms, self.tmpdir, self.job_results_dir)
