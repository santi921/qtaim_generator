# config.py
import os
from parsl.config import Config
from parsl.providers import PBSProProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.executors.threads import ThreadPoolExecutor


def alcf_config(
    threads_per_task: int = 8,
    safety_factor: int = 1,
    threads_per_node: int = 256,
    n_jobs: int = 64,
    queue: str = "debug",
    timeout: str = "00:30:00",
) -> Config:
    """
    Returns a Parsl config optimized for running on ALCF.

    Returns:
        Config: A Parsl configuration object for ALCF.
    """
    # These options will run work in 1 node batch jobs run one at a time

    # threads_per_node   = 256        # hardware threads
    # threads_per_task   = 8          # each job uses 8 threads
    workers_per_node = int(
        threads_per_node // threads_per_task // safety_factor
    )  # 256 // 8 = 32

    nodes_per_job = 1

    # The config will launch workers from this directory
    execute_dir = os.getcwd()

    aurora_single_tile_config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex_cpu",
                # Ensures one worker per GPU tile on each node
                max_workers_per_node=workers_per_node,
                cpu_affinity="block",
                prefetch_capacity=0,
                # Options that specify properties of PBS Jobs
                provider=PBSProProvider(
                    # Project name
                    account="generator",
                    # Submission queue
                    queue=queue,
                    # Commands run before workers launched
                    # Make sure to activate your environment where Parsl is installed
                    worker_init=(
                        "module use /soft/modulefiles; "
                        "module load conda; "
                        "conda activate generator; "
                        f"cd {execute_dir}; "
                        f"export OMP_NUM_THREADS={threads_per_task}; "
                        f"export OPENBLAS_NUM_THREADS={threads_per_task}; "
                        f"export MKL_NUM_THREADS={threads_per_task}; "
                        f"export NUMEXPR_MAX_THREADS={threads_per_task}; "
                        # set unlim memory
                        "ulimit -s unlimited; "
                        "export KMP_STACKSIZE=200M; "
                    ),
                    # Wall time for batch jobs
                    walltime=timeout,
                    # Change if data/modules located on other filesystem
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    # Ensures 1 manger per node; the manager will distribute work to its 12 workers, one per tile
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--ppn 1"
                    ),
                    # options added to #PBS -l select aside from ncpus
                    select_options="",
                    # How many nodes per PBS job:
                    nodes_per_block=nodes_per_job,
                    # Min/max *concurrent* PBS jobs (blocks) that Parsl can have in the queue:
                    min_blocks=1,
                    max_blocks=n_jobs,
                    # Tell Parsl / PBS how many hardware threads there are per node:
                    cpus_per_node=threads_per_node,
                ),
            ),
        ],
        # How many times to retry failed tasks
        # this is necessary if you have tasks that are interrupted by a PBS job ending
        # so that they will restart in the next job
        retries=1,
    )
    return aurora_single_tile_config


def base_config(n_workers: int = 4) -> Config:
    """Returns a basic Parsl config using local threads executor.

    Returns:
        Config: A Parsl configuration object.
    """
    local_threads = Config(
        executors=[ThreadPoolExecutor(max_threads=n_workers, label="local_threads")]
    )
    return local_threads
