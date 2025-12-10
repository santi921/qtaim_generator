# config.py
import os
from parsl.config import Config
from parsl.providers import PBSProProvider, LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher


from parsl.executors.threads import ThreadPoolExecutor
from parsl.addresses import address_by_hostname
from parsl.monitoring.monitoring import MonitoringHub

def alcf_config(
    threads_per_task: int = 8,
    safety_factor: float = 1.0,
    threads_per_node: int = 256,
    n_jobs: int = 2,
    queue: str = "debug",
    walltime: str = "00:30:00",
    monitoring: bool = False,
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

    if monitoring:
        monitoring = MonitoringHub(
            hub_address=address_by_hostname(),
            hub_port=55055,
            monitoring_debug=False,
        )
    else:
        monitoring = None

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
                    worker_init=( # Debugging
                            "set -x; "  # print every command as it runs
                            "echo '--- WORKER_INIT START ---'; "
                            "hostname; "
                            "date; "
                            "module use /soft/modulefiles; "
                            "source /soft/datascience/conda/2025-06-03/etc/profile.d/conda.sh; "
                            "echo 'Loaded conda module'; "
                            # Sanity check Python / Parsl
                            "which python || { echo 'python not found'; exit 1; }; "
                            "python -c 'import sys, parsl; print(\"PYOK\", sys.version); print(\"PARSL\", parsl.__version__)' || { echo 'Python/parsl import failed'; exit 1; }; "
                            # Go to working dir
                            f"cd {execute_dir} || {{ echo 'cd {execute_dir} failed'; exit 1; }}; "
                            "pwd; "
                            f"export OMP_NUM_THREADS={threads_per_task}; "
                            "export KMP_STACKSIZE=200M; "
                    ),
                    # Wall time for batch jobs
                    walltime=walltime, 
                    # Change if data/modules located on other filesystem
                    scheduler_options="#PBS -l filesystems=home:eagle",
                    # Ensures 1 manger per node; the manager will distribute work to its 12 workers, one per tile
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
                    # options added to #PBS -l select aside from ncpus
                    select_options="",
                    nodes_per_block=nodes_per_job,
                    min_blocks=1,
                    max_blocks=n_jobs,
                    cpus_per_node=threads_per_node,
                ),
            ),
        ],
        # How many times to retry failed tasks
        # this is necessary if you have tasks that are interrupted by a PBS job ending
        # so that they will restart in the next job
        retries=1,
        #monitoring=monitoring,
    )
    return aurora_single_tile_config, threads_per_node

"""
crux configs 

For CPU 0:
NUMA 0: cores 0-15,128-143
NUMA 1: cores 16-31,144-159
NUMA 2: cores 32-47,160-175
NUMA 3: cores 48-63,176-191

For CPU 1:
NUMA 4: cores 64-79,192-207
NUMA 5: cores 80-95,208-223
NUMA 6: cores 96-111,224-239
NUMA 7: cores 112-127,240-255
"""
cpu_affinity = "list:0-15,128-143;16-31,144-159;32-47,160-175;48-63,176-191;64-79,192-207;80-95,208-223;96-111,224-239;112-127,240-255"

def alcf_config_single_pbs(
    threads_per_task: int = 8,
    safety_factor: float = 1.0,
    threads_per_node: int = 256,
    n_jobs: int = 2,
    monitoring: bool = False,
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

    if monitoring:
        monitoring = MonitoringHub(
            hub_address=address_by_hostname(),
            hub_port=55055,
            monitoring_debug=False,
        )
    else:
        monitoring = None

    aurora_single_pbs_config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers_per_node=workers_per_node,
                cpu_affinity='block',
                prefetch_capacity=0,
                # Options that specify properties of PBS Jobs
                provider=LocalProvider(
                    nodes_per_block=n_jobs,
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
                    init_blocks=1,
                    max_blocks=1,
                    worker_init=( # Debugging
                            "set -x; "  # print every command as it runs
                            "echo '--- WORKER_INIT START ---'; "
                            "hostname; "
                            "date; "
                            "module use /soft/modulefiles; "
                            "source /soft/datascience/conda/2025-06-03/etc/profile.d/conda.sh; "
                            "echo 'Loaded conda module'; "
                            # Sanity check Python / Parsl
                            "which python || { echo 'python not found'; exit 1; }; "
                            "python -c 'import sys, parsl; print(\"PYOK\", sys.version); print(\"PARSL\", parsl.__version__)' || { echo 'Python/parsl import failed'; exit 1; }; "
                            # Go to working dir
                            f"cd {execute_dir} || {{ echo 'cd {execute_dir} failed'; exit 1; }}; "
                            "pwd; "
                            f"export OMP_NUM_THREADS={threads_per_task}; "
                            "export KMP_STACKSIZE=200M; "
                    ),
                ),
            ),
        ],
    )
    return aurora_single_pbs_config, threads_per_node

def base_config(n_workers: int = 4) -> Config:
    """Returns a basic Parsl config using local threads executor.

    Returns:
        Config: A Parsl configuration object.
    """
    local_threads = Config(
        executors=[ThreadPoolExecutor(max_threads=n_workers, label="local_threads")]
    )
    return local_threads
