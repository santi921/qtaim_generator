from quacc import change_settings
from qtaim_gen.source.quacc.runner import GeneratorRunner
from quacc import get_settings


def helper():
    run_root = "/home/santiagovargas/dev/qtaim_generator/data/parsl_test/1/"

    multiwfn_cmd = (
        "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
    )
    orca_2mkl_cmd = "/home/santiagovargas/orca_6_0_0/orca_2mkl"
    with change_settings(
        {
            "RESULTS_DIR": run_root,
            "SCRATCH_DIR": "./scratch/",
            "CREATE_UNIQUE_DIR": False,
        }
    ):
        print("Running GeneratorRunner test...")
        print("Setting RESULTS_DIR to ./results/ and SCRATCH_DIR to ./scratch/")

        full_set = 0
        n_threads = 5
        clean = True
        move_results = True
        overwrite = True
        overrun_running = True
        separate = True
        debug = False
        preprocess_compressed = False
        parse_only = False
        restart = False
        dry_run = False
        # print settings

        # not getting passed through
        # print("Current SETTINGS:", get_settings())
        runner = GeneratorRunner(
            command="generator-single-runner",
            folder=run_root,
            multiwfn_cmd=multiwfn_cmd,
            orca_2mkl_cmd=orca_2mkl_cmd,
            n_threads=n_threads,
            parse_only=parse_only,
            restart=restart,
            clean=clean,
            debug=debug,
            overrun_running=overrun_running,
            preprocess_compressed=preprocess_compressed,
            overwrite=overwrite,
            separate=separate,
            orca_6=True,
            full_set=full_set,
            move_results=move_results,
            dry_run=dry_run,
        )
        result = runner.run_cmd()
        # print("Command Output:", result.stdout)


helper()
