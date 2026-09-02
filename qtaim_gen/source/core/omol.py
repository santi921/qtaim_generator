from pathlib import Path
import os, stat, json, time, logging, re
from typing import Optional, Dict, Any, List
import subprocess

import zipfile

from qtaim_gen.source.utils.validation import (
    validation_checks,
    get_val_breakdown_from_folder,
    get_charge_spin_n_atoms_from_folder,
    get_expected_timing_keys,
    TIMINGS_PATCHED_KEY,
    TIMING_PLACEHOLDER,
)

from qtaim_gen.source.utils.io import check_results_exist
from qtaim_gen.source.utils.atomic_write import atomic_json_write
from qtaim_gen.source.core.horton import run_horton_analysis

from qtaim_gen.source.data.multiwfn import (
    charge_data,
    charge_data_dict,
    bond_order_data,
    bond_order_dict,
    fuzzy_data,
    other_data,
    other_data_dict,
    qtaim_data,
)
from qtaim_gen.source.core.parse_multiwfn import (
    parse_charge_doc,
    parse_charge_base,
    parse_charge_becke,
    parse_charge_doc_adch,
    parse_bond_order_doc,
    parse_bond_order_fuzzy,
    parse_bond_order_ibsi,
    parse_bond_order_laplace,
    parse_fuzzy_doc,
    parse_other_doc,
    parse_other_doc_esp, 
    parse_other_doc_geometry,
    parse_qtaim,
    parse_charge_doc_bader,
    parse_charge_chelpg,
    parse_fuzzy_real_space,
)

from qtaim_gen.source.utils.io import (
    pull_ecp_dict,
    overwrite_molden_w_ecp,
    check_spin,
    merge_zip_into,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORDER_OF_OPERATIONS = ["fuzzy_full", "qtaim", "bond", "charge", "other"]
ORDER_OF_OPERATIONS_separate = [
    "fuzzy_full",
    "charge_separate",
    "bond_separate",
    "qtaim",
    "other_separate",  # muting for meta
]


def write_settings_file(folder: str, mem: int = 400000000, n_threads: int = 3) -> None:
    # get loc of qtaim_embed folder
    qtaim_embed_loc = str(Path(__file__).parent.parent.parent)
    old_path = qtaim_embed_loc + "/source/data/settings.ini"
    # copy old path to current path
    # wwrite txt files line by line and add a line n_threads = str(n_threads) + "\n"
    # another line ompstacksize= str(mem) + "\n"
    with open(old_path, "r") as f:
        data = f.read()
    # write to new path
    # remove last line and save separately
    last_lines = data.split("\n")[-3:]
    data = "\n".join(data.split("\n")[:-3])  # remove last line

    new_path = str(Path(folder).joinpath("settings.ini"))
    # if new_path exists remove it
    if os.path.exists(new_path):
        os.remove(new_path)
    with open(new_path, "w") as f:
        f.write(data)
        f.write("  nthreads= {}\n".format(n_threads))
        f.write("  ompstacksize= {}\n".format(mem))
        f.write(last_lines[0] + "\n")  # write last line
        f.write(last_lines[1] + "\n")  # write last line without newline
        f.write(last_lines[2])  # write last line without newline
        f.write("\n")


def write_conversion(
    out_folder: str,
    read_file: str,
    overwrite: bool = False,
    name: str = "convert.in",
    orca_2mkl_cmd: str = "orca_2mkl",
) -> None:
    """
    Function to write a bash script that runs multiwfn on a given input file.
    Args:
        out_folder(str): folder to write the bash script to
        multi_wfn_cmd(str): command to run multiwfn
        multiwfn_input_file(str): input file for multiwfn
        overwrite(bool): whether to overwrite the file if it already exists
        name(str): name of the bash script
    """

    out_file = str(Path.home().joinpath(out_folder, name))
    # print("out_file: {}".format(out_file))

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    completed_tf = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if completed_tf and not overwrite:
        print("file exists and overwrite is true")
        return

    with open(out_file, "w") as f:
        f.write("#!/bin/bash\n")
        # remove .gbw from file name
        read_file = read_file.split(".gbw")[0]
        f.write(
            "{} ".format(orca_2mkl_cmd)
            + "'" + str(Path.home().joinpath(out_folder, read_file)) + "'"
            + " -molden\n"
        )
        # also have it clean up the gbw file, .molden.input file

        if os.path.exists(str(Path.home().joinpath(out_folder, read_file + ".molden.input"))):
            f.write("rm '{}.molden.input'\n".format(str(Path.home().joinpath(out_folder, read_file))))

        # check if gbw file exists
        if os.path.exists(str(Path.home().joinpath(out_folder, read_file + ".gbw"))):
            f.write("rm '{}.gbw'\n".format(str(Path.home().joinpath(out_folder, read_file))))

    st = os.stat(out_file)
    os.chmod(out_file, st.st_mode | stat.S_IEXEC)


def write_multiwfn_exe(
    out_folder: str,
    read_file: str,
    multi_wfn_cmd: str,
    multiwfn_input_file: str,
    convert_gbw: bool = False,
    overwrite: bool = False,
    mv_cpprop: bool = False,
    gbw_override: bool = False,
    name: str = "props.mfwn",
) -> None:
    """
    Function to write a bash script that runs multiwfn on a given input file.
    Args:
        out_folder(str): folder to write the bash script to
        read_file(str): file to read from
        multi_wfn_cmd(str): command to run multiwfn
        multiwfn_input_file(str): input file for multiwfn
        convert_gbw(bool): whether to convert the input file to a gbw file
        mv_cpprop(bool): whether to move the cpprop file to the output folder
        overwrite(bool): whether to overwrite the file if it already exists
        name(str): name of the bash script
        gbw_override(bool): whether to override the gbw file location
    """

    out_file = str(Path.home().joinpath(out_folder, name))
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    completed_tf = os.path.exists(out_file) and os.path.getsize(out_file) > 0

    if (not completed_tf) or overwrite:
        with open(out_file, "w") as f:
            f.write("#!/bin/bash\n")
            # Without pipefail the script's exit status is tee's (always 0), so
            # a Multiwfn crash or OOM kill was logged as "Completed" upstream.
            f.write("set -o pipefail\n")
            if convert_gbw:
                f.write("orca_2mkl '" + str(Path.home().joinpath(out_folder)) + "'\n")

            multiwfn_input_file_root = multiwfn_input_file.split("/")[-1].split(".")[0]
            bare_file = read_file.split("/")[-1]
            
            if gbw_override: 
                # 
                gbw_loc = bare_file
            else: 
                gbw_loc = str(Path.home().joinpath(out_folder, read_file))

            
            f.write(
                "{} ".format(multi_wfn_cmd)  # multiwfn command
                + "'" + gbw_loc + "'"  # wfn/gbw file
                + " < '{}' | tee ".format(multiwfn_input_file)  # multiwfn input file
                + "'" + str(
                    Path.home().joinpath(
                        out_folder, "{}.out".format(multiwfn_input_file_root)
                    )
                ) + "'"  # output file
                + "\n"
            )

            if mv_cpprop:
                f.write(
                    "mv CPprop.txt "
                    + "'" + str(Path.home().joinpath(out_folder, "CPprop.txt")) + "'"
                    + "\n"
                )

        st = os.stat(out_file)
        os.chmod(out_file, st.st_mode | stat.S_IEXEC)


def create_jobs(
    folder: str,
    multiwfn_cmd: str,
    orca_2mkl_cmd: str,
    separate: bool = False,
    debug: bool = True,
    logger: Optional[logging.Logger] = None,
    full_set: int = 0,
    patch_path: bool = False,
    wfx: bool = False,
    exhaustive_qtaim: bool = False,
) -> None:
    """
    Create job files for multiwfn analysis
    Takes:
        folder(str): folder to create jobs in
        multiwfn_cmd(str): command to run multiwfn
        orca_2mkl_cmd(str): command to run orca_2mkl
        separate(bool): whether to separate the analysis into different files
        overwrite(bool): whether to overwrite the output files
        orca_6(bool): whether calc is from orca6
        logger(logging.Logger): logger to log messages
        full_set(int): whether to use full set of analysis (1) or minimal (0)
        patch_path(bool): whether to patch the pathing in the multiwfn input files
        wfx(bool): whether to use .wfx format instead of .wfn for conversion

    """
    if logger is None:
        logger = logging.getLogger("gbw_analysis")

    if separate:
        routine_list = ORDER_OF_OPERATIONS_separate.copy()
        # routine_list = ["charge_separate"]
    else:
        routine_list = ORDER_OF_OPERATIONS
        # routine_list = ["qtaim", "bond", "charge"]

    if debug:  # just run qtaim in debug mode
        routine_list = ["qtaim"]

    # Determine target wavefunction extension based on --wfx flag
    wf_ext = ".wfx" if wfx else ".wfn"

    wf_present = False
    file_wf_search = None
    for file in os.listdir(folder):
        if file.endswith((".wfn", ".wfx")):
            wf_present = True
            file_wf_search = os.path.join(folder, file)

    bool_gbw = False
    file_read = None

    for file in os.listdir(folder):
        if file.endswith(".gbw"):
            bool_gbw = True
            file_gbw = os.path.join(folder, file)
            file_wf = file.replace(".gbw", wf_ext)
            # if there is a wfn/wfx, rename to match gbw prefix
            if wf_present:
                if file_wf not in os.listdir(folder):
                    logger.info(f"Renaming wavefunction file to: {file_wf}")
                    os.rename(
                        file_wf_search,
                        os.path.join(folder, file_wf),
                    )

            file_molden = file.replace(".gbw", ".molden.input")
            file_molden = os.path.join(folder, file_molden)
            file_read = os.path.join(folder, file_wf)

        if file.endswith((".wfn", ".wfx")):
            file_read = os.path.join(folder, file)

    if not wf_present and bool_gbw:
        logger.info(f"file_gbw: {file_gbw}")
        logger.info(f"out folder: {folder}")
        logger.info("wavefunction file not found - writing conversion script")
        try:
            write_conversion(
                out_folder=folder,
                read_file=file_gbw,
                overwrite=True,
                name="convert.in",
                orca_2mkl_cmd=orca_2mkl_cmd,
            )
            routine_list = ["convert"] + routine_list

        except Exception as e:
            logger.error(f"Error writing conversion script: {e}")

    job_dict = {}

    # print("folder: {}".format(folder))
    for routine in routine_list:
        try:

            if routine == "qtaim":
                job_dict["qtaim"] = os.path.join(folder, "qtaim.txt")
                # write qtaim data file
                with open(os.path.join(folder, "qtaim.txt"), "w") as f:
                    data = qtaim_data(exhaustive=exhaustive_qtaim)
                    f.write(data)

            elif routine == "fuzzy_full":
                # job_dict["fuzzy_full"] = os.path.join(folder, "fuzzy_full.txt")
                # with open(os.path.join(folder, "fuzzy_full.txt"), "w") as f:
                spin_tf = check_spin(folder)
                print("spin_tf: {}".format(spin_tf))
                fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
                for key, value in fuzzy_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)
                    # f.write(data)

            elif routine == "bond":
                job_dict["bond"] = os.path.join(folder, "bond.txt")
                with open(os.path.join(folder, "bond.txt"), "w") as f:
                    data = bond_order_data()
                    f.write(data)

            elif routine == "bond_separate":
                bond_dict = bond_order_dict(full_set=full_set)
                for key, value in bond_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)

            elif routine == "charge":
                job_dict["charge"] = os.path.join(folder, "charge.txt")
                with open(os.path.join(folder, "charge.txt"), "w") as f:
                    data = charge_data()
                    f.write(data)

            elif routine == "charge_separate":
                charge_dict = charge_data_dict(full_set=full_set)
                for key, value in charge_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)

            elif routine == "other":
                job_dict["other"] = os.path.join(folder, "other.txt")
                with open(os.path.join(folder, "other.txt"), "w") as f:
                    data = other_data()
                    f.write(data)

            elif routine == "other_separate":
                other_dict = other_data_dict(full_set=full_set)
                for key, value in other_dict.items():
                    job_dict[key] = os.path.join(folder, "{}.txt".format(key))
                    with open(os.path.join(folder, "{}.txt".format(key)), "w") as f:
                        f.write(value)

            elif routine == "convert":
                job_dict["convert"] = os.path.join(folder, "convert.txt")
                with open(os.path.join(folder, "convert.txt"), "w") as f:
                    file_wf_target = file_gbw.replace(".gbw", wf_ext)
                    file_wf_bare = file_wf_target.split("/")[-1]
                    if wfx:
                        # Multiwfn menu 100 -> 2 -> 4: export to .wfx
                        data = "100\n2\n4\n{}\n0\nq\n".format(file_wf_bare)
                    else:
                        # Multiwfn menu 100 -> 2 -> 5: export to .wfn
                        data = "100\n2\n5\n{}\n0\nq\n".format(file_wf_bare)
                    f.write(data)

            # else:
            #    logger.warning(f"Routine not recognized: {routine}")

        except Exception as e:
            logger.error(f"Error creating job for routine '{routine}': {e}")

    # print(job_dict)
    for key, value in job_dict.items():
        try:
            # if key == "qtaim":
            #    mv_cpprop = True
            # else:
            mv_cpprop = False

            if key == "charge_separate":
                charge_dict = charge_data_dict(full_set=full_set)
                for key, value in charge_dict.items():
                    write_multiwfn_exe(
                        out_folder=folder,
                        read_file=file_read,
                        multi_wfn_cmd=multiwfn_cmd,
                        multiwfn_input_file=value,
                        convert_gbw=False,
                        overwrite=True,
                        name="props_{}.mfwn".format(key),
                        mv_cpprop=mv_cpprop,
                        gbw_override=patch_path,

                    )

            elif key == "bond_separate":
                bond_dict = bond_order_dict(full_set=full_set)
                for key, value in bond_dict.items():
                    write_multiwfn_exe(
                        out_folder=folder,
                        read_file=file_read,
                        multi_wfn_cmd=multiwfn_cmd,
                        multiwfn_input_file=value,
                        convert_gbw=False,
                        overwrite=True,
                        name="props_{}.mfwn".format(key),
                        mv_cpprop=mv_cpprop,
                        gbw_override=patch_path,

                    )

            elif key == "other_separate":
                other_dict = other_data_dict(full_set=full_set)
                for key, value in other_dict.items():
                    write_multiwfn_exe(
                        out_folder=folder,
                        read_file=file_read,
                        multi_wfn_cmd=multiwfn_cmd,
                        multiwfn_input_file=value,
                        convert_gbw=False,
                        overwrite=True,
                        name="props_{}.mfwn".format(key),
                        mv_cpprop=mv_cpprop,
                        gbw_override=patch_path,
                    )
            
            elif key == "convert":
                # use ./ b/c multiwfn throws a fit sometimes
                write_multiwfn_exe(
                    out_folder=folder,
                    gbw_override=True,
                    read_file=file_molden,
                    multi_wfn_cmd=multiwfn_cmd,
                    multiwfn_input_file=value,
                    convert_gbw=False,
                    overwrite=True,
                    name="props_{}.mfwn".format(key),
                    mv_cpprop=mv_cpprop,
                )

            else:
                # print("key: {}".format(key))
                write_multiwfn_exe(
                    out_folder=folder,
                    read_file=file_read,
                    multi_wfn_cmd=multiwfn_cmd,
                    multiwfn_input_file=value,
                    convert_gbw=False,
                    overwrite=True,
                    name="props_{}.mfwn".format(key),
                    mv_cpprop=mv_cpprop,
                    gbw_override=patch_path,
                )

            logger.info(f"Created execution script for {key}")

        except Exception as e:
            logger.error(f"Error creating execution script for {key}: {e}")


def run_jobs(
    folder: str,
    separate: bool = False,
    orca_6: bool = True,
    restart: bool = False,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    prof_mem: bool = False,
    full_set: int = 0,
    move_results: bool = False,
    clean_jobs_tf: bool = False,
    subprocess_env: Optional[dict] = None,
    check_bcp_count: bool = False,
    bcp_tolerance: int = 2,
    require_qtaim_provenance: bool = False,
) -> None:
    """
    Run conversion and multiwfn jobs
    Takes:
        folder(str): folder to run jobs in
        separate(bool): whether to separate the analysis into different files
        logger(logging.Logger): logger to log messages
        prof_mem(bool): whether to profile memory usage
        full_set(int): whether to use full set of analysis (1) or minimal (0)
        move_results(bool): whether to move results to a separate folder
        clean_jobs_tf(bool): whether to remove job files after running

    """
    if logger is None:
        logger = logging.getLogger("gbw_analysis")

    spin_tf = check_spin(folder)

    if separate:
        order_of_operations = ORDER_OF_OPERATIONS_separate.copy()
        charge_dict = charge_data_dict(full_set=full_set)
        bond_dict = bond_order_dict(full_set=full_set)
        fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
        other_dict = other_data_dict(full_set=full_set)
        [order_of_operations.append(i) for i in charge_dict.keys()]
        [order_of_operations.append(i) for i in bond_dict.keys()]
        [order_of_operations.append(i) for i in fuzzy_dict.keys()]
        [order_of_operations.append(i) for i in other_dict.keys()]

        # Filter phantom keys that have no .mfwn files (used by create_jobs only)
        _phantom_keys = {"charge_separate", "bond_separate", "other_separate", "fuzzy_full"}
        order_of_operations = [o for o in order_of_operations if o not in _phantom_keys]
    else:
        order_of_operations = ORDER_OF_OPERATIONS
        charge_dict = {}
        bond_dict = {}
        fuzzy_dict = {}
        other_dict = {}

    if debug:
        order_of_operations = ["qtaim"]

    wf_present = False
    conv_file = None
    for file in os.listdir(folder):
        if file.endswith((".wfn", ".wfx")):
            wf_present = True
        if file.endswith("convert.in"):
            conv_file = os.path.join(folder, file)

    # run conversion script if wavefunction file is not present
    if not wf_present:
        logger.info("Running conversion script")
        if conv_file is None:
            logger.error("No conversion script (convert.in) found in folder.")
            return

        # run conversion script using subprocess with explicit cwd
        try:
            subprocess.run(["bash", conv_file], cwd=folder, check=True, env=subprocess_env)
        except Exception as e:
            logger.error(f"Error running conversion script: {e}")

        for file in os.listdir(folder):
            if file.endswith(".molden.input"):
                molden_file = os.path.join(folder, file)
            if file.endswith("orca.out") or file.endswith("output.out"):
                orca_out = os.path.join(folder, file)
        if not orca_6:
            try:
                dict_ecp = pull_ecp_dict(orca_out)
                overwrite_molden_w_ecp(molden_file, dict_ecp)
            except Exception as e:
                logger.error(f"Error replacing ECP values in molden file: {e}")

        order_of_operations = ["convert"] + order_of_operations
        # conversion script should read in .molden file and use multiwfn to convert to .wfn

    # create a json file to store job status

    # Maps each separate-mode operation → (compiled JSON filename, key to check).
    # key=None for other_dict ops because other.json is built via dict.update()
    # with scalar fields, not keyed by operation name.
    # In non-separate mode all four dicts are empty so _compiled_map is empty;
    # the {order}.json file check in the loop covers those ops by name.
    _compiled_map: dict = {}
    for _op in charge_dict:
        # charge.json stores {"<op>": {"charge": {...}, "dipole": [...], ...}}
        # so check the nested "charge" sub-key, not just the op-level dict
        _compiled_map[_op] = ("charge.json", _op, "charge")
    for _op in bond_dict:
        _compiled_map[_op] = ("bond.json", _op)
    for _op in fuzzy_dict:
        _compiled_map[_op] = ("fuzzy_full.json", _op)
    for _op in other_dict:
        _compiled_map[_op] = ("other.json", None)

    timings = {}
    # On restart, timings may be in generator/ (previous completed run) or
    # job root (interrupted run before move_results_to_folder ran).
    # Check generator/ first, fall back to job root.
    gen_timings = os.path.join(folder, "generator", "timings.json")
    root_timings = os.path.join(folder, "timings.json")
    if os.path.exists(gen_timings):
        timings_read_path = gen_timings
    else:
        timings_read_path = root_timings

    if os.path.exists(timings_read_path):
        if os.path.getsize(timings_read_path) > 0:
            try:
                with open(timings_read_path, "r") as f:
                    timings = json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    "Corrupted timings.json at %s -- starting fresh",
                    timings_read_path,
                )
                timings = {}

    # Lazy log-scrape cache for backfill_skip_timing -- only the first
    # missing-timing skip pays the parse cost.
    _log_timings_cache: list = []  # 0- or 1-element box for lazy init

    # n_atoms enables length-aware completeness checks in _compiled_data_present.
    # Without it a partial chelpg entry (e.g. 30 of 142 atoms) would be treated
    # as "data verified" and skipped forever while validation keeps failing.
    n_atoms_for_skip: Optional[int] = None
    charge_for_skip: Optional[int] = None
    dft_dict = None
    try:
        from qtaim_gen.source.utils.validation import (
            get_charge_spin_n_atoms_from_folder,
        )
        dft_dict = get_charge_spin_n_atoms_from_folder(folder, logger=logger)
        if dft_dict and dft_dict.get("mol"):
            n_atoms_for_skip = len(dft_dict["mol"])
            if dft_dict.get("charge") is not None:
                charge_for_skip = int(dft_dict["charge"])
    except Exception as e:
        logger.warning(
            "Could not determine n_atoms for skip-completeness check: %s", e
        )
    _fuzzy_routine_set = set(fuzzy_dict.keys()) if separate else set()

    # A wavefunction exported from a truncated molden (orca_2mkl killed
    # mid-write) loads cleanly in Multiwfn with a fraction of the orbitals and
    # every downstream step then computes garbage that still looks
    # "substantive". Two protein_core jobs ran 108-atom systems on a 53-orbital
    # wfx for two months. Check the electron count once, up front.
    if restart and dft_dict:
        wf_path = _wavefunction_path(folder)
        expected_e = _expected_electrons(dft_dict)
        observed_e = _wavefunction_electrons(wf_path) if wf_path else None
        if (
            expected_e is not None
            and observed_e is not None
            and abs(observed_e - expected_e) > 0.5
        ):
            logger.error(
                "%s declares %.1f electrons but the input implies %d; discarding "
                "it and every step output, rerunning all steps",
                wf_path,
                observed_e,
                expected_e,
            )
            _discard_wavefunction_and_step_outputs(folder, order_of_operations, logger)
            restart = False

    for order in order_of_operations:
        # Per-sub-job restart: data presence is the primary skip signal; timing
        # is secondary. This handles cases where timings.json was reset/corrupted
        # or a crash occurred between the mfwn script finishing and the timing write.
        if restart:
            has_files = _has_usable_step_output(
                folder,
                order,
                n_atoms=n_atoms_for_skip,
                check_bcp_count=check_bcp_count,
                bcp_tolerance=bcp_tolerance,
                require_qtaim_provenance=require_qtaim_provenance,
                charge=charge_for_skip,
                fuzzy_routines=_fuzzy_routine_set,
            )
            if has_files or _compiled_data_present(
                folder, order, _compiled_map,
                n_atoms=n_atoms_for_skip,
                fuzzy_routines=_fuzzy_routine_set,
            ):
                has_positive_timing = (
                    order in timings
                    and isinstance(timings[order], (int, float))
                    and timings[order] > 0
                )
                if has_positive_timing:
                    _timing_str = f", timing={timings[order]:.2f}s"
                else:
                    if not _log_timings_cache:
                        _log_timings_cache.append(
                            _parse_timings_from_log(
                                os.path.join(folder, "gbw_analysis.log")
                            )
                        )
                    backfill_skip_timing(timings, order, _log_timings_cache[0], logger)
                    atomic_json_write(
                        os.path.join(folder, "timings.json"), timings
                    )
                    _timing_str = f", timing backfilled={timings[order]:.2f}s"
                logger.info(
                    f"Skipping {order} in {folder}: data verified{_timing_str}"
                )
                continue
            if order in timings and timings[order] > 0:
                logger.warning(
                    f"Timing present for '{order}' in {folder} but output data not found — re-running"
                )

        if prof_mem:
            memory = {}

        mfwn_file = os.path.join(folder, "props_{}.mfwn".format(order))

        # Precondition: non-convert multiwfn sub-jobs need a wavefunction file.
        # Surfacing it here beats a downstream KeyError in parse_multiwfn.
        if order != "convert" and not _wavefunction_present(folder):
            logger.error(
                f"No orca.wfn or orca.wfx in {folder}; cannot run {order}."
            )
            timings[order] = -1
            continue

        logger.info(f"Running {mfwn_file}")
        start = time.time()
        step_failure = None
        try:
            # explicit cwd so the run does not depend on the process CWD
            subprocess.run(["bash", mfwn_file], cwd=folder, check=True, env=subprocess_env)
        except subprocess.CalledProcessError as e:
            step_failure = f"exit status {e.returncode}"
        except Exception as e:
            step_failure = str(e)
        elapsed = time.time() - start

        if clean_jobs_tf and os.path.exists(mfwn_file):
            logger.info(f"Removing file: {mfwn_file}")
            os.remove(mfwn_file)

        if step_failure is None:
            timings[order] = elapsed
            logger.info(f"Completed {order} in {elapsed:.2f} seconds")
        else:
            # A -1 timing fails validate_timing_dict, so the step is redone on
            # the next pass instead of its partial .out being parsed as a result.
            timings[order] = -1
            logger.error(
                f"Failed {order} after {elapsed:.2f} seconds ({step_failure}): {mfwn_file}"
            )

        # save timings to job root — move_results_to_folder() relocates at the end
        try:
            atomic_json_write(os.path.join(folder, "timings.json"), timings)
            logger.info(f"Saved timings.json in {folder}")

            # Heartbeat: touch lock file to keep mtime fresh for stale detection
            _lockfile = os.path.join(folder, ".processing.lock")
            if os.path.exists(_lockfile):
                try:
                    os.utime(_lockfile, None)
                except OSError:
                    pass

            if prof_mem:
                atomic_json_write(os.path.join(folder, "memory.json"), memory)

        except Exception as e:
            logger.error(f"Error saving timings.json: {e}")


def _parse_routine_out(
    routine: str, path: str, fuzzy_routines: Optional[set] = None
) -> Optional[dict]:
    """Parse one Multiwfn `<routine>.out` into the dict parse_multiwfn writes
    to `<routine>.json`. Returns None for a routine with no parser. Raises
    whatever the underlying parser raises, so callers decide what a failure
    means (parse_multiwfn logs it; the restart gate treats it as "rerun").
    """
    fuzzy_routines = fuzzy_routines or set()
    if routine == "fuzzy_full":
        return parse_fuzzy_doc(path)
    if routine in ("fuzzy_bond", "fuzzy"):
        return parse_bond_order_fuzzy(path)
    if routine == "ibsi_bond":
        return parse_bond_order_ibsi(path)
    if routine == "laplacian_bond":
        return parse_bond_order_laplace(path)
    if routine == "other":
        return parse_other_doc(path)
    if routine == "other_esp":
        return parse_other_doc_esp(path, ind_surface_prefix="ESP")
    if routine == "other_alie":
        return parse_other_doc_esp(path, ind_surface_prefix="ALIE")
    if routine == "other_geometry":
        return parse_other_doc_geometry(path)
    if routine == "charge":
        charges, atomic_dipoles, dipole_info = parse_charge_doc(path)
        return {"charge": charges, "dipole": dipole_info, "atomic_dipole": atomic_dipoles}
    if routine in ("hirshfeld", "vdd", "cm5"):
        charges, dipole_info = parse_charge_base(path, corrected=False)
        return {"charge": charges, "dipole": dipole_info}
    if routine == "mbis":
        charges = parse_charge_base(path, corrected=False, dipole=False)
        return {"charge": charges}
    if routine == "bader":
        charges, spin_info = parse_charge_doc_bader(path)
        return {"charge": charges, "spin": spin_info}
    if routine == "adch":
        charges, atomic_dipoles, dipole_info = parse_charge_doc_adch(path)
        return {"charge": charges, "dipole": dipole_info, "atomic_dipole": atomic_dipoles}
    if routine == "becke":
        charges, atomic_dipoles, dipole_info = parse_charge_becke(path)
        return {"charge": charges, "dipole": dipole_info, "atomic_dipole": atomic_dipoles}
    if routine == "chelpg":
        return {"charge": parse_charge_chelpg(path)}
    if routine in fuzzy_routines:
        return parse_fuzzy_real_space(path)
    return None


def parse_multiwfn(
    folder: str,
    separate: bool = False,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    full_set: int = 0,
) -> None:
    """
    Parse multiwfn output files to jsons and save them in folder
    Takes:
        folder(str): folder to parse
        separate(bool): whether to separate the analysis into different files
        debug(bool): whether to run a minimal set of jobs
        return_dicts(bool): return results as well as writing
    """
    # if return_dicts:
    #    compiled_dicts = {}

    if separate:
        routine_list = ORDER_OF_OPERATIONS_separate.copy()
        charge_dict = charge_data_dict(full_set=full_set)
        bond_dict = bond_order_dict(full_set=full_set)
        spin_tf = check_spin(folder)
        # print("spin_tf: {}".format(spin_tf))
        fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
        other_dict = other_data_dict(full_set=full_set)
        [routine_list.append(i) for i in charge_dict.keys()]
        [routine_list.append(i) for i in bond_dict.keys()]
        [routine_list.append(i) for i in fuzzy_dict.keys()]
        [routine_list.append(i) for i in other_dict.keys()]
        fuzzy_routines = set(fuzzy_dict.keys())

    else:
        routine_list = ORDER_OF_OPERATIONS
        fuzzy_routines = set()

    if debug:
        routine_list = ["qtaim"]

    for file in os.listdir(folder):
        if file.endswith(".out"):
            file_full_path = os.path.join(folder, file)
            for routine in routine_list:
                # Exact filename match: substring matching double-parsed
                # becke_fuzzy_density.out under both 'becke' and
                # 'becke_fuzzy_density' (same for mbis/*_spin variants),
                # writing wrong-parser output that only list order corrected.
                if file == routine + ".out":
                    if routine == "qtaim":
                        # qtaim.out is provenance only; qtaim.json is built
                        # from CPprop.txt below.
                        continue
                    json_file = file_full_path.replace(".out", ".json")
                    try:
                        data = _parse_routine_out(routine, file_full_path, fuzzy_routines)
                    except Exception as e:
                        logger.error(
                            f"Error parsing {routine} in {file_full_path}: {e}"
                        )
                        continue
                    if data is None:
                        logger.warning(
                            f"Unknown routine '{routine}' in file {file_full_path}"
                        )
                        continue
                    atomic_json_write(json_file, data)
                    logger.info(f"Parsed {routine} output to {json_file}")

        elif "CPprop.txt" in file and "qtaim" in routine_list:
            json_file = os.path.join(folder, "qtaim.json")
            cp_prop_path = os.path.join(folder, file)

            # A walltime kill during the CPprop.txt export leaves a partial
            # file. Parsing it here (the "reparse before rerun" path) replaced
            # a complete qtaim.json with a truncated one on every pass.
            from qtaim_gen.source.utils.validation import qtaim_run_status

            _qstat = qtaim_run_status(folder)
            if _qstat["have_qtaim_out"] and not _qstat["export_done"]:
                logger.error(
                    f"Skipping qtaim parse in {folder}: qtaim.out shows the "
                    f"CPprop.txt export never completed, so CPprop.txt is partial"
                )
                continue

            inp_loc = None
            inp_orca = None
            for file2 in os.listdir(folder):
                if file2.endswith(".inp"):
                    inp_loc = os.path.join(folder, file2)
                    inp_orca = True

                if file2.endswith("input.in"):
                    inp_loc = os.path.join(folder, file2)
                    inp_orca = False

            try:
                qtaim_dict = parse_qtaim(
                    cprop_file=cp_prop_path, inp_loc=inp_loc, orca_tf=inp_orca
                )
                if qtaim_dict is None or qtaim_dict == {}:
                    logger.error(
                        f"QTAIM parsing returned empty dictionary for {folder}"
                    )
                    continue

                atomic_json_write(json_file, qtaim_dict)
                # if return_dicts:
                #    compiled_dicts["qtaim"] = qtaim_dict
                logger.info(f"Parsed qtaim output to {json_file}")

            except Exception as e:
                logger.error(f"Error parsing qtaim: {e}")

    if separate:
        charge_routines = list(charge_dict.keys())
        bond_routines = list(bond_dict.keys())
        fuzzy_routines = list(fuzzy_data(spin=spin_tf, full_set=full_set).keys())
        other_routines = list(other_dict.keys())
        charge_dict_compiled = {}
        bond_dict_compiled = {}
        fuzzy_dict_compiled = {}
        other_dict_compiled = {}

        # remove double for loops by combining
        directory_files = os.listdir(folder)
        combined_routines = charge_routines + bond_routines + fuzzy_routines + other_routines
        for routine in combined_routines:
            # json name
            file = routine + ".json"
            if file not in directory_files:
                continue
            if routine != file.split(".json")[0]:
                continue
            full_path = os.path.join(folder, file)
            # Zero-byte or malformed intermediate (e.g. multiwfn crashed
            # between opening the json and writing content) must not kill
            # the compile loop -- the routine is just treated as missing.
            try:
                if os.path.getsize(full_path) == 0:
                    logger.warning("Skipping empty intermediate %s", full_path)
                    continue
                with open(full_path, "r") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping unreadable intermediate %s: %s", full_path, e)
                continue
            if routine in charge_routines:
                charge_dict_compiled[routine] = payload
            elif routine in bond_routines:
                bond_dict_compiled[routine] = payload
            elif routine in fuzzy_routines:
                try:
                    fuzzy_dict_compiled[routine] = payload[routine]
                except (KeyError, TypeError) as e:
                    logger.warning(
                        "Fuzzy intermediate %s missing top-level '%s' key: %s",
                        full_path, routine, e,
                    )
                    continue
            elif routine in other_routines:
                if isinstance(payload, dict):
                    other_dict_compiled.update(payload)
                            
                    # remove the file
                    # if os.path.exists(os.path.join(folder, file)):
                    #    logger.info(f"Removing file: {file}")
                    #    os.remove(os.path.join(folder, file))

        if charge_dict_compiled:
            atomic_json_write(os.path.join(folder, "charge.json"), charge_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["charge"] = charge_dict_compiled
            logger.info("Compiled charge.json")

        if bond_dict_compiled:
            atomic_json_write(os.path.join(folder, "bond.json"), bond_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["bond"] = bond_dict_compiled
            logger.info("Compiled bond.json")

        if fuzzy_dict_compiled:
            atomic_json_write(os.path.join(folder, "fuzzy_full.json"), fuzzy_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["fuzzy_full"] = fuzzy_dict_compiled
            logger.info("Compiled fuzzy_full.json")

        if other_dict_compiled:
            atomic_json_write(os.path.join(folder, "other.json"), other_dict_compiled)
            # if return_dicts:
            #    compiled_dicts["other"] = other_dict_compiled
            logger.info("Compiled other.json")

    # clean individual json files if separate AFTER compiled jsons writtent to ensure
    # data isn't lost if error occurs
    if separate:
        for routine in combined_routines:
            file = routine + ".json"
            if file in directory_files:
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")


def clean_jobs(
    folder: str,
    separate: bool = False,
    logger: Optional[logging.Logger] = None,
    full_set: int = 0,
    move_results: bool = True,
) -> None:
    """
    Clean up the mess of files created by the analysis
    Takes:
        folder(str): folder to clean
        separate(bool): whether to separate the analysis into different files
        logger(logging.Logger): logger to log messages
    Removes:
        - all .mfwn files
        - all .txt files that are not in the order of operations
        - all .out files that are not in the order of operations
        - all molden.input files
        - all convert.in files
    """
    if logger is None:
        logger = logging.getLogger("gbw_analysis")
    logger.info("Cleaning up jobs in folder: {}".format(folder))

    if separate:
        order_of_operations = ORDER_OF_OPERATIONS_separate.copy()
        charge_dict = charge_data_dict(full_set=full_set)
        [order_of_operations.append(i) for i in charge_dict.keys()]
        bond_dict = bond_order_dict(full_set=full_set)
        [order_of_operations.append(i) for i in bond_dict.keys()]
        spin_tf = check_spin(folder)
        fuzzy_dict = fuzzy_data(spin=spin_tf, full_set=full_set)
        other_dict = other_data_dict(full_set=full_set)
        [order_of_operations.append(i) for i in fuzzy_dict.keys()]
        [order_of_operations.append(i) for i in other_dict.keys()]
    else:
        order_of_operations = ORDER_OF_OPERATIONS

    txt_files = []
    txt_files = [i + ".txt" for i in order_of_operations] + ["convert.txt"]
    txt_files += [i + ".out" for i in order_of_operations] + ["convert.out"]
    txt_files += [i for i in ["settings.ini", "convert.out", "convert.txt"]]
    zip_file_out = os.path.join(folder, "out_files.zip")
    # print all jobs ending in .mfwn
    for file in os.listdir(folder):
        try:
            if file.endswith(".mfwn"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith(".txt"):
                if file in txt_files:
                    os.remove(os.path.join(folder, file))
                    logger.info(f"Removed {file}")                   
            if file.endswith(".molden.input"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith("settings.ini"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith("convert.in"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            # NOT CPprop.txt: it is collected into out_files.zip below and
            # deleted only after the zip is safely merged. Deleting it here made
            # the zip's CPprop.txt clause dead code, which is why no archived
            # job retained the one file needed to diagnose a lost critical
            # point (the per-CP property blocks live nowhere else -- qtaim.out
            # carries only the count).
            if file.endswith("fuzzy_full.txt"):
                os.remove(os.path.join(folder, file))
                logger.info(f"Removed {file}")
            if file.endswith((".wfn", ".wfx")):
                # if there is a gbw in the folder remove wfn/wfx
                if any(f.endswith(".gbw") for f in os.listdir(folder)):
                    os.remove(os.path.join(folder, file))
                    logger.info(f"Removed {file}")

        except Exception as e:
            logger.info(f"Couldn't rm file {file}: {e}")

    # zip all out files - collect first, delete only after zip is safely merged
    files_to_zip = [
        f for f in os.listdir(folder)
        if (f.endswith(".out") and f != "orca.out") or f.endswith("CPprop.txt")
    ]
    successfully_zipped = []
    with zipfile.ZipFile(zip_file_out, "w") as zipf:
        for file in files_to_zip:
            try:
                zipf.write(os.path.join(folder, file), arcname=file)
                successfully_zipped.append(file)
                logger.info(f"Zipped {file}")
            except Exception as e:
                logger.info(f"Couldn't zip {file}: {e}")

    if move_results:
        results_folder = os.path.join(folder, "generator")
        merge_zip_into(
            zip_file_out,
            os.path.join(results_folder, "out_files.zip"),
            logger=logger,
        )

    # delete only files that were successfully written to the zip
    for file in successfully_zipped:
        fp = os.path.join(folder, file)
        if os.path.exists(fp):
            try:
                os.remove(fp)
                logger.info(f"Removed {file} after zip")
            except Exception as e:
                logger.info(f"Couldn't remove {file}: {e}")


# Matches `Completed <key> in <s> seconds` lines emitted by
# qtaim_gen.source.core.omol.gbw_analysis (see logger.info call there).
_TIMING_LOG_PATTERN = re.compile(
    r"Completed\s+(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s+in\s+(?P<seconds>[-+]?\d*\.?\d+)\s+seconds"
)


def _parse_timings_from_log(log_path: str) -> dict:
    """Extract `Completed <key> in <seconds> seconds` entries from a log file.

    Returns a dict {key: seconds_float}. Last-occurrence wins (most recent run).
    Missing/unreadable log returns {}.
    """
    results: dict = {}
    if not os.path.isfile(log_path):
        return results
    try:
        with open(log_path, "r") as f:
            for line in f:
                m = _TIMING_LOG_PATTERN.search(line)
                if m:
                    try:
                        results[m.group("key")] = float(m.group("seconds"))
                    except ValueError:
                        continue
    except OSError:
        return {}
    return results


def backfill_skip_timing(
    timings: dict,
    order: str,
    log_timings: dict,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Stamp a backfilled timing entry for a restart-skipped routine.

    Used when the routine's data files exist on disk but its timing key is
    absent from timings.json -- without this, the next validate_timing_dict
    call fails on the missing key and the folder loops on HPC.

    Strategy:
      1. If `log_timings[order] > 0`, use that scraped value.
      2. Otherwise stamp TIMING_PLACEHOLDER.
    In both cases, record provenance under TIMINGS_PATCHED_KEY so the
    validator accepts the sentinel and downstream aggregates can filter it.
    Mutates *timings* in place. Caller is responsible for persisting.
    """
    log_val = log_timings.get(order)
    if isinstance(log_val, (int, float)) and log_val > 0:
        timings[order] = log_val
        source, value = "log", log_val
    else:
        timings[order] = TIMING_PLACEHOLDER
        source, value = "placeholder", TIMING_PLACEHOLDER
    marker = timings.get(TIMINGS_PATCHED_KEY)
    if not isinstance(marker, dict):
        marker = {}
    marker[order] = {"source": source, "value": value}
    timings[TIMINGS_PATCHED_KEY] = marker
    if logger:
        logger.info(
            "Backfilled missing timing for '%s' = %.2f (%s)", order, value, source,
        )


def patch_timings_from_log(
    folder: str,
    full_set: int = 0,
    spin_tf: bool = False,
    move_results: bool = True,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Fill missing keys in timings.json by recovering values from gbw_analysis.log.

    Strategy:
      1. Load existing timings from generator/timings.json (if move_results) or
         folder/timings.json.
      2. Parse `Completed <key> in <s> seconds` lines from gbw_analysis.log.
      3. For each expected key for the given full_set/spin_tf, if missing from
         timings dict, fill with log value if found, otherwise -1.0 sentinel.
         Validation accepts -1.0 only for keys listed in `_timings_patched`,
         so synthetic timings remain loudly flagged in downstream aggregates.
      4. Stamp a `_timings_patched` provenance marker listing patched keys and
         their source (log vs placeholder) so downstream consumers can filter
         out synthetic timings from aggregates.
      5. Atomic-write back to the same file.

    Returns True if any key was patched, False otherwise (including when file
    does not exist).
    """
    if move_results:
        timings_path = os.path.join(folder, "generator", "timings.json")
    else:
        timings_path = os.path.join(folder, "timings.json")

    if not os.path.isfile(timings_path):
        if logger:
            logger.warning("patch_timings: %s not found, skipping", timings_path)
        return False

    try:
        with open(timings_path, "r") as f:
            timings = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        if logger:
            logger.warning("patch_timings: cannot read %s: %s", timings_path, e)
        return False

    log_path = os.path.join(folder, "gbw_analysis.log")
    log_timings = _parse_timings_from_log(log_path)

    expected, _ = get_expected_timing_keys(full_set=full_set, spin_tf=spin_tf)
    patched: dict = {}
    for key in expected:
        # validate_timing_dict accepts 'other' or 'other_alie'; if asked for
        # 'other' but only 'other_alie' is present (or vice versa), don't patch
        if key == "other" and ("other" in timings or "other_alie" in timings):
            continue
        existing = timings.get(key)
        if isinstance(existing, (int, float)) and existing > 0:
            continue
        # 'other' is the canonical expected name; record it under 'other_alie'
        # since that's the actual key written by the analysis pipeline
        write_key = "other_alie" if key == "other" else key
        if key in log_timings and log_timings[key] > 0:
            timings[write_key] = log_timings[key]
            patched[write_key] = {"source": "log", "value": log_timings[key]}
        else:
            timings[write_key] = TIMING_PLACEHOLDER
            patched[write_key] = {"source": "placeholder", "value": TIMING_PLACEHOLDER}

    if not patched:
        if logger:
            logger.info("patch_timings: nothing to patch in %s", timings_path)
        return False

    # Provenance marker so W&B / tracking_db / aggregates can filter synthetic timings
    existing_marker = timings.get(TIMINGS_PATCHED_KEY) or {}
    if not isinstance(existing_marker, dict):
        existing_marker = {}
    existing_marker.update(patched)
    timings[TIMINGS_PATCHED_KEY] = existing_marker

    atomic_json_write(timings_path, timings)
    if logger:
        for key, info in patched.items():
            logger.info(
                "patch_timings: %s = %.2f (%s)", key, info["value"], info["source"]
            )
        logger.info(
            "patch_timings: wrote %d patched key(s) to %s", len(patched), timings_path
        )
    return True



def setup_logger(folder: str, name: str = "gbw_analysis") -> logging.Logger:
    logger = logging.getLogger(f"{name}-{folder}")
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(folder, "gbw_analysis.log"))
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def move_results_to_folder(
    folder: str, logger: logging.Logger, clean: bool = True
) -> None:
    """
    Docstring for move_results_to_folder

    :param folder: Description
    :type folder: str
    :param logger: Description
    :type logger: logging.Logger
    :param clean: Description
    :type clean: bool
    """
    results_list = [
        "timings.json",
        "charge.json",
        "bond.json",
        "fuzzy_full.json",
        "qtaim.json",
        "other.json",
        "orca.json",
        "horton.json",
    ]
    results_folder = os.path.join(folder, "generator")

    os.makedirs(results_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file in results_list:
            # if file exists in results folder, merge the jsons
            existing_path = os.path.join(results_folder, file)
            new_path = os.path.join(folder, file)
            if os.path.exists(existing_path):
                try:
                    with open(new_path, "r") as f:
                        data_new = json.load(f)
                    try:
                        with open(existing_path, "r") as f:
                            data_existing = json.load(f)
                    except (json.JSONDecodeError, OSError) as e:
                        # Quarantine corrupted destination so the fresh data
                        # can overwrite it. Without this, a malformed
                        # generator/<file>.json (e.g. trailing-comma
                        # charge.json from a pre-atomic-write era) is
                        # immortal: the merge keeps failing and the fresh
                        # data is silently dropped, so validation never recovers.
                        corrupt_path = existing_path + ".corrupt"
                        try:
                            os.replace(existing_path, corrupt_path)
                            logger.error(
                                "Quarantined unreadable %s -> %s (%s); "
                                "fresh data will overwrite",
                                existing_path, corrupt_path, e,
                            )
                        except OSError as mv_err:
                            logger.error(
                                "Failed to quarantine %s: %s",
                                existing_path, mv_err,
                            )
                        data_existing = {}
                    # merge the two dicts
                    if isinstance(data_existing, dict) and isinstance(data_new, dict):

                        data_merged = {**data_existing, **data_new}
                    else:
                        data_merged = data_new  # if not dict, just overwrite
                    atomic_json_write(existing_path, data_merged)

                    logger.info(f"Merged {file} into results folder")
                    # remove the original file

                    if clean:
                        os.remove(new_path)
                except Exception as e:
                    logger.error(f"Error merging file {file}: {e}")
            else:
                try:
                    os.rename(
                        os.path.join(folder, file),
                        os.path.join(results_folder, file),
                    )
                    logger.info(f"Moved {file} to results folder")
                except Exception as e:
                    logger.error(f"Error moving file {file}: {e}")


def _validate_parse_completeness(orca_dict: dict) -> bool:
    """Thin wrapper — canonical implementation lives in parse_orca."""
    from qtaim_gen.source.core.parse_orca import validate_parse_completeness
    return validate_parse_completeness(orca_dict)


def _extract_orca_out_from_archive(folder: str, logger: logging.Logger) -> bool:
    """Try to extract just orca.out from orca.tar.zst in *folder*.

    Tries `tar --zstd` first; on hosts whose tar lacks zstd support
    (e.g. some HPC login nodes), falls back to two-step
    `unzstd -k` + `tar -xf`. Leaves orca.tar.zst in place on success.

    Returns True if orca.out was successfully extracted.
    """
    tar_zst = os.path.join(folder, "orca.tar.zst")
    if not os.path.isfile(tar_zst):
        return False

    out_path = os.path.join(folder, "orca.out")

    try:
        subprocess.run(
            ["tar", "--zstd", "-xf", "orca.tar.zst", "orca.out"],
            cwd=folder,
            check=True,
            capture_output=True,
        )
        if os.path.isfile(out_path):
            logger.info("Extracted orca.out from orca.tar.zst")
            return True
        # tar exited 0 but orca.out is not on disk: member missing from archive,
        # not a tar capability problem. Don't try the fallback.
        logger.warning(
            "tar --zstd succeeded but orca.out not present in %s",
            tar_zst,
        )
        return False
    except FileNotFoundError as e:
        logger.warning("tar not available: %s", e)
        return False
    except subprocess.CalledProcessError as e:
        stderr_snip = (e.stderr or b"").decode(errors="replace")[:500]
        logger.info(
            "tar --zstd failed (exit %s): %s; falling back to unzstd+tar",
            e.returncode,
            stderr_snip.strip(),
        )

    tar_path = os.path.join(folder, "orca.tar")
    if os.path.isfile(tar_path):
        # Pre-existing orca.tar would be silently clobbered by `unzstd -k -f`.
        # Refuse rather than risk overwriting unrelated user data.
        logger.warning(
            "Refusing fallback: %s already exists; not overwriting", tar_path
        )
        return False

    created_tar = False
    try:
        subprocess.run(
            ["unzstd", "-k", "-f", "orca.tar.zst"],
            cwd=folder,
            check=True,
            capture_output=True,
        )
        created_tar = os.path.isfile(tar_path)
        subprocess.run(
            ["tar", "-xf", "orca.tar", "orca.out"],
            cwd=folder,
            check=True,
            capture_output=True,
        )
        extracted = os.path.isfile(out_path)
        if extracted:
            logger.info("Extracted orca.out via unzstd+tar fallback")
        return extracted
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        stderr_snip = ""
        if isinstance(e, subprocess.CalledProcessError):
            stderr_snip = (e.stderr or b"").decode(errors="replace")[:500].strip()
        logger.warning(
            "Could not extract orca.out from archive (fallback failed): %s %s",
            e,
            stderr_snip,
        )
        return False
    finally:
        if created_tar and os.path.isfile(tar_path):
            try:
                os.remove(tar_path)
            except OSError as e:
                logger.warning("Could not remove intermediate orca.tar: %s", e)


def _run_orca_parse(
    folder: str,
    move_results: bool,
    logger: logging.Logger,
    delete_out: bool = False,
) -> None:
    """Parse orca.out -> orca.json + merge into charge.json/bond.json.

    Handles:
    - Both 'orca.out' and 'output.out' filenames (via shared find_orca_output_file)
    - Extracts orca.out from orca.tar.zst if not already on disk
    - Missing orca.out (skip silently)
    - Writes orca_parse timing into timings.json for checkpoint support
    - Atomic writes throughout including timings.json (crash-safe)
    - Optionally deletes orca.out after successful parse (28-114MB files)
    - Validates parse completeness before allowing deletion
    - Merge is additive/idempotent -- safe to re-run
    """
    from qtaim_gen.source.core.parse_orca import (
        parse_orca_output,
        write_orca_json,
        merge_orca_into_charge_json,
        merge_orca_into_bond_json,
        find_orca_output_file,
    )

    orca_out_path = find_orca_output_file(folder)
    extracted_from_archive = False
    if orca_out_path is None:
        # orca.out may still be inside the compressed archive
        if _extract_orca_out_from_archive(folder, logger):
            orca_out_path = find_orca_output_file(folder)
            extracted_from_archive = True
    if orca_out_path is None:
        logger.info("No orca.out found in %s -- skipping ORCA parse", folder)
        return

    t_start = time.time()
    try:
        orca_dict = parse_orca_output(orca_out_path)

        # Validate completeness before committing (protects against truncated .out files)
        if not _validate_parse_completeness(orca_dict):
            logger.warning(
                "ORCA parse produced incomplete result (%d keys) for %s -- "
                "keeping orca.out for retry",
                len(orca_dict),
                orca_out_path,
            )
            # Still write partial orca.json (partial data better than none)
            write_orca_json(folder, orca_dict)
            # Clean up extracted file — archive still has it for retry
            if extracted_from_archive:
                try:
                    os.remove(orca_out_path)
                except OSError:
                    pass
            return  # Do NOT delete, do NOT merge

        write_orca_json(folder, orca_dict)
        # Resolve merge targets: prefer generator/ when move_results=True,
        # fall back to root (normal flow where parse_multiwfn just wrote them)
        charge_path = os.path.join(folder, "charge.json")
        bond_path = os.path.join(folder, "bond.json")
        if move_results:
            gen_charge = os.path.join(folder, "generator", "charge.json")
            gen_bond = os.path.join(folder, "generator", "bond.json")
            if os.path.isfile(gen_charge):
                charge_path = gen_charge
            if os.path.isfile(gen_bond):
                bond_path = gen_bond
        merge_orca_into_charge_json(orca_dict, charge_path)
        merge_orca_into_bond_json(orca_dict, bond_path)
        elapsed = round(time.time() - t_start, 2)
        logger.info("Parsed orca.out in %.2f s (%d keys)", elapsed, len(orca_dict))

        # Write orca_parse timing into timings.json (atomic write for crash safety).
        # Merge generator/ (prior-run keys) with root (current-run keys); root wins
        # on conflict so a Level 1 restart doesn't lose newly-written L1 sub-job
        # timings to a stale generator/timings.json.
        gen_timings = os.path.join(folder, "generator", "timings.json")
        root_timings = os.path.join(folder, "timings.json")
        timings = {}
        if os.path.isfile(gen_timings) and os.path.getsize(gen_timings) > 0:
            try:
                with open(gen_timings, "r") as f:
                    timings.update(json.load(f))
            except json.JSONDecodeError:
                pass
        if os.path.isfile(root_timings) and os.path.getsize(root_timings) > 0:
            try:
                with open(root_timings, "r") as f:
                    timings.update(json.load(f))
            except json.JSONDecodeError:
                pass

        if timings:
            timings["orca_parse"] = elapsed
            atomic_json_write(root_timings, timings)

        # Delete orca.out AFTER successful parse + merge + timing checkpoint.
        # Always clean up if we extracted it from the archive (archive still has it).
        if delete_out or extracted_from_archive:
            try:
                os.remove(orca_out_path)
                logger.info("Deleted %s after successful parse", orca_out_path)
            except OSError as e:
                logger.warning("Could not delete %s: %s", orca_out_path, e)

    except Exception as e:
        logger.error("Error parsing orca.out in %s: %s", folder, e)
        # On error, clean up extracted file (archive still has it for retry)
        if extracted_from_archive:
            try:
                os.remove(orca_out_path)
            except OSError:
                pass


# Multiwfn (v3.8) error signatures we want to catch. If multiwfn upgrades and
# rephrases these strings, the .out file will start passing the substantive
# check again — re-pin in tests when we update multiwfn.
_MULTIWFN_ERROR_SIGNATURES = (
    "Error:",                         # banner-line errors, e.g. "Error: Unable to find the input file"
    "cannot be found, input again",   # stuck on interactive prompt loop
)


# Per-routine positive completion markers. Present in a .out only after the
# routine finished writing its result section. Used to catch runs killed
# mid-computation (walltime, OOM) whose .out has a clean banner + partial
# progress but no result table — e.g. chelpg killed mid-LIBRETA-ESP.
# Markers are strings the routine's own parser uses to locate the result
# block, so by construction a parsed-without-error .out contains them.
_STEP_COMPLETION_MARKERS = {
    "chelpg": "Center       Charge",  # parse_charge_chelpg trigger
}


# Generic completion signal: multiwfn prints this banner once at startup and
# once more when the .mfwn script's final "0" returns to the main menu before
# "q". Every generated script is single-module (enter module -> compute ->
# print results -> "0" -> "q"), so a second occurrence proves the routine
# finished writing its results. A run killed mid-computation (walltime, OOM)
# dies in a progress loop and never reaches the second print. Verified on
# Multiwfn 3.8 noGUI: 12/12 complete .outs contain it twice, truncated .outs
# once (see docs re: elytes 274-atom edge case, Jul 2026).
_MULTIWFN_MENU_BANNER = b"Main function menu"
_MENU_BANNER_REQUIRED_COUNT = 2


def _is_substantive_step_out(path: str, order: str = None) -> bool:
    """True if a multiwfn .out file looks like a successful run.

    Rejects empty files and any file whose first 8 KB contains one of the
    multiwfn error signatures. Both signatures appear early in the file
    (banner + first prompt loop), so a single bounded read catches them.

    Completion is detected generically via `_MULTIWFN_MENU_BANNER`: the
    main-menu banner must appear at least `_MENU_BANNER_REQUIRED_COUNT`
    times (startup print + the script's final return-to-main-menu). This
    catches walltime-killed runs for every routine, whose head looks clean
    but whose result section was never reached.

    Routines in `_STEP_COMPLETION_MARKERS` must additionally contain their
    positive result-section marker.
    """
    if not os.path.isfile(path):
        return False
    try:
        size = os.path.getsize(path)
        if size == 0:
            return False
        with open(path, "rb") as f:
            head = f.read(8192).decode("utf-8", errors="replace")
    except OSError:
        return False
    if any(sig in head for sig in _MULTIWFN_ERROR_SIGNATURES):
        return False

    marker = _STEP_COMPLETION_MARKERS.get(order)
    marker_bytes = marker.encode("utf-8") if marker is not None else None

    # Stream in 1 MB chunks with per-pattern carries. A carry of
    # len(pattern)-1 bytes can never hold a complete occurrence, so
    # prepending it to the next chunk catches boundary-spanning matches
    # without double counting.
    banner_count = 0
    marker_found = marker_bytes is None
    banner_carry = b""
    marker_carry = b""
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return False
                buf = banner_carry + chunk
                banner_count += buf.count(_MULTIWFN_MENU_BANNER)
                banner_carry = buf[len(buf) - (len(_MULTIWFN_MENU_BANNER) - 1):]
                if not marker_found:
                    mbuf = marker_carry + chunk
                    if marker_bytes in mbuf:
                        marker_found = True
                    else:
                        marker_carry = mbuf[len(mbuf) - (len(marker_bytes) - 1):]
                if banner_count >= _MENU_BANNER_REQUIRED_COUNT and marker_found:
                    return True
    except OSError:
        return False


def _wavefunction_path(folder: str) -> Optional[str]:
    """Path of a non-empty orca.wfn or orca.wfx in folder/ or generator/, or None."""
    for base in (folder, os.path.join(folder, "generator")):
        for ext in (".wfn", ".wfx"):
            wf = os.path.join(base, f"orca{ext}")
            try:
                if os.path.isfile(wf) and os.path.getsize(wf) > 0:
                    return wf
            except OSError:
                continue
    return None


def _wavefunction_present(folder: str) -> bool:
    """True if a non-empty orca.wfn or orca.wfx exists in folder/ or generator/."""
    return _wavefunction_path(folder) is not None


def _expected_electrons(dft_dict: dict) -> Optional[int]:
    """sum(Z) - net charge from the parsed input file, or None."""
    try:
        from rdkit import Chem

        table = Chem.GetPeriodicTable()
        z = sum(table.GetAtomicNumber(a["element"]) for a in dft_dict["mol"].values())
        return z - int(dft_dict.get("charge", 0))
    except Exception:
        return None


def _wavefunction_electrons(path: str) -> Optional[float]:
    """Electron count a .wfx declares: <Number of Electrons> plus <Number of
    Core Electrons> (Multiwfn 3.8 writes both, the latter non-zero for ECP
    systems). Stops at the nuclear-names block, so only the header is read.
    .wfn carries no core count, so it is not judged. None if unreadable."""
    if not path.endswith(".wfx"):
        return None
    total = 0.0
    found = False
    tag = None
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if tag is not None:
                    total += float(s.split()[0])
                    found = True
                    tag = None
                    continue
                if s in ("<Number of Electrons>", "<Number of Core Electrons>"):
                    tag = s
                    continue
                if s.startswith("<Nuclear Names>") or s.startswith("<Primitive Centers>"):
                    break
    except (OSError, ValueError, IndexError):
        return None
    return total if found else None


def _discard_wavefunction_and_step_outputs(
    folder: str, orders: list, logger: logging.Logger
) -> None:
    """Remove the wavefunction and every per-step artifact in the job root so
    the whole analysis is recomputed from the .gbw."""
    targets = []
    for base in (folder, os.path.join(folder, "generator")):
        targets += [os.path.join(base, f"orca{ext}") for ext in (".wfn", ".wfx")]
    for order in orders:
        if order == "convert":
            continue
        targets += [os.path.join(folder, f"{order}.out"), os.path.join(folder, f"{order}.json")]
    targets += [os.path.join(folder, "CPprop.txt"), os.path.join(folder, "qtaim.json")]
    for path in targets:
        if os.path.isfile(path):
            try:
                os.remove(path)
                logger.info("Removed %s: derived from a wavefunction with the wrong electron count", path)
            except OSError as e:
                logger.warning("Could not remove %s: %s", path, e)


def _qtaim_output_complete(
    folder: str,
    n_atoms: Optional[int] = None,
    check_bcp_count: bool = False,
    bcp_tolerance: int = 2,
    require_qtaim_provenance: bool = False,
) -> bool:
    """Whether qtaim.json looks complete enough to skip the QTAIM step.

    A non-empty qtaim.json is not sufficient. Multiwfn numbers nuclear CPs
    first, so a record truncated in the bond-CP tail -- or one carrying every
    nuclear CP and no bond CPs at all -- is still a well-formed, non-empty file.
    Accepting it made the restart path contradict the validator: the step was
    skipped as "data verified" and the same job then failed validation for
    having no bond critical points, every pass, forever.

    Rejects (forcing a rerun) when the nuclear-CP count disagrees with the atom
    count, or a multi-atom system has no bond CPs. With check_bcp_count it also
    consults qtaim.out, rejecting a run whose CP search or CPprop.txt export
    never finished or whose stored bond-CP count falls short of what Multiwfn
    reported; conversely, an empty BCP set is then acceptable when a complete
    run itself reported none (genuinely non-interacting fragments).
    """
    from qtaim_gen.source.utils.validation import qtaim_run_status

    for base in (folder, os.path.join(folder, "generator")):
        path = os.path.join(base, "qtaim.json")
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not data:
            continue

        n_ncp = sum(1 for k in data if k != "_meta" and "_" not in k)
        n_bcp = sum(1 for k in data if k != "_meta" and "_" in k)
        if n_atoms is not None and n_ncp != n_atoms:
            return False
        status = None
        if (n_atoms if n_atoms is not None else 2) > 1 and n_bcp == 0:
            # Under check_bcp_count, an empty BCP set backed by a *complete*
            # run defers to the reported-count logic below: a run that itself
            # found zero (or only unstorable) bond CPs is deterministic, and
            # rerunning it forever cannot change the record (far-separated
            # fragments legitimately have none). Must match the validator or
            # the step reruns on every pass while validation keeps passing.
            if not check_bcp_count:
                return False
            status = qtaim_run_status(folder)
            if not (
                status["have_qtaim_out"]
                and status["search_done"]
                and status["export_done"]
            ):
                return False
        if require_qtaim_provenance:
            if status is None:
                status = qtaim_run_status(folder)
            if not status["have_qtaim_out"]:
                # no qtaim.out means completeness is unverifiable; must match
                # the validator or the step is skipped and then fails validation
                return False
        if check_bcp_count:
            if status is None:
                status = qtaim_run_status(folder)
            if status["have_qtaim_out"]:
                # search_done too, not just export_done: a qtaim.out with the
                # export marker but no parseable CP count line fails the
                # validator, so skipping here would be skip-then-fail forever
                if not status["search_done"] or not status["export_done"]:
                    return False
                # Must use the same tolerance the validator does, or the
                # restart path reruns records validation is happy to accept --
                # which is how a repair campaign ends up looping forever.
                reported = status["reported_bcp"]
                if reported is not None and reported - n_bcp > bcp_tolerance:
                    # Raw count first: it is an upper bound on the storable
                    # count, so a raw deficit inside the tolerance guarantees
                    # the exact one is too. Only past that is it worth reading
                    # CPprop.txt back out of out_files.zip.
                    from qtaim_gen.source.utils.validation import storable_bcp_count

                    storable = storable_bcp_count(folder)
                    expected = storable if storable is not None else reported
                    if expected - n_bcp > bcp_tolerance:
                        return False
        return True

    # No usable qtaim.json. parse_multiwfn only writes it after every step has
    # run, so a job killed between the qtaim step and the final parse holds a
    # complete qtaim.out + CPprop.txt and no json. Treating that as "not done"
    # made a 235-atom job redo its 10.5 ks qtaim step on four consecutive
    # restarts and never get past it inside the walltime.
    return _qtaim_raw_output_complete(folder, n_atoms=n_atoms)


def _qtaim_raw_output_complete(folder: str, n_atoms: Optional[int] = None) -> bool:
    """Root qtaim.out carries both completion markers and root CPprop.txt holds
    one nuclear CP per atom, so parse_multiwfn can build qtaim.json from it."""
    from qtaim_gen.source.utils.validation import qtaim_run_status
    from qtaim_gen.source.core.parse_qtaim import get_qtaim_descs, only_atom_cps

    cpprop = os.path.join(folder, "CPprop.txt")
    qtaim_out = os.path.join(folder, "qtaim.out")
    try:
        if not (
            os.path.isfile(qtaim_out)
            and os.path.isfile(cpprop)
            and os.path.getsize(cpprop) > 0
        ):
            return False
    except OSError:
        return False
    # qtaim_run_status reads the root qtaim.out first, which is the one that
    # exists here, so the status describes the same run as CPprop.txt.
    status = qtaim_run_status(folder)
    if not (status["search_done"] and status["export_done"]):
        return False
    if n_atoms is None:
        return True
    try:
        atoms, _ = only_atom_cps(get_qtaim_descs(cpprop))
    except Exception:
        return False
    return len(atoms) == n_atoms


# |sum(atomic charges) - net charge| above this means the run did not see the
# full electron density (e.g. a wavefunction exported from a truncated molden:
# Hirshfeld "charges" of 3.7 on carbon summing to hundreds). Grid partitions
# normally close to within 0.01 e.
_CHARGE_SUM_TOLERANCE = 0.5


def _step_out_parses(
    path: str,
    order: str,
    fuzzy_routines: Optional[set] = None,
    n_atoms: Optional[int] = None,
    charge: Optional[int] = None,
) -> bool:
    """Whether a banner-complete `.out` actually yields usable data.

    The menu-banner check only proves Multiwfn reached the end of the script.
    A run against a broken wavefunction gets there too, printing overflowed
    `************` fields that fail to parse or populations that sum to
    hundreds; skipping such a step as "data verified" meant the folder failed
    validation on every pass while nothing was ever recomputed.
    """
    try:
        data = _parse_routine_out(order, path, fuzzy_routines)
    except Exception:
        return False
    if data is None:
        return True  # no parser for this routine; the banner check is all we have
    if not data:
        return False
    charges = data.get("charge") if isinstance(data, dict) else None
    if not isinstance(charges, dict):
        return True
    if n_atoms is not None and len(charges) != n_atoms:
        return False
    if charge is not None:
        try:
            total = sum(float(v) for v in charges.values())
        except (TypeError, ValueError):
            return False
        if abs(total - charge) > _CHARGE_SUM_TOLERANCE:
            return False
    return True


def _has_usable_step_output(
    folder: str,
    order: str,
    n_atoms: Optional[int] = None,
    check_bcp_count: bool = False,
    bcp_tolerance: int = 2,
    require_qtaim_provenance: bool = False,
    charge: Optional[int] = None,
    fuzzy_routines: Optional[set] = None,
) -> bool:
    """Check whether a sub-job appears to have produced usable output on disk.

    Primary signal: `.out` file must be substantive (see `_is_substantive_step_out`).
    Fallback: if `.out` is absent (cleaned up after a prior successful run), a
    non-empty per-step `.json` is accepted. A bad `.out` (error signature) blocks
    the `.json` fallback — stale intermediate JSONs must not mask a failed run.

    Special case for `convert`: this step has no analytical output — its
    purpose is to produce `orca.wfn`/`orca.wfx` from `orca.molden.input`. Skip
    only if a non-empty wavefunction file exists; otherwise re-run regardless
    of whether `convert.out` looks substantive.
    """
    # TODO: if more side-effect steps are added (convert is the only one
    # today), replace this branch with a {step: artifact} registry instead of
    # accumulating elif's here.
    if order == "convert":
        return _wavefunction_present(folder)

    # qtaim.json can be non-empty yet incomplete, so presence is not enough
    if order == "qtaim":
        return _qtaim_output_complete(
            folder,
            n_atoms=n_atoms,
            check_bcp_count=check_bcp_count,
            bcp_tolerance=bcp_tolerance,
            require_qtaim_provenance=require_qtaim_provenance,
        )

    for base in (folder, os.path.join(folder, "generator")):
        out_path = os.path.join(base, f"{order}.out")
        if os.path.isfile(out_path):
            if _is_substantive_step_out(out_path, order=order) and _step_out_parses(
                out_path,
                order,
                fuzzy_routines=fuzzy_routines,
                n_atoms=n_atoms,
                charge=charge,
            ):
                return True
            # .out present but bad — don't trust stale .json in this location
        else:
            # .out absent (cleaned up after a prior successful run) — .json is the only artifact
            json_path = os.path.join(base, f"{order}.json")
            try:
                if os.path.isfile(json_path) and os.path.getsize(json_path) > 0:
                    with open(json_path, "r") as _f:
                        if json.load(_f):
                            return True
            except (OSError, json.JSONDecodeError):
                pass
    return False


# Fuzzy compiled entries store n_atoms regular atoms plus two summary rows
# ("sum" and "abs_sum"); validate_fuzzy_dict pins the count at n_atoms + 2.
_FUZZY_EXTRA_ENTRIES = 2


def _compiled_data_present(
    folder: str,
    order: str,
    compiled_map: dict,
    n_atoms: Optional[int] = None,
    fuzzy_routines: Optional[set] = None,
) -> bool:
    """Return True if compiled JSON output for `order` exists and looks complete.

    Checks both the job root (in-progress run) and the generator/ subfolder
    (completed prior run after move_results_to_folder).

    When `n_atoms` is supplied, the per-atom count is verified against the
    same expectation the validator enforces (charge: exactly `n_atoms`;
    fuzzy: exactly `n_atoms + _FUZZY_EXTRA_ENTRIES`). Without this length
    check a partial write -- e.g. chelpg killed mid-table after writing 30
    of 142 atoms -- would be treated as "data verified", the restart would
    skip the routine, and the validator would fail forever on the next pass.

    Args:
        folder: Job folder path.
        order: Operation name (e.g. 'hirshfeld', 'becke_fuzzy_density').
        compiled_map: Maps operation name → (compiled_json_filename, key_or_None).
            key=None for other_dict ops where other.json stores scalar fields
            via dict.update(), not keyed by operation name.
        n_atoms: Atom count from the input file; enables length-based
            completeness checks. None falls back to legacy bool-only check.
        fuzzy_routines: Set of routine names that go into fuzzy_full.json.
            Required to apply the `n_atoms + 2` fuzzy length expectation
            (compiled_map alone can't distinguish bond vs. fuzzy ops since
            both use the `else` branch below).
    """
    if order not in compiled_map:
        return False
    entry = compiled_map[order]
    json_name = entry[0]
    key = entry[1]
    sub_key = entry[2] if len(entry) > 2 else None
    fuzzy_routines = fuzzy_routines or set()
    for base in [folder, os.path.join(folder, "generator")]:
        json_path = os.path.join(base, json_name)
        if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if key is None:
                # other ops: just verify the compiled JSON is non-empty
                if data:
                    return True
            elif sub_key is not None:
                # charge ops: {"<op>": {"charge": {...}, ...}} — check the nested sub-key
                # avoids false-positive when op-level dict exists but charge dict is empty
                charges = data.get(key, {}).get(sub_key)
                if not charges:
                    continue
                if n_atoms is not None and len(charges) != n_atoms:
                    continue  # partial entry — treat as missing, force rerun
                return True
            else:
                # bond/fuzzy ops: key value is the data dict directly
                payload = data.get(key)
                if not payload:
                    continue
                if (
                    n_atoms is not None
                    and order in fuzzy_routines
                    and len(payload) != n_atoms + _FUZZY_EXTRA_ENTRIES
                ):
                    continue  # partial fuzzy table — force rerun
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def gbw_analysis(
    folder: str,
    multiwfn_cmd: str,
    orca_2mkl_cmd: str,
    separate: bool = True,
    parse_only: bool = False,
    clean: bool = True,
    overwrite: bool = True,
    orca_6: bool = True,
    restart: bool = False,
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    mem: int = 400000000,
    n_threads: int = 4,
    prof_mem: bool = False,
    preprocess_compressed: bool = False,
    full_set: int = 0,
    move_results: bool = True,
    patch_path: bool= False,
    check_orca: bool = False,
    check_bcp_count: bool = False,
    bcp_tolerance: int = 2,
    require_qtaim_provenance: bool = False,
    wfx: bool = False,
    exhaustive_qtaim: bool = False,
    subprocess_env: Optional[dict] = None,
    patch_timings: bool = False,
    horton_python: str = "",
) -> None:
    """
    Run a full analysis on a folder of gbw files
    Takes:
        folder(str): folder to analyze
        multiwfn_cmd(str): command to run multiwfn
        orca_2mkl_cmd(str): command to run orca_2mkl
        separate(bool): whether to separate the analysis into different files
        parse_only(bool): whether to only parse the files
        clean(bool): whether to clean
        overwrite(bool): whether to overwrite the output files
        orca_6(bool): whether calc is from orca6
        restart(bool): whether to restart from the last step using timings.json
        debug(bool): whether to run a minimal set of jobs
        logger(logging.Logger): logger to log messages
        mem(int): memory to use for the analysis in bytes
        n_threads(int): number of threads to use for the analysis
        prof_mem(bool): whether to profile memory usage during the analysis
        preprocess_compressed(bool): whether to preprocess compressed files (not implemented yet)
        full_set(int): refined set of cheaper calcs or full set of analysis
        move_results(bool): whether to move results to a single results folder after analysis
        wfx(bool): whether to use .wfx format instead of .wfn for conversion
        horton_python(str): python interpreter of the separate horton environment;
            non-empty enables the HORTON charge engine post-step (default off)
        check_bcp_count(bool): reject qtaim.json records holding fewer bond
            critical points than Multiwfn reported in qtaim.out
    Writes:
        - settings.ini file with memory and n_threads
        - jobs for conversion to wfn/wfx and multiwfn analysis
        - timings.json file with timings for each step
        - bond.json, charge.json, fuzzy_full.json, qtaim.json, other.json files with parsed data
    Returns:
        - tf_validation(bool): whether the analysis was successful
    """

    if logger is None:
        logger = setup_logger(folder)
    logger.info("Starting gbw_analysis in folder: {}".format(folder))

    if not os.path.exists(folder):
        print("Folder does not exist")
        logger.error("Folder does not exist: {}".format(folder))
        return

    # check if there is a .wfn or .gbw file in the folder. If there is an
    # option to preprocess compressed files
    if preprocess_compressed:
        logger.info("Preprocessing compressed files in folder: {}".format(folder))
        # check if the required files are already uncompressed - .inp, .wfn
        required_files = [".inp", ".wfn", ".wfx"]
        uncompressed_files = [
            f for f in os.listdir(folder) if f.endswith(tuple(required_files))
        ]
        # also check these files are not empty
        uncompressed_files = [
            f
            for f in uncompressed_files
            if os.path.getsize(os.path.join(folder, f)) > 0
        ]

        if uncompressed_files:
            logger.info("Found uncompressed files: {}".format(uncompressed_files))
            logger.info("Skipping uncompression step")

        else:
            logger.warning("No uncompressed files found - will attempt to uncompress")
        # skip if uncompressed files are present

        if len(uncompressed_files) < 2:
            # run unstd and extract in the target folder so resulting files land there
            for file in os.listdir(folder):
                if file.endswith(".tar.zst") or file.endswith(".tgz"):
                    logger.info(f"Found compressed file: {file}")
                    zstd_file = file

                    # run unzstd with cwd=folder so outputs land directly in folder
                    try:
                        subprocess.run(
                            ["unzstd", "-f", zstd_file], cwd=folder, check=True
                        )
                        # remove the zstd file after successful extraction
                        os.remove(os.path.join(folder, zstd_file))
                    except Exception as e:
                        logger.error(f"Error running unzstd on {zstd_file}: {e}")

                    # untar resulting file (tar filename is zstd_file with .tar)
                    if zstd_file.endswith(".tar.zst"):
                        tar_file_name = zstd_file.replace(".tar.zst", ".tar")
                        tar_cmd = ["tar", "-xf", tar_file_name, "--directory", folder]
                        
                    else:
                        tar_file_name = zstd_file.replace(".tgz", ".tar")
                        tar_cmd = ["tar", "-xf", tar_file_name, "--directory", folder]

                    tar_file_out = tar_file_name
                    # extract tar in the folder
                    try:
                        subprocess.run(tar_cmd, cwd=folder, check=True)
                        # remove the tar file after successful extraction
                        os.remove(os.path.join(folder, tar_file_name))
                    except Exception as e:
                        logger.error(f"Error extracting tar {tar_file_name}: {e}")

                    # After extracting in-place (cwd=folder), expected files should be in folder
                    found_any = False
                    for file2 in os.listdir(folder):
                        if (
                            file2.startswith("orca.engrad")
                            or file2.startswith("orca.out")
                            or file2.startswith("orca.inp")
                            or file2.startswith("orca.property.inp")
                            or file2.startswith("orca.property.txt")
                            or file2.startswith("orca_stderr")
                        ):
                            logger.info(f"Found extracted file in folder: {file2}")
                            found_any = True
                    if not found_any:
                        logger.warning(
                            "No expected extracted files found in %s after extraction",
                            folder,
                        )

                if file.endswith(".gbw.zstd0"):
                    logger.info(f"Found compressed gbw file: {file}")
                    zstd_file = file
                    gbw_file = zstd_file.replace(".zstd0", "")
                    # run unzstd to produce gbw_file inside folder
                    try:
                        subprocess.run(
                            ["unzstd", "-o", gbw_file, "-f", zstd_file],
                            cwd=folder,
                            check=True,
                        )
                    except Exception as e:
                        logger.error(f"Error running unzstd for gbw {zstd_file}: {e}")

            # Legacy orca5.* files are only safe to drop when the canonical
            # orca.gbw is present (produced from orca.gbw.zstd0 above). If the
            # folder only has orca5.gbw, removing it would strip the sole
            # wavefunction source and brick downstream orca_2mkl/Multiwfn.
            canonical_gbw_path = os.path.join(folder, "orca.gbw")
            canonical_gbw_present = (
                os.path.isfile(canonical_gbw_path)
                and os.path.getsize(canonical_gbw_path) > 0
            )
            always_intermediate = [".tar", ".tar.zst", ".tgz", ".gbw.zstd0", ".zstd", ".npz"]
            legacy_orca5 = ["orca5.gbw", "orca5.wfn", "orca5.wfx"]
            for file in os.listdir(folder):
                is_intermediate = any(file.endswith(ext) for ext in always_intermediate)
                is_legacy_orca5 = file in legacy_orca5
                if is_intermediate or (is_legacy_orca5 and canonical_gbw_present):
                    try:
                        os.remove(os.path.join(folder, file))
                        logger.info(f"Removed intermediate file: {file}")
                    except Exception as e:
                        logger.error(f"Error removing intermediate file {file}: {e}")
                elif is_legacy_orca5 and not canonical_gbw_present:
                    logger.warning(
                        "Keeping legacy %s -- no canonical orca.gbw present in %s",
                        file,
                        folder,
                    )

    if restart:
        # Check both locations: generator/ (previous completed run) and
        # job root (interrupted run before move_results_to_folder ran)
        gen_timings = os.path.join(folder, "generator", "timings.json")
        root_timings = os.path.join(folder, "timings.json")

        if os.path.exists(gen_timings) and os.path.getsize(gen_timings) > 0:
            timings_path = gen_timings
        elif os.path.exists(root_timings) and os.path.getsize(root_timings) > 0:
            timings_path = root_timings
        else:
            timings_path = None

        if timings_path is None:
            logger.warning("No timings file found - starting from scratch!")
            restart = False
        else:
            logger.info("Timings file found at %s - restarting.", timings_path)

    # check if output already exists
    if not overwrite:
        # if move_results:
        #    folder_check = os.path.join(folder, "generator")
        # else:
        folder_check = folder

        if check_results_exist(folder_check, move_results=move_results):
            print("Output already exists")
            try:
                tf_validation = validation_checks(
                    folder_check,
                    full_set=full_set,
                    verbose=False,
                    move_results=move_results,
                    logger=logger,
                    check_orca=check_orca,
                    check_bcp_count=check_bcp_count,
                    bcp_tolerance=bcp_tolerance,
                    require_qtaim_provenance=require_qtaim_provenance,
                )
            except Exception as e:
                logger.error(f"Error during validation checks: {e}")
                tf_validation = False

            # we might change level-of-analysis so only return if all requested analyses are present
            if tf_validation:
                logger.info("Output already exists and is valid - skipping analysis")
                if horton_python:
                    run_horton_analysis(
                        folder=folder,
                        horton_python=horton_python,
                        subprocess_env=subprocess_env,
                        logger=logger,
                    )
                logger.info("gbw_analysis completed in folder: {}".format(folder))
                logger.info("Validation status: {}".format(tf_validation))
                return

            # attempt to reparse if output exists but validation failed
            else:
                # Check if orca.json is the ONLY missing piece — if so, just
                # run _run_orca_parse without parse_multiwfn (which would try
                # to read .txt files that may have been cleaned up already)
                try:
                    tf_without_orca = validation_checks(
                        folder_check,
                        full_set=full_set,
                        verbose=False,
                        move_results=move_results,
                        logger=logger,
                        check_orca=False,
                        check_bcp_count=check_bcp_count,
                        bcp_tolerance=bcp_tolerance,
                        require_qtaim_provenance=require_qtaim_provenance,
                    )
                except Exception:
                    tf_without_orca = False

                if tf_without_orca:
                    # Everything passes except orca — orca-only reparse
                    logger.info(
                        "Validation passes without orca check - running orca-only parse"
                    )
                    _run_orca_parse(folder, move_results, logger)
                    if horton_python:
                        run_horton_analysis(
                            folder=folder,
                            horton_python=horton_python,
                            subprocess_env=subprocess_env,
                            logger=logger,
                        )
                    if move_results:
                        move_results_to_folder(folder, logger=logger, clean=clean)
                    # Clean up orca.out after successful orca-only parse (28-114 MB)
                    if clean:
                        from qtaim_gen.source.core.parse_orca import find_orca_output_file
                        orca_out_path = find_orca_output_file(folder)
                        if orca_out_path is not None:
                            try:
                                os.remove(orca_out_path)
                                logger.info("Deleted %s after orca-only parse", orca_out_path)
                            except OSError as e:
                                logger.warning("Could not delete %s: %s", orca_out_path, e)
                        clean_jobs(
                            folder,
                            separate=separate,
                            logger=logger,
                            full_set=full_set,
                            move_results=move_results,
                        )
                    logger.info("gbw_analysis completed (orca-only) in folder: %s", folder)
                    return

                try:
                    logger.warning(
                        "Output exists but validation failed - attempting to reparse before re-running analysis"
                    )
                    parse_multiwfn(
                        folder,
                        separate=separate,
                        debug=debug,
                        logger=logger,
                        full_set=full_set,
                    )

                    # Also reparse ORCA output
                    _run_orca_parse(folder, move_results, logger)

                    if move_results:
                        move_results_to_folder(folder, logger=logger, clean=clean)

                    tf_validation = validation_checks(
                        folder_check,
                        full_set=full_set,
                        verbose=False,
                        move_results=move_results,
                        logger=logger,
                        check_orca=check_orca,
                        check_bcp_count=check_bcp_count,
                        bcp_tolerance=bcp_tolerance,
                        require_qtaim_provenance=require_qtaim_provenance,
                    )

                    if tf_validation:
                        logger.info(
                            "Reparsing successful on 2nd try - skipping analysis"
                        )
                        if horton_python:
                            run_horton_analysis(
                                folder=folder,
                                horton_python=horton_python,
                                subprocess_env=subprocess_env,
                                logger=logger,
                            )
                        logger.info(
                            "gbw_analysis completed in folder: {}".format(folder)
                        )
                        logger.info("Validation status: {}".format(tf_validation))
                        return
                except Exception as e:
                    logger.error(f"Error during reparsing attempt: {e}")
                    logger.info("Proceeding to re-run full analysis.")

    write_settings_file(mem=mem, n_threads=n_threads, folder=folder)

    if not parse_only:
        print("... Creating jobs")
        # create jobs for conversion to wfn and multiwfn analysis
        create_jobs(
            folder=folder,
            multiwfn_cmd=multiwfn_cmd,
            orca_2mkl_cmd=orca_2mkl_cmd,
            separate=separate,
            debug=debug,
            logger=logger,
            full_set=full_set,
            patch_path=patch_path,
            wfx=wfx,
            exhaustive_qtaim=exhaustive_qtaim,
        )
        # run jobs
        run_jobs(
            folder=folder,
            separate=separate,
            orca_6=orca_6,
            restart=restart,
            debug=debug,
            logger=logger,
            prof_mem=prof_mem,
            full_set=full_set,
            move_results=move_results,
            clean_jobs_tf=clean,
            subprocess_env=subprocess_env,
            check_bcp_count=check_bcp_count,
            bcp_tolerance=bcp_tolerance,
            require_qtaim_provenance=require_qtaim_provenance,
        )

    print("... Parsing multiwfn output")
    # parse those jobs to jsons for 5 categories
    parse_multiwfn(
        folder, separate=separate, debug=debug, logger=logger, full_set=full_set
    )

    # Parse ORCA output file (if present)
    _run_orca_parse(folder, move_results, logger)

    # HORTON charge engine post-step (separate python env, see core/horton.py)
    if horton_python:
        run_horton_analysis(
            folder=folder,
            horton_python=horton_python,
            subprocess_env=subprocess_env,
            logger=logger,
        )

    # move all results to a results folder

    if move_results:
        move_results_to_folder(folder, logger=logger, clean=clean)

    tf_validation = validation_checks(
        folder,
        full_set=full_set,
        verbose=True,
        move_results=move_results,
        logger=logger,
        check_orca=check_orca,
        check_bcp_count=check_bcp_count,
        bcp_tolerance=bcp_tolerance,
        require_qtaim_provenance=require_qtaim_provenance,
    )

    # Optional repair pass: if validation failed and patch_timings is on,
    # recover missing timing keys from gbw_analysis.log (or stamp -1.0
    # placeholders). patch_timings_from_log only writes positive timing
    # values, so if the only validation failure was missing/zero timing
    # keys, the patch necessarily satisfies validate_timing_dict — skip
    # the second full validation_checks pass. Other validation failures
    # (missing JSONs, n_atoms mismatch) are not patched and remain failures.
    if not tf_validation and patch_timings:
        dft_dict = get_charge_spin_n_atoms_from_folder(
            folder, logger=logger, verbose=False
        )
        if not dft_dict:
            logger.warning(
                "patch_timings: could not read charge/spin/n_atoms; "
                "skipping spin keys"
            )
        spin_tf = bool(dft_dict and dft_dict.get("spin", 1) != 1)
        did_patch = patch_timings_from_log(
            folder,
            full_set=full_set,
            spin_tf=spin_tf,
            move_results=move_results,
            logger=logger,
        )
        if did_patch:
            tf_validation = True

    logger.info("gbw_analysis completed in folder: {}".format(folder))
    logger.info("Validation status: {}".format(tf_validation))
    # move log file to results folder

    # ONLY CLEAN IF VALIDATION PASSED
    if clean and tf_validation:
        logger.info("... Cleaning up")
        # Delete orca.out after validated parse (28-114 MB files)
        from qtaim_gen.source.core.parse_orca import find_orca_output_file
        orca_out_path = find_orca_output_file(folder)
        if orca_out_path is not None:
            try:
                os.remove(orca_out_path)
                logger.info("Deleted %s after validated parse", orca_out_path)
            except OSError as e:
                logger.warning("Could not delete %s: %s", orca_out_path, e)
        clean_jobs(
            folder,
            separate=separate,
            logger=logger,
            full_set=full_set,
            move_results=move_results,
        )


# /global/scratch/users/santiagovargas/gbws_cleaning_lean/ml_elytes/elytes_md_eqv2_electro_512_C3H8O_3_group_133_shell_0_0_1_1341
#!/bin/bash
# SBATCH --job-name=conj_systems
# SBATCH --partition=cm2
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=6
# SBATCH --cpus-per-task=1
# SBATCH --time=40:00:00
# SBATCH -C lr6_m192
# SBATCH -p lr6
# SBATCH --account=lr_blau
# SBATCH --qos=condo_blau
