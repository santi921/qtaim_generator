import os
from concurrent.futures import ThreadPoolExecutor, as_completed

FILE_NAME = "orca.gbw.zstd0"  # change here if it's actually "orca.gbw.zst0"

SPECIAL_OMOL_PATHS = [
    "/lus/eagle/projects/OMol25/omol/torsion_profiles/outputs_120324/",
    "/lus/eagle/projects/OMol25/omol/electrolytes/md_based/outputs_241029/",
    "/lus/eagle/projects/OMol25/omol/electrolytes/outputs_unsolvated_120424/",
    "/lus/eagle/projects/OMol25/omol/electrolytes/solvated_090624/",
    "/lus/eagle/projects/OMol25/omol/redo_orca6/metal_organics/outputs_062424/",
    "/lus/eagle/projects/OMol25/omol/metal_organics/outputs_072324",
    "/lus/eagle/projects/OMol25/omol/metal_organics/outputs_ln_082524",
    "/lus/eagle/projects/OMol25/omol/metal_organics/outputs_low_spin_241118",
    "/lus/eagle/projects/OMol25/omol/metal_organics/restart5to6",
]


def find_zst_folders(
    root_dir, output_file, counts_file, alternative_root=None, count_level="top"
):
    """
    Scan:
      - All top-level dirs under root_dir EXCEPT 'omol'
      - All special OMol paths explicitly

    For each scanned directory:
      - Check if FILE_NAME exists in that dir or in its immediate subdirs
      - If yes, write the directory path (or remapped alternative_root) to output_file
      - Always write a count (0 or 1) for that directory to counts_file
    """
    total_found = 0

    def scan_dir(dir_path, out_file, counts_out):
        nonlocal total_found

        # Count logic depends on count_level
        found_here = False
        hit_count = 0
        if count_level == "top":
            # Only check in the directory itself
            found_here = os.path.isfile(os.path.join(dir_path, FILE_NAME))
            hit_count = 1 if found_here else 0
        elif count_level == "sub":
            # Count how many immediate subdirs contain the file
            try:
                with os.scandir(dir_path) as it:
                    for entry in it:
                        if entry.is_dir():
                            sub_path = os.path.join(entry.path, FILE_NAME)
                            if os.path.isfile(sub_path):
                                hit_count += 1
            except PermissionError:
                counts_out.write(f"{dir_path} : ERROR_PERMISSION\n")
                return
            found_here = hit_count > 0
        else:
            raise ValueError(f"Unknown count_level: {count_level}")

        if found_here:
            if alternative_root:
                rel = os.path.relpath(dir_path, root_dir)
                out_path = os.path.join(alternative_root, rel)
                out_file.write(out_path + "\n")
            else:
                out_file.write(dir_path + "\n")
            total_found += 1

        counts_out.write(f"{dir_path} : {hit_count}\n")

    # ----------------------------------------------------
    # MAIN LOGIC
    # ----------------------------------------------------
    with open(output_file, "w") as out_file, open(counts_file, "w") as counts_out:

        # 1. Scan top-level dirs except omol
        try:
            for entry in os.scandir(root_dir):
                if not entry.is_dir():
                    continue
                if entry.name == "omol":
                    continue
                scan_dir(entry.path, out_file, counts_out)
        except FileNotFoundError:
            print(f"Root directory not found: {root_dir}")
            return
        """
        # 2. Scan special OMol directories
        for path in SPECIAL_OMOL_PATHS:
            clean = path.rstrip("/")
            if os.path.exists(clean):
                scan_dir(clean, out_file, counts_out)
            else:
                counts_out.write(f"{clean} : MISSING\n")
        """
    print(f"[DONE] Found {total_found} matching folders.")
    print(f"Paths written to: {output_file}")
    print(f"Counts written to: {counts_file}")


if __name__ == "__main__":
    # can you iterate over top-level dirs in /lus/eagle/projects/OMol25
    root_directory = "/lus/eagle/projects/OMol25"
    # save to different files in one directory
    project_dir = "/lus/eagle/projects/generator/job_lists/"
    folder_list = [
        "pdb_pockets_300K",
        "trans1x",
        "electrolytes_scaled_sep",
        "low_spin_23",
        "protein_core",
        "pdb_pockets_400K",
        "pdb_fragments_400K",
        "rgd_uks",
        "dna",
        "ani1xbb",
        "tm_react",
        "noble_gas_compounds",
        "mo_hydrides",
        "noble_gas",
        "pmechdb",
        "nakb",
        "electrolytes_redox",
        "orbnet_denali",
        "5A_elytes",
        "rmechdb",
        "droplet",
        "ani2x",
        "electrolytes_reactivity",
        "protein_interface",
        "rna",
        "spice",
        "geom_orca6",
        "ml_elytes",
        "ml_mo",
        "rpmd",
        "ml_protein_interface",
        "pdb_fragments_300K",
        "scaled_separations_exp",
        "omol/torsion_profiles/outputs_120324",
        "omol/electrolytes/md_based/outputs_241029",
        "omol/electrolytes/outputs_unsolvated_120424",
        "omol/electrolytes/solvated_090624",
        "omol/redo_orca6/metal_organics/outputs_062424",
        "omol/metal_organics/outputs_072324",
        "omol/metal_organics/outputs_ln_082524",
        "omol/metal_organics/outputs_low_spin_241118",
        "omol/metal_organics/restart5to6",
    ]

    print("Starting job folder scan... number of folder lists:", len(folder_list))

    def run_scan(folder, count_level="top"):
        root_dir = os.path.join(root_directory, folder)
        output_file = os.path.join(
            project_dir, f"{folder.replace('/', '_')}_zst_folders.txt"
        )
        counts_file = os.path.join(
            project_dir, f"{folder.replace('/', '_')}_zst_counts.txt"
        )
        find_zst_folders(root_dir, output_file, counts_file, count_level=count_level)
        return folder

    with ThreadPoolExecutor(max_workers=8) as executor:  # adjust max_workers as needed
        # Change count_level here: "top" for top-level, "sub" for immediate subdirs
        count_level = "top"  # or "sub"
        futures = {
            executor.submit(run_scan, folder, count_level): folder
            for folder in folder_list
        }
        for future in as_completed(futures):
            folder = futures[future]
            try:
                future.result()
                print(f"Completed scan for folder: {folder}")
            except Exception as e:
                print(f"Error scanning folder {folder}: {e}")

    print("All done.")
