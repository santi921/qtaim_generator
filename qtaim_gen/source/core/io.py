def write_input_file(folder, lines, n_atoms, options):
    """
    Write input file for Multiwfn.
    Takes:
        folder: folder to write input file to
        lines: lines from xyz file
        n_atoms: number of atoms in xyz file
        options: dictionary of options for input file
    Returns:
        None
    """
    with open(folder + "/input.in", "w") as f:
        f.write("!{} {} AIM\n\n".format(options["functional"], options["basis"]))
        f.write("*xyz {} {}\n".format(options["charge"], options["spin"]))
        for ind in range(n_atoms):
            f.write(
                str(lines[ind + 2].split()[0])
                + "\t"
                + str(lines[ind + 2].split()[1])
                + "\t"
                + str(lines[ind + 2].split()[1])
                + "\t"
                + str(lines[ind + 2].split()[2])
                + "\n"
            )
        f.write("*\n")


def write_input_file_from_pmg_molecule(folder, molecule, options):
    try:
        sites = molecule.sites
        charge = molecule.charge
        spin = molecule.spin_multiplicity
        pmg = True
    except:
        sites = molecule["sites"]
        charge = molecule["charge"]
        spin = molecule["spin_multiplicity"]
        pmg = False

    n_atoms = int(len(sites))

    # print(folder)
    with open(folder + "/input.in", "w") as f:
        # for relativistic set functional to "TPSS ZORA" and basis to "ZORA-def2-TZVP SARC/J"
        f.write("!{} {} AIM\n\n".format(options["functional"], options["basis"]))
        if "basis_atoms" in options:
            f.write("%basis\n")
            for atom in options["basis_atoms"]:
                f.write(
                    'NewGTO    {} "{}" end\n'.format(atom["element"], atom["basis"])
                )
            f.write("end\n")
        if "relativistic" in options:
            if options["relativistic"] == True:
                f.write("%rel\n")
                f.write("picturechange  true\n")
                f.write("end\n")
        if "parallel_procs" in options:
            f.write("{}/n".format(options["parallel_procs"]))

        f.write("%SCF\n")
        f.write("    MaxIter 1000\n")
        f.write("END\n")
        f.write(
            "* xyz {} {}\n".format(
                int(charge),
                int(spin),
            )
        )
        for ind in range(n_atoms):
            if pmg:
                xyz = sites[ind].coords
                atom = sites[ind].specie.symbol
            else:
                xyz = sites[ind]["xyz"]
                atom = sites[ind]["element"]
            f.write(
                "{}\t{: .4f}\t{: .4f}\t{: .4f}\n".format(atom, xyz[0], xyz[1], xyz[2])
            )
        f.write("*\n")


def complete_folder_molecule_to_json(folder):
    """
    Takes:
        folder: folder to complete
    Returns:
        row: row of pandas dataframe - contains all information molecule in folder
    """
    # TODO
