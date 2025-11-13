from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict
from typing import Dict, Sequence, Any, Union


def write_input_file(folder: str, lines: Sequence[str], n_atoms: int, options: Dict[str, Any]) -> None:
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


def convert_inp_to_xyz(orca_path: str, output_path: str) -> None:
    """
    Convert an ORCA input file to an XYZ file.
    Takes:
        orca_path: path to ORCA input file
        output_path: path to write XYZ file to
    Returns:
        None
    """

    mol_dict: Dict[str, Any] = dft_inp_to_dict(orca_path, parse_charge_spin=True)

    n_atoms: int = len(mol_dict["mol"])

    xyz_str: str = "{}\n".format(n_atoms)
    spin_charge_line: str = "{} {}\n".format(mol_dict["charge"], mol_dict["spin"])
    xyz_str += spin_charge_line
    # write the atom positions
    for ind, atom in mol_dict["mol"].items():

        atom_line: str = "{} {} {} {}\n".format(
            atom["element"], atom["pos"][0], atom["pos"][1], atom["pos"][2]
        )
        xyz_str += atom_line

    with open(output_path, "w") as f:
        f.write(xyz_str)


def write_input_file_from_pmg_molecule(folder: str, molecule: Union[Any, Dict[str, Any]], options: Dict[str, Any]) -> None:
    try:
        sites = molecule.sites
        charge = molecule.charge
        spin = molecule.spin_multiplicity
        pmg = True
    except Exception:
        sites = molecule["sites"]
        charge = molecule["charge"]
        spin = molecule["spin_multiplicity"]
        pmg = False

    n_atoms: int = int(len(sites))

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
            f.write("{}\n".format(options["parallel_procs"]))

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
