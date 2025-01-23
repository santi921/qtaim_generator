# pseudo code 

# ask for chunk size
# iterate through dataset for chunksize 
# split and save global statistics
# create utilities to merge global statistics

import os
import argparse
import pandas as pd
from tqdm import tqdm

from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph

from qtaim_gen.source.core.bonds import get_bonds_from_rdkit
from qtaim_gen.source.core.io import convert_inp_to_xyz
from qtaim_gen.source.core.parse_json import get_data


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_folder", type=str, default="./ORCA/")
    parser.add_argument(
        "-lmdb_folder", type=str, default="./lmdb/"
    )
    parser.add_argument(
        "-chunk_size", type=int, default=1000
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge data at the end"
    )
    parser.add_argument(
        "--gather_json_data", action="store_true", help="Gather data from json files"
    )
    
    args = parser.parse_args()
    root_folder = args.root_folder
    pkl_file = args.pkl_file
    gather_tf = bool(args.gather_json_data)
    merge_tf = bool(args.merge)
    chunk_size = int(args.chunk_size)
    lmdb_folder = args.lmdb_folder
    