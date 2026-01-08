"""
CLASSES for converting qtaim_gen lmdb files to qtaim_embed format. This includes the following classes:
- Converter: Base class for converting qtaim_gen lmdb files to qtaim_embed format. This class contains methods for connecting to lmdb files, pulling lmdb files, and scaling
graphs.
- QTAIMConverter: Class for converting qtaim_gen lmdb files to qtaim_embed format. This class inherits from Converter and contains methods for processing the lmdb files and generating the graphs.
- GeneralConverter: Class for converting qtaim_gen lmdb files to qtaim_embed format. This class inherits from Converter and contains methods for processing the lmdb files and generating the graphs.
- ASELMDBConverter: Class for converting ase_lmdb formatted files to qtaim_embed format. This class inherits from Converter and contains methods for processing the lmdb files and generating the graphs.
"""

import os
import pickle
import lmdb
import json
from glob import glob
import numpy as np

from copy import deepcopy
from typing import Dict, List, Tuple, Union, Any
import bisect
import logging
from logging.handlers import RotatingFileHandler


from qtaim_embed.data.processing import (
    HeteroGraphStandardScalerIterative,
    HeteroGraphLogMagnitudeScaler,
)
from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.lmdb import serialize_dgl_graph, load_dgl_graph_from_serialized

from qtaim_gen.source.utils.lmdbs import (
    get_elements_from_structure_lmdb,
    get_elements_from_structure_lmdb_folder_list,
    parse_charge_data,
    parse_qtaim_data,
)
from qtaim_gen.source.core.qtaim_embed import (
    split_graph_labels,
    get_include_exclude_indices,
    build_and_featurize_graph,
)


def clean_id(key: bytes) -> str:
    """
    Clean the ID of the molecule.

    Args:
        key (bytes): The ID of the molecule in bytes.

    Returns:
        str: The cleaned ID as a string.
    """
    if "'" in key.decode("ascii"):
        id = key.decode("ascii").split("'")[1]
    elif '"' in key.decode("ascii"):
        id = key.decode("ascii").split('"')[1]
    else:
        id = key.decode("ascii")
    return id


def setup_logger(lmdb_path: str, logger_name: str = "Converter") -> logging.Logger:
    """
    Setup logger for the converter class.

    Args:
        lmdb_path (str): The path to the LMDB directory where logs will be saved.
        logger_name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    
    # Remove existing handlers to avoid duplicate logs
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(lmdb_path, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create file handler - use provided logger_name (sanitize extension if present)
    base_name = os.path.splitext(logger_name)[0]
    log_file = os.path.join(logs_dir, f"{base_name}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class Converter:
    def __init__(self, config_dict: Dict[str, Any], config_path: str = None):
        self.config_dict = config_dict
        self.config_path = config_path
        self.restart = config_dict["restart"]
        
        
        # Setup logging
        self.logger = self._setup_logger(
            config_dict["lmdb_path"], 
            logger_name=config_dict.get("lmdb_name", "converter")
        )

        # keys
        self.keys_data = {
            "atom": [],
            "bond": [],
            "global": [],
        }
        self.keys_target = {
            "atom": [],
            "bond": [],
            "global": [],
        }

        self.keys_target = config_dict.get("keys_target", self.keys_target)
        self.keys_data = config_dict.get("keys_data", self.keys_data)

        self.index_dict = {}
        
        # data
        self.single_lmdb_in = True
        self.lmdb_dict = self.pull_lmdbs()
        
        # grapher        
        self.grapher = None

        # scaler settings
        if "save_scaler" in self.config_dict.keys():
            self.save_scaler = self.config_dict["save_scaler"]
        else:
            self.save_scaler = False

        self.skip_keys = config_dict.get("filter_list", ["length", "scaled"])

        ####################### Element Set ########################
        if "element_set" in self.config_dict.keys():
            element_set = self.config_dict["element_set"]
        else:
            if self.single_lmdb_in:
                #print("geom_lmdb locs", self.lmdb_dict["geom_lmdb"]["env"])
                element_set = get_elements_from_structure_lmdb(
                    self.lmdb_dict["geom_lmdb"]["env"]

                )
            else:
                element_set = get_elements_from_structure_lmdb_folder_list(
                    self.lmdb_dict["geom_lmdb"]["lmdb_paths"]
                )
        #print(f"Element set: {element_set}")
        self.element_set = sorted(list(element_set))
        self.logger.info(f"Element set: {self.element_set}")


        ####################### LMDB Dir Stuff ########################
        # make directory for lmdb
        if not os.path.exists(self.config_dict["lmdb_path"]):
            os.makedirs(self.config_dict["lmdb_path"])
            self.logger.info(f"Created LMDB directory: {self.config_dict['lmdb_path']}")

        if "filter_list" not in self.config_dict.keys():
            self.config_dict["filter_list"] = ["scaled", "length"]

        if "chunk" not in self.config_dict.keys():
            self.config_dict["chunk"] = -1
        
        # output lmdb file handling - TODO: chunked outputs
        self.file = os.path.join(
            self.config_dict["lmdb_path"], 
            self.config_dict["lmdb_name"] if self.config_dict["lmdb_name"].endswith(".lmdb") else f"{self.config_dict['lmdb_name']}.lmdb"
        )

        if os.path.exists(self.file) and not(self.restart):
            os.remove(self.file)
            self.logger.info(f"Removed existing LMDB file: {self.file}")
            # also remove locked files if they exist
            lock_file = self.file + "-lock"
            if os.path.exists(lock_file):
                os.remove(lock_file)
                self.logger.info(f"Removed existing LMDB lock file: {lock_file}")
        
        

        self.db = self.connect_db(self.file)
        self.logger.info(f"Connected to output LMDB: {self.file}")

        if self.restart and os.path.exists(self.file):
            # get all existing keys from the existing LMDB file and store in self.existing_keys to reference against 
            with self.db.begin(write=False) as txn:
                self.existing_keys = set()

                cursor = txn.cursor()
                for key, _ in cursor:
                    if key.decode("ascii") not in self.skip_keys:
                        self.existing_keys.add(key.decode("ascii"))
                

                # handle scaled info
                scaled_entry = txn.get("scaled".encode("ascii"))   
                
                if scaled_entry is not None:
                    self.scaled = pickle.loads(scaled_entry)
                else: 
                    self.scaled = False
                         
                if self.scaled:
                    self.logger.info(f"Existing LMDB file '{self.file}' is already scaled. Skipping scaling.")
                else:
                    self.logger.info(f"Existing LMDB file '{self.file}' is not scaled. Will scale after processing.")

        # construct scalers for features and labels
        self.feature_scaler_iterative = HeteroGraphStandardScalerIterative(
            features_tf=True, mean={}, std={}
        )
        self.label_scaler_iterative = HeteroGraphStandardScalerIterative(
            features_tf=False, mean={}, std={}
        )

    def _setup_logger(self, lmdb_path: str, logger_name: str) -> logging.Logger:
        """
        Setup logger for the converter instance.

        Args:
            lmdb_path (str): The path to the LMDB directory where logs will be saved.

        Returns:
            logging.Logger: The configured logger instance.
        """
        return setup_logger(lmdb_path, logger_name=logger_name)
        

    def pull_lmdbs(self) -> Dict[str, Any]:
        """
        General method for pulling LMDB files from the config file. This method checks
        if the paths in the config file are valid and returns a dictionary of LMDB environments.
        This should work for any of the converters as long as the keys in the config file are correct.

        Returns:
            Dict[str, Any]: A dictionary containing the LMDB environments and related information.
        """

        lmdb_dict = {}

        assert (
            "geom_lmdb" in self.config_dict["lmdb_locations"].keys()
        ), "The config file must contain a key 'geom_lmdb'"

        for key in self.config_dict["lmdb_locations"].keys():

            lmdb_path = self.config_dict["lmdb_locations"][key]

            # check if the path is a file or directory
            if not os.path.exists(lmdb_path):
                raise ValueError(f"Path '{lmdb_path}' does not exist.")

            if os.path.isdir(lmdb_path):
                self.logger.info(f"Connecting to LMDB folder: {lmdb_path} (key: {key})")
                # if the path is a directory, check if it contains lmdb files
                lmdb_dict[key] = self.connect_folder_of_dbs(
                    db_path=lmdb_path, key=key.split("_")[0]
                )
                self.single_lmdb_in = False
            else:
                # single lmdb file
                self.logger.info(f"Connecting to single LMDB file: {lmdb_path} (key: {key})")
                lmdb_dict[key] = self.connect_db(lmdb_path, with_meta=True)

        if self.single_lmdb_in:
            assert (
                lmdb_dict[key]["env"].stat()["entries"] > 0
            ), f"The LMDB file is empty: {lmdb_dict[key]}"
        else:
            for key in lmdb_dict.keys():
                assert (
                    lmdb_dict[key]["num_samples"] > 0
                ), f"The LMDB file is empty: {lmdb_dict[key]}"

        self.logger.info(f"Successfully loaded {len(lmdb_dict)} LMDB source(s)")
        return lmdb_dict


    def connect_db(
        self, 
        lmdb_path=None, 
        map_size=int(1099511627776 * 2), 
        with_meta=False, 
        skip_keys=["length", "scaled"]
    ):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=False,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
            map_async=True,
            map_size=map_size,
        )
        if with_meta:
            length_entry = env.begin().get("length".encode("ascii"))
            # convert length entry to int if it exists, otherwise set to None
            length_entry = pickle.loads(length_entry) if length_entry is not None else None
            _keys = []
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    if key.decode("ascii") not in skip_keys:
                        _keys.append(key)
            
            if length_entry is None:
                length_entry = len(_keys)

            return {
                "env": env,
                "keys": _keys,
                "num_samples": length_entry,
                "lmdb_path": lmdb_path,
            }
        return env


    def connect_folder_of_dbs(self, db_path, key=None, skip_keys=["length", "scaled"]):
        """
        Connect to a folder of LMDB files and return a dictionary containing the environments and related information.
        Takes:
            db_path (str): The path to the folder containing the LMDB files.
            key (str): The key to look for in the LMDB file names. Use if all lmdbs are in a single folder. 

        """
        # print("Connecting to folder of LMDBs")
        # print("key: ", key)
        if os.path.isdir(db_path):
            db_paths = sorted(glob(f"{db_path}/*{key}*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{db_path}'"
            self.logger.info(f"Found {len(db_paths)} LMDB files in {db_path}")
            # self.metadata_path = self.path / "metadata.npz"

            _keys = []
            envs = []
            for db_path in db_paths:
                self.logger.debug(f"Connecting to LMDB: {db_path}")
                cur_env = self.connect_db(db_path, with_meta=True, skip_keys=skip_keys)
                envs.append(cur_env)
                _keys.append(list(cur_env["keys"]))
                #_keys.append(list(range(cur_env["num_samples"])))
                
            keylens = [len(k) for k in _keys]  # need to be dicts
            _keylen_cumulative = np.cumsum(keylens).tolist()  # need to be dicts
            num_samples = sum(keylens)  # need to be dicts
            # keys will just be a global index for the entire folder of LMDBs, so we need to adjust the keys to be cumulative across all LMDBs
            keys = list(range(num_samples))  
            self.logger.info(f"Total samples across all LMDBs: {num_samples}")
            return {
                "envs": envs,
                "keys": keys,
                "keys_raw": _keys,
                "keylens": keylens,
                "keylen_cumulative": _keylen_cumulative,
                "num_samples": num_samples,
                "lmdb_paths": db_paths,
            }

        else:
            # Throw error if the path is a file
            raise ValueError(f"Path '{db_path}' is a file, not a directory.")


    def scale_graphs_single(
        self,
        lmdb_file: lmdb,
        label_scaler: Union[
            HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler
        ],
        feature_scaler: Union[
            HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler
        ],
        return_info=False,
    ) -> None:
        """
        Scale the graphs using the provided scaler. Read from LMDB file and apply the scaler to each graph.
        """

        with lmdb_file.begin(write=False) as txn_in:
            # go through keys first and check that "scaled" isn't true, if so skip scaling
            scaled_entry = txn_in.get("scaled".encode("ascii"))
            #print("scaled entry: ", pickle.loads(scaled_entry))

            if scaled_entry is not None:
                scaled = pickle.loads(scaled_entry)
                if scaled:
                    self.logger.info("Graphs are already scaled. Skipping scaling.")
                    return

            cursor = txn_in.cursor()
            scaled_count = 0
            
            for key, value in cursor:
                # normalize key to string so comparisons and writes are consistent
                if isinstance(key, bytes):
                    key_str = key.decode("ascii")
                else:
                    key_str = str(key)

                if key_str in self.skip_keys:
                    self.logger.debug(f"Skipping key {key_str} as it is in the skip list.")
                    continue

                if key_str not in self.config_dict["filter_list"]:
                    #print("key_str: ", key_str)
                    # print(key.decode("ascii"))
                    graph = load_dgl_graph_from_serialized(pickle.loads(value))
                    #print(graph)
                    graph = feature_scaler([graph])
                    graph = label_scaler(graph)

                    txn = lmdb_file.begin(write=True)
                    txn.put(
                        f"{key_str}".encode("ascii"),
                        pickle.dumps(
                            serialize_dgl_graph(graph[0], ret=True), protocol=-1
                        ),
                    )
                    txn.commit()
                    scaled_count += 1

            # set the scaled flag to True in the LMDB file and on the class 
            self.scaled = True
            txn = lmdb_file.begin(write=True)
            txn.put(
                "scaled".encode("ascii"),
                pickle.dumps(True, protocol=-1),
            )
            txn.commit()
            self.logger.info(f"Scaled {scaled_count} graphs.")
            if return_info:
                return {
                    "scaled_count": scaled_count,
                    "total_count": len(list(cursor)),
                }


    def scale_graph_lmdb(self, return_info=False) -> None:

        if self.config_dict["chunk"] == -1:
            self.logger.info("Starting graph scaling...")
            db = lmdb.open(
                self.file,
                map_size=int(1099511627776 * 2),
                subdir=False,
                meminit=False,
                map_async=True,
            )

            
            ret_dict = self.scale_graphs_single(
                lmdb_file=db,
                label_scaler=self.label_scaler_iterative,
                feature_scaler=self.feature_scaler_iterative,
                return_info=return_info,
            )
            if return_info:
                return ret_dict
            
            self.logger.info("Graph scaling completed.")

        else:
            # TODO: implement chunking for folder of LMDBs
            self.logger.error("Scaling/Chunking for folder of LMDBs is not implemented yet.")
            raise NotImplementedError("Scaling/Chunking for folder of LMDBs is not implemented yet.")
            """for key in self.lmdb_dict[key_lmdb]["lmdb_paths"]:
                print(f"Scaling LMDB file: {key}")
                db = lmdb.open(
                    key,
                    map_size=int(1099511627776 * 2),
                    subdir=False,
                    meminit=False,
                    map_async=True,
                )

                self.scale_graphs_single(
                    lmdb_file=db,
                    label_scaler=self.label_scaler_iterative_qtaim,
                    feature_scaler=self.feature_scaler_iterative_qtaim,
                )"""



    def finalize(self, return_info=False, keys_to_iterate=None, processed_count=0):
        # indicates that the scalers should not be updated anymore 
        self.feature_scaler_iterative.finalize()
        self.label_scaler_iterative.finalize()

        if self.save_scaler:
            lmdb_path_qtaim = self.config_dict["lmdb_path"]

            self.feature_scaler_iterative.save_scaler(
                lmdb_path_qtaim + "/feature_scaler_iterative.pt"
            )
            self.label_scaler_iterative.save_scaler(
                lmdb_path_qtaim + "/label_scaler_iterative.pt"
            )
        
        # last info on whether the graphs were scaled or not
        txn = self.db.begin(write=True)
        txn.put("scaled".encode("ascii"), pickle.dumps(False, protocol=-1))
        txn.commit()
        self.db.close()

        print("error dict stats:")
        for key in self.fail_log_dict.keys():
            print(f"{key}: \t\t {len(self.fail_log_dict[key])} errors")
            if len(self.fail_log_dict[key]) > 0:
                print(f"error keys: \t{self.fail_log_dict[key]}")
        
        # number of keys in the lmdb file
        print(f"Total number of keys in LMDB: {len(keys_to_iterate)}")
        
        if return_info:
            return {
                "fail_log_dict": self.fail_log_dict,
                "keys_to_iterate": keys_to_iterate,
                "processed_count": processed_count,
            }


    def overwrite_config(self, file_location=None):
        """
        Overwrite the config file with the current config_dict.
        """
        if file_location is None:
            file_location = self.config_path
        self.logger.info(f"Overwriting config file at {file_location}")
        with open(file_location, "w") as f:
            json.dump(self.config_dict, f, indent=4)
            

    
    def __getitem__(self, key_lmdb, idx):
        """
        Get the item from the LMDB file or folder. Idx is an index for folders but a key for single lmdbs.
        Args:
            env: The LMDB environment.
            key_lmdb: The key for the LMDB file or folder.
            idx: The index of the item to get.
        Returns:
            data_object: The data object retrieved from the LMDB.
        """
        if not self.single_lmdb_in:
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect_right(self.lmdb_dict[key_lmdb]["keylen_cumulative"], idx)
            if db_idx == 0:
                idx_in_db = idx
            else:
                idx_in_db = idx - self.lmdb_dict[key_lmdb]["keylen_cumulative"][db_idx - 1]
                
            key_in_db = self.lmdb_dict[key_lmdb]["keys_raw"][db_idx][idx_in_db]
            #print(self.lmdb_dict[key_lmdb]["keys_raw"][db_idx])
            #print(f"Retrieving key {key_in_db} from LMDB {key_lmdb} at index {idx}.")

            datapoint_pickled = self.lmdb_dict[key_lmdb]["envs"][db_idx]["env"].begin().get(key_in_db)
            
            if datapoint_pickled is None:
                self.logger.warning(f"Key {key_in_db} not found in LMDB {key_lmdb} at index {idx}.")
                return None
            data_object = pickle.loads(datapoint_pickled)

        else:
            # value_charge = pickle.loads(self.lmdb_dict["charge_lmdb"].get(key))
            datapoint_pickled = self.lmdb_dict[key_lmdb]["env"].begin().get(idx)
            # throw
            # handle if the key is not in the lmdb
            if datapoint_pickled is None:
                self.logger.warning(f"Key {idx} not found in LMDB {key_lmdb}.")
                # assert False, f"Key {idx} not found in LMDB {key_lmdb}."
                return None
            # assert False, f"LMDB file {key_lmdb} is not a folder. Use the key directly."
            data_object = pickle.loads(datapoint_pickled)
        #print("data object retrieved: ", data_object)
        return data_object


class BaseConverter(Converter):
    def __init__(self, config_dict: Dict[str, Any], config_path: str = None):

        super().__init__(config_dict, config_path)
        self.fail_log_dict = {
            "structure": [],
            "graph": [],
            "scaler": [],
        }

    
        key = "geom_lmdb"
        
        assert (
            key in self.config_dict["lmdb_locations"].keys()
        ), f"The config file must contain a key '{key}'"

    def process(
        self,
        return_info=False,
    ) -> Dict[
        str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler]
    ]:
        """
        Main loop for processing the LMDB files and generating the graphs.
        """
        self.logger.info("Starting BaseConverter processing...")
        keys_to_iterate = self.lmdb_dict["geom_lmdb"]["keys"]
        self.logger.info(f"Total keys to iterate: {len(keys_to_iterate)}")
        
        processed_count = 0
        for key in keys_to_iterate:
            
            if type(key) == bytes:
                key_str = key.decode("ascii")
            else:
                key_str = key

            if self.restart and key_str in self.existing_keys:
                self.logger.info(f"Key {key_str} already exists in LMDB. Skipping.")
                continue

            try:
                # structure keys
                value_structure = self.__getitem__("geom_lmdb", key)
                
                if value_structure != None:
                    mol_graph = value_structure["molecule_graph"]
                    #print(mol_graph)
                    n_atoms = len(mol_graph)
                    spin = value_structure["spin"]
                    charge = value_structure["charge"]
                    bonds = value_structure["bonds"]
                    
                    if self.single_lmdb_in:
                        id = clean_id(key)
                    else: 
                        id = key_str
                    

                    # global features
                    global_feats = {
                        "charge": charge,
                        "spin": spin,
                        "n_atoms": n_atoms,
                        "n_bonds": len(bonds),
                    }

                    # initialize atom_feats with dictionary whos keys are atom inds and empty dict as values
                    atom_feats = {i: {} for i in range(n_atoms)}
                    bond_feats = {}
                    bond_list = {
                        tuple(sorted(b)): None for b in bonds if b[0] != b[1]
                    }

                else:
                    self.fail_log_dict["structure"].append(key)
                    self.logger.debug(f"Failed to retrieve structure for key: {key}")
                    continue
            except Exception as e:
                self.fail_log_dict["structure"].append(key)
                self.logger.debug(f"Exception retrieving structure for key {key}: {str(e)}")
                continue


            ############################# Molwrapper ####################################
            try:
                mol_wrapper = MoleculeWrapper(
                    mol_graph,
                    functional_group=None,
                    free_energy=None,
                    id=id,
                    bonds=bond_list,
                    non_metal_bonds=bond_list,
                    atom_features=atom_feats,
                    bond_features=bond_feats,
                    global_features=global_feats,
                    original_atom_ind=None,
                    original_bond_mapping=None,
                )

                if not self.grapher:
                    self.grapher = get_grapher(
                        element_set=self.element_set,
                        atom_keys=self.keys_data["atom"],
                        bond_keys=self.keys_data["bond"],
                        global_keys=self.keys_data["global"],
                        allowed_ring_size=self.config_dict["allowed_ring_size"],
                        allowed_charges=self.config_dict["allowed_charges"],
                        allowed_spins=self.config_dict["allowed_spins"],
                        self_loop=True,
                        atom_featurizer_tf=True,
                        bond_featurizer_tf=True,
                        global_featurizer_tf=True,
                    )
                    # add grapher info to log
                    self.logger.info(f"Grapher initialized with the following settings:")
                    self.logger.info(f"Element set: {self.element_set}")
                    self.logger.info(f"feat_names: {self.grapher.feat_names}")
                    # global feature info
                    self.logger.info(f">>>>>   Global features info   <<<<<")
                    self.logger.info(f"Allowed charges: {self.grapher.global_featurizer.allowed_charges}")
                    self.logger.info(f"Allowed spins: {self.grapher.global_featurizer.allowed_spins}")
                    self.logger.info(f"Feature size: {self.grapher.global_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.global_featurizer._feature_name}")
                    # print out the atom and bond feature info
                    self.logger.info(f">>>>>   Atom features info   <<<<<")
                    self.logger.info(f"Elements set: {self.grapher.atom_featurizer.element_set}")
                    self.logger.info(f"Feature size: {self.grapher.atom_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.atom_featurizer._feature_name}")
                    self.logger.info(f">>>>>   Bond features info   <<<<<")
                    self.logger.info(f"Allowed ring sizes: {self.grapher.bond_featurizer.allowed_ring_size}")
                    self.logger.info(f"Feature size: {self.grapher.bond_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.bond_featurizer._feature_name}")


                graph = build_and_featurize_graph(
                    self.grapher, mol_wrapper
                )

                if self.index_dict == {}:

                    self.index_dict = get_include_exclude_indices(
                        feat_names=self.grapher.feat_names,
                        target_dict=self.keys_target,
                    )
                    # save self.index_dict to config
                    self.config_dict["index_dict"] = self.index_dict
                    self.overwrite_config()
                
                split_graph_labels(
                    graph,
                    include_names=self.index_dict["include_names"],
                    include_locs=self.index_dict["include_locs"],
                    exclude_locs=self.index_dict["exclude_locs"],
                )
                
            except:
                self.fail_log_dict["graph"].append(key.decode("ascii"))
                continue

            try:
                self.feature_scaler_iterative.update([graph])
                self.label_scaler_iterative.update([graph])

                txn = self.db.begin(write=True)
                txn.put(
                    f"{key_str}".encode("ascii"),
                    pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                )
                txn.commit()
                processed_count += 1

            except Exception as e:
                self.fail_log_dict["scaler"].append(key.decode("ascii"))
                self.logger.debug(f"Exception adding graph to scaler for key {key}: {str(e)}")
                continue

        ret_dict = self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=processed_count)
        if return_info:
            return ret_dict


class QTAIMConverter(Converter):
    def __init__(self, config_dict: Dict[str, Any], config_path: str = None):

        super().__init__(config_dict, config_path)

        self.fail_log_dict = {
            "structure": [],
            "qtaim": [],
            "graph": [],
            "scaler": [],
        }

        for data_input in ["geom", "qtaim"]:
            key = f"{data_input}_lmdb"
            
            assert (
                key in self.config_dict["lmdb_locations"].keys()
            ), f"The config file must contain a key '{key}'"
   
    def process(
        self,
        return_info=False,
    ) -> Dict[
        str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler], 

    ]:
        """
        Main loop for processing the LMDB files and generating the graphs.
        """
        self.logger.info("Starting QTAIMConverter processing...")
        keys_to_iterate = self.lmdb_dict["geom_lmdb"]["keys"]
        self.logger.info(f"Total keys to iterate: {len(keys_to_iterate)}")
        
        processed_count = 0
        for key in keys_to_iterate:
            
            if type(key) == bytes:
                key_str = key.decode("ascii")
            else:
                key_str = key
            
            if self.restart and key_str in self.existing_keys:
                self.logger.info(f"Key {key_str} already exists in LMDB. Skipping.")
                continue

            try:
                # structure keys
                value_structure = self.__getitem__("geom_lmdb", key)

                if value_structure != None:
                    mol_graph = value_structure["molecule_graph"]
                    n_atoms = len(mol_graph)
                    spin = value_structure["spin"]
                    charge = value_structure["charge"]
                    bonds = value_structure["bonds"]

                    if self.single_lmdb_in:
                        id = clean_id(key)
                    else: 
                        id = key_str

                    # global features
                    global_feats = {
                        "charge": charge,
                        "spin": spin,
                        "n_atoms": n_atoms,
                        "n_bonds": len(bonds),
                    }

                    # initialize atom_feats with dictionary whos keys are atom inds and empty dict as values
                    atom_feats = {i: {} for i in range(n_atoms)}
                    bond_feats = {}

                    
                else:
                    self.fail_log_dict["structure"].append(key.decode("ascii"))
                    self.logger.debug(f"Failed to retrieve structure for key: {key}")
                    continue
            except Exception as e:
                self.fail_log_dict["structure"].append(key.decode("ascii"))
                self.logger.debug(f"Exception retrieving structure for key {key}: {str(e)}")
                continue
            

            try:
                dict_qtaim_raw = self.__getitem__("qtaim_lmdb", key)
                #print("qtaim raw data: ", dict_qtaim_raw["0"].keys())
                if dict_qtaim_raw != None:
                    # parse qtaim data
                    (
                        atom_keys_qtaim,
                        bond_keys_qtaim,
                        atom_feats,
                        bond_feats,
                        connected_bond_paths,
                    ) = parse_qtaim_data(
                        dict_qtaim_raw,
                        atom_feats,
                        bond_feats,
                        atom_keys=self.keys_data["atom"],
                        bond_keys=self.keys_data["bond"],
                    )
                    #print("qtaim data")
                    #print(atom_feats)
                    #print(bond_feats)
                    #print(connected_bond_paths)
                else:
                    self.fail_log_dict["qtaim"].append(key.decode("ascii"))
                    self.logger.debug(f"Failed to retrieve QTAIM data for key: {key}")
                    continue

            except Exception as e:
                self.fail_log_dict["qtaim"].append(key.decode("ascii"))
                self.logger.debug(f"Exception retrieving QTAIM data for key {key}: {str(e)}")
                continue
        
            ############################# Molwrapper ####################################

            try:
                mol_wrapper = MoleculeWrapper(
                    mol_graph,
                    functional_group=None,
                    free_energy=None,
                    id=id,
                    bonds=connected_bond_paths,
                    non_metal_bonds=connected_bond_paths,
                    atom_features=atom_feats,
                    bond_features=bond_feats,
                    global_features=global_feats,
                    original_atom_ind=None,
                    original_bond_mapping=None,
                )

                if not self.grapher:
                    self.grapher = get_grapher(
                        element_set=self.element_set,
                        atom_keys=self.keys_data["atom"],
                        bond_keys=self.keys_data["bond"],
                        global_keys=self.keys_data["global"],
                        allowed_ring_size=self.config_dict["allowed_ring_size"],
                        allowed_charges=self.config_dict["allowed_charges"],
                        allowed_spins=self.config_dict["allowed_spins"],
                        self_loop=True,
                        atom_featurizer_tf=True,
                        bond_featurizer_tf=True,
                        global_featurizer_tf=True,
                    )
                    # print("\nbond keys qtaim: ", self.bond_keys)
                    # add grapher info to log
                    self.logger.info(f"Grapher initialized with the following settings:")
                    self.logger.info(f"Element set: {self.element_set}")
                    self.logger.info(f"feat_names: {self.grapher.feat_names}")
                    # global feature info
                    self.logger.info(f">>>>>   Global features info   <<<<<")
                    self.logger.info(f"Allowed charges: {self.grapher.global_featurizer.allowed_charges}")
                    self.logger.info(f"Allowed spins: {self.grapher.global_featurizer.allowed_spins}")
                    self.logger.info(f"Feature size: {self.grapher.global_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.global_featurizer._feature_name}")
                    # print out the atom and bond feature info
                    self.logger.info(f">>>>>   Atom features info   <<<<<")
                    self.logger.info(f"Elements set: {self.grapher.atom_featurizer.element_set}")
                    self.logger.info(f"Feature size: {self.grapher.atom_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.atom_featurizer._feature_name}")
                    self.logger.info(f">>>>>   Bond features info   <<<<<")
                    self.logger.info(f"Allowed ring sizes: {self.grapher.bond_featurizer.allowed_ring_size}")
                    self.logger.info(f"Feature size: {self.grapher.bond_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.bond_featurizer._feature_name}")

                graph = build_and_featurize_graph(
                    self.grapher, mol_wrapper
                )

                if self.index_dict == {}:

                    self.index_dict = get_include_exclude_indices(
                        feat_names=self.grapher.feat_names,
                        target_dict=self.keys_target,
                    )
                    # save self.index_dict to config
                    self.config_dict["index_dict"] = self.index_dict
                    self.overwrite_config()
                
                split_graph_labels(
                    graph,
                    include_names=self.index_dict["include_names"],
                    include_locs=self.index_dict["include_locs"],
                    exclude_locs=self.index_dict["exclude_locs"],
                )
                
            except:
                self.fail_log_dict["graph"].append(key.decode("ascii"))
                continue

            try:
                self.feature_scaler_iterative.update([graph])
                self.label_scaler_iterative.update([graph])

                txn = self.db.begin(write=True)
                txn.put(
                    f"{key_str}".encode("ascii"),
                    pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                )
                txn.commit()
                processed_count += 1

            except:
                self.fail_log_dict["scaler"].append(key.decode("ascii"))
                continue

        ret_info = self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=processed_count)
        if return_info:
            return ret_info


class GeneralConverter(Converter):
    def __init__(self, config_dict: Dict[str, Any], config_path: str = None):

        super().__init__(config_dict, config_path)

        self.fail_log_dict = {
            "structure": [],
            "graph": [],
            "scaler": [],
            "qtaim": [], # possible set of dicts to draw from
            "charge": [],
            "fuzzy_full": [], 
            "bonds": [], 
            "other": []
        }

        self.bonding_scheme = self.config_dict.get("bonding_scheme", "qtaim")
        self.data_inputs = self.config_dict.get("data_inputs", ["geom", "qtaim", "charge"]) # add fuzzy_full, bonds, other as possible data inputs

        # assert that for each data input, the corresponding LMDB is in the config dict in "lmdb_locations"
    
        for data_input in self.data_inputs:
            key = f"{data_input}_lmdb"
            
            assert (
                key in self.config_dict["lmdb_locations"].keys()
            ), f"The config file must contain a key '{key}'"



    def process(
        self,
        return_info=False,
    ) -> Dict[
        str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler], 

    ]:
        """
        Main loop for processing the LMDB files and generating the graphs.
        """
        self.logger.info("Starting GeneralConverter processing...")
        
        keys_to_iterate = self.lmdb_dict["geom_lmdb"]["keys"]
        self.logger.info(f"Total keys to iterate: {len(keys_to_iterate)}")
        
        processed_count = 0
        for key in keys_to_iterate:
            
            if type(key) == bytes:
                key_str = key.decode("ascii")
            else:
                key_str = key
            
            if self.restart and key_str in self.existing_keys:
                self.logger.info(f"Key {key_str} already exists in LMDB. Skipping.")
                continue

            try:
                # structure keys
                value_structure = self.__getitem__("geom_lmdb", key)

                if value_structure != None:
                    mol_graph = value_structure["molecule_graph"]
                    n_atoms = len(mol_graph)
                    spin = value_structure["spin"]
                    charge = value_structure["charge"]
                    bonds = value_structure["bonds"]

                    if self.single_lmdb_in:
                        id = clean_id(key)
                    else: 
                        id = key_str

                    # global features
                    global_feats = {
                        "charge": charge,
                        "spin": spin,
                        "n_atoms": n_atoms,
                        "n_bonds": len(bonds),
                    }

                    # initialize atom_feats with dictionary whos keys are atom inds and empty dict as values
                    atom_feats = {i: {} for i in range(n_atoms)}
                    bond_feats = {}

                    
                else:
                    self.fail_log_dict["structure"].append(key.decode("ascii"))
                    self.logger.debug(f"Failed to retrieve structure for key: {key}")
                    continue

            except Exception as e:
                self.fail_log_dict["structure"].append(key.decode("ascii"))
                self.logger.debug(f"Exception retrieving structure for key {key}: {str(e)}")
                continue
            

            try:
                dict_qtaim_raw = self.__getitem__("qtaim_lmdb", key)
                #print("qtaim raw data: ", dict_qtaim_raw["0"].keys())
                if dict_qtaim_raw != None:
                    # parse qtaim data
                    (
                        atom_keys_qtaim,
                        bond_keys_qtaim,
                        atom_feats,
                        bond_feats,
                        connected_bond_paths,
                    ) = parse_qtaim_data(
                        dict_qtaim_raw,
                        atom_feats,
                        bond_feats,
                        atom_keys=self.keys_data["atom"],
                        bond_keys=self.keys_data["bond"],
                    )
                    #print("qtaim data")
                    #print(atom_feats)
                    #print(bond_feats)
                    #print(connected_bond_paths)
                else:
                    self.fail_log_dict["qtaim"].append(key.decode("ascii"))
                    self.logger.debug(f"Failed to retrieve QTAIM data for key: {key}")
                    continue

            except Exception as e:
                self.fail_log_dict["qtaim"].append(key.decode("ascii"))
                self.logger.debug(f"Exception retrieving QTAIM data for key {key}: {str(e)}")
                continue
        
            ############################# Molwrapper ####################################

            try:
                mol_wrapper = MoleculeWrapper(
                    mol_graph,
                    functional_group=None,
                    free_energy=None,
                    id=id,
                    bonds=connected_bond_paths,
                    non_metal_bonds=connected_bond_paths,
                    atom_features=atom_feats,
                    bond_features=bond_feats,
                    global_features=global_feats,
                    original_atom_ind=None,
                    original_bond_mapping=None,
                )

                if not self.grapher:
                    self.grapher = get_grapher(
                        element_set=self.element_set,
                        atom_keys=self.keys_data["atom"],
                        bond_keys=self.keys_data["bond"],
                        global_keys=self.keys_data["global"],
                        allowed_ring_size=self.config_dict["allowed_ring_size"],
                        allowed_charges=self.config_dict["allowed_charges"],
                        allowed_spins=self.config_dict["allowed_spins"],
                        self_loop=True,
                        atom_featurizer_tf=True,
                        bond_featurizer_tf=True,
                        global_featurizer_tf=True,
                    )
                    # print("\nbond keys qtaim: ", self.bond_keys)
                    # add grapher info to log
                    self.logger.info(f"Grapher initialized with the following settings:")
                    self.logger.info(f"Element set: {self.element_set}")
                    self.logger.info(f"feat_names: {self.grapher.feat_names}")
                    # global feature info
                    self.logger.info(f">>>>>   Global features info   <<<<<")
                    self.logger.info(f"Allowed charges: {self.grapher.global_featurizer.allowed_charges}")
                    self.logger.info(f"Allowed spins: {self.grapher.global_featurizer.allowed_spins}")
                    self.logger.info(f"Feature size: {self.grapher.global_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.global_featurizer._feature_name}")
                    # print out the atom and bond feature info
                    self.logger.info(f">>>>>   Atom features info   <<<<<")
                    self.logger.info(f"Elements set: {self.grapher.atom_featurizer.element_set}")
                    self.logger.info(f"Feature size: {self.grapher.atom_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.atom_featurizer._feature_name}")
                    self.logger.info(f">>>>>   Bond features info   <<<<<")
                    self.logger.info(f"Allowed ring sizes: {self.grapher.bond_featurizer.allowed_ring_size}")
                    self.logger.info(f"Feature size: {self.grapher.bond_featurizer.feature_size}")
                    self.logger.info(f"Feature names: {self.grapher.bond_featurizer._feature_name}")

                graph = build_and_featurize_graph(
                    self.grapher, mol_wrapper
                )

                if self.index_dict == {}:

                    self.index_dict = get_include_exclude_indices(
                        feat_names=self.grapher.feat_names,
                        target_dict=self.keys_target,
                    )
                    # save self.index_dict to config
                    self.config_dict["index_dict"] = self.index_dict
                    self.overwrite_config()
                
                split_graph_labels(
                    graph,
                    include_names=self.index_dict["include_names"],
                    include_locs=self.index_dict["include_locs"],
                    exclude_locs=self.index_dict["exclude_locs"],
                )
                
            except:
                self.fail_log_dict["graph"].append(key.decode("ascii"))
                continue

            try:
                self.feature_scaler_iterative.update([graph])
                self.label_scaler_iterative.update([graph])

                txn = self.db.begin(write=True)
                txn.put(
                    f"{key_str}".encode("ascii"),
                    pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                )
                txn.commit()
                processed_count += 1

            except:
                self.fail_log_dict["scaler"].append(key.decode("ascii"))
                continue
            
        ret_info = self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=processed_count)
        if return_info:
            return ret_info

    


class ASELMDBConverter:
    # TODO: last class to implement
    pass
