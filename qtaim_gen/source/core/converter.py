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
from concurrent.futures import ThreadPoolExecutor, as_completed

from copy import deepcopy
from typing import Dict, List, Tuple, Union, Any, Optional
import bisect
import logging
from logging.handlers import RotatingFileHandler


from qtaim_embed.data.processing import (
    HeteroGraphStandardScalerIterative,
    HeteroGraphLogMagnitudeScaler,
)
from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.grapher import get_grapher
#try: 

from qtaim_embed.data.lmdb import serialize_dgl_graph, load_dgl_graph_from_serialized
serial_func = serialize_dgl_graph
#except: 
#    from qtaim_embed.data.lmdb import serialize_graph
#    serial_func = serialize_graph

from qtaim_gen.source.utils.lmdbs import (
    get_elements_from_structure_lmdb,
    get_elements_from_structure_lmdb_folder_list,
    parse_charge_data,
    parse_qtaim_data,
    parse_fuzzy_data,
    parse_other_data,
    parse_bond_data,
    gather_structure_info
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

def load_scaler(scaler_path: str, features_tf: bool):
    """Helper to load a scaler from file."""
    from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative
    return HeteroGraphStandardScalerIterative(
        features_tf=features_tf,
        load=True,
        load_path=scaler_path
    )


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
        self.single_lmdb_in = True # TODO allow multiple lmdb inputs
        self.lmdb_dict = self.pull_lmdbs()
        
        # grapher        
        self.grapher = None

        # scaler settings
        if "save_scaler" in self.config_dict.keys():
            self.save_scaler = self.config_dict["save_scaler"]
        else:
            self.save_scaler = False

        self.skip_keys = config_dict.get("filter_list", ["length", "scaled"])

        # Parallelization settings
        self.n_workers = config_dict.get("n_workers", 8)
        self.batch_size = config_dict.get("batch_size", 500)

        # Sharding settings for parallel processing of large datasets
        self.shard_index = config_dict.get("shard_index", 0)
        self.total_shards = config_dict.get("total_shards", 1)
        self.skip_scaling = config_dict.get("skip_scaling", False)
        self.save_unfinalized_scaler = config_dict.get("save_unfinalized_scaler", False)
        self.auto_merge = config_dict.get("auto_merge", False)

        if self.total_shards > 1:
            self.logger.info(f"Sharding enabled: shard {self.shard_index + 1} of {self.total_shards}")
            if self.auto_merge and self.shard_index == self.total_shards - 1:
                self.logger.info(f"Auto-merge enabled: will merge all shards after processing")

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
        
        # output lmdb file handling - supports sharding for parallel processing
        lmdb_name = self.config_dict["lmdb_name"]
        if self.total_shards > 1:
            base_name = lmdb_name.replace(".lmdb", "")
            lmdb_name = f"{base_name}_shard_{self.shard_index}.lmdb"
            self.logger.info(f"Shard output file: {lmdb_name}")

        self.file = os.path.join(
            self.config_dict["lmdb_path"],
            lmdb_name if lmdb_name.endswith(".lmdb") else f"{lmdb_name}.lmdb"
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
            max_readers=126,
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
            items = list(cursor)
            scaled_count = 0

            # Diagnostic logging: report key counts and examples
            try:
                key_strs = [k.decode("ascii") if isinstance(k, bytes) else str(k) for k, _ in items]
            except Exception:
                key_strs = [str(k) for k, _ in items]
            self.logger.debug(f"scale_graphs_single: total keys in cursor: {len(key_strs)}")
            self.logger.debug(f"scale_graphs_single: example keys: {key_strs[:20]}")

            skipped_by_skip_keys = 0
            skipped_by_filter_list = 0

            for key, value in items:
                # normalize key to string so comparisons and writes are consistent
                if isinstance(key, bytes):
                    key_str = key.decode("ascii")
                else:
                    key_str = str(key)
                if key_str in self.skip_keys:
                    skipped_by_skip_keys += 1
                    self.logger.debug(f"Skipping key {key_str} as it is in the skip list.")
                    continue

                if key_str not in self.config_dict["filter_list"]:
                    # process graph
                    try:
                        graph = load_dgl_graph_from_serialized(pickle.loads(value))
                    except Exception as e:
                        self.logger.exception(f"Failed to load graph for key {key_str}: {e}")
                        continue
                    # apply scalers
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
                else:
                    skipped_by_filter_list += 1

            # Report diagnostics
            self.logger.debug(f"scale_graphs_single: processed {scaled_count} keys; skipped_by_skip_keys={skipped_by_skip_keys}, skipped_by_filter_list={skipped_by_filter_list}")

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
                    "total_count": len(items),
                }


    def scale_graph_lmdb(self, return_info=False) -> None:
        # If the converter already knows the LMDB is scaled (e.g. restart mode), skip scaling
        if getattr(self, "scaled", False):
            self.logger.info("Existing LMDB already scaled. Skipping scaling.")
            if return_info:
                return None
            return

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


    def _partition_keys(self, keys: list) -> list:
        """
        Partition keys for this shard using deterministic modulo assignment.

        Args:
            keys: List of all keys to partition

        Returns:
            List of keys assigned to this shard
        """
        if self.total_shards <= 1:
            return keys

        partitioned = [k for i, k in enumerate(keys) if i % self.total_shards == self.shard_index]
        self.logger.info(f"Shard {self.shard_index}: {len(partitioned)} keys (of {len(keys)} total)")
        return partitioned


    def _auto_merge_shards(self):
        """
        Automatically merge all shards after this shard completes processing.

        This method is called by the last shard (shard_index == total_shards - 1)
        when auto_merge is enabled. It constructs the list of shard directories
        and calls merge_shards() to combine them.
        """
        self.logger.info("=" * 60)
        self.logger.info("AUTO-MERGE: Starting automatic shard merging...")
        self.logger.info("=" * 60)

        # Construct shard directory list
        base_lmdb_path = self.config_dict["lmdb_path"]

        # Get parent directory (remove current shard suffix)
        if base_lmdb_path.endswith(f"shard_{self.shard_index}"):
            parent_dir = os.path.dirname(base_lmdb_path)
        else:
            # Try to infer parent from path
            parent_dir = os.path.dirname(base_lmdb_path)

        # Build list of all shard directories
        shard_dirs = []
        for i in range(self.total_shards):
            shard_dir = os.path.join(parent_dir, f"shard_{i}")
            if os.path.exists(shard_dir):
                shard_dirs.append(shard_dir)
            else:
                self.logger.warning(f"Shard directory not found: {shard_dir}")

        if len(shard_dirs) != self.total_shards:
            self.logger.error(
                f"Expected {self.total_shards} shards but found {len(shard_dirs)}. "
                "Skipping auto-merge. Run merge manually."
            )
            return

        # Create merged output directory
        merged_dir = os.path.join(parent_dir, "merged")

        self.logger.info(f"Merging {len(shard_dirs)} shards into: {merged_dir}")

        try:
            merged_path = self.__class__.merge_shards(
                shard_dirs=shard_dirs,
                output_dir=merged_dir,
                output_name="merged.lmdb",
                skip_scaling=self.skip_scaling,  # Respect config setting
                logger=self.logger
            )
            self.logger.info("=" * 60)
            self.logger.info(f"AUTO-MERGE COMPLETE: {merged_path}")
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"Auto-merge failed: {e}", exc_info=True)
            self.logger.error("You can manually merge with: Converter.merge_shards()")


    def finalize(self, return_info=False, keys_to_iterate=None, processed_count=0):
        lmdb_path = self.config_dict["lmdb_path"]

        # For sharded runs, save unfinalized scalers for later merging
        if self.save_unfinalized_scaler:
            shard_suffix = f"_shard_{self.shard_index}" if self.total_shards > 1 else ""
            self.feature_scaler_iterative.save_scaler(
                f"{lmdb_path}/feature_scaler_unfinalized{shard_suffix}.pt"
            )
            self.label_scaler_iterative.save_scaler(
                f"{lmdb_path}/label_scaler_unfinalized{shard_suffix}.pt"
            )
            self.logger.info(f"Saved unfinalized scalers for merging (shard {self.shard_index})")

        # Finalize scalers (compute final mean/std)
        self.feature_scaler_iterative.finalize()
        self.label_scaler_iterative.finalize()

        if self.save_scaler:
            shard_suffix = f"_shard_{self.shard_index}" if self.total_shards > 1 else ""
            self.feature_scaler_iterative.save_scaler(
                f"{lmdb_path}/feature_scaler_iterative{shard_suffix}.pt"
            )
            self.label_scaler_iterative.save_scaler(
                f"{lmdb_path}/label_scaler_iterative{shard_suffix}.pt"
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


    @classmethod
    def merge_shards(
        cls,
        shard_dirs: List[str],
        output_dir: str,
        output_name: str = "merged.lmdb",
        skip_scaling: bool = False,
        logger: Optional[logging.Logger] = None
    ) -> str:
        """
        Merge multiple sharded converter outputs into a single LMDB.

        Args:
            shard_dirs: List of shard directory paths to merge
            output_dir: Output directory for merged LMDB
            output_name: Name of output LMDB file (default: "merged.lmdb")
            skip_scaling: Skip applying merged scalers to LMDB (default: False)
            logger: Logger instance (creates new one if None)

        Returns:
            Path to merged LMDB file

        Example:
            >>> shard_dirs = ["output/shard_0", "output/shard_1", "output/shard_2"]
            >>> merged_path = QTAIMConverter.merge_shards(
            ...     shard_dirs=shard_dirs,
            ...     output_dir="output/merged"
            ... )
        """
        from qtaim_embed.data.processing import merge_scalers

        if logger is None:
            logger = logging.getLogger("MergeShards")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                ))
                logger.addHandler(handler)

        os.makedirs(output_dir, exist_ok=True)

        # Find LMDB files in shard directories
        logger.info(f"Merging {len(shard_dirs)} shards...")
        shard_lmdbs = []
        for shard_dir in shard_dirs:
            lmdb_files = list(glob(os.path.join(shard_dir, "*.lmdb")))
            if not lmdb_files:
                logger.warning(f"No LMDB files found in {shard_dir}, skipping")
                continue
            shard_lmdbs.append(lmdb_files[0])

        if not shard_lmdbs:
            raise ValueError("No LMDB files found in any shard directory")

        # Create merged LMDB
        output_path = os.path.join(output_dir, output_name)
        # Keep .lmdb extension for consistency
        logger.info(f"Creating merged LMDB: {output_path}")

        # Calculate total entries
        total_entries = 0
        for lmdb_path in shard_lmdbs:
            env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
            with env.begin() as txn:
                total_entries += txn.stat()["entries"]
            env.close()

        logger.info(f"Total entries to merge: {total_entries}")

        # Create output LMDB
        map_size = max(1099511627776, total_entries * 100000)
        merged_env = lmdb.open(
            output_path,
            subdir=False,
            map_size=map_size,
            writemap=True,
            map_async=True
        )

        # Copy all entries from shards
        total_copied = 0
        with merged_env.begin(write=True) as dst_txn:
            for i, lmdb_path in enumerate(shard_lmdbs):
                logger.info(f"Copying shard {i+1}/{len(shard_lmdbs)}")
                src_env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
                with src_env.begin() as src_txn:
                    cursor = src_txn.cursor()
                    for key, value in cursor:
                        dst_txn.put(key, value)
                        total_copied += 1
                src_env.close()

        merged_env.close()
        logger.info(f"Merged {total_copied} entries")

        # Merge scalers
        logger.info("Merging feature scalers...")
        feature_scaler_paths = []
        for shard_dir in shard_dirs:
            scaler_path = os.path.join(shard_dir, "feature_scaler_unfinalized.pt")
            if not os.path.exists(scaler_path):
                pattern = os.path.join(shard_dir, "feature_scaler_unfinalized_shard_*.pt")
                matches = glob(pattern)
                if matches:
                    scaler_path = matches[0]
            if os.path.exists(scaler_path):
                feature_scaler_paths.append(scaler_path)

        if feature_scaler_paths:
            scalers = [load_scaler(p, features_tf=True) for p in feature_scaler_paths]
            merged_feature_scaler = merge_scalers(scalers, features_tf=True, finalize_merged=True)
            feature_out = os.path.join(output_dir, "feature_scaler_iterative.pt")
            merged_feature_scaler.save_scaler(feature_out)
            logger.info(f"Saved merged feature scaler")
        else:
            logger.warning("No feature scalers found")
            merged_feature_scaler = None

        # Merge label scalers
        logger.info("Merging label scalers...")
        label_scaler_paths = []
        for shard_dir in shard_dirs:
            scaler_path = os.path.join(shard_dir, "label_scaler_unfinalized.pt")
            if not os.path.exists(scaler_path):
                pattern = os.path.join(shard_dir, "label_scaler_unfinalized_shard_*.pt")
                matches = glob(pattern)
                if matches:
                    scaler_path = matches[0]
            if os.path.exists(scaler_path):
                label_scaler_paths.append(scaler_path)

        if label_scaler_paths:
            scalers = [load_scaler(p, features_tf=False) for p in label_scaler_paths]
            merged_label_scaler = merge_scalers(scalers, features_tf=False, finalize_merged=True)
            label_out = os.path.join(output_dir, "label_scaler_iterative.pt")
            merged_label_scaler.save_scaler(label_out)
            logger.info(f"Saved merged label scaler")
        else:
            logger.warning("No label scalers found")
            merged_label_scaler = None

        # Apply scalers to merged LMDB
        if not skip_scaling and merged_feature_scaler and merged_label_scaler:
            logger.info("Applying merged scalers to LMDB...")
            env = lmdb.open(output_path, subdir=False, map_size=map_size)
            count = 0
            metadata_keys = {b'scaled', b'scaler_finalized'}
            with env.begin(write=True) as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    # Skip metadata keys
                    if key in metadata_keys:
                        continue

                    try:
                        # Deserialize: pickle.loads returns bytes, then deserialize to DGLGraph
                        serialized_bytes = pickle.loads(value)
                        graph = load_dgl_graph_from_serialized(serialized_bytes)

                        # Apply scalers - feature scaler expects a list
                        graph = merged_feature_scaler([graph])
                        graph = merged_label_scaler(graph)

                        # Serialize and write back
                        serialized_bytes = serialize_dgl_graph(graph[0], ret=True)
                        txn.put(key, pickle.dumps(serialized_bytes, protocol=-1))
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to scale graph {key}: {e}")
            env.close()
            logger.info(f"Applied scalers to {count} graphs")

        # Merge configs
        first_config = os.path.join(shard_dirs[0], "config.json")
        if os.path.exists(first_config):
            with open(first_config, "r") as f:
                config = json.load(f)
            config.pop("shard_index", None)
            config.pop("total_shards", None)
            config.pop("skip_scaling", None)
            config.pop("save_unfinalized_scaler", None)
            config["lmdb_path"] = os.path.abspath(output_dir)
            config["lmdb_name"] = output_name
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"Merge complete: {output_path}")
        return output_path


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

    def _build_graph_for_key_base(
        self,
        key: bytes,
        grapher,
        index_dict: dict,
    ) -> Tuple[str, Optional[Any], Dict[str, List[str]]]:
        """Build a graph for a single key. Thread-safe for parallel processing."""
        failures = {k: [] for k in self.fail_log_dict.keys()}

        if isinstance(key, bytes):
            key_str = key.decode("ascii")
        else:
            key_str = str(key)

        if self.restart and key_str in self.existing_keys:
            return (key_str, None, failures)

        try:
            value_structure = self.__getitem__("geom_lmdb", key)
            if value_structure is None:
                failures["structure"].append(key_str)
                return (key_str, None, failures)

            mol_graph, global_feats = gather_structure_info(value_structure)
            bond_feats = {}
            atom_feats = {i: {} for i in range(global_feats["n_atoms"])}
            bonds = value_structure["bonds"]
            bond_list = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}
            id = clean_id(key) if self.single_lmdb_in else key_str
        except Exception as e:
            failures["structure"].append(key_str)
            return (key_str, None, failures)

        try:
            mol_wrapper = MoleculeWrapper(
                mol_graph, functional_group=None, free_energy=None, id=id,
                bonds=bond_list, non_metal_bonds=bond_list,
                atom_features=atom_feats, bond_features=bond_feats,
                global_features=global_feats, original_atom_ind=None, original_bond_mapping=None,
            )
            graph = build_and_featurize_graph(grapher, mol_wrapper)
            split_graph_labels(
                graph,
                include_names=index_dict["include_names"],
                include_locs=index_dict["include_locs"],
                exclude_locs=index_dict["exclude_locs"],
            )
            return (key_str, graph, failures)
        except Exception as e:
            failures["graph"].append(key_str)
            return (key_str, None, failures)

    def process(
        self,
        return_info=False,
    ) -> Dict[
        str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler]
    ]:
        """
        Main loop for processing the LMDB files and generating the graphs.
        Uses parallel processing with ThreadPoolExecutor.
        """
        self.logger.info("Starting BaseConverter processing...")
        keys_to_iterate = self.lmdb_dict["geom_lmdb"]["keys"]
        self.logger.info(f"Total keys before partitioning: {len(keys_to_iterate)}")

        # Partition keys for sharding
        keys_to_iterate = self._partition_keys(keys_to_iterate)
        self.logger.info(f"Keys to process in this shard: {len(keys_to_iterate)}")
        self.logger.info(f"Using {self.n_workers} workers for parallel processing")

        write_buffer = []
        processed_count = 0

        # Phase 1: Initialize grapher with first successful key (sequential)
        self.logger.info("Phase 1: Initializing grapher...")
        first_key_idx = 0

        for idx, key in enumerate(keys_to_iterate):
            if isinstance(key, bytes):
                key_str = key.decode("ascii")
            else:
                key_str = str(key)

            if self.restart and key_str in self.existing_keys:
                first_key_idx = idx + 1
                continue

            try:
                value_structure = self.__getitem__("geom_lmdb", key)
                if value_structure is None:
                    first_key_idx = idx + 1
                    continue

                mol_graph, global_feats = gather_structure_info(value_structure)
                bond_feats = {}
                atom_feats = {i: {} for i in range(global_feats["n_atoms"])}
                bonds = value_structure["bonds"]
                bond_list = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}
                id = clean_id(key) if self.single_lmdb_in else key_str

                mol_wrapper = MoleculeWrapper(
                    mol_graph, functional_group=None, free_energy=None, id=id,
                    bonds=bond_list, non_metal_bonds=bond_list,
                    atom_features=atom_feats, bond_features=bond_feats,
                    global_features=global_feats, original_atom_ind=None, original_bond_mapping=None,
                )
                # we only need to build the grapher once
                if self.grapher is None:
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
                    self.logger.info(f"Grapher initialized with element_set: {self.element_set}")

                first_graph = build_and_featurize_graph(self.grapher, mol_wrapper)

                self.index_dict = get_include_exclude_indices(
                    feat_names=self.grapher.feat_names,
                    target_dict=self.keys_target,
                )
                self.config_dict["index_dict"] = self.index_dict
                self.overwrite_config()

                split_graph_labels(
                    first_graph,
                    include_names=self.index_dict["include_names"],
                    include_locs=self.index_dict["include_locs"],
                    exclude_locs=self.index_dict["exclude_locs"],
                )

                self.feature_scaler_iterative.update([first_graph])
                self.label_scaler_iterative.update([first_graph])
                write_buffer.append((
                    f"{key_str}".encode("ascii"),
                    pickle.dumps(serialize_dgl_graph(first_graph, ret=True), protocol=-1),
                ))
                processed_count += 1
                first_key_idx = idx + 1
                break

            except Exception as e:
                self.logger.debug(f"Failed to initialize with key {key_str}: {e}")
                first_key_idx = idx + 1
                continue

        if self.grapher is None:
            self.logger.error("Failed to initialize grapher with any key")
            return self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=0)

        # Phase 2: Process remaining keys in parallel
        remaining_keys = keys_to_iterate[first_key_idx:]
        self.logger.info(f"Phase 1: Initialized grapher after {first_key_idx} attempts, 1 key processed")
        self.logger.info(f"Phase 2: Processing {len(remaining_keys)} remaining keys with {self.n_workers} workers...")

        def process_key(key):
            return self._build_graph_for_key_base(key, self.grapher, self.index_dict)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(process_key, key): key for key in remaining_keys}

            for future in as_completed(futures):
                try:
                    key_str, graph, failures = future.result()

                    for fail_type, fail_list in failures.items():
                        self.fail_log_dict[fail_type].extend(fail_list)

                    if graph is not None:
                        self.feature_scaler_iterative.update([graph])
                        self.label_scaler_iterative.update([graph])

                        write_buffer.append((
                            f"{key_str}".encode("ascii"),
                            pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                        ))
                        processed_count += 1

                        if len(write_buffer) >= self.batch_size:
                            txn = self.db.begin(write=True)
                            for buf_key, buf_value in write_buffer:
                                txn.put(buf_key, buf_value)
                            txn.commit()
                            self.logger.debug(f"Committed batch of {len(write_buffer)} items")
                            write_buffer.clear()

                except Exception as e:
                    self.logger.warning(f"Error processing future: {e}")

        # Flush remaining items
        if write_buffer:
            txn = self.db.begin(write=True)
            for buf_key, buf_value in write_buffer:
                txn.put(buf_key, buf_value)
            txn.commit()
            self.logger.debug(f"Committed final batch of {len(write_buffer)} items")

        ret_dict = self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=processed_count)

        # Auto-merge shards if this is the last shard
        if self.auto_merge and self.shard_index == self.total_shards - 1:
            self._auto_merge_shards()

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
   
    def _build_graph_for_key_qtaim(
        self,
        key: bytes,
        grapher,
        index_dict: dict,
    ) -> Tuple[str, Optional[Any], Dict[str, List[str]]]:
        """Build a graph for a single key. Thread-safe for parallel processing."""
        failures = {k: [] for k in self.fail_log_dict.keys()}

        if isinstance(key, bytes):
            key_str = key.decode("ascii")
        else:
            key_str = str(key)

        if self.restart and key_str in self.existing_keys:
            return (key_str, None, failures)

        try:
            value_structure = self.__getitem__("geom_lmdb", key)
            if value_structure is None:
                failures["structure"].append(key_str)
                return (key_str, None, failures)

            mol_graph, global_feats = gather_structure_info(value_structure)
            bond_feats = {}
            atom_feats = {i: {} for i in range(global_feats["n_atoms"])}
            id = clean_id(key) if self.single_lmdb_in else key_str
        except Exception as e:
            failures["structure"].append(key_str)
            return (key_str, None, failures)

        try:
            dict_qtaim_raw = self.__getitem__("qtaim_lmdb", key)
            if dict_qtaim_raw is None:
                failures["qtaim"].append(key_str)
                return (key_str, None, failures)

            (_, _, atom_feats, bond_feats, connected_bond_paths) = parse_qtaim_data(
                dict_qtaim_raw, atom_feats, bond_feats,
                atom_keys=self.keys_data["atom"],
                bond_keys=self.keys_data["bond"],
            )
        except Exception as e:
            failures["qtaim"].append(key_str)
            return (key_str, None, failures)

        try:
            mol_wrapper = MoleculeWrapper(
                mol_graph, functional_group=None, free_energy=None, id=id,
                bonds=connected_bond_paths, non_metal_bonds=connected_bond_paths,
                atom_features=atom_feats, bond_features=bond_feats,
                global_features=global_feats, original_atom_ind=None, original_bond_mapping=None,
            )
            graph = build_and_featurize_graph(grapher, mol_wrapper)
            split_graph_labels(
                graph,
                include_names=index_dict["include_names"],
                include_locs=index_dict["include_locs"],
                exclude_locs=index_dict["exclude_locs"],
            )
            return (key_str, graph, failures)
        except Exception as e:
            failures["graph"].append(key_str)
            return (key_str, None, failures)

    def process(
        self,
        return_info=False,
    ) -> Dict[
        str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler],
    ]:
        """
        Main loop for processing the LMDB files and generating the graphs.
        Uses parallel processing with ThreadPoolExecutor.
        """
        self.logger.info("Starting QTAIMConverter processing...")
        keys_to_iterate = self.lmdb_dict["geom_lmdb"]["keys"]
        self.logger.info(f"Total keys before partitioning: {len(keys_to_iterate)}")

        # Partition keys for sharding
        keys_to_iterate = self._partition_keys(keys_to_iterate)
        self.logger.info(f"Keys to process in this shard: {len(keys_to_iterate)}")
        self.logger.info(f"Using {self.n_workers} workers for parallel processing")

        write_buffer = []
        processed_count = 0

        # Phase 1: Initialize grapher with first successful key (sequential)
        self.logger.info("Phase 1: Initializing grapher...")
        first_key_idx = 0

        for idx, key in enumerate(keys_to_iterate):
            if isinstance(key, bytes):
                key_str = key.decode("ascii")
            else:
                key_str = str(key)

            if self.restart and key_str in self.existing_keys:
                first_key_idx = idx + 1
                continue

            try:
                value_structure = self.__getitem__("geom_lmdb", key)
                if value_structure is None:
                    first_key_idx = idx + 1
                    continue

                mol_graph, global_feats = gather_structure_info(value_structure)
                bond_feats = {}
                atom_feats = {i: {} for i in range(global_feats["n_atoms"])}
                id = clean_id(key) if self.single_lmdb_in else key_str

                dict_qtaim_raw = self.__getitem__("qtaim_lmdb", key)
                if dict_qtaim_raw is None:
                    first_key_idx = idx + 1
                    continue

                (_, _, atom_feats, bond_feats, connected_bond_paths) = parse_qtaim_data(
                    dict_qtaim_raw, atom_feats, bond_feats,
                    atom_keys=self.keys_data["atom"],
                    bond_keys=self.keys_data["bond"],
                )

                mol_wrapper = MoleculeWrapper(
                    mol_graph, functional_group=None, free_energy=None, id=id,
                    bonds=connected_bond_paths, non_metal_bonds=connected_bond_paths,
                    atom_features=atom_feats, bond_features=bond_feats,
                    global_features=global_feats, original_atom_ind=None, original_bond_mapping=None,
                )
                if self.grapher is None:
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
                    self.logger.info(f"Grapher initialized with element_set: {self.element_set}")

                first_graph = build_and_featurize_graph(self.grapher, mol_wrapper)

                self.index_dict = get_include_exclude_indices(
                    feat_names=self.grapher.feat_names,
                    target_dict=self.keys_target,
                )
                self.config_dict["index_dict"] = self.index_dict
                self.overwrite_config()

                split_graph_labels(
                    first_graph,
                    include_names=self.index_dict["include_names"],
                    include_locs=self.index_dict["include_locs"],
                    exclude_locs=self.index_dict["exclude_locs"],
                )

                self.feature_scaler_iterative.update([first_graph])
                self.label_scaler_iterative.update([first_graph])
                write_buffer.append((
                    f"{key_str}".encode("ascii"),
                    pickle.dumps(serialize_dgl_graph(first_graph, ret=True), protocol=-1),
                ))
                processed_count += 1
                first_key_idx = idx + 1
                break

            except Exception as e:
                self.logger.debug(f"Failed to initialize with key {key_str}: {e}")
                first_key_idx = idx + 1
                continue

        if self.grapher is None:
            self.logger.error("Failed to initialize grapher with any key")
            return self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=0)

        # Phase 2: Process remaining keys in parallel
        remaining_keys = keys_to_iterate[first_key_idx:]
        self.logger.info(f"Phase 1: Initialized grapher after {first_key_idx} attempts, 1 key processed")
        self.logger.info(f"Phase 2: Processing {len(remaining_keys)} remaining keys with {self.n_workers} workers...")

        def process_key(key):
            return self._build_graph_for_key_qtaim(key, self.grapher, self.index_dict)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(process_key, key): key for key in remaining_keys}

            for future in as_completed(futures):
                try:
                    key_str, graph, failures = future.result()

                    for fail_type, fail_list in failures.items():
                        self.fail_log_dict[fail_type].extend(fail_list)

                    if graph is not None:
                        self.feature_scaler_iterative.update([graph])
                        self.label_scaler_iterative.update([graph])

                        write_buffer.append((
                            f"{key_str}".encode("ascii"),
                            pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                        ))
                        processed_count += 1

                        if len(write_buffer) >= self.batch_size:
                            txn = self.db.begin(write=True)
                            for buf_key, buf_value in write_buffer:
                                txn.put(buf_key, buf_value)
                            txn.commit()
                            self.logger.debug(f"Committed batch of {len(write_buffer)} items")
                            write_buffer.clear()

                except Exception as e:
                    self.logger.warning(f"Error processing future: {e}")

        # Flush remaining items
        if write_buffer:
            txn = self.db.begin(write=True)
            for buf_key, buf_value in write_buffer:
                txn.put(buf_key, buf_value)
            txn.commit()
            self.logger.debug(f"Committed final batch of {len(write_buffer)} items")

        ret_info = self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=processed_count)

        # Auto-merge shards if this is the last shard
        if self.auto_merge and self.shard_index == self.total_shards - 1:
            self._auto_merge_shards()

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

        self.bonding_scheme = self.config_dict.get("bonding_scheme", "structural")
        self.data_inputs = self.config_dict.get("data_inputs", ["geom", "qtaim", "charge"]) # add fuzzy_full, bonds, other as possible data inputs
        
        if config_dict.get("charge_filter", None) is not None:
            self.charge_filter = config_dict["charge_filter"]

        # Optional filters for fuzzy and other data
        self.fuzzy_filter = config_dict.get("fuzzy_filter", None)
        self.other_filter = config_dict.get("other_filter", None)

        # Bond parsing options (for bonding_scheme="bonding")
        self.bond_filter = config_dict.get("bond_filter", None)  # e.g., ["fuzzy", "ibsi"]
        self.bond_cutoff = config_dict.get("bond_cutoff", None)  # e.g., 0.3 for fuzzy threshold
        self.bond_list_definition = config_dict.get("bond_list_definition", "fuzzy")  # which bond type defines connectivity

        # Missing data strategy: "skip" (default) or "sentinel" (fill with NaN)
        self.missing_data_strategy = config_dict.get("missing_data_strategy", "skip")
        self.sentinel_value = config_dict.get("sentinel_value", float("nan"))

        # assert that for each data input, the corresponding LMDB is in the config dict in "lmdb_locations"
    
        for data_input in self.data_inputs:
            key = f"{data_input}_lmdb"
            
            assert (
                key in self.config_dict["lmdb_locations"].keys()
            ), f"The config file must contain a key '{key}'"



    def _build_graph_for_key(
        self,
        key: bytes,
        grapher,
        index_dict: dict,
    ) -> Tuple[str, Optional[Any], Dict[str, List[str]]]:
        """
        Build a graph for a single key. Thread-safe for parallel processing.

        Returns:
            Tuple of (key_str, graph or None, failures_dict)
        """
        failures = {k: [] for k in self.fail_log_dict.keys()}

        if isinstance(key, bytes):
            key_str = key.decode("ascii")
        else:
            key_str = str(key)

        # Check restart
        if self.restart and key_str in self.existing_keys:
            return (key_str, None, failures)

        try:
            # Structure
            value_structure = self.__getitem__("geom_lmdb", key)
            if value_structure is None:
                failures["structure"].append(key_str)
                return (key_str, None, failures)

            mol_graph, global_feats = gather_structure_info(value_structure)
            bond_feats = {}
            atom_feats = {i: {} for i in range(global_feats["n_atoms"])}
            bonds = value_structure["bonds"]
            bond_list = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}

            if self.single_lmdb_in:
                id = clean_id(key)
            else:
                id = key_str
        except Exception as e:
            logging.error(f"Error parsing structure for key {key_str}: {e}")
            failures["structure"].append(key_str)
            return (key_str, None, failures)

        # Charge data
        if "charge" in self.data_inputs:
            try:
                dict_charge_raw = self.__getitem__("charge_lmdb", key)
                if dict_charge_raw is not None:
                    atom_feats_charge, global_dipole_feats = parse_charge_data(
                        dict_charge_raw, global_feats["n_atoms"], self.charge_filter
                    )
                    global_feats.update(global_dipole_feats)
                    atom_feats.update(atom_feats_charge)
                else:
                    failures["charge"].append(key_str)
                    if self.missing_data_strategy == "skip":
                        return (key_str, None, failures)
            except Exception as e:
                logging.error(f"Error parsing charge data for key {key_str}: {e}")
                failures["charge"].append(key_str)
                if self.missing_data_strategy == "skip":
                    return (key_str, None, failures)

        # QTAIM data
        connected_bond_paths = None
        if "qtaim" in self.data_inputs:
            try:
                dict_qtaim_raw = self.__getitem__("qtaim_lmdb", key)
                if dict_qtaim_raw is not None:
                    (_, _, atom_feats, bond_feats, connected_bond_paths) = parse_qtaim_data(
                        dict_qtaim_raw, atom_feats, bond_feats,
                        atom_keys=self.keys_data["atom"],
                        bond_keys=self.keys_data["bond"],
                    )
                else:
                    failures["qtaim"].append(key_str)
                    if self.missing_data_strategy == "skip":
                        return (key_str, None, failures)
            except Exception as e:
                logging.error(f"Error parsing QTAIM data for key {key_str}: {e}")
                failures["qtaim"].append(key_str)
                if self.missing_data_strategy == "skip":
                    return (key_str, None, failures)

        # Fuzzy data
        if "fuzzy" in self.data_inputs:
            try:
                dict_fuzzy_raw = self.__getitem__("fuzzy_lmdb", key)
                if dict_fuzzy_raw is not None:
                    atom_feats_fuzzy, global_fuzzy_feats = parse_fuzzy_data(
                        dict_fuzzy_raw, global_feats["n_atoms"], self.fuzzy_filter
                    )
                    global_feats.update(global_fuzzy_feats)
                    for atom_idx, fuzzy_feats in atom_feats_fuzzy.items():
                        if atom_idx in atom_feats:
                            atom_feats[atom_idx].update(fuzzy_feats)
                        else:
                            atom_feats[atom_idx] = fuzzy_feats
                else:
                    failures["fuzzy"].append(key_str)
                    if self.missing_data_strategy == "skip":
                        return (key_str, None, failures)
            except Exception as e:
                logging.error(f"Error parsing fuzzy data for key {key_str}: {e}")
                failures["fuzzy"].append(key_str)
                if self.missing_data_strategy == "skip":
                    return (key_str, None, failures)

        # Other data
        if "other" in self.data_inputs:
            try:
                dict_other_raw = self.__getitem__("other_lmdb", key)
                if dict_other_raw is not None:
                    global_other_feats = parse_other_data(dict_other_raw, self.other_filter)
                    global_feats.update(global_other_feats)
                else:
                    failures["other"].append(key_str)
                    if self.missing_data_strategy == "skip":
                        return (key_str, None, failures)
            except Exception as e:
                logging.error(f"Error parsing other data for key {key_str}: {e}")
                failures["other"].append(key_str)
                if self.missing_data_strategy == "skip":
                    return (key_str, None, failures)

        # Bonds LMDB
        bonds_from_lmdb = None
        if "bond" in self.data_inputs:
            try:
                dict_bonds_raw = self.__getitem__("bond_lmdb", key)
                if dict_bonds_raw is not None:
                    bond_feats_from_lmdb, bonds_from_lmdb = parse_bond_data(
                        dict_bonds_raw,
                        bond_filter=self.bond_filter,
                        cutoff=self.bond_cutoff,
                        bond_list_definition=self.bond_list_definition,
                        bond_feats=None,
                        clean=True,
                        as_lists=False,
                    )
                    if bond_feats_from_lmdb:
                        for bond_key, bond_value in bond_feats_from_lmdb.items():
                            if bond_key in bond_feats:
                                bond_feats[bond_key].update(bond_value)
                            else:
                                bond_feats[bond_key] = bond_value
                else:
                    failures["bonds"].append(key_str)
                    if self.missing_data_strategy == "skip":
                        return (key_str, None, failures)
            except Exception as e:
                logging.error(f"Error parsing bonds for key {key_str}: {e}")
                failures["bonds"].append(key_str)
                if self.missing_data_strategy == "skip":
                    return (key_str, None, failures)

        # Select bond definitions
        if self.bonding_scheme == "qtaim":
            selected_bond_definitions = connected_bond_paths
        elif self.bonding_scheme == "bonding":
            if bonds_from_lmdb is not None:
                selected_bond_definitions = {tuple(sorted(b)): None for b in bonds_from_lmdb}
            else:
                selected_bond_definitions = bond_list
        else:
            selected_bond_definitions = bond_list

        # Build graph
        try:
            mol_wrapper = MoleculeWrapper(
                mol_graph,
                functional_group=None,
                free_energy=None,
                id=id,
                bonds=selected_bond_definitions,
                non_metal_bonds=selected_bond_definitions,
                atom_features=atom_feats,
                bond_features=bond_feats,
                global_features=global_feats,
                original_atom_ind=None,
                original_bond_mapping=None,
            )

            graph = build_and_featurize_graph(grapher, mol_wrapper)

            split_graph_labels(
                graph,
                include_names=index_dict["include_names"],
                include_locs=index_dict["include_locs"],
                exclude_locs=index_dict["exclude_locs"],
            )

            return (key_str, graph, failures)
        except Exception as e:
            self.logger.error(f"Error building graph for key {key_str}: {e}")
            failures["graph"].append(key_str)
            return (key_str, None, failures)

    def process(
        self,
        return_info=False,
    ) -> Dict[
        str, Union[HeteroGraphStandardScalerIterative, HeteroGraphLogMagnitudeScaler],

    ]:
        """
        Main loop for processing the LMDB files and generating the graphs.
        Uses parallel processing with ThreadPoolExecutor.
        """
        self.logger.info("Starting GeneralConverter processing...")

        keys_to_iterate = self.lmdb_dict["geom_lmdb"]["keys"]
        self.logger.info(f"Total keys before partitioning: {len(keys_to_iterate)}")

        # Partition keys for sharding
        keys_to_iterate = self._partition_keys(keys_to_iterate)
        self.logger.info(f"Keys to process in this shard: {len(keys_to_iterate)}")
        self.logger.info(f"Using {self.n_workers} workers for parallel processing")

        write_buffer = []
        processed_count = 0

        # Phase 1: Initialize grapher with first successful key (sequential)
        # We need the grapher and index_dict before parallel processing
        self.logger.info("Phase 1: Initializing grapher with first successful key...")
        first_graph = None
        first_key_idx = 0

        for idx, key in enumerate(keys_to_iterate):
            if isinstance(key, bytes):
                key_str = key.decode("ascii")
            else:
                key_str = str(key)

            if self.restart and key_str in self.existing_keys:
                first_key_idx = idx + 1
                continue

            # Try to build first graph to initialize grapher
            try:
                # Step 1: Get geometry data
                try:
                    value_structure = self.__getitem__("geom_lmdb", key)
                    if value_structure is None:
                        self.logger.debug(f"Key {key_str}: geometry data is None")
                        first_key_idx = idx + 1
                        continue
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to get geometry data: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 2: Parse structure info
                try:
                    mol_graph, global_feats = gather_structure_info(value_structure)
                    bond_feats = {}
                    atom_feats = {i: {} for i in range(global_feats["n_atoms"])}
                    bonds = value_structure["bonds"]
                    bond_list = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}
                    id = clean_id(key) if self.single_lmdb_in else key_str
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to parse structure info: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 3: QTAIM data
                connected_bond_paths = None
                if "qtaim_lmdb" in self.lmdb_dict:
                    try:
                        # hypothesis 1 - the raw dict is broken - unlikely bc this is working on qtaim normal
                        dict_qtaim_raw = self.__getitem__("qtaim_lmdb", key)
                        if dict_qtaim_raw is not None:
                            (_, _, atom_feats, bond_feats, connected_bond_paths) = parse_qtaim_data(
                                dict_qtaim_raw, atom_feats, bond_feats,
                                atom_keys=self.keys_data["atom"],
                                bond_keys=self.keys_data["bond"],
                            )
                        else:
                            self.logger.debug(f"Key {key_str}: QTAIM data missing in LMDB, skipping")
                            first_key_idx = idx + 1
                            continue
                    
                    except Exception as e:
                        self.logger.warning(f"Key {key_str}: Failed to parse QTAIM data: {e}", exc_info=True)
                        first_key_idx = idx + 1
                        continue


                # Step 4: Charge data
                if "charge_lmdb" in self.lmdb_dict:
                    try:
                        dict_charge_raw = self.__getitem__("charge_lmdb", key)
                        if dict_charge_raw is not None:
                            atom_feats_charge, global_dipole_feats = parse_charge_data(
                                dict_charge_raw, global_feats["n_atoms"], self.charge_filter
                            )
                            for feat_key in atom_feats_charge[0].keys():
                                if feat_key not in self.keys_data["atom"]:
                                    self.keys_data["atom"].append(feat_key)
                            for feat_key in global_dipole_feats.keys():
                                if feat_key not in self.keys_data["global"]:
                                    self.keys_data["global"].append(feat_key)
                            global_feats.update(global_dipole_feats)
                            atom_feats.update(atom_feats_charge)

                        elif self.missing_data_strategy == "skip":
                            self.logger.debug(f"Key {key_str}: charge data missing, skipping")
                            first_key_idx = idx + 1
                            continue
                    
                    except Exception as e:
                        self.logger.warning(f"Key {key_str}: Failed to parse charge data: {e}", exc_info=True)
                        first_key_idx = idx + 1
                        continue



                # Step 5: Fuzzy data
                if "fuzzy_lmdb" in self.lmdb_dict:
                    try:
                        dict_fuzzy_raw = self.__getitem__("fuzzy_lmdb", key)
                        if dict_fuzzy_raw is not None:
                            atom_feats_fuzzy, global_fuzzy_feats = parse_fuzzy_data(
                                dict_fuzzy_raw, global_feats["n_atoms"], self.fuzzy_filter
                            )
                            for feat_key in atom_feats_fuzzy.get(0, {}).keys():
                                if feat_key not in self.keys_data["atom"]:
                                    self.keys_data["atom"].append(feat_key)
                            for feat_key in global_fuzzy_feats.keys():
                                if feat_key not in self.keys_data["global"]:
                                    self.keys_data["global"].append(feat_key)
                            global_feats.update(global_fuzzy_feats)
                            for atom_idx, fuzzy_feats in atom_feats_fuzzy.items():
                                atom_feats.setdefault(atom_idx, {}).update(fuzzy_feats)
                        elif self.missing_data_strategy == "skip":
                            self.logger.debug(f"Key {key_str}: fuzzy data missing in LMDB, skipping")
                            first_key_idx = idx + 1
                            continue
                    except Exception as e:
                        self.logger.warning(f"Key {key_str}: Failed to parse fuzzy data: {e}", exc_info=True)
                        first_key_idx = idx + 1
                        continue


                # Step 6: Other data
                if "other_lmdb" in self.lmdb_dict:
                    try:
                        dict_other_raw = self.__getitem__("other_lmdb", key)
                        if dict_other_raw is not None:
                            global_other_feats = parse_other_data(dict_other_raw, self.other_filter)
                            for feat_key in global_other_feats.keys():
                                if feat_key not in self.keys_data["global"]:
                                    self.keys_data["global"].append(feat_key)
                            global_feats.update(global_other_feats)
                        elif self.missing_data_strategy == "skip":
                            self.logger.debug(f"Key {key_str}: other data missing in LMDB, skipping")
                            first_key_idx = idx + 1
                            continue
                    except Exception as e:
                        self.logger.warning(f"Key {key_str}: Failed to parse other data: {e}", exc_info=True)
                        first_key_idx = idx + 1
                        continue


                # Step 7: Bonds LMDB
                bonds_from_lmdb = None
                if "bonds_lmdb" in self.lmdb_dict:
                    try:
                        dict_bonds_raw = self.__getitem__("bonds_lmdb", key)
                        if dict_bonds_raw is not None:
                            bond_feats_from_lmdb, bonds_from_lmdb = parse_bond_data(
                                dict_bonds_raw, bond_filter=self.bond_filter,
                                cutoff=self.bond_cutoff, bond_list_definition=self.bond_list_definition,
                                bond_feats=None, clean=True, as_lists=False,
                            )
                            if bond_feats_from_lmdb:
                                sample_bond = next(iter(bond_feats_from_lmdb.values()), {})
                                for feat_key in sample_bond.keys():
                                    if feat_key not in self.keys_data["bond"]:
                                        self.keys_data["bond"].append(feat_key)
                                for bond_key, bond_value in bond_feats_from_lmdb.items():
                                    bond_feats.setdefault(bond_key, {}).update(bond_value)
                    except Exception as e:
                        self.logger.warning(f"Key {key_str}: Failed to parse bond data: {e}", exc_info=True)
                        first_key_idx = idx + 1
                        continue

                # Step 8: Select bond definitions
                try:
                    if self.bonding_scheme == "qtaim":
                        selected_bond_definitions = connected_bond_paths
                    elif self.bonding_scheme == "bonding" and bonds_from_lmdb is not None:
                        selected_bond_definitions = {tuple(sorted(b)): None for b in bonds_from_lmdb}
                    else:
                        selected_bond_definitions = bond_list
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to select bond definitions: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 9: Build MoleculeWrapper
                try:
                    mol_wrapper = MoleculeWrapper(
                        mol_graph, functional_group=None, free_energy=None, id=id,
                        bonds=selected_bond_definitions, non_metal_bonds=selected_bond_definitions,
                        atom_features=atom_feats, bond_features=bond_feats,
                        global_features=global_feats, original_atom_ind=None, original_bond_mapping=None,
                    )
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to build MoleculeWrapper: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 10: Initialize grapher
                try:
                    if self.grapher is None:
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
                        self.logger.info(f"Grapher initialized with keys_data: {self.keys_data}")
                except Exception as e:
                    self.logger.error(f"Key {key_str}: Failed to initialize grapher: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 11: Build and featurize graph
                try:
                    first_graph = build_and_featurize_graph(self.grapher, mol_wrapper)
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to build and featurize graph: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 12: Get include/exclude indices
                try:
                    self.index_dict = get_include_exclude_indices(
                        feat_names=self.grapher.feat_names,
                        target_dict=self.keys_target,
                    )
                    self.config_dict["index_dict"] = self.index_dict
                    self.overwrite_config()
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to get include/exclude indices: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 13: Split graph labels
                try:
                    split_graph_labels(
                        first_graph,
                        include_names=self.index_dict["include_names"],
                        include_locs=self.index_dict["include_locs"],
                        exclude_locs=self.index_dict["exclude_locs"],
                    )
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to split graph labels: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

                # Step 14: Update scalers and buffer
                try:
                    self.feature_scaler_iterative.update([first_graph])
                    self.label_scaler_iterative.update([first_graph])
                    write_buffer.append((
                        f"{key_str}".encode("ascii"),
                        pickle.dumps(serialize_dgl_graph(first_graph, ret=True), protocol=-1),
                    ))
                    processed_count += 1
                    first_key_idx = idx + 1
                    self.logger.info(f"Successfully initialized grapher with key {key_str}")
                    break
                except Exception as e:
                    self.logger.warning(f"Key {key_str}: Failed to update scalers or serialize: {e}", exc_info=True)
                    first_key_idx = idx + 1
                    continue

            except Exception as e:
                # Outer catch-all for truly unexpected errors
                self.logger.error(f"Key {key_str}: Unexpected error in grapher initialization: {e}", exc_info=True)
                first_key_idx = idx + 1
                continue

        if self.grapher is None:
            self.logger.error("Failed to initialize grapher with any key")
            return self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=0)

        # Phase 2: Process remaining keys in parallel
        remaining_keys = keys_to_iterate[first_key_idx:]
        self.logger.info(f"Phase 1: Initialized grapher after {first_key_idx} attempts, 1 key processed")
        self.logger.info(f"Phase 2: Processing {len(remaining_keys)} remaining keys with {self.n_workers} workers...")

        def process_key(key):
            return self._build_graph_for_key(key, self.grapher, self.index_dict)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(process_key, key): key for key in remaining_keys}

            for future in as_completed(futures):
                try:
                    key_str, graph, failures = future.result()

                    # Merge failures into main dict
                    for fail_type, fail_list in failures.items():
                        self.fail_log_dict[fail_type].extend(fail_list)

                    if graph is not None:
                        # Phase 3: Sequential scaler update and buffering
                        self.feature_scaler_iterative.update([graph])
                        self.label_scaler_iterative.update([graph])

                        write_buffer.append((
                            f"{key_str}".encode("ascii"),
                            pickle.dumps(serialize_dgl_graph(graph, ret=True), protocol=-1),
                        ))
                        processed_count += 1

                        # Batch commit
                        if len(write_buffer) >= self.batch_size:
                            txn = self.db.begin(write=True)
                            for buf_key, buf_value in write_buffer:
                                txn.put(buf_key, buf_value)
                            txn.commit()
                            self.logger.debug(f"Committed batch of {len(write_buffer)} items")
                            write_buffer.clear()

                except Exception as e:
                    self.logger.warning(f"Error processing future: {e}")

        # Flush remaining items in buffer
        if write_buffer:
            txn = self.db.begin(write=True)
            for buf_key, buf_value in write_buffer:
                txn.put(buf_key, buf_value)
            txn.commit()
            self.logger.debug(f"Committed final batch of {len(write_buffer)} items")
            write_buffer.clear()

        ret_info = self.finalize(return_info=return_info, keys_to_iterate=keys_to_iterate, processed_count=processed_count)

        # Auto-merge shards if this is the last shard
        if self.auto_merge and self.shard_index == self.total_shards - 1:
            self._auto_merge_shards()

        if return_info:
            return ret_info


class ASELMDBConverter:
    # TODO: last class to implement
    pass
