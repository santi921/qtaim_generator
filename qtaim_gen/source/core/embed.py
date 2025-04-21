from typing import Dict, List, Tuple, Union, Any

from qtaim_embed.data.processing import (
    HeteroGraphStandardScalerIterative,
    HeteroGraphLogMagnitudeScaler,
)
from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.grapher import get_grapher


def get_include_exclude_indices(
    feat_names: Dict[str, List[str]], target_dict: Dict[str, List[str]]
) -> Dict[str, Dict[str, Union[List[int], List[str]]]]:
    """
    Get the indices of the features to include and exclude.

    Args:
        feat_names (Dict[str, List[str]]): Dictionary of feature names.
        target_dict (Dict[str, List[str]]): Dictionary of target features.

    Returns:
        Dict[str, Dict[str, Union[List[int], List[str]]]]:
            A dictionary containing include and exclude indices and names.
    """
    target_locs = {}
    for node_type, value_list in target_dict.items():
        if node_type not in target_locs:
            target_locs[node_type] = []

        for value in value_list:
            target_locs[node_type].append(feat_names[node_type].index(value))

    include_locs, exclude_locs = {}, {}
    include_names, exclude_names = {}, {}

    for node_type, value_list in feat_names.items():
        for i, value in enumerate(value_list):
            if node_type in target_locs.keys():
                if i in target_locs[node_type]:
                    if node_type not in include_names:
                        include_names[node_type] = []
                        include_locs[node_type] = []

                    include_locs[node_type].append(i)
                    include_names[node_type].append(value)
                else:
                    if node_type not in exclude_names:
                        exclude_names[node_type] = []
                        exclude_locs[node_type] = []
                    exclude_locs[node_type].append(i)
                    exclude_names[node_type].append(value)
            else:
                if node_type not in exclude_names:
                    exclude_names[node_type] = []
                    exclude_locs[node_type] = []
                exclude_locs[node_type].append(i)
                exclude_names[node_type].append(value)

    info_dict = {
        "include_locs": include_locs,
        "exclude_locs": exclude_locs,
        "include_names": include_names,
        "exclude_names": exclude_names,
    }
    return info_dict


def split_graph_labels(
    graph: Any,
    include_names: Dict[str, List[str]],
    include_locs: Dict[str, List[int]],
    exclude_locs: Dict[str, List[int]],
) -> None:
    """
    Splits the graph into features and labels.

    Args:
        graph (Any): The DGL graph.
        include_names (Dict[str, List[str]]): Names of features to include in labels.
        include_locs (Dict[str, List[int]]): Indices of features to include in labels.
        exclude_locs (Dict[str, List[int]]): Indices of features to exclude from labels.

    Returns:
        None
    """
    labels = {}
    features_new = {}

    for key, value in graph.ndata["feat"].items():
        if key in include_names.keys():
            graph_features = {}

            graph_features[key] = graph.ndata["feat"][key][:, exclude_locs[key]]

            features_new.update(graph_features)

            labels[key] = graph.ndata["feat"][key][:, include_locs[key]]

        graph.ndata["feat"] = features_new
        graph.ndata["labels"] = labels


def initialize_grapher(
    grapher, element_set, atom_keys, bond_keys, global_keys, config_dict
):
    """
    Initialize the grapher object if not already initialized.
    """
    if not grapher:
        grapher = get_grapher(
            element_set=element_set,
            atom_keys=atom_keys,
            bond_keys=bond_keys,
            global_keys=global_keys,
            allowed_ring_size=config_dict["allowed_ring_size"],
            allowed_charges=config_dict["allowed_charges"],
            allowed_spins=config_dict["allowed_spins"],
            self_loop=True,
            atom_featurizer_tf=True,
            bond_featurizer_tf=True,
            global_featurizer_tf=True,
        )
    return grapher


def build_and_featurize_graph(grapher, mol_wrapper):
    """
    Build and featurize a graph using the grapher.
    """
    graph = grapher.build_graph(mol_wrapper)
    graph, _ = grapher.featurize(graph, mol_wrapper, ret_feat_names=True)
    return graph


def split_and_scale_graph(graph, index_dict, feature_scaler, label_scaler):
    """
    Split the graph into features and labels, and update scalers.
    """
    split_graph_labels(
        graph,
        include_names=index_dict["include_names"],
        include_locs=index_dict["include_locs"],
        exclude_locs=index_dict["exclude_locs"],
    )
    feature_scaler.update([graph])
    label_scaler.update([graph])
