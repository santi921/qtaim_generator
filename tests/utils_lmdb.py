import pickle as pkl
from copy import deepcopy
from qtaim_embed.data.lmdb import load_graph_from_serialized
import numpy as np
import lmdb
import pickle


def get_first_graph(converter):
    db_test = converter.connect_db(converter.file)
    graph = None
    with db_test.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                graph = deepcopy(load_graph_from_serialized(pkl.loads(value)))
                break
            except Exception:
                pass
    return graph


def check_graph_equality(graph1, graph2):
    """Check that two PyG HeteroData graphs have same-shaped but different features."""
    for node_type in graph1.node_types:
        for attr in ("feat", "labels"):
            if hasattr(graph1[node_type], attr) and hasattr(graph2[node_type], attr):
                ft_1 = getattr(graph1[node_type], attr)
                ft_2 = getattr(graph2[node_type], attr)
                assert (
                    ft_1.shape == ft_2.shape
                ), f"Graph features shapes changed! {node_type}.{attr}. Graph 1: {ft_1.shape}, Graph 2: {ft_2.shape}"
                assert not np.array_equal(
                    ft_1.cpu().numpy(), ft_2.cpu().numpy()
                ), f"Graph features are the same! {node_type}.{attr}."


def get_benchmark_info(converter):
    info_process = converter.process(return_info=True)
    first_graph_pre = get_first_graph(converter)
    info_scale = converter.scale_graph_lmdb(return_info=True)
    info_scale_restart = converter.scale_graph_lmdb(return_info=True)
    first_graph_post = get_first_graph(converter)
    return (
        info_process,
        first_graph_pre,
        info_scale,
        info_scale_restart,
        first_graph_post,
    )


def _make_lmdb(path, with_length=True):
    env = lmdb.open(path, map_size=1 << 24, subdir=False)
    with env.begin(write=True) as txn:
        # optional length entry
        if with_length:
            txn.put("length".encode("ascii"), pickle.dumps(2))
        txn.put("0".encode("ascii"), pickle.dumps({"val": 0}))
        txn.put("1".encode("ascii"), pickle.dumps({"val": 1}))
    env.sync()
    env.close()
