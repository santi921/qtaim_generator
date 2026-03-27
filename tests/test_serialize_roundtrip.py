import pytest
import numpy as np
import pickle as pkl

torch = pytest.importorskip(
    "torch", reason="torch not installed; skipping serialize roundtrip test"
)
pyg = pytest.importorskip(
    "torch_geometric", reason="torch_geometric not installed; skipping serialize roundtrip test"
)

from qtaim_embed.data.lmdb import serialize_graph, load_graph_from_serialized


def test_serialize_deserialize_roundtrip_synthetic():
    import torch
    from torch_geometric.data import HeteroData

    # create a small heterograph with atom and bond node types
    g = HeteroData()

    # atom nodes with features and labels
    g["atom"].feat = torch.randn(3, 4)
    g["atom"].labels = torch.randint(0, 2, (3, 1)).float()

    # bond nodes with features and labels
    g["bond"].feat = torch.randn(2, 5)
    g["bond"].labels = torch.randint(0, 2, (2, 1)).float()

    # edges: atom -> bond and bond -> atom
    g["atom", "a2b", "bond"].edge_index = torch.tensor([[0, 1], [0, 1]])
    g["bond", "b2a", "atom"].edge_index = torch.tensor([[0, 1], [1, 2]])

    serialized = serialize_graph(g, ret=True)
    blob = pkl.dumps(serialized, protocol=-1)
    loaded = pkl.loads(blob)

    g_rt = load_graph_from_serialized(loaded)
    assert g_rt is not None

    # verify node features roundtrip
    for ntype in ["atom", "bond"]:
        for attr in ["feat", "labels"]:
            original = getattr(g[ntype], attr).cpu().numpy()
            roundtripped = getattr(g_rt[ntype], attr).cpu().numpy()
            assert original.shape == roundtripped.shape, \
                f"Shape mismatch for {ntype}.{attr}"
            assert np.array_equal(original, roundtripped), \
                f"Value mismatch for {ntype}.{attr}"

    # verify edge structure roundtrip
    for etype in [("atom", "a2b", "bond"), ("bond", "b2a", "atom")]:
        original_ei = g[etype].edge_index.cpu().numpy()
        roundtripped_ei = g_rt[etype].edge_index.cpu().numpy()
        assert np.array_equal(original_ei, roundtripped_ei), \
            f"Edge index mismatch for {etype}"
