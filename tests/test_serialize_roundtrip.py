import pytest
import numpy as np
import pickle as pkl

pyg = pytest.importorskip(
    "dgl", reason="dgl not installed; skipping serialize roundtrip test"
)
torch = pytest.importorskip(
    "torch", reason="torch not installed; skipping serialize roundtrip test"
)


from qtaim_embed.data.lmdb import serialize_dgl_graph, load_dgl_graph_from_serialized


def test_serialize_deserialize_roundtrip_synthetic():
    import dgl
    import torch

    # create a small heterograph with one node type and one edge type

    g = dgl.heterograph(
        {("atom", "bond", "atom"): (torch.tensor([0, 1]), torch.tensor([1, 2]))},
        num_nodes_dict={"atom": 3},
    )

    # attach node-level feat and label tensors
    g.ndata["feat"] = torch.randn(3, 4)
    g.ndata["labels"] = torch.randint(0, 2, (3, 1)).float()

    # attach edge-level data similarly
    g.edata["feat"] = torch.randn(g.num_edges(), 5)
    g.edata["labels"] = torch.randint(0, 2, (g.num_edges(), 1)).float()

    serialized = serialize_dgl_graph(g, ret=True)
    blob = pkl.dumps(serialized, protocol=-1)
    loaded = pkl.loads(blob)

    g_rt = load_dgl_graph_from_serialized(loaded)
    assert g_rt is not None

    # helper to normalize ndata/edata access across possibly dict-like or tensor storage
    def _extract_node_level(mapping, graph):
        if mapping is None:
            return {}
        # if mapping is a dict-like, return as-is
        if isinstance(mapping, dict):
            return {k: v.cpu().numpy() for k, v in mapping.items()}
        # else mapping is a tensor corresponding to the single node type
        nt = graph.ntypes[0]
        return {nt: mapping.cpu().numpy()}

    def _extract_edge_level(mapping, graph):
        if mapping is None:
            return {}
        if isinstance(mapping, dict):
            return {k: v.cpu().numpy() for k, v in mapping.items()}
        # tensor case: single edge type
        et = graph.canonical_etypes[0]
        return {et: mapping.cpu().numpy()}

    for level in ["feat", "labels"]:
        a_nodes = _extract_node_level(g.ndata.get(level, None), g)
        b_nodes = _extract_node_level(g_rt.ndata.get(level, None), g_rt)
        assert set(a_nodes.keys()) == set(b_nodes.keys())
        for k in a_nodes.keys():
            assert a_nodes[k].shape == b_nodes[k].shape
            assert np.array_equal(a_nodes[k], b_nodes[k])

    for level in ["feat", "labels"]:
        a_edges = _extract_edge_level(g.edata.get(level, None), g)
        b_edges = _extract_edge_level(g_rt.edata.get(level, None), g_rt)
        assert set(a_edges.keys()) == set(b_edges.keys())
        for k in a_edges.keys():
            assert a_edges[k].shape == b_edges[k].shape
            assert np.array_equal(a_edges[k], b_edges[k])
