"""
End-to-end integration test: generator LMDB -> qtaim_embed training.

Verifies that graphs produced by BaseConverter can be:
  1. Loaded by LMDBMoleculeDataset with TransformMol
  2. Batched by DataLoaderLMDB
  3. Passed through one optimizer step (forward + backward) of a model
     that reads graph.feat and graph.labels from each node type

This is the minimal proof that the generator output format is
training-compatible with qtaim_embed.
"""

import pytest
import torch
from pathlib import Path

try:
    from qtaim_gen.source.core.converter import BaseConverter
    from qtaim_embed.core.dataset import LMDBMoleculeDataset
    from qtaim_embed.data.dataloader import DataLoaderLMDB
    from qtaim_embed.data.lmdb import TransformMol
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="qtaim_embed or converter not available"
)

MERGED = Path(__file__).parent / "test_files" / "lmdb_tests" / "generator_lmdbs_merged"


def _converter_config(tmp_path):
    return {
        "chunk": -1,
        "filter_list": ["length"],
        "restart": False,
        "allowed_ring_size": [3, 4, 5, 6, 7, 8],
        "allowed_charges": None,
        "allowed_spins": None,
        # n_atoms is both a feature and a regression target so labels are non-empty
        "keys_target": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "keys_data": {"atom": [], "bond": [], "global": ["n_atoms"]},
        "lmdb_path": str(tmp_path) + "/",
        "lmdb_name": "train.lmdb",
        "lmdb_locations": {"geom_lmdb": str(MERGED / "merged_geom.lmdb")},
        "n_workers": 1,
        "batch_size": 100,
    }


@pytest.fixture(scope="module")
def converter_lmdb(tmp_path_factory):
    """Run BaseConverter once and return path to output LMDB."""
    tmp = tmp_path_factory.mktemp("embed_training")
    config = _converter_config(tmp)
    info = BaseConverter(config).process(return_info=True)
    assert info["processed_count"] > 0, "Converter produced no graphs"
    return tmp / "train.lmdb"


def test_lmdb_loads_via_dataset(converter_lmdb):
    """LMDBMoleculeDataset with TransformMol can load all graphs."""
    ds = LMDBMoleculeDataset(
        config={"src": str(converter_lmdb)},
        transform=TransformMol,
    )
    assert len(ds) > 0, "Dataset reports zero length"
    graph = ds[0]
    assert hasattr(graph, "node_types"), "Item is not a PyG HeteroData graph"
    assert "atom" in graph.node_types
    assert hasattr(graph["atom"], "feat"), "atom.feat missing"


def test_dataloader_batching(converter_lmdb):
    """DataLoaderLMDB can batch graphs into feat/labels tensors."""
    ds = LMDBMoleculeDataset(
        config={"src": str(converter_lmdb)},
        transform=TransformMol,
    )
    dl = DataLoaderLMDB(dataset=ds, batch_size=2, shuffle=False, num_workers=0)
    batched_graphs, batched_labels = next(iter(dl))

    assert hasattr(batched_graphs["atom"], "feat"), "batch missing atom.feat"
    assert batched_graphs["atom"].feat.ndim == 2, "feat should be 2-D"

    for ntype, labels in batched_labels.items():
        assert labels.ndim >= 1, f"{ntype}.labels should be at least 1-D"


def test_one_training_step(converter_lmdb):
    """Full forward + backward pass through a minimal linear model."""
    ds = LMDBMoleculeDataset(
        config={"src": str(converter_lmdb)},
        transform=TransformMol,
    )
    dl = DataLoaderLMDB(dataset=ds, batch_size=len(ds), shuffle=False, num_workers=0)
    batched_graphs, batched_labels = next(iter(dl))

    # Only test node types that have both feat and non-empty labels
    trainable_types = [
        nt for nt in batched_graphs.node_types
        if hasattr(batched_graphs[nt], "feat")
        and nt in batched_labels
        and batched_labels[nt].numel() > 0
    ]
    assert len(trainable_types) > 0, "No node type has both feat and labels"

    params = []
    losses = []
    for nt in trainable_types:
        feat = batched_graphs[nt].feat.float()
        labels = batched_labels[nt].float()
        in_dim = feat.shape[1]
        out_dim = labels.shape[1] if labels.ndim > 1 else 1

        layer = torch.nn.Linear(in_dim, out_dim)
        params += list(layer.parameters())

        pred = layer(feat)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        losses.append(torch.nn.functional.mse_loss(pred, labels))

    total_loss = sum(losses)
    optim = torch.optim.Adam(params, lr=1e-3)
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    assert total_loss.item() >= 0, "Loss must be non-negative"
    assert not torch.isnan(total_loss), "Loss is NaN"
