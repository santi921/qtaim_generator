"""SOAP featurization across OMol4M and the four comparator datasets.

See docs/plans/2026-05-04-soap-featurization-plan.md for the locked
species lists, hyperparameters, and per-source loader contracts.
"""

from qtaim_gen.source.analysis.comparator_embedding.loaders import (
    LOADERS,
    SourceSpec,
)

__all__ = ["LOADERS", "SourceSpec"]
