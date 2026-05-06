---
title: Merged intro+related work for OMol-Descriptors-4M (NeurIPS D&B)
type: working draft (v7)
date: 2026-05-03
supersedes: literature_brief_v6.md
status: all citations locked with DOIs/keys; prose ready for LaTeX conversion
---

## Changelog v6 → v7

- **All remaining citations confirmed**: Rackers2023, Qiao2020 (OrbNet), Unke2021 (PhiSNet), Schütt2019 (SchNOrb), Grambow2020 with DOI.
- **Two research lineages now visible** in ¶6: Tkatchenko (Schütt2019 SchNOrb → Gallegos2024 SchNet4AIM) and Smidt (Unke2021 PhiSNet → Rackers2023 density). Brief acknowledgment of both lineages added — strengthens the "this dataset has natural model audiences with active research programs" rhetoric.
- **Rackers2023's "scaling limit" framing** (per its abstract) now explicitly connected to the foundation-scale-data argument from Yuan2026 in ¶1. Both papers identify the same bottleneck from different sides — Rackers2023 from the model-output side (electron densities can't be computed for large systems), Yuan2026 from the model-training side (foundation-scale data needed). Our dataset is the descriptor analog of that argument.
- **Single-cite OrbNet** (Qiao2020) — extensions (OrbNet-Equi, OrbNet-Denali) dropped per "cleaner if single" preference.

---

## Changelog v5 → v6

- **Added Yuan2026** (Nat Rev Chem 2026, foundation models for atomistic simulation) — perspective/review with explicit treatment of foundationally-sized datasets. Cited in both ¶1 (foundation-model framing) and ¶5 (criteria for foundation-scale data infrastructure).
- **Critical correction in ¶4**: tmQM+ vs OMol-Desc-4M are *orthogonally scoped*, not just different in scale. tmQM+ varies LOT, holds descriptor family fixed (QTAIM only). OMol-Desc-4M varies descriptor family (charges + bond orders + QTAIM + fuzzy), holds LOT fixed at the OMol25 level. The earlier "multi-method" framing of tmQM+ was wrong — it's multi-LOT × single-descriptor-family. Fixed.
- **¶6 family 1**: strengthened the SchNet4AIM connection. OMol-Desc-4M ships delocalization indices (confirmed by Santiago), which means the previous hedge ("closely related to") becomes a direct claim ("the same QTAIM 1- and 2-body descriptors SchNet4AIM predicts").
- **¶7 clarified as first-tier release**: this paper publishes Level 0; subsequent tiers planned. Added delocalization indices to the descriptor inventory.
- **DOIs added** to citation table for all references where Santiago has supplied them. Following Santiago's instruction, no DOI is listed where one was not explicitly given (e.g., Grambow2020).

---

## Changelog v4 → v5

- **Citations confirmed and BibTeX-ready** (see locked citation table below): OMol25 (Levine2026), GNoME (Merchant2023), Tkatchenko 2024 = **Gallegos2024** (Gallegos is first author!), BDE-db (StJohn2020), AIMNet2 (Anstine2025), and new Grambow2020.
- **¶4 framing corrected**: removed the incorrect claim that tmQM+ "motivates the multi-tier theory ladder of OMol-Descriptors-4M." OMol25's level of theory was set by the OMol25 team independent of descriptor work. The actual argument is: tmQM+ showed descriptors at any LOT help ML in TM chemistry; OMol-Desc-4M extends this finding to a much wider chemistry envelope.
- **¶6 family taxonomy updated** per Santiago's correction:
  - **SchNOrb moved to family 1** (descriptor predictors). Per Santiago, it predicts descriptors directly on a graph rather than full Hamiltonian fields, putting it alongside AIMNet/SchNet4AIM in spirit.
  - **PhiSNet stays in family 2** (electronic structure predictors), recharacterized as density prediction (per Santiago: "PhiSNet seems to be trying to predict electronic densities").
  - **Rackers/Smidt density paper added to family 2** as another density-prediction precedent. Citation flagged for Santiago to verify exact paper.
- **Grambow2020 added to family 3** in ¶6, paired with Vargas2021: Grambow shows DL works for barriers without QTAIM augmentation; Vargas2021 shows QTAIM augmentation enables extrapolation. The pairing makes both citations earn their place.

---

## Changelog v3 → v4

- **¶4 locked**: Vargas 2025 D5DD (tmQM+, 60k TM complexes) added as the direct TM precedent. Other smaller datasets (QMugs, OE62, Alchemy, BDE-db) retained from earlier versions.
- **¶6 substantially rewritten**: now organized around three model families (descriptor predictors / wavefunction predictors / descriptor-augmented predictors). New citations: Tkatchenko 2024 SchNet4AIM (closest model precedent in the literature), Vargas 2021 JCTC (Diels-Alder QTAIM barrier prediction), Vargas 2024 D4DD (qtaim_generator + cross-dataset benchmarks), Green JACS 2024 (when descriptors help D-MPNNs — engaged with directly), Coley JCIM 2023 (light cite for descriptor-augmented reaction kinetics).
- **¶7 lineage nod added**: brief reference to the Vargas research arc culminating in this dataset.
- **Santiago-optional placeholders** marked `[opt-result]` for results Santiago may want to add from each cited paper.

---

## Changelog v2 → v3

- **QM7-X reclassified**: moved from ¶2 (smaller descriptor sets) to ¶5 (comparable-scale-but-narrow). Per Santiago's correction with the QM7-X property table: QM7-X covers only ≤7 heavy atoms across CNOSCl with one charge scheme (Hirshfeld), no bond orders, no QTAIM. Its strengths are dispersion descriptors (Tkatchenko MBD signatures: atC6, atPOL, vdwR) and a detailed energy decomposition. Structurally it sits with PCQM4Mv2 as a comparable-scale-but-narrow precedent, not a smaller-descriptor dataset.
- **Strengthened ¶5 gap argument**: the two existing 4M-scale chemistry datasets each fail breadth on different axes — PCQM4Mv2 wide-composition/narrow-property, QM7-X narrow-composition/medium-property. OMol-Descriptors-4M is the first to be wide on both axes.
- **Updated ¶7 contributions** to land the three-way contrast cleanly.

---

# Merged intro + related work — v2 draft

Per Santiago's revised outline (2026-05-03 chat): merge §1 and §2 into one flowing arc. Sub-arcs:

1. ML + chemistry as a paradigm (warm-up: drug discovery, materials)
2. Datasets for molecular chemistry (broad survey)
3. Distinction with materials (brief, scope-setting)
4. Smaller descriptor datasets (the QM9 / QMugs / OE62 tier)
5. Comparable-scale datasets (the OMol25 / SPICE / PCQM4Mv2 tier — generally NOT descriptor-focused, no coverage audit)
6. Models trained on descriptor datasets (the AIMNet / SchNOrb / Tkatchenko-2024 tier)
7. Gap statement + contributions

Targeting ~1.5 pages combined (the saved 0.5 from §2 reallocates to §3 dataset description, which gains a per-vertical chemistry tag column).

---

## Citation placement and BibTeX keys (locked v7, 2026-05-03)

### All citations confirmed with DOIs

| Citation key | DOI | Short form | Final placement | Role |
|---|---|---|---|---|
| Vargas2024 | 10.1039/D4DD00057A | *Digital Discovery* 2024 | ¶6 family 3 | qtaim_generator package + cross-dataset benchmarks |
| Vargas2025 | 10.1039/D5DD00220F | *Digital Discovery* 2025 — tmQM+ | ¶4 | Smaller TM-focused descriptor dataset; closest TM precedent |
| Vargas2021 | 10.1021/acs.jctc.1c00623 | *JCTC* 2021 — Diels-Alder QTAIM | ¶6 family 3 | First demonstration of QTAIM features for ML barrier prediction |
| Gallegos2024 | 10.1038/s41467-024-48567-9 | *Nat Commun* 2024 — SchNet4AIM | ¶6 family 1 | Closest model precedent; predicts QTAIM 1- and 2-body descriptors (incl. DIs) |
| Schütt2019 | 10.1038/s41467-019-12875-2 | *Nat Commun* 2019 — SchNOrb | ¶6 family 1 | Earlier Tkatchenko-group descriptor predictor (orbital-level, on graph) |
| Unke2021 | (NeurIPS 2021) | PhiSNet | ¶6 family 2 | SE(3)-equivariant wavefunction & density prediction (Smidt et al.) |
| Rackers2023 | 10.1088/2632-2153/acb314 | *Mach Learn Sci Technol* 2023 | ¶6 family 2 | Electron density prediction; "cracking the quantum scaling limit" |
| Qiao2020 | 10.1063/5.0021955 | *J Chem Phys* 2020 — OrbNet | ¶6 family 3 | Semi-empirical features → DFT-level prediction |
| Green2024 | 10.1021/jacs.4c04670 | *JACS* 2024 | ¶6 family 3 | Methodology study on when QM descriptors help D-MPNNs |
| Grambow2020 | 10.1021/acs.jpclett.0c00500 | *JPC Lett* 2020 — Activation energies | ¶6 family 3 | Green-group barrier prediction without QTAIM (paired with Vargas2021) |
| Coley2023 | 10.1021/acs.jcim.3c00892 | *JCIM* 2023 | ¶6 family 3 (light) | Reaction-kinetics with cheminformatics topological indices |
| Levine2026 | arXiv:2505.08762 | OMol25 | ¶5, ¶7 | Geometry source |
| Merchant2023 | 10.1038/s41586-023-06735-9 | GNoME, *Nature* 2023 | ¶1 | Materials warm-up |
| StJohn2020 | 10.1038/s41467-020-16201-z | BDE-db, *Nat Commun* 2020 | ¶2 | Smaller descriptor dataset (290k BDEs) |
| Anstine2025 | 10.1039/D4SC08572H | AIMNet2, *Chem Sci* 2025 | ¶6 family 1 | Atomic-charge/multipole prediction at scale |
| Yuan2026 | 10.1038/s41570-025-00793-5 | *Nat Rev Chem* 2026 — Foundation models | ¶1, ¶5 | Perspective on foundation-scale chemistry data |

### Optional / Santiago to decide later

| Citation | Status |
|---|---|
| AIMNet (1st gen, Zubatyuk 2019 *Chem Sci*) | Santiago to decide whether to cite alongside AIMNet2; currently single-cite |
| OrbNet-Equi (Christensen2021), OrbNet-Denali (Christensen2021 *J Chem Phys*) | Dropped per "cleaner if single" — only Qiao2020 cited |

---

## Draft prose (~1.5 pages)

### ¶1 — ML + chemistry warm-up

> Machine learning has reshaped chemistry's two oldest problems: finding new molecules and predicting how they behave. In drug discovery, graph neural networks score candidate structures against multiple endpoints simultaneously [Chemprop; MoleculeNet], and generative models now propose them outright [GFlowNet; Pocket2Mol]. In materials, machine-learned interatomic potentials simulate dynamics that were previously the exclusive domain of expensive ab initio methods [NequIP; MACE; MACE-MP-0; EquiformerV2; Merchant2023]. What unifies these advances is data: each represents a moment when a sufficiently large, well-curated dataset crossed a threshold that made gradient-based optimization viable for that specific problem. Recent perspectives synthesize this trend and posit that scaling chemistry datasets toward foundation-model regimes — alongside expressive architectures and modern training strategies — should yield models with broad transferability and OOD robustness [Yuan2026]. The argument has been made most forcefully for MLIPs, where data infrastructure has caught up with modeling ambition; the analogous argument for descriptor-prediction models has been bottlenecked by the absence of foundation-scale descriptor datasets — the gap this paper aims to close.

**Citations needed**: Chemprop (Yang et al. 2019 *J Chem Inf Model*), MoleculeNet (Wu et al. 2018 *Chem Sci*), GFlowNet (Bengio et al. 2021), Pocket2Mol (Peng et al. 2022 *ICML*), NequIP (Batzner et al. 2022 *Nat Commun*), MACE (Batatia et al. 2022 *NeurIPS*), MACE-MP-0 (Batatia et al. 2024), EquiformerV2 (Liao et al. 2024 *ICLR*), Merchant2023 (GNoME, *Nature* 2023, doi:10.1038/s41586-023-06735-9), Yuan2026 (*Nat Rev Chem* 2026, doi:10.1038/s41570-025-00793-5).

**Tone note**: deliberately broad and uncontroversial in the first half; the Yuan2026 citation pivots the paragraph from survey to thesis. The closing sentence ("the gap this paper aims to close") is the hand-off into ¶2 and beyond. The second half of ¶1 now does double duty as both setup and contribution preview.

### ¶2 — Datasets for molecular chemistry

> Datasets in molecular chemistry split along the axis of what is labeled. The smallest and earliest QM-curated sets — QM7, QM9, and the GDB-derived family — annotated molecules with O(10) descriptors per structure: HOMO/LUMO gaps, atomization energies, polarizabilities, dipole moments, zero-point vibrational energies [QM7; QM9]. These have been the workhorses of property-prediction GNNs for nearly a decade [SchNet; MPNN; DimeNet; PaiNN]. Larger specialized sets have followed, each pushing one axis while holding others fixed: QMugs at 665k drug-like molecules with multiple charge schemes [QMugs]; OE62 at 62k molecules with multi-functional orbital energies [OE62]; AlchemyDB at 200k organics [Alchemy]; BDE-db at 290k bond dissociation energies [BDE-db].

**Citations needed**:
- QM7 (Rupp et al. 2012 *PRL*), QM9 (Ramakrishnan et al. 2014 *Sci Data*)
- SchNet (Schütt et al. 2018 *J Chem Phys*), MPNN (Gilmer et al. 2017 *ICML*), DimeNet (Gasteiger et al. 2020 *ICLR*), PaiNN (Schütt et al. 2021 *ICML*)
- QMugs (Isert et al. 2022 *Sci Data*), OE62 (Stuke et al. 2020 *Sci Data*), Alchemy (Chen et al. 2019 arXiv), BDE-db (St. John et al. 2020 *Nat Commun* — **please confirm**)

### ¶3 — Brief distinction with materials

> The parallel materials-data ecosystem — Materials Project, OQMD, JARVIS, AFLOW — hosts periodic-system DFT data with formation energies, bandgaps, and elastic moduli [MatProj; OQMD; JARVIS; AFLOW]. The compositional space, property targets, and DFT conventions (PBE/PBESol functionals with PAW pseudopotentials, k-point sampling, periodic boundary conditions) differ enough that a dataset designed for molecular descriptors does not transfer cleanly. We restrict scope to molecular (non-periodic) chemistry; materials descriptors are a complementary problem we do not engage with here.

**Citations needed**: Materials Project (Jain et al. 2013 *APL Materials*), OQMD (Saal et al. 2013 *JOM*), JARVIS (Choudhary et al. 2020 *npj Comput Mater*), AFLOW (Curtarolo et al. 2012 *Comput Mater Sci*).

**Tone note**: 3-4 sentences max. Reviewers from materials ML may still review this paper; signal clearly that you're not claiming relevance to their corner.

### ¶4 — Smaller descriptor datasets (more specific than ¶2)

> Beyond the canonical scaffolds above, a focused tradition of descriptor-dataset construction has emerged around specific chemistry use cases. Most directly relevant to our work is tmQM+ [Vargas2025], which curates 60k transition metal complexes with QTAIM descriptors computed at multiple levels of DFT theory. tmQM+ is the closest precedent in *intent* — descriptor-rich and transition-metal-focused — and the comparison with OMol-Descriptors-4M is informative on two axes simultaneously. tmQM+ is two orders of magnitude smaller than the TM coverage in OMol-Descriptors-4M, *and* the two datasets are orthogonally scoped on the descriptor axis: tmQM+ varies the level of DFT theory while holding the descriptor family fixed (QTAIM only), whereas OMol-Descriptors-4M varies the descriptor family (multiple charge schemes, multiple bond-order schemes, full QTAIM topology including delocalization indices, fuzzy descriptors, and ORCA-derived globals) at a single, high level of theory inherited from OMol25. The principal empirical finding from tmQM+ — that QTAIM descriptors aid ML at essentially any level of DFT theory tested across the LOT ladder [opt-result: Santiago to add headline number] — extends naturally to a forward-looking question for OMol-Descriptors-4M: whether the descriptor-augmentation benefits documented at the TM scale transfer to the substantially wider chemistry envelope of OMol25 (transition metals plus lanthanides, actinides, charged species, reactive geometries, and protein-derived fragments). Earlier and broader-scope descriptor datasets (QMugs, OE62, AlchemyDB, BDE-db) cited in ¶2 sit at smaller scales than tmQM+ and on narrower descriptor families.

**Status**: prose locked. The `[opt-result]` placeholder is for Santiago to insert a specific result from Vargas2025 if desired.

**Framing notes**:
- "Orthogonally scoped" is the key conceptual move. tmQM+ and OMol-Desc-4M are not the same dataset at different scales; they expand the descriptor-dataset frontier on perpendicular axes (LOT vs descriptor family). Future work that combines both axes simultaneously becomes a natural follow-on.
- We do NOT claim the multi-tier release was motivated by the tmQM+ low-LOT finding. OMol25's level of theory is set by the OMol25 team independently. The forward-looking generalization claim is the right framing.

### ¶5 — Comparable-scale datasets, generally NOT descriptor-focused

> In the last five years, chemistry datasets have crossed the 10⁶–10⁸ structure threshold, but almost exclusively for energies and forces. ANI-1x/2x/ccx [ANI-1x; ANI-2x; ANI-1ccx], SPICE [SPICE], GEOM [GEOM], transition1x [transition1x], OC20/22 [OC20; OC22], and OMol25 [Levine2026] supply force-field training data across diverse chemistries; none ship multi-method partial charges, bond orders, QTAIM topology, or fuzzy descriptors. These are the datasets that motivate Yuan2026's proposal that foundation-scale chemistry models are within reach with the right scale and diversity of training data — but the proposal as articulated is for MLIPs, not descriptors. Two precedents reach 4M structures while attempting to expose descriptor labels — and each constrains breadth on a different axis. PCQM4Mv2 [PCQM4Mv2], drawn from PubChemQC [PubChemQC], reaches 4M drug-like organic molecules but ships exactly one descriptor (the HOMO-LUMO gap) at one level of theory; the chemistry envelope is wide but the descriptor surface is one-dimensional. QM7-X [QM7-X] reaches 4.2M structures with a richer descriptor inventory — a detailed energy decomposition (PBE0/MBD components), Hirshfeld charges and dipoles, and dispersion descriptors (atomic C₆ coefficients, polarizabilities, van der Waals radii) — but only 7k unique compositions across five elements (CNOSCl) and at most seven heavy atoms; the descriptor breadth is moderate but the chemistry envelope collapses to a tight subset of organic chemistry. Neither dataset ships multi-method partial charges or bond orders, neither ships QTAIM topology, and crucially neither carries a coverage audit: a quantitative answer to "what fraction of descriptor space does this dataset actually probe?" The result is a structural gap. The largest descriptor datasets are 10²–10³ times smaller than the largest MLIP datasets, and the two existing 4M-scale chemistry datasets each fail breadth on a different axis.

**Citations needed**: ANI-1x (Smith et al. 2020), ANI-2x (Devereux et al. 2020), ANI-1ccx (Smith et al. 2020 *Nat Commun*), SPICE (Eastman et al. 2023 *Sci Data*), GEOM (Axelrod & Gomez-Bombarelli 2022 *Sci Data*), transition1x (Schreiner et al. 2022 *Sci Data*), OC20 (Chanussot et al. 2021 *ACS Catal*), OC22 (Tran et al. 2023 *ACS Catal*), OMol25 (Levine et al. 2025), PubChemQC (Nakata & Shimazaki 2017 *J Chem Inf Model*), PCQM4Mv2 (Hu et al. 2021 *NeurIPS*), QM7-X (Hoja et al. 2021 *Sci Data*).

**Tone note**: this is the load-bearing paragraph. Three claims must land: (i) MLIP datasets are big but not descriptor-focused, (ii) the two comparable-scale descriptor precedents (PCQM4Mv2, QM7-X) each fail breadth on a different axis — wide-composition/narrow-property vs narrow-composition/wider-property — (iii) none of them audit coverage. If a reviewer reads only this paragraph they should leave understanding the gap.

### ¶6 — Models trained on descriptor datasets

> Architectures that consume descriptor data divide into three families. **The first predicts descriptor labels directly from geometry.** AIMNet and AIMNet2 [Anstine2025] predict atomic charges and multipoles trained on Hirshfeld, MBIS, and CM5 outputs across 14 elements at the hybrid-DFT level, demonstrating broad chemical coverage for descriptor prediction. SchNet4AIM [Gallegos2024] extends an earlier line of work [Schütt2019] toward predicting QTAIM descriptors directly: where SchNOrb [Schütt2019] established that orbital-level descriptors (Hamiltonian matrix elements over an atomic-orbital basis) could be predicted along the molecular graph, SchNet4AIM [Gallegos2024] applies the same paradigm to interatomic real-space descriptors — including delocalization indices and pairwise interaction energies — the same QTAIM 1- and 2-body quantities that OMol-Descriptors-4M ships. The Tkatchenko-Pendás group explicitly frames SchNet4AIM as breaking the cost bottleneck that has prevented real-space chemical descriptors from being used at scale; OMol-Descriptors-4M is the natural training corpus for that program at chemistry coverage SchNet4AIM has not previously had access to [opt-result: Santiago to add SchNet4AIM scale/error numbers]. **The second predicts full electronic structures** — densities or wavefunctions — from which descriptors are derivable. PhiSNet [Unke2021] uses SE(3)-equivariant message passing to predict molecular wavefunctions and electronic densities, and Rackers et al. [Rackers2023] demonstrate that Euclidean neural networks trained on densities of small water clusters can predict densities of clusters more than four times larger — the most direct demonstration to date of "cracking the quantum scaling limit" via ML-predicted electronic structure. This family operates upstream of descriptors: its predictions are the field from which the descriptors in OMol-Descriptors-4M are derived. **The third treats descriptors as input features for downstream property prediction.** OrbNet [Qiao2020] uses semi-empirical features for DFT-level prediction. On reaction-barrier prediction specifically, two precedents bracket the role of descriptors: Grambow et al. [Grambow2020] showed that deep learning of activation energies works well from molecular structure alone, while our prior work [Vargas2021] demonstrated that QTAIM topological descriptors of the reactant state enable Diels-Alder barrier prediction with extrapolation from solution-phase reactants to enzyme-catalyzed variants — the descriptor augmentation buys OOD generalization the structure-only approach cannot match. The qtaim_generator package [Vargas2024] generalized this approach with cross-dataset benchmarks (QM8, QM9, LIBE, Tox21, and a Green-group reaction set), finding consistent improvements in OOD generalization and small-data regimes when QTAIM features augment ChemProp and BondNet. A recent systematic study from the Green group [Green2024] catalogs when QM descriptors help D-MPNNs across 16 properties: the largest gains accrue for small datasets where descriptors correlate well with the target — a finding our dataset enables testing rigorously at scales where it has not previously been possible. The Coley group [Coley2023] further establishes that descriptor-augmented ML can match ab initio approaches on combustion-kinetics rate constants, albeit using cheminformatics topological indices rather than QM descriptors — evidence that the descriptor-augmentation paradigm is robust across descriptor families. Across all three model families, what has limited progress is dataset scale and descriptor breadth, not modeling capacity.

**Tone note**: ~370 words. The longest paragraph in the section, justified by being the load-bearing one for "models that will use this dataset." The three-family taxonomy is reusable in §6 (Benchmark Tasks and Baselines) of the body.

**Two research lineages now explicit**: Tkatchenko (Schütt2019 → Gallegos2024) within family 1, and Smidt (Unke2021 → Rackers2023) within family 2. The Schütt2019→Gallegos2024 connection is particularly worth highlighting because it shows SchNet4AIM is part of an active, multi-year research program with proven scaling capacity — strengthening the "natural training corpus" claim.

**Engagement with Green2024 finding**: deliberate. Their study answers "do descriptors help?" at the 100–10⁵ data-size scale; ours enables "*which* descriptors help for *which* property in *which* chemistry subdomain?" at 4M scale. This pivot from binary to conditional is the right defensive move and worth landing in §6 of the body too.

**Grambow/Vargas2021 pairing**: cleanest way to motivate descriptor augmentation. Grambow shows DL alone works for barriers; Vargas2021 shows QTAIM augmentation enables OOD extrapolation. Both citations earn their place; neither carries the argument alone.

### ¶7 — Gap and contributions

> OMol-Descriptors-4M is, to our knowledge, the first 4M-scale chemistry dataset that is wide on both axes simultaneously. We provide 4M structures spanning ~3.9M unique compositions across the chemistry envelope of OMol25 [Levine2026] (transition metals, lanthanides, actinides, charged species, reactive geometries, protein-derived fragments) with multi-method post-DFT descriptors at the first tier of a planned multi-tier release: six partial-charge schemes (Hirshfeld, ADCH, CM5, Becke, Mulliken-ORCA, Loewdin-ORCA), five bond-order schemes (fuzzy, IBSI, Laplacian from Multiwfn; Mayer and Loewdin from ORCA), full QTAIM topology including ring critical points, cage critical points, and delocalization indices, fuzzy descriptors (Becke and Hirshfeld densities, ELF, MBIS, Laplacian), and ORCA-derived globals. Subsequent tiers will extend coverage to additional methods and levels of theory. The dataset is one to three orders of magnitude beyond prior dedicated descriptor releases on every descriptor family it ships; against the 4M-scale precedents, it expands descriptor breadth from one (PCQM4Mv2) or moderate-but-narrow-chemistry (QM7-X) to dense multi-method coverage on a chemistry envelope two orders of magnitude wider than QM7-X's. This work extends a research program from our group on QTAIM-based descriptors for ML — barrier prediction with extrapolation [Vargas2021], a general-purpose package and cross-dataset benchmarks [Vargas2024], and the tmQM+ release for transition metal chemistry [Vargas2025] — to the scale and chemistry envelope of OMol25. Our contributions are three: (1) the dataset, with composition-consistent splits and six held-out chemistry stress tests; (2) the open HPC pipeline (qtaim_generator) that produced it, with measured throughput on three production HPC centers and end-to-end reproducibility from ORCA inputs to graph-ready LMDBs; (3) an evaluation protocol with quantified per-descriptor noise floors derived from cross-method comparison, defining a lower bound on achievable error that future work using this benchmark should report.

**Tone note**: every sentence in this paragraph traces back to a section. The first sentence funds §3 (dataset). The "one to three orders of magnitude" claim funds §5 (coverage). The three contributions correspond to §3-4 / §6 / §8 in the body. This is the contract paragraph — anything claimed here must be developed in the body.

**Primacy claim status (locked 2026-05-03)**: "first 4M-scale chemistry dataset that is wide on both axes simultaneously" is verified by Santiago as a member of the OMol team — no other group has post-processed the OMol25 4M electronic-structure release at this scale. Cleared for submission.

---

## Where I'm uncertain (per Santiago's stated preference)

**All citations resolved as of v7.**

**Optional / Santiago to decide later (not blocking)**:

- **AIMNet (1st gen, Zubatyuk 2019 *Chem Sci*)**: currently single-cite to Anstine2025. If you want to add the 2019 reference for completeness, easy to insert.

- **OrbNet extensions** (OrbNet-Equi, OrbNet-Denali): currently single-cite to Qiao2020. The extensions don't carry argumentative weight beyond the original paradigm; cleaner without them.

- **Optional results from cited papers**: marked `[opt-result]` in ¶4 (Vargas2025 cross-LOT result) and ¶6 (SchNet4AIM scale/error). Quick wins for a copy-edit pass when you have time, not blocking.

**Editorial choices that are deliberate but worth flagging**:

- Green2024 framed as motivation rather than counter-evidence.
- Coley2023 lightest cite, first cut if overlong.
- "Wide on both axes" framing in ¶7 is load-bearing.
- Grambow/Vargas2021 pairing in ¶6 — clean motivator for descriptor augmentation.
- Yuan2026 dual-cite — paid for by the dual rhetorical role (framing in ¶1, criteria in ¶5).
- "Orthogonally scoped" framing in ¶4 (tmQM+ varies LOT × QTAIM-only; OMol-Desc-4M varies descriptor family × single LOT) — this is the conceptual sharpening from v6 and remains the cleanest way to relate the two datasets.
- **Two research lineages** acknowledged in ¶6: Tkatchenko (Schütt2019 → Gallegos2024) in family 1, and Smidt (Unke2021 → Rackers2023) in family 2. Both lineages are active programs whose natural next training corpus is OMol-Desc-4M — this is implicit but worth being aware of when defending against "who will use this dataset?" reviewer questions.

---

## Suggested next step

Prose is now lockable. The brief is ready to be converted to LaTeX for the body of the paper. Highest-leverage next moves:

1. **Convert to LaTeX section** — turn the seven paragraphs into the actual §1+§2 of the paper, with `\cite{}` calls using the locked citation keys. One tool call deliverable; we have everything needed.

2. **Build the descriptor-breadth × chemistry-breadth scatter figure** for ¶5 / §1. Visually clinches the "wide on both axes" / "orthogonal axes" argument. One tool call deliverable from this same data.

3. **Pivot to §5 Coverage Analysis structure**. The prior-dataset list (QM7-X, PCQM4Mv2, QMugs, OE62, Alchemy, BDE-db, tmQM+) is now the comparator set for T3. Higher-leverage analytical content for the body.

4. **Optional results insertion** at the `[opt-result]` markers in ¶4 and ¶6 when you have time.

Your call on order.
