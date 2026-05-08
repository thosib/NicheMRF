# NicheMRF

NicheMRF detects functional cell niches in spatial transcriptomics data using Markov Random Fields.

A functional niche is a spatially coherent group of neighboring cells with similar gene expression and intercellular signaling profiles. Detecting these is useful for understanding tissue formation, immune function, and the tumor microenvironment.

**Authors:** Jada Dixon, Thomas Sibilly, Jonah Stockwell

**Data:** All raw data is available at https://doi.org/10.6019/S-BIAD2146. We have also provided sample files at two steps of the pipeline: "lymph_node_50k_sample.h5ad" (produced by nicheMRF_start) and "lymph_node_50k_with_signaling.h5ad" (produced by nicheMRF_preprocessing)

## Overview

Existing tools for niche detection use graph community detection (SquidPy / Leiden) or variational autoencoders (CellCharter, NicheCompass). MRFs have been applied to structured tissues like the brain but not to less-structured tissues. NicheMRF applies this framework to lymph node tissue, using ligand-receptor signaling as the primary feature rather than raw gene expression.

## Pipeline

**Data.** STHELAR Xenium lymph node (Giraud-Sauveur et al. 2026): ~500k cells, ~5k genes. Downsampled to 50k cells; analysis run on a vertical spatial strip of ~7k cells.

**Features.** COMMOT infers per-cell signaling activity across all LR pairs in the CellChat human database. Each cell's feature vector encodes total communication activity per pathway.

**Inference.** A Gaussian Mixture Model provides initial niche assignments. EM then iterates: the E-step computes soft memberships combining GMM likelihoods with a Potts spatial coupling term; the M-step updates GMM parameters via weighted MLE. Convergence is declared when the max change in soft assignments falls below 1e-5.

Notebooks should be run in the following order: nicheMRF_start (produces "lymph_node_50k_sample.h5ad") > nicheMRF_preprocessing (produces "lymph_node_50k_with_signaling.h5ad") > nicheMRF_inference (produces AnnData object and figures)

## Results

**Cluster quality.** Each niche's mean signaling vector was compared against 1,000 same-sized random subsamples of the dataset. P-values were computed from a beta distribution fit to the random correlations. Almost all extracted niches were statistically significant at p < 0.002.

**Cell type spread.** Cell types were distributed across niches rather than concentrated within them, consistent with niches being organized by signaling context rather than cell identity.

**Benchmarking vs. SquidPy** (38 niches on the same data):

| Tool                 | p < 0.005 | 0.005 < p < 0.05 | p > 0.05 |
| -------------------- | --------- | ---------------- | -------- |
| SquidPy              | 28        | 6                | 4        |
| NicheMRF (this work) | 9         | 1                | 1        |

NicheMRF produced fewer, spatially compact niches; all passed the significance threshold. SquidPy produced finer-grained partitions with three that did not.

## Limitations

- Analysis was restricted to a ~7k-cell vertical crop, not the full tissue section.
- Only one tissue type (lymph node) was evaluated.
