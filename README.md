# Multi-Modal Cell Type Classifier

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive Streamlit demo that illustrates how a multi-modal deep learning
pipeline could classify neurons by fusing electrophysiology, morphology, and
transcriptomics data. The app generates synthetic traces, morphological maps,
and gene-expression profiles on the fly, lets users toggle modalities on or
off, and visualises how attention-based late fusion combines the three streams.

## What the App Demonstrates

- **Synthetic data generation** for three neuroscience modalities
  (patch-clamp traces, 2-D dendritic projections, log-scaled expression matrices).
- **Late-fusion concept**: each modality is processed by a notional encoder
  (LSTM, 3-D CNN, Transformer) before an attention layer merges them.
- **Graceful degradation**: users can disable any subset of modalities and
  see how predicted confidence and attention weights shift.
- **Interactive classification**: a simulated classifier produces per-class
  probability distributions and highlights the predicted cell type.

## Limitations

- **All data is synthetic.** No real Allen Cell Types Database or Patch-seq
  recordings are loaded. The traces and images are procedurally generated
  with NumPy / SciPy.
- **No trained model.** Classification probabilities are sampled from a
  Dirichlet distribution, not computed by a neural network. The accuracy
  figures shown (85.2 %, 78.3 %, 71.1 %) are hard-coded illustrative
  targets, not measured results.
- **No PyTorch / TensorFlow dependency.** The app runs entirely on Streamlit,
  NumPy, SciPy, and Matplotlib.

## Running Locally

```bash
pip install streamlit numpy scipy matplotlib
streamlit run app.py
```

## References

- Gouwens, N. W. et al. (2020). Integrated morphoelectric and transcriptomic
  classification of cortical GABAergic cells. *Cell*, 183(4), 935-953.
- Scala, F. et al. (2021). Phenotypic variation of transcriptomic cell types
  in mouse motor cortex. *Nature*, 598, 144-150.
- Allen Institute for Brain Science. Allen Cell Types Database.
  https://celltypes.brain-map.org/

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

Built by Kiran Shay | Johns Hopkins University
