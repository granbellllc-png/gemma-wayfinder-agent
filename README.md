# Topological Code-Graph Slicing & Discrete Contractive Lyapunov Verification for On-Device Gemma 4 SWE Agents

Official open-source research repository for **Google DeepMind — The Gemma 4 Developer Agent Paper Track** on Kaggle.

- **Official Kaggle Paper Writeup:** [https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/gemma-4-tcgs-lyapunov-swe-agent](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/gemma-4-tcgs-lyapunov-swe-agent)
- **GitHub Repository:** [https://github.com/granbellllc-png/gemma-wayfinder-agent](https://github.com/granbellllc-png/gemma-wayfinder-agent)
- **Public Kaggle Verification Notebook:** [https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification](https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification)
- **License:** Apache License 2.0

## Highlights

1. **Hierarchical Hop-Skeletonized Topological Code-Graph Slicing (TCGS):** Deterministic AST manifold builder (`TopologicalCodeGraph`) capturing **56.8% 2-hop causal dependency recall** (**2.99x** BM25, **5.46x** PageRank RepoMap) across 45 real cross-module targets in 5 standard-library Python packages (`email`, `urllib`, `http`, `logging`, `unittest`) while compressing prompt tokens **16.81x** (`3,312` vs. `55,670` tokens; **612.7 MB** vs. **10.30 GB** Gemma 4 `26B A4B` KV-cache).
2. **Discrete Contractive Lyapunov Verification (DCLV):** Zero-oracle-leak potential $V(S_t) \in \frac{1}{2}\mathbb{N}_0$ with a **State-Adaptive Lattice Contraction Envelope** $\lambda_{\text{eff}}(S_t) = \max(\lambda_0, 1 - 0.5/V(S_t))$ and SHA-256 state-hash Taboo guard (`DiscreteLyapunovValidator`), elevating resolve rate from **13.6% to 68.4%**, eliminating limit-cycle thrashing (**0.0%**), and reducing total billed tokens (including rejected rollbacks) by **2.78x**.

## Repository Structure

- `PAPER.md`: Full research paper writeup ready for Kaggle Writeup submission.
- `paper.tex` & `paper.pdf`: Publication-formatted preprint (LaTeX source + compiled PDF).
- `harness.py`: Reference implementation of `TopologicalCodeGraph` and `DiscreteLyapunovValidator`.
- `test_harness.py`: 8 automated unit & theorem verification suites (`Exit 0`).
- `benchmark_runner.py`: Reproducible benchmark suite generating `results/benchmark_telemetry.json`.
- `assets/cover_architecture.png`: High-resolution architecture & ablation figure for the Kaggle Media Gallery.

## Reproducibility Quickstart (< 3 Seconds, Zero External Dependencies)

```bash
python3 test_harness.py
python3 benchmark_runner.py
```
