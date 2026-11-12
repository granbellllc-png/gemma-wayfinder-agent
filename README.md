# Topological Code-Graph Slicing, Discrete Contractive Lyapunov Verification, and QF_RDL SMT Grounding for On-Device Gemma 4 SWE Agents

Official open-source research & verification repository (**v3.0**) for **Google DeepMind — The Gemma 4 Developer Agent Paper Track** on Kaggle.

- **Competition Page:** [https://www.kaggle.com/competitions/gemma-4-developer-agent-paper](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
- **GitHub Repository:** [https://github.com/granbellllc-png/gemma-wayfinder-agent](https://github.com/granbellllc-png/gemma-wayfinder-agent)
- **Public Kaggle Verification Notebook:** [https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification](https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification)
- **License:** Apache License 2.0

## Key Contributions & Verified Results (v3.0)

1. **Hierarchical Hop-Skeletonized Topological Code-Graph Slicing (TCGS):** Deterministic AST manifold builder (`TopologicalCodeGraph`) achieving **56.8% 2-hop causal dependency recall** (**2.99x** BM25, **5.46x** PageRank RepoMap) across 45 real cross-module targets in 5 standard-library Python packages (`email`, `urllib`, `http`, `logging`, `unittest`; 1,917 AST nodes, 5,997 directed edges) while compressing prompt tokens **16.81x** (`3,312` vs. `55,670` tokens; **612.7 MB** vs. **10.30 GB** Gemma 4 `26B A4B` KV-cache).
2. **Discrete Contractive Lyapunov Verification (DCLV) + `DPLL(QF_RDL)` Formal SMT Verifier:** Zero-oracle-leak potential $V(S_t) \in \frac{1}{2}\mathbb{N}_0$ with a **State-Adaptive Lattice Contraction Envelope** $\lambda_{\text{eff}}(S_t) = \max(\lambda_0, 1 - 0.5/V(S_t))$, SHA-256 state-hash Taboo guard (`DiscreteLyapunovValidator`), and exact Bellman-Ford difference-logic SMT solver (`DifferenceLogicSMTVerifier`), elevating multi-turn resolve rate from **13.6% to 68.4%**, eliminating cyclic thrashing (**0.0%**), and cutting total billed tokens (including rejected rollbacks) by **2.78x**.
3. **Live Executable Multi-File Python Bug-Fix Suite (`N = 15` Repositories):** End-to-end AST slicing + live `unittest` verification across 5 canonical multi-file defect families, achieving **100.0% 2-hop cross-file recall** (`157` mean slice tokens), **15/15 (100.0%) test-verified bug resolution**, **36 automatic rollbacks**, and **100.0% `QF_RDL` SAT verification**.

## Repository Structure

- `PAPER.md`: Full research paper writeup (`< 3,000` words, Kaggle Writeup compliant).
- `paper.pdf`: Publication-formatted PDF preprint generated directly from `results/benchmark_telemetry.json`.
- `harness.py`: Reference implementation of `TopologicalCodeGraph`, `DiscreteLyapunovValidator`, `DifferenceLogicSMTVerifier`, and `TRIZContradictionMatrixRouter`.
- `test_harness.py`: **11 automated unit, theorem, and `QF_RDL` SMT verification suites** (`Exit 0`).
- `benchmark_runner.py`: Reproducible benchmark suite (Experiments A, B, C, D) generating `results/benchmark_telemetry.json` (`SHA-256: 99c1504723884779c2e347cf7fc6bb9ab4be0fe0f96bfe72b90760b90c73462f`).
- `assets/cover_architecture.png` & `assets/ablation_and_live_benchmark.png`: High-resolution figures for the Kaggle Media Gallery.

## Reproducibility Quickstart (< 3 Seconds, Zero External Dependencies)

```bash
python3 test_harness.py
python3 benchmark_runner.py
```
