# Topological Code-Graph Slicing and Contractive Lyapunov Verification for On-Device Gemma 4 SWE Agents

Open-source research and GPU verification repository (**Wayfinder v3.1**) for **Google DeepMind — The Gemma 4 Developer Agent Paper Track** on Kaggle.

- **Competition Page:** [https://www.kaggle.com/competitions/gemma-4-developer-agent-paper](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
- **GitHub Repository:** [https://github.com/granbellllc-png/gemma-wayfinder-agent](https://github.com/granbellllc-png/gemma-wayfinder-agent)
- **Public GPU Kaggle Verification Notebook:** [https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification](https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification)
- **License:** Apache License 2.0

## Summary of Factorial Empirical Results (v3.1)

1. **Token-Budgeted AST Graph Textualization (`TopologicalCodeGraph`, TCGS $k=2$):** Execution-free AST dependency textualizer achieving 54.7% 2-hop causal dependency recall vs. 18.7% for Okapi BM25 (+36.0 pp, 95% CI [30.5, 41.9], paired Wilcoxon $p = 7.62 \times 10^{-9}$) and 10.4% for PageRank RepoMap across 45 cross-module targets in five Python standard-library packages (1,917 AST nodes, 5,997 directed edges), compressing context 16.69x (3,335 vs. 55,670 tokens; 616.98 MB vs. 10.30 GB KV-cache).
2. **Factorial Live On-GPU `google/gemma-4` Neural Repository Repair ($N = 20$ Seven-File Diagnostic Repositories, 100 Held-Out Unit Tests):** Evaluated on 2 Tesla T4 GPUs (PyTorch 2.11.0+cu130, Transformers 5.20.2) using official Gemma 4 E2B-it and Gemma 4 E4B-it weights under identical prompts and a 6-LLM-call ceiling:
   - **Gemma 4 E2B-it:** Wayfinder v3.1 raises resolution from 5.0% (1/20) to 20.0% (4/20) on E2B-it (vs. 15.0% under Fixed-$\lambda_0$ DCLV), with 26,905 billed tokens per solved repository and 0.0% cyclic thrashing.
   - **Gemma 4 E4B-it:** Wayfinder v3.1 raises resolution from 40.0% (8/20) to 60.0% (12/20) on E4B-it and outperforms Fixed-$\lambda_0$ DCLV (50.0%, 55 rollbacks vs. 25 rollbacks; paired exact McNemar $p = 0.092285$ vs. BM25 and $p = 0.375$ vs. Fixed DCLV across $N=40$ pairs), achieving 5,733 vs. 11,250 billed tokens per solved repository with 0.0% cyclic thrashing and 100.0% verified `QF_RDL` SAT witnesses.
3. **Sub-Millisecond Bellman-Ford `DPLL(QF_RDL)` Certificate & Five-Gate Forensic Audit:** Certifies Lyapunov contraction, non-negativity, file-blast radius, and prompt-token ceilings (`verify_witness(pi)`), backed by 12 unit/theorem suites (`test_harness.py`) and 5 automated artifact-parity gates (`verify_submission_integrity.py`).

## Repository Structure

- `PAPER.md`: Full research paper (<3,000 words, Kaggle Writeup compliant).
- `paper.pdf` and `paper.tex`: Unicode academic PDF preprint and synchronized LaTeX source (`build_publication_assets.py`).
- `harness.py`: Reference implementation of `TopologicalCodeGraph`, `DiscreteLyapunovValidator`, and `DifferenceLogicSMTVerifier`.
- `live_gemma4_gpu_benchmark.py`: 20-repository (7 files per repo, 100 held-out unit tests) Dual-GPU `google/gemma-4` (`E2B-it` and `E4B-it`) factorial repair benchmark (`results/live_gemma4_gpu_telemetry.json`, SHA-256 `acf9de0b44c37be30448068d0d40a89c6e6e3139925acc307de0ae3b1955c407`).
- `benchmark_runner.py`: Reproducible Python `stdlib` AST slicing ($N = 45$), non-LLM scripted `QF_RDL` stress suite ($N = 15$), and explicitly labeled synthetic Markov horizon sweep (`results/benchmark_telemetry.json`, SHA-256 `47032dd3e673b4dd26aaefcd0b581692fc9b2b17d93890d4aa48c653dbaf8f0d`).
- `test_harness.py` and `verify_submission_integrity.py`: 12 unit/theorem suites + 5-gate forensic audit (`Exit 0`).
- `assets/cover_architecture.png` and `assets/ablation_and_live_benchmark.png`: High-resolution figures generated directly from verified telemetry.

## Reproducibility Quickstart

```bash
python3 test_harness.py
python3 benchmark_runner.py
python3 live_gemma4_gpu_benchmark.py --self-check
python3 verify_submission_integrity.py
```
