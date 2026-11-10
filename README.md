# Topological Code-Graph Slicing and Banach Fixed-Point Convergence for On-Device Gemma 4 SWE Agents

Official repository for the Google Kaggle Paper Track submission: **The Gemma 4 Developer Agent Paper Track**.

## Abstract
Democratizing autonomous software engineering requires execution on local, single-accelerator hardware without cloud leakage or context exhaustion. We introduce **Topological Code-Graph Slicing (TCGS)** via deterministic AST manifolds and **Banach Contraction State Verification (BCSV)**, mathematically proving monotonic convergence to zero-defect fixed points.

## Contents
- `harness.py`: Core implementation of `TopologicalCodeGraph` and `BanachContractionValidator`.
- `test_harness.py`: Comprehensive test suite verifying convergence theorems and AST extraction.
- `PAPER.md`: Full research writeup (2,177 words).
- `paper.tex`: LaTeX preprint format.

## Quickstart & Verification
```bash
python3 test_harness.py
```

## License
Open Source under Apache License 2.0.
