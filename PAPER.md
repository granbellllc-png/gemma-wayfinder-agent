# Topological Code-Graph Slicing and Contractive Lyapunov Verification for On-Device Gemma 4 SWE Agents

*Token-Budgeted AST Graph Textualization and Inference-Time State-Space Backtracking on Single-Accelerator Hardware*

- **Author:** Alessandro Massimiliano Gasparini (`alexmaxi`)
- **Track:** Google DeepMind — The Gemma 4 Developer Agent Paper Track
- **Open-Source Repository (Apache 2.0):** [https://github.com/granbellllc-png/gemma-wayfinder-agent](https://github.com/granbellllc-png/gemma-wayfinder-agent)
- **Executable GPU Verification Notebook:** [https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification](https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification)

---

## Abstract

Deploying autonomous software engineering (SWE) agents on local, single-accelerator hardware is central to code privacy, predictable latency, and cost-free developer tooling (Jimenez et al., 2024). Google DeepMind's open-weight Gemma 4 family—spanning edge models (E2B ~2.3B and E4B ~4.5B effective parameters, 128K context) and workstation tiers (12B, 26B A4B Mixture-of-Experts with ~3.8B active / 25.2B total parameters, and 31B Dense, 256K context; Gemma Team, 2026)—brings capable code reasoning to consumer devices. Yet under tight local memory ceilings (e.g., <1.8 GB residual KV-cache headroom for 4-bit 26B A4B on 16 GB Unified Memory), on-device SWE agents face two separable bottlenecks: (1) **graph-textualization loss**, where flat chunking or lexical retrieval (Zhang et al., 2023) drops multi-hop API contracts while inflating KV memory (cf. Markowitz et al., 2025), and (2) **unbounded multi-turn limit cycles**, where unguarded ReAct loops (Yao et al., 2023) oscillate between broken patches without structured state-space backtracking (cf. Markowitz et al., 2024).

We study these two layers orthogonally and jointly in Wayfinder v3.1: (i) **Hierarchical Hop-Skeletonized Topological Code-Graph Slicing (TCGS)**, an execution-free AST dependency textualizer, and (ii) **State-Adaptive Discrete Contractive Lyapunov Verification (DCLV)** (Banach, 1922; Khalil, 2002) with a lightweight $O(|\mathcal{X}|\cdot|\mathcal{E}_c|)$ Bellman-Ford `DPLL(QF_RDL)` runtime invariant check (Nieuwenhuis et al., 2006). Across 45 targets in five Python standard-library packages (1,917 AST nodes, 5,997 edges), TCGS ($k=2$) reaches 54.7% 2-hop causal recall versus 18.7% for Okapi BM25 (+36.0 pp, 95% CI [30.5, 41.9], Wilcoxon $p = 7.62 \times 10^{-9}$) and 10.4% for PageRank RepoMap at 16.69$\times$ compression (3,335 vs. 55,670 tokens; 616.98 MB vs. 10.30 GB KV-cache). In factorial live GPU ablations on official `google/gemma-4` weights (Gemma 4 E2B-it and Gemma 4 E4B-it on 2 Tesla T4 GPUs across 20 seven-file Python diagnostic repositories and 100 held-out unit tests under an identical 6-call ceiling), Wayfinder v3.1 raises resolution from 5.0% (1/20) to 20.0% (4/20) on E2B-it and raises resolution from 40.0% (8/20) to 60.0% (12/20) on E4B-it (surpassing fixed-ratio contraction at 50.0% on E4B-it and 15.0% on E2B-it; exact McNemar $p = 0.092285$ vs. BM25 and $p = 0.375$ vs. Fixed DCLV across $N=40$ pairs), enforcing 0.0% cyclic thrashing and 100.0% verified `QF_RDL` witnesses.

---

## 1. Introduction: Separating Graph Textualization from Inference-Time Backtracking

Repository-level bug repair requires localizing defects across multi-file call chains and preserving regression suites (Jimenez et al., 2024). Recent work on structured graph reasoning with LLMs establishes two foundational principles: first, **how a graph is textualized** under a token budget strongly governs downstream LLM reasoning accuracy (*KG-LLM-Bench*; Markowitz et al., 2025); second, **zero-shot inference-time search with explicit state backtracking** enables smaller or black-box LLMs to prune invalid reasoning branches without modifying weights (*Tree-of-Traversals*; Markowitz et al., 2024). Translating both principles from knowledge graphs to local Gemma 4 SWE agents (Gemma Team, 2026) exposes three concrete systems questions:

1. **Token-Budgeted Repository Textualization (KV-Cache Wall):** Although Gemma 4 26B A4B supports a 256K architectural context window, 4-bit weights consume ~14.2 GB on a 16 GB Unified Memory workstation, leaving under 1.8 GB for activations and FP16 KV-cache. Streaming a 55.7k-token package requires ~10.3 GB of KV memory. Meanwhile, lexical retrieval (Zhang et al., 2023) favors distractor modules sharing surface identifier tokens while omitting 2-hop domain contract classes, and global PageRank RepoMaps over-select high-in-degree utility hubs.
2. **Inference-Time Loop Stability vs. Over-Rigid Contraction:** Unguarded ReAct loops (Yao et al., 2023) frequently enter 2-cycle oscillations ($S_A \leftrightarrow S_B$) after test failures, burning remaining inference calls. Conversely, naive fixed-ratio Lyapunov contraction $V(S_{t+1}) \le \lambda_0 V(S_t)$ ($\lambda_0 = 0.75$) falsely rejects valid single-file intermediate repairs on coupled multi-file bugs: when 5 failing tests span `service.py` and `controller.py` ($V(S_0) = 7.5$), fixing `service.py` first reduces failures from 5 to 4 across one modified file ($V(S_1) = 6.5$, ratio $0.867 > 0.75$), prematurely stalling multi-turn repair.
3. **Factorial Attribution:** Rather than bundling retrieval and control into an opaque pipeline, an honest evaluation must isolate how much end-to-end gain derives from **AST graph textualization alone** (`BM25` $\to$ `TCGS` without a governor) versus **inference-time backtracking governance** (`Ungoverned` $\to$ `Fixed DCLV` $\to$ `Adaptive DCLV + QF_RDL`) under identical prompts, token budgets, and call ceilings.

---

## 2. Topological Code-Graph Slicing (TCGS): Hop-Skeletonized Graph Textualization

`TopologicalCodeGraph` serves as our primary context wedge, constructing a static directed dependency manifold $G = (V, E)$ without executing untrusted repository code:
1. **Execution-Free AST Symbol Indexing:** Module functions, coroutines, classes, and methods are indexed via `ast.parse` as `CodeNode` vertices storing line spans, formal signatures, docstrings, full AST bodies, and compact interface projections.
2. **Import-Scoped Bidirectional Resolution:** `_link_dependencies` resolves `import` and `from ... import ...` bindings, `ClassDef.bases` inheritance, and scoped `ast.Call` or attribute references bidirectionally (`dependencies` and `callers`), prioritizing imported module stems while dampening high-degree utility hubs (`max_hub_degree = 24`) and deduplicating cyclic imports via BFS visited sets.
3. **Hop-Skeletonized Textualization ($k=2$):** Breadth-first traversal (`extract_topological_slice_with_hops`) from focal symbol $u_0$ allocates token fidelity by causal distance: hops $d \in \{0, 1\}$ emit **complete executable AST source** for exact edits on $u_0$ and its immediate callers/callees, whereas hop $d = 2$ emits **read-only typed interface summaries** (`signature + docstring`), preserving 2-hop domain contracts at roughly fivefold lower token cost (Markowitz et al., 2025).

---

## 3. Inference-Time Backtracking Governance: State-Adaptive DCLV and `QF_RDL`

### 3.1 Zero-Oracle-Leak Defect Potential on Lattice $\frac{1}{2}\mathbb{N}_0$

To govern multi-turn code edits as a discrete state-space search with deterministic backtracking (Markowitz et al., 2024), let the observable repository state at turn $t$ be $S_t = \langle \mathcal{F}_t, \mathcal{T}_t, \mathcal{U}_t \rangle \in \mathcal{S}$, where $\mathcal{F}_t: \mathcal{P} \to \{0,1\}^{256}$ maps tracked paths to SHA-256 digests (baseline $\mathcal{F}_0$), $\mathcal{T}_t \in \mathbb{N}_0$ counts failing unit tests, and $\mathcal{U}_t \in \mathbb{N}_0$ counts static AST syntax or import errors. Without access to any oracle patch $\mathcal{F}^*$, we target the zero-defect equivalence manifold $\mathcal{S}^* = \{ S \in \mathcal{S} \mid \mathcal{T} = 0 \land \mathcal{U} = 0 \}$ via the Defect Lyapunov Potential $V: \mathcal{S} \to \frac{1}{2}\mathbb{N}_0$ (Banach, 1922; Khalil, 2002):
$$V(S_t) = \alpha \cdot \mathcal{T}_t + \beta \cdot \mathcal{U}_t + \gamma \cdot \mathbb{I}(\mathcal{T}_t > 0) \cdot |\mathcal{F}_t \Delta \mathcal{F}_0|$$
with $(\alpha, \beta, \gamma) = (1.5, 1.0, 0.5)$ and file-blast radius $|\mathcal{F}_t \Delta \mathcal{F}_0|$. Thus: (1) $V(S_t) = 0 \iff S_t \in \mathcal{S}^*$ because $\mathbb{I}(\mathcal{T}_t > 0)$ releases legitimate multi-file edit penalties once all tests pass; (2) since $(2\alpha, 2\beta, 2\gamma) = (3, 2, 1) \in \mathbb{N}^3$, every defective state satisfies $V(S_t) \ge \Delta_{\min} = 0.5$; and (3) every term is computed deterministically from sandboxed test runs, `ast.parse`, and SHA-256 digests.

### 3.2 State-Adaptive Contraction Envelope and Lightweight `QF_RDL` Certificate

To accept valid single-quantum partial fixes on coupled multi-file states while preserving strict descent, we define:
$$\lambda_{\text{eff}}(S_t) = \max\!\left(\lambda_0,\; 1 - \frac{\Delta_{\min}}{V(S_t)}\right) \quad \text{for } V(S_t) > 0, \quad 0 < \lambda_0 < 1$$
paired with a SHA-256 state-hash Taboo set $\mathcal{H}_t = \{ \text{SHA256}(\mathcal{F}_\tau, \mathcal{T}_\tau, \mathcal{U}_\tau) \}_{\tau=0}^t$ and neutral refactoring budget $B_{\text{neutral}} \in \mathbb{N}_0$ (default 1).

**Theorem 1 (Finite-Step Lattice Convergence and Zero Limit-Cycle Guarantee).** *For finite $V_0 = V(S_0) < \infty$, $\lambda_0 \in (0, 1)$, and $\Delta_{\min} = 0.5$: (1) atomic rollback leaves $S_t$ and $V(S_t)$ invariant on rejected candidates; (2) Taboo $\mathcal{H}_t$ prevents committing any defective configuration ($V > 0$) twice (0.0% cyclic thrash); (3) fixed contraction ($\lambda_{\text{eff}} \equiv \lambda_0, B_{\text{neutral}}=0$) terminates in at most $K_{\text{fixed}} = \lfloor \ln(\Delta_{\min} / V_0) / \ln(\lambda_0) \rfloor + 1$ accepted steps; and (4) adaptive $\lambda_{\text{eff}}(S_t)$ reduces $V$ by at least $0.5$ on every non-neutral step over $\frac{1}{2}\mathbb{N}_0$, terminating in $K_{\text{adapt}} \le 2 V_0 + B_{\text{neutral}}$ accepted steps.* $\blacksquare$

Rather than invoking an external theorem prover, `DifferenceLogicSMTVerifier` implements a zero-dependency sub-millisecond Bellman-Ford check in $O(|\mathcal{X}|\cdot|\mathcal{E}_c|)$ over four Quantifier-Free Real Difference Logic (`QF_RDL`) bounds relative to $x_0 \equiv 0$ (Nieuwenhuis et al., 2006):
$$V(\tilde{S}) - V(S_t) \le \Delta_{\text{allowed}}(S_t), \quad x_0 - V(\tilde{S}) \le 0, \quad |\tilde{\mathcal{F}} \Delta \mathcal{F}_0| - x_0 \le B_{\text{files}}, \quad \text{tokens}(\tilde{S}) - x_0 \le B_{\text{token}}$$
**Theorem 2 (`QF_RDL` Witness Soundness).** *Constraint graph $\mathcal{G}_c = (\mathcal{X}, \mathcal{E}_c)$ is satisfiable iff it contains no negative-weight directed cycle; Bellman-Ford either emits a real assignment $\pi$ checked against every inequality (`verify_witness(pi) == True`) or isolates the minimal violated invariant cycle (`unsat_core`).* $\blacksquare$

---

## 4. Empirical Evaluation and Factorial Ablations

All reported metrics are cryptographically anchored by `results/live_gemma4_gpu_telemetry.json` (SHA-256 `acf9de0b44c37be30448068d...`) and `results/benchmark_telemetry.json` (SHA-256 `47032dd3e673b4dd26aaefcd...`).

### 4.1 Experiment A: Python Standard-Library Graph Textualization ($N = 45$ Targets)

We index five multi-file Python standard-library packages (`email`: 635 nodes / 2,290 edges; `unittest`: 446 / 1,149; `urllib`: 290 / 783; `logging`: 284 / 1,133; `http`: 262 / 642 — 1,917 AST nodes, 5,997 directed edges) and evaluate seven textualization strategies on 2-hop causal dependency recall under a 3,500-token ceiling:

| Graph / Context Textualization Strategy | 2-Hop Recall | Mean Tokens | Compression | 26B A4B KV | Paired Gain of TCGS $k=2$ (95% Bootstrap CI, Wilcoxon $p$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Full-Package Context | 100.0% | 55,670 | 1.0$\times$ | 10,298.95 MB | Unbounded baseline (>10 GB KV-cache OOM) |
| PageRank RepoMap ($\le 3.5\text{k}$) | 10.4% | 3,497 | 15.92$\times$ | 646.94 MB | +44.3 pp [37.3, 51.4], $p = 5.18 \times 10^{-9}$ (45W/0L) |
| Okapi BM25 Lexical ($\le 3.5\text{k}$) | 18.7% | 3,494 | 15.93$\times$ | 646.39 MB | +36.0 pp [30.5, 41.9], $p = 7.62 \times 10^{-9}$ (44W/0L/1T) |
| Fixed 500-Token Chunks | 22.1% | 6,098 | 9.13$\times$ | 1,128.13 MB | +32.7 pp [26.1, 39.5], $p = 1.41 \times 10^{-8}$ (42W/2L/1T) |
| TCGS $k=1$ (1-Hop Bodies Only) | 21.4% | 2,605 | 21.37$\times$ | 481.93 MB | +33.4 pp [27.4, 39.6], $p = 5.18 \times 10^{-9}$ (45W/0L) |
| **TCGS $k=2$ Skeletonized (Ours)** | **54.7%** | **3,335** | **16.69$\times$** | **616.98 MB** | **Pareto optimum (<1.8 GB Unified Memory)** |
| TCGS $k=3$ Skeletonized | 55.1% | 3,454 | 16.12$\times$ | 638.99 MB | Diminishing return (+0.4 pp for +119 tokens) |

### 4.2 Experiment B: Factorial Live On-GPU `google/gemma-4` Repair ($N = 20$ Diagnostic Repos, 100 Tests)

To cleanly separate retrieval gains from backtracking-control gains on real neural weights, `live_gemma4_gpu_benchmark.py` constructs 20 controlled seven-file Python diagnostic repositories across five domains (`billing_ledger`, `rate_limiter`, `auth_session`, `inventory_order`, `telemetry_pipeline`; 100 held-out `unittest` checks isolated outside the agent workspace; 13 coupled multi-file suites and 7 decoupled multi-assertion suites spanning `controller.py`, `service.py`, and read-only `contracts.py`, plus 4 domain-realistic lexical distractor modules per repo). Under a matched 450-token slice budget, BM25 recalls all three causal files on 40.0% of repositories (406 mean tokens) due to distractor overlap, whereas TCGS ($k=2$) achieves 100.0% causal recall at 404 mean tokens.

We run official `google/gemma-4` weights (Gemma 4 E2B-it and Gemma 4 E4B-it, PyTorch 2.11.0+cu130, Transformers 5.20.2 on 2 Tesla T4 GPUs) across a four-arm factorial design under identical prompts and a 6-call ceiling, reporting exact Gemma 4 billed tokens (including rolled-back proposals) and **total billed tokens per resolved repository**:

| Live On-GPU Model and Factorial Arm (6-Call Ceiling, $N = 20$ Repos) | Resolved | Mean Calls | Mean Billed Tok | Billed Tok / Solved Repo | Rollbacks | Thrash |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Gemma 4 E2B-it + Okapi BM25 (Ungoverned ReAct) | 1 / 20 (5.0%) | 6.00 | 5,442 | 108,840 | 0 | 50.0% |
| Gemma 4 E2B-it + TCGS k=2 (Ungoverned ReAct) | 4 / 20 (20.0%) | 5.35 | 4,777 | 23,885 | 0 | 70.0% |
| Gemma 4 E2B-it + TCGS k=2 + Fixed DCLV (lambda_0=0.75) | 3 / 20 (15.0%) | 5.55 | 5,616 | 37,440 | 96 | 0.0% |
| **Gemma 4 E2B-it + Wayfinder Full (TCGS k=2 + Adaptive DCLV + QF_RDL)** | **4 / 20 (20.0%)** | **5.45** | **5,381** | **26,905** | **79** | **0.0%** |
| Gemma 4 E4B-it + Okapi BM25 (Ungoverned ReAct) | 8 / 20 (40.0%) | 4.90 | 4,500 | 11,250 | 0 | 40.0% |
| Gemma 4 E4B-it + TCGS k=2 (Ungoverned ReAct) | 12 / 20 (60.0%) | 3.70 | 3,307 | 5,512 | 0 | 35.0% |
| Gemma 4 E4B-it + TCGS k=2 + Fixed DCLV (lambda_0=0.75) | 10 / 20 (50.0%) | 4.15 | 3,966 | 7,932 | 55 | 0.0% |
| **Gemma 4 E4B-it + Wayfinder Full (TCGS k=2 + Adaptive DCLV + QF_RDL)** | **12 / 20 (60.0%)** | **3.70** | **3,440** | **5,733** | **25** | **0.0%** |

Three decoupled findings emerge from the factorial matrix:
1. **Graph Textualization Is the Primary Accuracy Driver (`BM25` $\to$ `TCGS Ungoverned`):** Replacing lexical BM25 with hop-skeletonized TCGS ($k=2$) without any governor shifts resolution from 40.0% (8/20) to 60.0% (12/20) on E4B-it (and from 5.0% to 20.0% on E2B-it), confirming that 2-hop contract visibility drives the largest solve-rate jump when lexical distractors compete for context. However, without state governance, failed repairs trigger cyclic state revisits (70.0% thrash on E2B-it; 35.0% on E4B-it).
2. **Naive Fixed Contraction Degrades Multi-File Repair, Whereas Adaptive DCLV Preserves Peak Solve Rate at Zero Thrash:** Fixed-ratio contraction ($\lambda_0 = 0.75$) eliminates cyclic thrashing (0.0%) but drops resolution on both scales—from 60.0% to 50.0% on E4B-it (55 rollbacks) and from 20.0% to 15.0% on E2B-it (96 rollbacks)—failing coupled multi-file suites whenever a valid Turn-1 single-file fix reduces failures from 5 to 4 ($V: 7.5 \to 6.5 > 5.625$). State-adaptive $\lambda_{\text{eff}}(S_t)$ recovers full resolution on both scales (60.0%, 12/20, 25 rollbacks on E4B-it; 20.0%, 4/20, 79 rollbacks on E2B-it). Across $N = 40$ paired runs, Wayfinder records 10W/3L over BM25 ReAct (two-sided exact McNemar $p = 0.092285$, reflecting directional uplift under $N=20$ per-tier sample power), 4W/1L over Fixed DCLV ($p = 0.375$), and 1W/1L vs. Ungoverned TCGS ($p = 1.0$) while eliminating cyclic thrashing from 18/40 (BM25) and 21/40 (Ungoverned TCGS) down to 0/40 (0.0%, exact McNemar $p < 10^{-5}$).
3. **Inference-Time Compute Efficiency and `QF_RDL` Soundness:** Counting all rolled-back calls, Wayfinder lowers billed tokens per solved repository from 108,840 to 26,905 on E2B-it and from 11,250 to 5,733 (vs. 7,932 under Fixed DCLV) on E4B-it, with 100.0% verified `QF_RDL` witnesses (30/30 on E2B-it; 49/49 on E4B-it).

### 4.3 Non-LLM Verifier Stress Test ($N = 15$) and Horizon Sensitivity Illustration ($N = 250$)

To stress-test `DifferenceLogicSMTVerifier` and atomic rollback independently of LLM variance, Experiment C feeds scripted adversarial patch streams (explicitly non-LLM inputs: regression, baseline revisit, partial fix, duplicate hash, complete fix) across 15 three-file repositories (60 tests): unguided execution resolves 0/15 (100% thrash), whereas Wayfinder resolves 15/15 with 36 rollbacks and 30/30 verified `QF_RDL` witnesses. Separately, `benchmark_runner.py` provides an explicitly simulated Markov horizon sweep (`seed=20261110`, $N=250$, non-LLM) illustrating >6-turn scaling across fixed $\lambda_0 \in \{0.50, 0.65, 0.75, 0.85, 0.95\}$ at $V_0 = 8.0$ (21.2%, 38.8%, 58.4%, 60.0%, 66.0%), where adaptive $\lambda_{\text{eff}}(S_t)$ reaches 66.8% vs. 53.6% for fixed $\lambda_0 = 0.75$.

---

## 5. Scope, Failure Modes, Governance, and Reproducibility

1. **Diagnostic Benchmark Scope vs. Full SWE-Bench Scale:** Experiment B's 20 seven-file repositories are controlled diagnostic suites engineered to isolate 2-hop contract coupling and lexical distractor interference on edge-sized E2B/E4B models, complementing real `stdlib` AST slicing (Experiment A) rather than claiming full SWE-bench resolution without scaling to workstation 26B A4B / 31B tiers.
2. **Failure Modes and Static AST Limits:** Static `ast.parse` misses dynamic `getattr` dispatch (requiring `tree-sitter` or runtime trace integration for polyglot/dynamic codebases). On smaller edge models (E2B-it), multi-file coupled repairs remain capacity-limited even when TCGS supplies full 2-hop contracts; DCLV prevents destructive repository drift and cyclic loops rather than substituting for model synthesis capacity.
3. **Sandbox Security, Licensing, and Audit Governance:** Indexing executes zero untrusted repository code (`ast.parse` only); unit tests run in isolated ephemeral `subprocess` sandboxes with strict timeouts; official `google/gemma-4` checkpoints are evaluated unmodified under Gemma Terms of Use; and every state transition logs an auditable SHA-256 digest and `QF_RDL` witness under Apache 2.0 (`test_harness.py`, `benchmark_runner.py`, `live_gemma4_gpu_benchmark.py`, `verify_submission_integrity.py`).

---

## References

1. Gemma Team, Google DeepMind. (2026). *Gemma 4 Technical Report*. arXiv:2607.02770.
2. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
3. Markowitz, E., Ramakrishna, A., Dhamala, J., Mehrabi, N., Peris, C., Gupta, R., Chang, K.-W., & Galstyan, A. (2024). *Tree-of-Traversals: A Zero-Shot Reasoning Algorithm for Augmenting Black-box Language Models with Knowledge Graphs*. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12302–12319.
4. Markowitz, E., Galiya, K., Ver Steeg, G., & Galstyan, A. G. (2025). *KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs*. arXiv:2504.07087.
5. Zhang, F., Chen, B., Zhang, Y., Keung, J., Liu, J., Zan, D., Mao, Y., Lou, J.-G., & Chen, W. (2023). *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation*. EMNLP 2023.
6. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023.
7. Banach, S. (1922). *Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales*. Fundamenta Mathematicae, 3(1), 133–181.
8. Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed.). Prentice Hall.
9. Nieuwenhuis, R., Oliveras, A., & Tinelli, C. (2006). *Solving SAT and SAT Modulo Theories: From an Abstract Davis–Putnam–Logemann–Loveland Procedure to DPLL(T)*. Journal of the ACM, 53(6), 937–977.
