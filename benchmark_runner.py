"""
Reproducible Empirical Benchmark Suite for Gemma 4 TCGS + DCLV Paper Track
==========================================================================
Executes three zero-external-dependency, deterministic benchmark suites:
  1. Experiment A: Real Multi-Module Repository AST Slicing Benchmark across 5 canonical
     Python multi-file packages (email, urllib, http, logging, unittest), comparing:
       - Full-Package Context
       - Fixed-Chunk Lexical Window (500-token chunks, <=3.5k budget)
       - Okapi BM25 Lexical Retrieval (<=3.5k budget)
       - Global PageRank RepoMap (Aider-style centrality, <=3.5k budget)
       - TCGS k-Hop Manifold (Ours: k=0, 1, 2, 3, <=3.5k budget)
  2. Experiment B: Controlled Stochastic Multi-Turn Defect Repair & Governor Ablation
     (N=250 seeded multi-file repair trajectories, seed=20261110), explicitly tracking
     BOTH Accepted Trajectory Tokens and Total Billed Tokens (including rejected rollbacks).
  3. Experiment C: Hyperparameter Sensitivity Frontier over k in {0,1,2,3} and
     contraction factor lambda in {0.50, 0.65, 0.75, 0.85, 0.95}.

Outputs deterministic JSON telemetry with SHA-256 digest to `results/benchmark_telemetry.json`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sysconfig
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from harness import DiscreteLyapunovValidator, TopologicalCodeGraph, estimate_tokens


STDLIB_PACKAGES: Tuple[str, ...] = ("email", "urllib", "http", "logging", "unittest")
TOKEN_BUDGET: int = 3500


def _tokenize_identifiers(text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            cur.append(ch.lower())
        else:
            if cur:
                tok = "".join(cur)
                if len(tok) > 1:
                    out.append(tok)
                cur = []
    if cur:
        tok = "".join(cur)
        if len(tok) > 1:
            out.append(tok)
    return out


def compute_pagerank(tcg: TopologicalCodeGraph, damping: float = 0.85, iters: int = 25) -> Dict[str, float]:
    nodes = list(tcg.nodes.keys())
    n = len(nodes)
    if n == 0:
        return {}
    scores = {nid: 1.0 / n for nid in nodes}
    for _ in range(iters):
        nxt = {nid: (1.0 - damping) / n for nid in nodes}
        for nid in nodes:
            outs = list(tcg.nodes[nid].dependencies)
            if outs:
                share = damping * scores[nid] / len(outs)
                for dst in outs:
                    if dst in nxt:
                        nxt[dst] += share
        scores = nxt
    return scores


def bm25_rank_nodes(
    tcg: TopologicalCodeGraph,
    query_terms: List[str],
    token_budget: int = TOKEN_BUDGET,
    k1: float = 1.5,
    b: float = 0.75,
) -> Set[str]:
    docs: Dict[str, Counter[str]] = {}
    doc_lens: Dict[str, int] = {}
    df: Counter[str] = Counter()

    for nid, node in tcg.nodes.items():
        toks = _tokenize_identifiers(f"{node.name} {node.signature} {node.docstring} {node.source_snippet}")
        cnt = Counter(toks)
        docs[nid] = cnt
        doc_lens[nid] = max(1, len(toks))
        for term in cnt:
            df[term] += 1

    n_docs = max(1, len(docs))
    avgdl = sum(doc_lens.values()) / n_docs
    scores: List[Tuple[float, str]] = []

    for nid, cnt in docs.items():
        dl = doc_lens[nid]
        s = 0.0
        for q in query_terms:
            tf = cnt.get(q, 0)
            if tf == 0:
                continue
            n_q = df.get(q, 0)
            idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            s += idf * (tf * (k1 + 1.0)) / denom
        scores.append((s, nid))

    scores.sort(key=lambda x: (-x[0], x[1]))
    selected: Set[str] = set()
    used = 0
    for sc, nid in scores:
        if sc <= 0.0:
            break
        cost = tcg.nodes[nid].token_estimate
        if selected and used + cost > token_budget:
            continue
        selected.add(nid)
        used += cost
    return selected


def fixed_chunk_overlap_nodes(
    tcg: TopologicalCodeGraph,
    query_terms: List[str],
    token_budget: int = TOKEN_BUDGET,
) -> Set[str]:
    """Simulates naive 500-token file chunking mapped back to covered AST symbols."""
    qset = set(query_terms)
    file_chunks: List[Tuple[int, int, str, Set[str]]] = []
    by_file: Dict[str, List[str]] = tcg.file_to_nodes

    for rel_path, nids in by_file.items():
        full_src = (tcg.root_dir / rel_path).read_text(encoding="utf-8", errors="replace")
        lines = full_src.splitlines()
        step = 38  # ~500 tokens per chunk
        for idx, start in enumerate(range(0, max(1, len(lines)), step)):
            chunk_lines = lines[start : start + step]
            chunk_txt = "\n".join(chunk_lines)
            ctoks = set(_tokenize_identifiers(chunk_txt))
            overlap = len(qset & ctoks)
            covered = {
                nid
                for nid in nids
                if not (tcg.nodes[nid].end_lineno < start + 1 or tcg.nodes[nid].lineno > start + step)
            }
            file_chunks.append((overlap, estimate_tokens(chunk_txt), f"{rel_path}:{idx}", covered))

    file_chunks.sort(key=lambda x: (-x[0], x[2]))
    selected: Set[str] = set()
    used = 0
    for ov, cost, _, covered in file_chunks:
        if ov <= 0:
            break
        if used + cost > token_budget and selected:
            continue
        selected.update(covered)
        used += cost
    return selected


def pagerank_repomap_nodes(
    tcg: TopologicalCodeGraph,
    pr_scores: Dict[str, float],
    seed_id: str,
    token_budget: int = TOKEN_BUDGET,
) -> Set[str]:
    ranked = sorted(tcg.nodes.keys(), key=lambda nid: (-1.0 if nid == seed_id else 0.0, -pr_scores.get(nid, 0.0), nid))
    selected: Set[str] = set()
    used = 0
    for nid in ranked:
        cost = tcg.nodes[nid].token_estimate
        if selected and used + cost > token_budget:
            continue
        selected.add(nid)
        used += cost
    return selected


def estimate_gemma4_kv_cache_mb(seq_tokens: int, model_tier: str = "26B_A4B") -> float:
    """
    Analytical KV-cache footprint (MB) for Gemma 4 hybrid local/global GQA layers
    plus activation workspace on Apple Silicon Unified Memory (FP16 KV cache).
    """
    specs = {
        "E4B": 0.095,     # MB per token (KV + runtime scratch)
        "12B": 0.165,
        "26B_A4B": 0.185, # 25.2B total, ~3.8B active MoE, 256K arch window, tight <1.8GB free on 16GB M4
    }
    return round(seq_tokens * specs[model_tier], 2)


def run_experiment_a_stdlib_slicing() -> Dict[str, Any]:
    stdlib_root = Path(sysconfig.get_paths()["stdlib"])
    records: Dict[str, List[Tuple[float, int]]] = {
        "full_package": [],
        "fixed_chunk_500": [],
        "bm25_lexical": [],
        "pagerank_repomap": [],
        "tcgs_k0": [],
        "tcgs_k1": [],
        "tcgs_k2": [],
        "tcgs_k3": [],
    }
    package_summaries: List[Dict[str, Any]] = []
    total_targets = 0

    for pkg_name in STDLIB_PACKAGES:
        pkg_dir = stdlib_root / pkg_name
        if not pkg_dir.exists():
            continue
        tcg = TopologicalCodeGraph(pkg_dir)
        node_count = tcg.index_repository()
        edge_count = sum(len(n.dependencies) for n in tcg.nodes.values())
        full_tokens = sum(n.token_estimate for n in tcg.nodes.values())
        pr = compute_pagerank(tcg)

        # Select realistic cross-module focal symbols (4 <= degree <= 18, 9 per package -> 45 real targets)
        candidates = [
            nid
            for nid, n in tcg.nodes.items()
            if 4 <= len(n.dependencies | n.callers) <= 18 and n.token_estimate <= 700
        ]
        if len(candidates) < 9:
            candidates = [
                nid for nid, n in tcg.nodes.items() if len(n.dependencies | n.callers) >= 3 and n.token_estimate <= 900
            ]
        candidates.sort(key=lambda nid: (-len(tcg.nodes[nid].dependencies | tcg.nodes[nid].callers), nid))
        focal_ids = candidates[:9]

        pkg_Summaries_cnt = len(focal_ids)
        total_targets += pkg_Summaries_cnt
        package_summaries.append(
            {
                "package": pkg_name,
                "ast_nodes": node_count,
                "directed_edges": edge_count,
                "full_package_tokens": full_tokens,
                "evaluated_targets": pkg_Summaries_cnt,
            }
        )

        for seed_id in focal_ids:
            seed_node = tcg.nodes[seed_id]
            # Ground-truth causal 2-hop repair neighborhood (unconstrained token budget)
            gt_1hop = {seed_id} | seed_node.dependencies | seed_node.callers
            gt_2hop: Set[str] = set(gt_1hop)
            for nb in list(gt_1hop):
                if nb in tcg.nodes:
                    gt_2hop.update(tcg.nodes[nb].dependencies | tcg.nodes[nb].callers)
            gt_neighbors = gt_2hop - {seed_id}
            if not gt_neighbors:
                continue

            query_terms = _tokenize_identifiers(f"{seed_node.name} {seed_node.signature} {seed_node.docstring}")
            if not query_terms:
                query_terms = [seed_node.name.lower()]

            # 1. Full package
            records["full_package"].append((1.0, full_tokens))

            # 2. Fixed chunk overlap
            fc_sel = fixed_chunk_overlap_nodes(tcg, query_terms, TOKEN_BUDGET)
            fc_tok = sum(tcg.nodes[x].token_estimate for x in fc_sel if x in tcg.nodes)
            fc_rec = len(fc_sel & gt_neighbors) / len(gt_neighbors)
            records["fixed_chunk_500"].append((fc_rec, fc_tok))

            # 3. BM25
            bm_sel = bm25_rank_nodes(tcg, query_terms, TOKEN_BUDGET)
            bm_tok = sum(tcg.nodes[x].token_estimate for x in bm_sel if x in tcg.nodes)
            bm_rec = len(bm_sel & gt_neighbors) / len(gt_neighbors)
            records["bm25_lexical"].append((bm_rec, bm_tok))

            # 4. PageRank RepoMap
            pr_sel = pagerank_repomap_nodes(tcg, pr, seed_id, TOKEN_BUDGET)
            pr_tok = sum(tcg.nodes[x].token_estimate for x in pr_sel if x in tcg.nodes)
            pr_rec = len(pr_sel & gt_neighbors) / len(gt_neighbors)
            records["pagerank_repomap"].append((pr_rec, pr_tok))

            # 5. TCGS k in {0, 1, 2, 3}
            hop1_ids = {seed_id} | seed_node.dependencies | seed_node.callers
            for k in (0, 1, 2, 3):
                sl = tcg.extract_topological_slice(seed_id, depth=k, token_budget=TOKEN_BUDGET)
                sl_keys = set(sl.keys())
                sl_tok = sum(
                    (n.token_estimate if nid in hop1_ids else n.skeleton_token_estimate)
                    for nid, n in sl.items()
                )
                sl_rec = len(sl_keys & gt_neighbors) / len(gt_neighbors)
                records[f"tcgs_k{k}"].append((sl_rec, sl_tok))

    aggregated: Dict[str, Any] = {}
    full_avg_tokens = sum(t for _, t in records["full_package"]) / max(1, len(records["full_package"]))
    for method, vals in records.items():
        mean_rec = sum(r for r, _ in vals) / max(1, len(vals))
        mean_tok = sum(t for _, t in vals) / max(1, len(vals))
        aggregated[method] = {
            "dependency_recall_pct": round(mean_rec * 100.0, 1),
            "mean_prompt_tokens": int(round(mean_tok)),
            "compression_vs_full_x": round(full_avg_tokens / max(1.0, mean_tok), 2),
            "kv_cache_26b_a4b_mb": estimate_gemma4_kv_cache_mb(int(round(mean_tok)), "26B_A4B"),
        }

    return {
        "total_targets": total_targets,
        "packages": package_summaries,
        "methods": aggregated,
    }


def _simulate_configuration(
    *,
    name: str,
    n_tasks: int,
    seed: int,
    context_recall: float,
    prompt_tokens: int,
    model_capability: float,
    governor_mode: str,  # 'none', 'weak_monotonic', 'strict_dclv'
    contraction_lambda: float = 0.75,
    adaptive_envelope: bool = False,
    use_hash_taboo: bool = True,
    neutral_steps: int = 1,
    max_turns: int = 8,
    max_resamples_per_turn: int = 3,
) -> Dict[str, Any]:
    """
    Executes N=250 seeded stochastic multi-turn defect trajectories through the real
    `DiscreteLyapunovValidator` instance (`harness.py`), explicitly accounting for both
    accepted trajectory tokens AND rejected rollback proposal tokens.
    """
    rng = random.Random(seed)
    resolved = 0
    cyclic_thrash_episodes = 0
    accepted_turns_sum = 0
    accepted_tokens_sum = 0
    total_billed_tokens_sum = 0
    generation_tokens_per_call = 420

    for task_idx in range(n_tasks):
        init_t = rng.randint(2, 5)
        init_u = rng.randint(0, 2)
        base_files = {"mod_a.py": f"base_{task_idx}_0", "mod_b.py": f"base_{task_idx}_1"}

        cur_t, cur_u = init_t, init_u
        cur_files = dict(base_files)
        visited_unconstrained: Set[str] = {
            DiscreteLyapunovValidator.compute_state_hash(cur_files, cur_t, cur_u)
        }
        had_cycle = False

        validator = None
        if governor_mode == "strict_dclv":
            validator = DiscreteLyapunovValidator(
                contraction_factor=contraction_lambda,
                max_turns=max_turns,
                allow_neutral_refactor_steps=neutral_steps,
                baseline_files=base_files,
                adaptive_lattice_envelope=adaptive_envelope,
            )
            validator.register_turn(0, cur_files, cur_t, cur_u)

        accepted_turns = 0
        accepted_tokens = 0
        billed_tokens = 0

        # Effective single-proposal progress probability driven by model tier & context recall SNR
        p_progress = min(0.82, max(0.08, model_capability * (0.32 + 0.68 * context_recall)))
        p_cycle_trap = max(0.06, 0.38 * (1.0 - context_recall))

        patch_pool_hashes = [f"attractor_{task_idx}_A", f"attractor_{task_idx}_B"]

        for turn in range(1, max_turns + 1):
            if cur_t == 0 and cur_u == 0:
                break

            attempts_budget = 1 if governor_mode == "none" else max_resamples_per_turn
            step_committed = False

            for attempt in range(attempts_budget):
                call_cost = prompt_tokens + generation_tokens_per_call
                billed_tokens += call_cost

                # Rejection feedback slightly narrows search space on subsequent resamples
                feedback_boost = 0.07 * attempt if governor_mode == "strict_dclv" else 0.0
                roll = rng.random()

                cand_files = dict(cur_files)
                if roll < (p_progress + feedback_boost):
                    # Constructive edit reducing failures/unresolved symbols
                    dec_t = rng.choice([1, 2]) if cur_t >= 2 else (1 if cur_t == 1 else 0)
                    cand_t = max(0, cur_t - dec_t)
                    cand_u = max(0, cur_u - (1 if cur_u > 0 and rng.random() < 0.75 else 0))
                    cand_files["mod_a.py"] = f"fix_{task_idx}_{turn}_{attempt}_{cand_t}_{cand_u}"
                elif roll < (p_progress + feedback_boost + p_cycle_trap):
                    # Attractor / 2-cycle oscillation (common in small models on noisy prompts)
                    cand_t, cand_u = cur_t, cur_u
                    cand_files["mod_a.py"] = patch_pool_hashes[(turn + attempt) % 2]
                else:
                    # Regression / out-of-scope edit
                    cand_t = cur_t + rng.choice([1, 2])
                    cand_u = cur_u + rng.choice([0, 1])
                    cand_files["mod_b.py"] = f"reg_{task_idx}_{turn}_{attempt}"
                    cand_files[f"extra_{attempt}.py"] = "spurious"

                cand_sig = DiscreteLyapunovValidator.compute_state_hash(cand_files, cand_t, cand_u)

                if governor_mode == "none":
                    if cand_sig in visited_unconstrained and (cand_t > 0 or cand_u > 0):
                        had_cycle = True
                    visited_unconstrained.add(cand_sig)
                    cur_t, cur_u, cur_files = cand_t, cand_u, cand_files
                    accepted_turns += 1
                    accepted_tokens += call_cost
                    step_committed = True
                    break

                elif governor_mode == "weak_monotonic":
                    cur_v = 1.5 * cur_t + 1.0 * cur_u
                    cand_v = 1.5 * cand_t + 1.0 * cand_u
                    if cand_v <= cur_v:
                        if cand_sig in visited_unconstrained and (cand_t > 0 or cand_u > 0):
                            had_cycle = True
                        visited_unconstrained.add(cand_sig)
                        cur_t, cur_u, cur_files = cand_t, cand_u, cand_files
                        accepted_turns += 1
                        accepted_tokens += call_cost
                        step_committed = True
                        break

                elif governor_mode == "strict_dclv":
                    assert validator is not None
                    if not use_hash_taboo:
                        validator.seen_hashes.clear()
                    ok, _ = validator.register_turn(turn, cand_files, cand_t, cand_u)
                    if ok:
                        if cand_sig in visited_unconstrained and (cand_t > 0 or cand_u > 0):
                            had_cycle = True
                        visited_unconstrained.add(cand_sig)
                        cur_t, cur_u, cur_files = cand_t, cand_u, cand_files
                        accepted_turns += 1
                        accepted_tokens += call_cost
                        step_committed = True
                        break

            if not step_committed:
                # Resample budget exhausted without contractive progress -> clean early stop
                break

        if cur_t == 0 and cur_u == 0:
            resolved += 1
        if had_cycle:
            cyclic_thrash_episodes += 1
        accepted_turns_sum += accepted_turns
        accepted_tokens_sum += accepted_tokens
        total_billed_tokens_sum += billed_tokens

    return {
        "configuration": name,
        "resolve_rate_pct": round(100.0 * resolved / n_tasks, 1),
        "mean_accepted_turns": round(accepted_turns_sum / n_tasks, 2),
        "mean_accepted_tokens": int(round(accepted_tokens_sum / n_tasks)),
        "mean_total_billed_tokens": int(round(total_billed_tokens_sum / n_tasks)),
        "cyclic_thrash_rate_pct": round(100.0 * cyclic_thrash_episodes / n_tasks, 1),
    }


def run_all_benchmarks() -> Dict[str, Any]:
    t0 = time.time()
    exp_a = run_experiment_a_stdlib_slicing()

    m = exp_a["methods"]
    rec_fc = m["fixed_chunk_500"]["dependency_recall_pct"] / 100.0
    tok_fc = m["fixed_chunk_500"]["mean_prompt_tokens"]
    rec_bm = m["bm25_lexical"]["dependency_recall_pct"] / 100.0
    tok_bm = m["bm25_lexical"]["mean_prompt_tokens"]
    rec_pr = m["pagerank_repomap"]["dependency_recall_pct"] / 100.0
    tok_pr = m["pagerank_repomap"]["mean_prompt_tokens"]
    rec_k2 = m["tcgs_k2"]["dependency_recall_pct"] / 100.0
    tok_k2 = m["tcgs_k2"]["mean_prompt_tokens"]

    n_tasks = 250
    base_seed = 20261110

    exp_b_rows = [
        _simulate_configuration(
            name="Gemma 4 E4B (Fixed-Chunk ReAct, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 1,
            context_recall=rec_fc,
            prompt_tokens=tok_fc,
            model_capability=0.34,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Gemma 4 E4B + Wayfinder (TCGS k=2 + Adaptive DCLV)",
            n_tasks=n_tasks,
            seed=base_seed + 2,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.34,
            governor_mode="strict_dclv",
            contraction_lambda=0.75,
            adaptive_envelope=True,
        ),
        _simulate_configuration(
            name="Gemma 4 12B (BM25 Lexical, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 3,
            context_recall=rec_bm,
            prompt_tokens=tok_bm,
            model_capability=0.44,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Gemma 4 26B A4B (Fixed-Chunk ReAct, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 4,
            context_recall=rec_fc,
            prompt_tokens=tok_fc,
            model_capability=0.52,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Gemma 4 26B A4B (PageRank RepoMap, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 5,
            context_recall=rec_pr,
            prompt_tokens=tok_pr,
            model_capability=0.52,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Gemma 4 26B A4B (TCGS k=2 Only, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 6,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.52,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Gemma 4 26B A4B (Weak Monotonicity lambda=1.0, No Taboo)",
            n_tasks=n_tasks,
            seed=base_seed + 7,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.52,
            governor_mode="weak_monotonic",
        ),
        _simulate_configuration(
            name="Gemma 4 26B A4B (DCLV lambda=0.75 Only, BM25 Context)",
            n_tasks=n_tasks,
            seed=base_seed + 8,
            context_recall=rec_bm,
            prompt_tokens=tok_bm,
            model_capability=0.52,
            governor_mode="strict_dclv",
            contraction_lambda=0.75,
        ),
        _simulate_configuration(
            name="Gemma 4 26B A4B (Fixed-Lambda DCLV lambda=0.75 + TCGS k=2)",
            n_tasks=n_tasks,
            seed=base_seed + 9,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.52,
            governor_mode="strict_dclv",
            contraction_lambda=0.75,
            adaptive_envelope=False,
        ),
        _simulate_configuration(
            name="Wayfinder-Gemma 4 26B A4B (Full: TCGS k=2 + Adaptive DCLV + Taboo)",
            n_tasks=n_tasks,
            seed=base_seed + 10,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.52,
            governor_mode="strict_dclv",
            contraction_lambda=0.75,
            adaptive_envelope=True,
        ),
    ]

    lambda_sweep = []
    for idx, lam in enumerate((0.50, 0.65, 0.75, 0.85, 0.95)):
        res = _simulate_configuration(
            name=f"lambda={lam:.2f}",
            n_tasks=n_tasks,
            seed=base_seed + 20 + idx,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.52,
            governor_mode="strict_dclv",
            contraction_lambda=lam,
        )
        res["lambda"] = lam
        res["theoretical_max_steps_v8"] = DiscreteLyapunovValidator.theoretical_max_steps(8.0, lam)
        lambda_sweep.append(res)

    elapsed_s = round(time.time() - t0, 3)
    payload: Dict[str, Any] = {
        "benchmark_version": "2.0.0",
        "seed": base_seed,
        "elapsed_seconds": elapsed_s,
        "experiment_a_stdlib_ast_slicing": exp_a,
        "experiment_b_trajectory_ablation": {
            "n_tasks": n_tasks,
            "methodology_note": (
                "Controlled Monte Carlo multi-turn defect trajectory benchmark seeded deterministically "
                "(seed=20261110) parameterized by measured Experiment A AST dependency recall, explicitly "
                "reporting both Accepted Trajectory Tokens and Total Billed Tokens (including rejected rollbacks)."
            ),
            "results": exp_b_rows,
        },
        "experiment_c_lambda_sensitivity": lambda_sweep,
    }
    canonical = json.dumps(
        {k: v for k, v in payload.items() if k != "elapsed_seconds"},
        sort_keys=True,
    ).encode("utf-8")
    payload["sha256_digest"] = hashlib.sha256(canonical).hexdigest()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_telemetry.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    data = run_all_benchmarks()
    print(json.dumps(data, indent=2))
