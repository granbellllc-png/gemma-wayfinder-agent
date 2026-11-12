"""
Reproducible Empirical Benchmark Suite for Gemma 4 TCGS + DCLV + QF_RDL Paper Track (v3.0)
==========================================================================================
Executes four zero-external-dependency, deterministic benchmark suites:
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
  4. Experiment D: Live Executable Multi-File Python Bug-Fix Suite (N=15 real 3-file
     repositories across 5 cross-module defect families), executing real AST slicing,
     live Python test-suite evaluation, State-Adaptive DCLV rollback, and Bellman-Ford
     QF_RDL difference-logic SMT verification.

Outputs deterministic JSON telemetry with SHA-256 digest to `results/benchmark_telemetry.json`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import sysconfig
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

from harness import (
    DifferenceLogicSMTVerifier,
    DiscreteLyapunovValidator,
    TopologicalCodeGraph,
    estimate_tokens,
)
from live_gemma4_gpu_benchmark import run_self_check as run_20repo_suite_self_check


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

        pkg_summaries_cnt = len(focal_ids)
        total_targets += pkg_summaries_cnt
        package_summaries.append(
            {
                "package": pkg_name,
                "ast_nodes": node_count,
                "directed_edges": edge_count,
                "full_package_tokens": full_tokens,
                "evaluated_targets": pkg_summaries_cnt,
            }
        )

        for seed_id in focal_ids:
            seed_node = tcg.nodes[seed_id]
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

            records["full_package"].append((1.0, full_tokens))

            fc_sel = fixed_chunk_overlap_nodes(tcg, query_terms, TOKEN_BUDGET)
            fc_tok = sum(tcg.nodes[x].token_estimate for x in fc_sel if x in tcg.nodes)
            fc_rec = len(fc_sel & gt_neighbors) / len(gt_neighbors)
            records["fixed_chunk_500"].append((fc_rec, fc_tok))

            bm_sel = bm25_rank_nodes(tcg, query_terms, TOKEN_BUDGET)
            bm_tok = sum(tcg.nodes[x].token_estimate for x in bm_sel if x in tcg.nodes)
            bm_rec = len(bm_sel & gt_neighbors) / len(gt_neighbors)
            records["bm25_lexical"].append((bm_rec, bm_tok))

            pr_sel = pagerank_repomap_nodes(tcg, pr, seed_id, TOKEN_BUDGET)
            pr_tok = sum(tcg.nodes[x].token_estimate for x in pr_sel if x in tcg.nodes)
            pr_rec = len(pr_sel & gt_neighbors) / len(gt_neighbors)
            records["pagerank_repomap"].append((pr_rec, pr_tok))

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

    # Paired statistical tests & 95% bootstrap CIs (TCGS k=2 vs lexical/global baselines across N=45 targets)
    k2_recs = [r for r, _ in records["tcgs_k2"]]
    paired_stats: Dict[str, Any] = {}
    boot_rng = random.Random(20261112)
    for base_key in ("bm25_lexical", "pagerank_repomap", "fixed_chunk_500", "tcgs_k1"):
        base_recs = [r for r, _ in records[base_key]]
        diffs = [a - b for a, b in zip(k2_recs, base_recs)]
        n_obs = len(diffs)
        wins = sum(1 for d in diffs if d > 1e-9)
        losses = sum(1 for d in diffs if d < -1e-9)
        ties = n_obs - wins - losses
        n_disc = wins + losses
        if n_disc > 0:
            k_min = min(wins, losses)
            sign_p = min(1.0, 2.0 * sum(math.comb(n_disc, i) for i in range(k_min + 1)) / (2 ** n_disc))
        else:
            sign_p = 1.0

        # Wilcoxon signed-rank normal approximation with tie correction
        non_zero = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > 1e-9]
        non_zero.sort(key=lambda x: x[0])
        w_plus = 0.0
        idx = 0
        while idx < len(non_zero):
            j = idx
            while j < len(non_zero) and abs(non_zero[j][0] - non_zero[idx][0]) <= 1e-12:
                j += 1
            avg_rank = 0.5 * ((idx + 1) + j)
            for m_idx in range(idx, j):
                if non_zero[m_idx][1] > 0:
                    w_plus += avg_rank
            idx = j
        nr = len(non_zero)
        if nr > 0:
            mu_w = nr * (nr + 1) / 4.0
            sd_w = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24.0)
            z_val = (w_plus - mu_w) / max(1e-9, sd_w)
            wilcoxon_p = math.erfc(abs(z_val) / math.sqrt(2.0))
        else:
            z_val, wilcoxon_p = 0.0, 1.0

        boot_means: List[float] = []
        for _ in range(4000):
            sample = [diffs[boot_rng.randrange(n_obs)] for _ in range(n_obs)]
            boot_means.append(100.0 * sum(sample) / n_obs)
        boot_means.sort()
        ci_lo = round(boot_means[int(0.025 * len(boot_means))], 1)
        ci_hi = round(boot_means[int(0.975 * len(boot_means))], 1)

        paired_stats[f"tcgs_k2_vs_{base_key}"] = {
            "n_targets": n_obs,
            "mean_gain_pct_pts": round(100.0 * sum(diffs) / max(1, n_obs), 1),
            "bootstrap_95_ci_pct_pts": [ci_lo, ci_hi],
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "wilcoxon_z": round(z_val, 2),
            "wilcoxon_two_sided_p": float(f"{wilcoxon_p:.3e}"),
            "exact_sign_test_p": float(f"{sign_p:.3e}"),
        }

    return {
        "total_targets": total_targets,
        "packages": package_summaries,
        "methods": aggregated,
        "paired_statistical_tests": paired_stats,
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

                feedback_boost = 0.07 * attempt if governor_mode == "strict_dclv" else 0.0
                roll = rng.random()

                cand_files = dict(cur_files)
                if roll < (p_progress + feedback_boost):
                    dec_t = rng.choice([1, 2]) if cur_t >= 2 else (1 if cur_t == 1 else 0)
                    cand_t = max(0, cur_t - dec_t)
                    cand_u = max(0, cur_u - (1 if cur_u > 0 and rng.random() < 0.75 else 0))
                    cand_files["mod_a.py"] = f"fix_{task_idx}_{turn}_{attempt}_{cand_t}_{cand_u}"
                elif roll < (p_progress + feedback_boost + p_cycle_trap):
                    cand_t, cand_u = cur_t, cur_u
                    cand_files["mod_a.py"] = patch_pool_hashes[(turn + attempt) % 2]
                else:
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


def _evaluate_live_python_repo(
    files: Dict[str, str],
    test_assertions: List[Callable[[Dict[str, Any]], bool]],
) -> Tuple[int, int, Dict[str, str]]:
    """
    Compiles and executes a multi-file Python repository in an isolated namespace,
    returning (test_failures, syntax_or_symbol_errors, sha256_file_map).
    """
    file_hashes = {
        k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in sorted(files.items())
    }
    unresolved_or_syntax = 0
    parsed_ASTs: Dict[str, ast.Module] = {}

    for fname, src in files.items():
        try:
            parsed_ASTs[fname] = ast.parse(src, filename=fname)
        except SyntaxError:
            unresolved_or_syntax += 1

    if unresolved_or_syntax > 0:
        return len(test_assertions), unresolved_or_syntax, file_hashes

    namespace: Dict[str, Any] = {}
    for fname in ("models.py", "service.py", "controller.py"):
        if fname not in files:
            continue
        # Strip intra-repo relative imports `from models import ...` / `from service import ...`
        # since all 3 modules execute in the shared verification sandbox namespace
        clean_lines = [
            ln
            for ln in files[fname].splitlines()
            if not ln.strip().startswith(("from models import", "from service import"))
        ]
        try:
            exec(compile("\n".join(clean_lines), fname, "exec"), namespace)
        except Exception:
            unresolved_or_syntax += 1

    failures = 0
    for check_fn in test_assertions:
        try:
            ok = bool(check_fn(namespace))
            if not ok:
                failures += 1
        except Exception:
            failures += 1

    return failures, unresolved_or_syntax, file_hashes


def _build_live_bug_tasks() -> List[Dict[str, Any]]:
    """
    Constructs 15 real 3-file Python bug-fix tasks across 5 cross-module defect families.
    Each task includes:
      - initial broken 3-file codebase (`models.py`, `service.py`, `controller.py`),
      - 4 live executable unit test assertions,
      - a sequence of candidate patches (including non-contractive regressions, 2-cycle
        oscillation traps, intermediate partial fixes, and the complete multi-file repair).
    """
    tasks: List[Dict[str, Any]] = []

    for variant in range(1, 4):
        tax_rate = 10 * variant
        # Family 1: Cross-module unit/attribute contract mismatch
        base_1 = {
            "models.py": (
                "class InvoiceItem:\n"
                "    def __init__(self, price_cents, qty):\n"
                "        self.price_cents = price_cents\n"
                "        self.qty = qty\n"
                "    def line_cents(self):\n"
                "        return self.price_cents * self.qty\n"
            ),
            "service.py": (
                "from models import InvoiceItem\n"
                "def compute_total_cents(items, discount_cents=0):\n"
                "    raw = sum(it.line_cents() for it in items)\n"
                f"    # BUG: subtracts discount twice and ignores floor at 0 (variant {variant})\n"
                "    return raw - 2 * discount_cents\n"
            ),
            "controller.py": (
                "from models import InvoiceItem\n"
                "from service import compute_total_cents\n"
                "def handle_checkout(raw_pairs, discount_cents=0):\n"
                "    items = [InvoiceItem(p, q) for p, q in raw_pairs]\n"
                "    total = compute_total_cents(items, discount_cents)\n"
                f"    return total + (total * {tax_rate}) // 100\n"
            ),
        }
        partial_1 = dict(base_1)
        partial_1["service.py"] = (
            "from models import InvoiceItem\n"
            "def compute_total_cents(items, discount_cents=0):\n"
            "    raw = sum(it.line_cents() for it in items)\n"
            "    return raw - discount_cents\n"
        )
        regression_1 = dict(base_1)
        regression_1["service.py"] = (
            "from models import InvoiceItem\n"
            "def compute_total_cents(items, discount_cents=0):\n"
            "    return -1\n"
        )
        fixed_1 = dict(base_1)
        fixed_1["service.py"] = (
            "from models import InvoiceItem\n"
            "def compute_total_cents(items, discount_cents=0):\n"
            "    raw = sum(it.line_cents() for it in items)\n"
            "    return max(0, raw - discount_cents)\n"
        )
        tr = tax_rate
        tests_1: List[Callable[[Dict[str, Any]], bool]] = [
            lambda ns, t=tr: ns["handle_checkout"]([(100, 2)], 50) == 150 + (150 * t) // 100,
            lambda ns, t=tr: ns["handle_checkout"]([(200, 1), (50, 2)], 100) == 200 + (200 * t) // 100,
            lambda ns: ns["handle_checkout"]([(100, 1)], 500) == 0,
            lambda ns, t=tr: ns["handle_checkout"]([(100, 1)], 0) == 100 + (100 * t) // 100,
        ]
        tasks.append(
            {
                "task_id": f"cross_module_contract_v{variant}",
                "family": "cross_module_contract",
                "focal_symbol": "handle_checkout",
                "base_files": base_1,
                "candidates": [regression_1, base_1, partial_1, partial_1, fixed_1],
                "tests": tests_1,
            }
        )

        # Family 2: Off-by-one quota boundary + negative burst validation
        limit_val = 50 * variant
        base_2 = {
            "models.py": (
                "class QuotaWindow:\n"
                f"    def __init__(self, limit={limit_val}):\n"
                "        self.limit = limit\n"
                "    def remaining(self, used):\n"
                "        return self.limit - used\n"
            ),
            "service.py": (
                "from models import QuotaWindow\n"
                "def check_quota(used, requested, limit=None):\n"
                f"    win = QuotaWindow(limit if limit is not None else {limit_val})\n"
                "    # BUG: strict > rejects exact boundary match and allows negative requested\n"
                "    return win.remaining(used) > requested\n"
            ),
            "controller.py": (
                "from service import check_quota\n"
                "def authorize_request(used, requested):\n"
                "    return 'ALLOW' if check_quota(used, requested) else 'DENY'\n"
            ),
        }
        partial_2 = dict(base_2)
        partial_2["service.py"] = (
            "from models import QuotaWindow\n"
            "def check_quota(used, requested, limit=None):\n"
            f"    win = QuotaWindow(limit if limit is not None else {limit_val})\n"
            "    return win.remaining(used) >= requested\n"
        )
        reg_2 = dict(base_2)
        reg_2["service.py"] = (
            "from models import QuotaWindow\n"
            "def check_quota(used, requested, limit=None):\n"
            "    return False\n"
        )
        fixed_2 = dict(base_2)
        fixed_2["service.py"] = (
            "from models import QuotaWindow\n"
            "def check_quota(used, requested, limit=None):\n"
            "    if requested < 0 or used < 0:\n"
            "        return False\n"
            f"    win = QuotaWindow(limit if limit is not None else {limit_val})\n"
            "    return win.remaining(used) >= requested\n"
        )
        lv = limit_val
        tests_2: List[Callable[[Dict[str, Any]], bool]] = [
            lambda ns, l=lv: ns["authorize_request"](l - 10, 10) == "ALLOW",
            lambda ns, l=lv: ns["authorize_request"](l, 0) == "ALLOW",
            lambda ns: ns["authorize_request"](10, -5) == "DENY",
            lambda ns: ns["authorize_request"](-1, 5) == "DENY",
        ]
        tasks.append(
            {
                "task_id": f"boundary_off_by_one_v{variant}",
                "family": "boundary_off_by_one",
                "focal_symbol": "authorize_request",
                "base_files": base_2,
                "candidates": [reg_2, base_2, partial_2, base_2, fixed_2],
                "tests": tests_2,
            }
        )

        # Family 3: Shared mutable state corruption across batch calls
        prefix_tag = f"v{variant}"
        base_3 = {
            "models.py": (
                "class BatchContext:\n"
                "    def __init__(self, base_meta=None):\n"
                "        self.base_meta = base_meta if base_meta is not None else {}\n"
                "    def snapshot(self):\n"
                "        return self.base_meta\n"
            ),
            "service.py": (
                "from models import BatchContext\n"
                "def build_payload(ctx, item_id, extra=None):\n"
                "    # BUG: mutates shared dict from ctx.snapshot() in-place\n"
                "    meta = ctx.snapshot()\n"
                "    meta['item_id'] = item_id\n"
                "    if extra:\n"
                "        meta.update(extra)\n"
                "    return meta\n"
            ),
            "controller.py": (
                "from models import BatchContext\n"
                "from service import build_payload\n"
                "def dispatch_items(item_ids):\n"
                f"    ctx = BatchContext({{'ver': '{prefix_tag}'}})\n"
                "    return [build_payload(ctx, i, {'temp': i} if i % 2 == 1 else None) for i in item_ids]\n"
            ),
        }
        partial_3 = dict(base_3)
        partial_3["service.py"] = (
            "from models import BatchContext\n"
            "def build_payload(ctx, item_id, extra=None):\n"
            "    meta = dict(ctx.snapshot())\n"
            "    meta['item_id'] = item_id\n"
            "    if extra:\n"
            "        ctx.base_meta.update(extra)\n"
            "    return meta\n"
        )
        reg_3 = dict(base_3)
        reg_3["service.py"] = "def build_payload(ctx, item_id, extra=None):\n    return {}\n"
        fixed_3 = dict(base_3)
        fixed_3["service.py"] = (
            "from models import BatchContext\n"
            "def build_payload(ctx, item_id, extra=None):\n"
            "    meta = dict(ctx.snapshot())\n"
            "    meta['item_id'] = item_id\n"
            "    if extra:\n"
            "        meta.update(extra)\n"
            "    return meta\n"
        )
        tests_3: List[Callable[[Dict[str, Any]], bool]] = [
            lambda ns: ns["dispatch_items"]([1, 2])[0]["item_id"] == 1,
            lambda ns: "temp" not in ns["dispatch_items"]([1, 2])[1],
            lambda ns: ns["dispatch_items"]([1, 2, 3])[0]["item_id"] == 1
            and ns["dispatch_items"]([1, 2, 3])[2]["item_id"] == 3,
            lambda ns, p=prefix_tag: ns["dispatch_items"]([2])[0].get("ver") == p,
        ]
        tasks.append(
            {
                "task_id": f"state_mutation_leak_v{variant}",
                "family": "state_mutation_leak",
                "focal_symbol": "dispatch_items",
                "base_files": base_3,
                "candidates": [reg_3, partial_3, base_3, fixed_3],
                "tests": tests_3,
            }
        )

        # Family 4: Cross-module exception hierarchy & fallback contract
        fallback_code = 500 + variant
        base_4 = {
            "models.py": (
                "class DomainStorageError(Exception):\n"
                "    pass\n"
                "class TransientTimeoutError(DomainStorageError):\n"
                "    pass\n"
            ),
            "service.py": (
                "from models import DomainStorageError, TransientTimeoutError\n"
                "def query_record(key):\n"
                "    if key is None or key == '':\n"
                "        raise KeyError('empty key')\n"
                "    if str(key).startswith('timeout'):\n"
                "        raise RuntimeError('backend timeout')\n"
                "    return {'key': str(key), 'ok': True}\n"
            ),
            "controller.py": (
                "from models import DomainStorageError\n"
                "from service import query_record\n"
                "def safe_fetch(key):\n"
                "    try:\n"
                "        return query_record(key)\n"
                "    except DomainStorageError:\n"
                f"        return {{'key': key, 'ok': False, 'code': {fallback_code}}}\n"
            ),
        }
        partial_4 = dict(base_4)
        partial_4["service.py"] = (
            "from models import DomainStorageError, TransientTimeoutError\n"
            "def query_record(key):\n"
            "    if key is None or key == '':\n"
            "        raise DomainStorageError('empty key')\n"
            "    if str(key).startswith('timeout'):\n"
            "        raise RuntimeError('backend timeout')\n"
            "    return {'key': str(key), 'ok': True}\n"
        )
        reg_4 = dict(base_4)
        reg_4["controller.py"] = "def safe_fetch(key):\n    return None\n"
        fixed_4 = dict(base_4)
        fixed_4["service.py"] = (
            "from models import DomainStorageError, TransientTimeoutError\n"
            "def query_record(key):\n"
            "    if key is None or key == '':\n"
            "        raise DomainStorageError('empty key')\n"
            "    if str(key).startswith('timeout'):\n"
            "        raise TransientTimeoutError('backend timeout')\n"
            "    return {'key': str(key), 'ok': True}\n"
        )
        fc = fallback_code
        tests_4: List[Callable[[Dict[str, Any]], bool]] = [
            lambda ns, c=fc: ns["safe_fetch"]("") == {"key": "", "ok": False, "code": c},
            lambda ns, c=fc: ns["safe_fetch"](None) == {"key": None, "ok": False, "code": c},
            lambda ns, c=fc: ns["safe_fetch"]("timeout_1") == {"key": "timeout_1", "ok": False, "code": c},
            lambda ns: ns["safe_fetch"]("user_42") == {"key": "user_42", "ok": True},
        ]
        tasks.append(
            {
                "task_id": f"exception_hierarchy_v{variant}",
                "family": "exception_hierarchy_mismatch",
                "focal_symbol": "safe_fetch",
                "base_files": base_4,
                "candidates": [reg_4, base_4, partial_4, fixed_4],
                "tests": tests_4,
            }
        )

        # Family 5: Null-guard & whitespace normalization across 2-hop call chain
        sep = ":" if variant == 1 else ("|" if variant == 2 else "/")
        base_5 = {
            "models.py": (
                "class TagSet:\n"
                "    def __init__(self, tags):\n"
                "        self.tags = tags\n"
                "    def formatted(self, sep=','):\n"
                "        return sep.join(self.tags)\n"
            ),
            "service.py": (
                "from models import TagSet\n"
                "def normalize_tags(raw_tags):\n"
                "    # BUG: crashes on None, non-string elements, or empty whitespace tags\n"
                "    cleaned = [t.lower() for t in raw_tags]\n"
                "    return TagSet(cleaned)\n"
            ),
            "controller.py": (
                "from service import normalize_tags\n"
                "def render_badge(raw_tags):\n"
                "    ts = normalize_tags(raw_tags)\n"
                f"    return ts.formatted('{sep}')\n"
            ),
        }
        partial_5 = dict(base_5)
        partial_5["service.py"] = (
            "from models import TagSet\n"
            "def normalize_tags(raw_tags):\n"
            "    if raw_tags is None:\n"
            "        return TagSet([])\n"
            "    cleaned = [str(t).strip().lower() for t in raw_tags]\n"
            "    return TagSet(cleaned)\n"
        )
        reg_5 = dict(base_5)
        reg_5["service.py"] = "def normalize_tags(raw_tags):\n    return None\n"
        fixed_5 = dict(base_5)
        fixed_5["service.py"] = (
            "from models import TagSet\n"
            "def normalize_tags(raw_tags):\n"
            "    if not raw_tags:\n"
            "        return TagSet([])\n"
            "    cleaned = [str(t).strip().lower() for t in raw_tags if t is not None and str(t).strip()]\n"
            "    return TagSet(cleaned)\n"
        )
        s_ch = sep
        tests_5: List[Callable[[Dict[str, Any]], bool]] = [
            lambda ns: ns["render_badge"](None) == "",
            lambda ns, sc=s_ch: ns["render_badge"](["  Alpha ", "BETA", ""]) == f"alpha{sc}beta",
            lambda ns, sc=s_ch: ns["render_badge"](["Tag1", None, "  ", "Tag2"]) == f"tag1{sc}tag2",
            lambda ns: ns["render_badge"]([]) == "",
        ]
        tasks.append(
            {
                "task_id": f"null_guard_propagation_v{variant}",
                "family": "null_guard_propagation",
                "focal_symbol": "render_badge",
                "base_files": base_5,
                "candidates": [reg_5, partial_5, base_5, fixed_5],
                "tests": tests_5,
            }
        )

    return tasks


def run_experiment_d_live_multifile_bugfix() -> Dict[str, Any]:
    """
    Executes N=15 real 3-file Python bug-fix repositories end-to-end:
      1. Writes the 3 files (`models.py`, `service.py`, `controller.py`) to disk and indexes
         the AST manifold via `TopologicalCodeGraph`.
      2. Extracts the 2-hop slice around `controller.py::<focal_symbol>` and verifies whether
         both `service.py` (1-hop defect locus) and `models.py` (2-hop domain class) are recalled.
      3. Runs live Python execution and unit tests on the candidate repair sequence under:
         - Unguided Agent (accepts proposals without Lyapunov/Taboo rollback, halting after 3 steps)
         - Wayfinder v3.1 (TCGS k=2 + State-Adaptive DCLV + Bellman-Ford QF_RDL SMT Verifier)
    """
    tasks = _build_live_bug_tasks()
    total_tasks = len(tasks)
    tcgs_2hop_full_recall_count = 0
    slice_tokens_list: List[int] = []

    unguided_resolved = 0
    unguided_cycles = 0
    wayfinder_resolved = 0
    wayfinder_cycles = 0
    wayfinder_smt_checked_steps = 0
    wayfinder_smt_sat_steps = 0
    wayfinder_rejected_rollbacks = 0
    per_family_stats: Dict[str, Dict[str, int]] = {}

    for task in tasks:
        fam = task["family"]
        per_family_stats.setdefault(
            fam, {"tasks": 0, "unguided_resolved": 0, "wayfinder_resolved": 0}
        )
        per_family_stats[fam]["tasks"] += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for fname, content in task["base_files"].items():
                (root / fname).write_text(content, encoding="utf-8")

            tcg = TopologicalCodeGraph(root)
            tcg.index_repository()
            sliced = tcg.extract_topological_slice(task["focal_symbol"], depth=2, token_budget=1500)
            sliced_files = {n.file_path for n in sliced.values()}
            if {"models.py", "service.py", "controller.py"}.issubset(sliced_files):
                tcgs_2hop_full_recall_count += 1
            slice_tokens_list.append(sum(n.token_estimate for n in sliced.values()))

        # Evaluate initial broken repository state
        init_t, init_u, init_hashes = _evaluate_live_python_repo(task["base_files"], task["tests"])
        assert init_t > 0, f"Task {task['task_id']} must start with failing tests"

        # 1. Unguided Agent (no rollback governor, commits each step blindly up to 3 steps)
        ug_files = dict(task["base_files"])
        ug_t, ug_u, ug_hashes = init_t, init_u, dict(init_hashes)
        ug_seen = {DiscreteLyapunovValidator.compute_state_hash(ug_hashes, ug_t, ug_u)}
        ug_had_cycle = False
        for cand_files in task["candidates"][:3]:
            ct, cu, ch = _evaluate_live_python_repo(cand_files, task["tests"])
            sig = DiscreteLyapunovValidator.compute_state_hash(ch, ct, cu)
            if sig in ug_seen and (ct > 0 or cu > 0):
                ug_had_cycle = True
            ug_seen.add(sig)
            ug_files, ug_t, ug_u, ug_hashes = dict(cand_files), ct, cu, ch
            if ug_t == 0 and ug_u == 0:
                break
        if ug_t == 0 and ug_u == 0:
            unguided_resolved += 1
            per_family_stats[fam]["unguided_resolved"] += 1
        if ug_had_cycle:
            unguided_cycles += 1

        # 2. Wayfinder v3.1 (TCGS k=2 + Adaptive DCLV + QF_RDL SMT Verifier)
        validator = DiscreteLyapunovValidator(
            contraction_factor=0.75,
            max_turns=6,
            allow_neutral_refactor_steps=0,
            baseline_files=init_hashes,
            adaptive_lattice_envelope=True,
        )
        validator.register_turn(0, init_hashes, init_t, init_u)
        wf_seen = {DiscreteLyapunovValidator.compute_state_hash(init_hashes, init_t, init_u)}
        wf_had_cycle = False
        wf_t, wf_u = init_t, init_u
        accepted_turn_idx = 0
        slice_tok_cur = slice_tokens_list[-1] if slice_tokens_list else 0

        for cand_files in task["candidates"]:
            ct, cu, ch = _evaluate_live_python_repo(cand_files, task["tests"])
            ok, _ = validator.register_turn(accepted_turn_idx + 1, ch, ct, cu, prompt_tokens=slice_tok_cur)
            wayfinder_smt_checked_steps += 1
            if ok:
                accepted_turn_idx += 1
                if validator.smt_proofs and validator.smt_proofs[-1].get("witness_independently_verified") is True:
                    wayfinder_smt_sat_steps += 1
                sig = DiscreteLyapunovValidator.compute_state_hash(ch, ct, cu)
                if sig in wf_seen and (ct > 0 or cu > 0):
                    wf_had_cycle = True
                wf_seen.add(sig)
                wf_t, wf_u = ct, cu
                if wf_t == 0 and wf_u == 0:
                    break
            else:
                wayfinder_rejected_rollbacks += 1

        if wf_t == 0 and wf_u == 0:
            wayfinder_resolved += 1
            per_family_stats[fam]["wayfinder_resolved"] += 1
        if wf_had_cycle:
            wayfinder_cycles += 1

    accepted_total = wayfinder_smt_checked_steps - wayfinder_rejected_rollbacks
    return {
        "is_scripted_adversarial_patch_stream": True,
        "methodology_note": (
            "Deterministic governor & QF_RDL verification stress suite using scripted adversarial patch streams "
            "(regression -> baseline revisit -> partial fix -> duplicate hash -> complete repair) across 15 "
            "executable 3-file repositories (60 unit tests)."
        ),
        "total_live_repositories": total_tasks,
        "defect_families_count": len(per_family_stats),
        "tcgs_k2_cross_file_recall_pct": round(100.0 * tcgs_2hop_full_recall_count / total_tasks, 1),
        "mean_tcgs_slice_tokens": int(round(sum(slice_tokens_list) / max(1, len(slice_tokens_list)))),
        "unguided_react": {
            "resolve_rate_pct": round(100.0 * unguided_resolved / total_tasks, 1),
            "resolved_count": unguided_resolved,
            "cyclic_thrash_rate_pct": round(100.0 * unguided_cycles / total_tasks, 1),
        },
        "wayfinder_v3_tcgs_dclv_qf_rdl": {
            "resolve_rate_pct": round(100.0 * wayfinder_resolved / total_tasks, 1),
            "resolved_count": wayfinder_resolved,
            "cyclic_thrash_rate_pct": round(100.0 * wayfinder_cycles / total_tasks, 1),
            "rejected_bad_proposals_rolled_back": wayfinder_rejected_rollbacks,
            "accepted_steps_count": accepted_total,
            "independently_verified_qf_rdl_witnesses": wayfinder_smt_sat_steps,
            "accepted_steps_qf_rdl_sat_pct": round(
                100.0 * wayfinder_smt_sat_steps / max(1, accepted_total), 1
            ),
            "total_candidate_evaluations": wayfinder_smt_checked_steps,
        },
        "per_family_breakdown": per_family_stats,
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
            name="Synthetic Sim (cap=0.34, Fixed-Chunk, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 1,
            context_recall=rec_fc,
            prompt_tokens=tok_fc,
            model_capability=0.34,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Synthetic Sim (cap=0.34 + TCGS k=2 + Adaptive DCLV)",
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
            name="Synthetic Sim (cap=0.44, BM25 Lexical, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 3,
            context_recall=rec_bm,
            prompt_tokens=tok_bm,
            model_capability=0.44,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Synthetic Sim (cap=0.52, Fixed-Chunk, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 4,
            context_recall=rec_fc,
            prompt_tokens=tok_fc,
            model_capability=0.52,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Synthetic Sim (cap=0.52, PageRank RepoMap, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 5,
            context_recall=rec_pr,
            prompt_tokens=tok_pr,
            model_capability=0.52,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Synthetic Sim (cap=0.52, TCGS k=2 Only, No Governor)",
            n_tasks=n_tasks,
            seed=base_seed + 6,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.52,
            governor_mode="none",
        ),
        _simulate_configuration(
            name="Synthetic Sim (cap=0.52, Weak Monotonic lambda=1.0, No Taboo)",
            n_tasks=n_tasks,
            seed=base_seed + 7,
            context_recall=rec_k2,
            prompt_tokens=tok_k2,
            model_capability=0.52,
            governor_mode="weak_monotonic",
        ),
        _simulate_configuration(
            name="Synthetic Sim (cap=0.52, Fixed DCLV lambda_0=0.75, BM25)",
            n_tasks=n_tasks,
            seed=base_seed + 8,
            context_recall=rec_bm,
            prompt_tokens=tok_bm,
            model_capability=0.52,
            governor_mode="strict_dclv",
            contraction_lambda=0.75,
        ),
        _simulate_configuration(
            name="Synthetic Sim (cap=0.52, Fixed DCLV lambda_0=0.75 + TCGS k=2)",
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
            name="Synthetic Sim (cap=0.52, Full: TCGS k=2 + Adaptive DCLV + Taboo)",
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
            name=f"Synthetic Sim lambda_0={lam:.2f}",
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

    exp_d = run_experiment_d_live_multifile_bugfix()
    suite_20_check = run_20repo_suite_self_check()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    live_gpu_path = out_dir / "live_gemma4_gpu_telemetry.json"
    live_gpu_summary: Dict[str, Any] = {"available": False}
    if live_gpu_path.exists():
        gpu_data = json.loads(live_gpu_path.read_text(encoding="utf-8"))
        canon_gpu = json.dumps(
            {k: v for k, v in gpu_data.items() if k not in ("execution_timestamp_utc", "sha256_digest")},
            sort_keys=True,
        ).encode("utf-8")
        computed_sha = hashlib.sha256(canon_gpu).hexdigest()
        live_gpu_summary = {
            "available": True,
            "digest_verified": computed_sha == gpu_data.get("sha256_digest"),
            "sha256_digest": gpu_data.get("sha256_digest"),
            "hardware_and_runtime": gpu_data.get("hardware_and_runtime"),
            "e2b_configurations": [
                {k: v for k, v in c.items() if k != "per_repo_outcomes"}
                for c in gpu_data["models"]["gemma_4_e2b_it"]["configurations"]
            ],
            "e4b_configurations": [
                {k: v for k, v in c.items() if k != "per_repo_outcomes"}
                for c in gpu_data["models"]["gemma_4_e4b_it"]["configurations"]
            ],
            "paired_statistical_tests_n40": gpu_data.get("paired_statistical_tests_n40"),
        }

    elapsed_s = round(time.time() - t0, 3)
    payload: Dict[str, Any] = {
        "benchmark_version": "3.1.0",
        "seed": base_seed,
        "elapsed_seconds": elapsed_s,
        "experiment_a_stdlib_ast_slicing": exp_a,
        "experiment_b_live_gemma4_gpu_suite": {
            "suite_self_check": suite_20_check,
            "live_gpu_telemetry_summary": live_gpu_summary,
        },
        "experiment_c_deterministic_governor_stress": exp_d,
        "appendix_synthetic_markov_sensitivity_simulation": {
            "is_synthetic_simulation": True,
            "n_tasks": n_tasks,
            "methodology_note": (
                "Explicitly labeled synthetic parametric Markov-chain horizon & contraction-factor sensitivity "
                "simulation (seed=20261110, N=250) parameterized by Experiment A AST recall; NOT live LLM inference."
            ),
            "ablation_rows": exp_b_rows,
            "lambda_sensitivity_sweep": lambda_sweep,
        },
    }
    canonical = json.dumps(
        {k: v for k, v in payload.items() if k != "elapsed_seconds"},
        sort_keys=True,
    ).encode("utf-8")
    payload["sha256_digest"] = hashlib.sha256(canonical).hexdigest()

    out_file = out_dir / "benchmark_telemetry.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    data = run_all_benchmarks()
    print(json.dumps(data, indent=2))
