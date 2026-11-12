"""
Live On-GPU Multi-Turn Repository Repair Benchmark on Official Google Gemma 4 Weights
=====================================================================================
Evaluates 4 SWE Agent Configurations under identical LLM call & context budgets across
N = 20 multi-file Python repositories (7 modules/repo, 5 held-out unit tests/repo = 100 tests)
across 5 software engineering domains using real neural inference on `google/gemma-4`
(`gemma-4-e2b-it` and `gemma-4-e4b-it`):

  1. Okapi BM25 Lexical + Ungoverned ReAct (<=450 slice tokens, max 6 LLM calls)
  2. TCGS (k=2) + Ungoverned ReAct         (<=450 slice tokens, max 6 LLM calls)
  3. TCGS (k=2) + Fixed-Ratio DCLV         (lambda_0 = 0.75, B_neutral = 0, max 6 LLM calls)
  4. Wayfinder Full: TCGS (k=2) + State-Adaptive DCLV + SHA-256 Taboo + QF_RDL SMT (max 6 LLM calls)

Held-out `test_suite.py` is never exposed in retrieved context and cannot be edited by the agent.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from harness import (
    DiscreteLyapunovValidator,
    TopologicalCodeGraph,
    estimate_tokens,
)

SLICE_TOKEN_BUDGET: int = 450
MAX_LLM_CALLS_PER_REPO: int = 6
MAX_NEW_TOKENS: int = 512
EDITABLE_FILES: Tuple[str, ...] = ("controller.py", "service.py")


def _tokenize_lex(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(t) > 1]


def retrieve_bm25_context(
    repo_dir: Path,
    query_text: str,
    token_budget: int = SLICE_TOKEN_BUDGET,
) -> Tuple[str, Set[str], int]:
    """
    Retrieves module/node context using Okapi BM25 over AST symbols across all repository files,
    respecting `token_budget`. Returns (formatted_prompt_context, covered_files, token_estimate).
    """
    tcg = TopologicalCodeGraph(repo_dir)
    tcg.index_repository()
    q_terms = _tokenize_lex(query_text)

    docs: Dict[str, Counter[str]] = {}
    doc_lens: Dict[str, int] = {}
    df: Counter[str] = Counter()
    for nid, node in tcg.nodes.items():
        toks = _tokenize_lex(f"{node.file_path} {node.name} {node.signature} {node.docstring} {node.source_snippet}")
        cnt = Counter(toks)
        docs[nid] = cnt
        doc_lens[nid] = max(1, len(toks))
        for term in cnt:
            df[term] += 1

    n_docs = max(1, len(docs))
    avgdl = sum(doc_lens.values()) / n_docs
    k1, b = 1.5, 0.75
    scored: List[Tuple[float, str]] = []
    for nid, cnt in docs.items():
        dl = doc_lens[nid]
        score = 0.0
        for q in q_terms:
            tf = cnt.get(q, 0)
            if tf == 0:
                continue
            nq = df.get(q, 0)
            idf = math.log(1.0 + (n_docs - nq + 0.5) / (nq + 0.5))
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)))
        scored.append((score, nid))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected_files: List[str] = []
    used = 0
    blocks: List[str] = []
    for sc, nid in scored:
        if sc <= 0:
            break
        fp = tcg.nodes[nid].file_path
        if fp in selected_files:
            continue
        tag = "Editable Full Source" if fp in EDITABLE_FILES else "Repository Module"
        blk = f"# === FILE: {fp} ({tag}) ===\n{(repo_dir / fp).read_text(encoding='utf-8').strip()}"
        cost = estimate_tokens(blk)
        if selected_files and used + cost > token_budget:
            continue
        selected_files.append(fp)
        blocks.append(blk)
        used += cost

    ctx = "\n\n".join(blocks)
    return ctx, set(selected_files), estimate_tokens(ctx)


def retrieve_tcgs_k2_context(
    repo_dir: Path,
    focal_symbol: str,
    token_budget: int = SLICE_TOKEN_BUDGET,
) -> Tuple[str, Set[str], int]:
    """
    Retrieves Hop-Skeletonized TCGS (k=2) slice starting at `focal_symbol`.
    Returns (formatted_prompt_context, covered_files, token_estimate).
    """
    tcg = TopologicalCodeGraph(repo_dir)
    tcg.index_repository()
    ctx, selected = tcg.render_slice_context(
        target_symbol=focal_symbol,
        depth=2,
        token_budget=token_budget,
        skeletonize_distant_hops=True,
    )
    covered_files = {n.file_path for n in selected.values()}
    return ctx, covered_files, estimate_tokens(ctx)


def _make_distractors(
    domain_kw: Tuple[str, str, str, str],
    wrong_const_line: str,
    issue_vocab: str = "",
) -> Dict[str, str]:
    k1, k2, k3, k4 = domain_kw
    return {
        "legacy_v1.py": (
            f'"""Deprecated v1 {k1} {k2} compatibility helpers ({issue_vocab})."""\n'
            f"{wrong_const_line}\n\n"
            f"def legacy_{k1}_{k2}(payload: dict) -> dict:\n"
            f'    """Legacy {k1} adapter for {k2} {k3} payloads and {k4} normalization."""\n'
            f"    out = dict(payload)\n"
            f"    out['legacy_{k4}'] = True\n"
            f"    out['{k2}_mode'] = 'v1_compat'\n"
            f"    return out\n\n"
            f"def format_legacy_{k3}(record: dict) -> str:\n"
            f'    """Formats {k1} {k2} {k3} summary text."""\n'
            f"    return f\"{k1}:{{record.get('{k2}', 'na')}}:{k4}\"\n"
        ),
        "audit_log.py": (
            f'"""Structured audit trail recorder for {k1} {k2} {k3} operations."""\n\n'
            f"class AuditRecorder:\n"
            f'    """Records {k1} {k2} and {k3} events for {k4} compliance."""\n'
            f"    def __init__(self) -> None:\n"
            f"        self.events: list[dict] = []\n\n"
            f"    def record_{k1}_event(self, actor: str, {k2}_id: str, detail: str) -> int:\n"
            f'        """Appends {k1} {k2} {k3} audit event and returns count."""\n'
            f"        self.events.append({{'actor': actor.strip(), '{k2}': {k2}_id, '{k4}': detail}})\n"
            f"        return len(self.events)\n"
        ),
        "metrics_exporter.py": (
            f'"""Telemetry counter aggregation for {k1} {k2} {k3} {k4}."""\n\n'
            f"def build_{k1}_metrics(total: int, rejected: int) -> dict:\n"
            f'    """Computes {k1} {k2} rejection ratio for {k3} {k4}."""\n'
            f"    denom = max(1, total)\n"
            f"    return {{'{k1}_total': total, '{k2}_rejected': rejected, '{k3}_rate': round(rejected / denom, 4)}}\n"
        ),
        "admin_cli.py": (
            f'"""Operator CLI diagnostic helper for {k1} {k2} {k3} inspection."""\n\n'
            f"def render_{k1}_banner(env: str, {k2}_label: str) -> str:\n"
            f'    """Renders operator {k1} {k2} {k3} {k4} status header."""\n'
            f"    return f'[{{env.upper()}}] {k1}-{k2}-{k3}: {{{k2}_label}}'\n"
        ),
    }


def build_20_repo_specs() -> List[Dict[str, Any]]:
    """
    Constructs 20 realistic 7-file Python repositories across 5 engineering domains:
      - 12 coupled multi-file integration repositories (T_0 = 5 -> T_1 = 4 -> T_2 = 0)
      - 8 decoupled multi-assertion repositories       (T_0 = 4 or 5 -> T_1 <= 2 -> T_2 = 0)
    """
    specs: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # DOMAIN 1: Billing & Ledger Settlement (BL-01 .. BL-04)
    # -------------------------------------------------------------------------
    bl_configs = [
        ("BL-01", 2500, 800, "EXEMPT", True),
        ("BL-02", 2000, 1000, "NONPROFIT", True),
        ("BL-03", 3000, 600, "GOV", True),
        ("BL-04", 1500, 750, "EDU", False),
    ]
    for rid, cap_bps, std_tax_bps, exempt_tier, coupled in bl_configs:
        distractors = _make_distractors(
            ("invoice", "discount", "tax", "subtotal"),
            f"LEGACY_DISCOUNT_CAP_BPS = {cap_bps * 2}  # Obsolete v1 cap, do not use",
            "discounted taxed totals items list tier strings",
        )
        contracts_py = (
            '"""Domain contract specifications for invoice discount and tax basis points."""\n\n'
            "class BillingContract:\n"
            f'    """[service.py] settle_subtotal: clamp discount via clamp_discount_bps (max {cap_bps}, <0 ValueError) and use // 10000. '
            f"[controller.py] generate_invoice: copy items (do not mutate), empty or <0 item raises ValueError, "
            f"normalize tier via strip().upper() ('{exempt_tier}'->0, 'STANDARD'->{std_tax_bps}).\"\"\"\n\n"
            "    @staticmethod\n"
            "    def clamp_discount_bps(discount_bps: int) -> int:\n"
            f'        """Raises ValueError if discount_bps < 0; returns min(discount_bps, {cap_bps})."""\n'
            "        if discount_bps < 0:\n"
            "            raise ValueError('negative discount')\n"
            f"        return min(discount_bps, {cap_bps})\n\n"
            "    @staticmethod\n"
            "    def tax_rate_bps(tier: str) -> int:\n"
            f'        """Returns 0 for {exempt_tier!r}, {std_tax_bps} for \'STANDARD\', else raises ValueError."""\n'
            f"        if tier == {exempt_tier!r}:\n"
            "            return 0\n"
            "        if tier == 'STANDARD':\n"
            f"            return {std_tax_bps}\n"
            "        raise ValueError(f'unknown tier: {tier}')\n"
        )
        service_buggy = (
            "from contracts import BillingContract\n\n"
            "def settle_subtotal(gross_cents: int, discount_bps: int, tier: str) -> int:\n"
            '    """Computes net invoice cents after discount and tier tax."""\n'
            "    disc = (gross_cents * discount_bps) // 1000\n"
            "    net = gross_cents - disc\n"
            "    rate = BillingContract.tax_rate_bps(tier)\n"
            "    return net + (net * rate) // 10000\n"
        )
        service_fixed = (
            "from contracts import BillingContract\n\n"
            "def settle_subtotal(gross_cents: int, discount_bps: int, tier: str) -> int:\n"
            '    """Computes net invoice cents after discount and tier tax."""\n'
            "    capped = BillingContract.clamp_discount_bps(discount_bps)\n"
            "    disc = (gross_cents * capped) // 10000\n"
            "    net = gross_cents - disc\n"
            "    rate = BillingContract.tax_rate_bps(tier)\n"
            "    return net + (net * rate) // 10000\n"
        )
        controller_buggy = (
            "from service import settle_subtotal\n\n"
            "def generate_invoice(order: dict) -> dict:\n"
            '    """Builds settled invoice summary from incoming order payload."""\n'
            "    items = order['items']\n"
            "    items.sort()\n"
            "    tier = order['tier']\n"
            "    total = settle_subtotal(sum(items), order.get('discount_bps', 0), tier)\n"
            "    return {'tier': tier, 'total_cents': total}\n"
        )
        controller_fixed = (
            "from service import settle_subtotal\n\n"
            "def generate_invoice(order: dict) -> dict:\n"
            '    """Validates order items (non-empty, non-negative ints), normalizes tier, without mutating order."""\n'
            "    items = list(order['items'])\n"
            "    if not items or any(x < 0 for x in items):\n"
            "        raise ValueError('invalid items')\n"
            "    tier = order['tier'].strip().upper()\n"
            "    total = settle_subtotal(sum(items), order.get('discount_bps', 0), tier)\n"
            "    return {'tier': tier, 'total_cents': total}\n"
        )
        if coupled:
            test_py = (
                "import unittest\n"
                "from controller import generate_invoice\n"
                "from service import settle_subtotal\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_discount_clamp(self):\n"
                f"        expected = 10000 - (10000 * {cap_bps}) // 10000\n"
                f"        self.assertEqual(settle_subtotal(10000, {cap_bps + 2000}, {exempt_tier!r}), expected)\n\n"
                "    def test_2_controller_rejects_negative_item(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            generate_invoice({'items': [500, -10], 'tier': 'STANDARD', 'discount_bps': 0})\n\n"
                "    def test_3_e2e_normalized_exempt_with_cap(self):\n"
                "        src = [3000, 1000, 6000]\n"
                f"        res = generate_invoice({{'items': src, 'tier': '  {exempt_tier.lower()} ', 'discount_bps': {cap_bps + 1500}}})\n"
                "        self.assertEqual(src, [3000, 1000, 6000])\n"
                f"        self.assertEqual(res['total_cents'], 10000 - (10000 * {cap_bps}) // 10000)\n\n"
                "    def test_4_e2e_standard_tier_tax_and_discount(self):\n"
                "        src = [8000, 2000]\n"
                "        res = generate_invoice({'items': src, 'tier': ' standard ', 'discount_bps': 1000})\n"
                "        self.assertEqual(src, [8000, 2000])\n"
                f"        self.assertEqual(res['total_cents'], 9000 + (9000 * {std_tax_bps}) // 10000)\n\n"
                "    def test_5_e2e_negative_discount_rejected(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            generate_invoice({'items': [2000, 1000], 'tier': 'STANDARD', 'discount_bps': -50})\n"
            )
        else:
            test_py = (
                "import unittest\n"
                "from controller import generate_invoice\n"
                "from service import settle_subtotal\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_clamp(self):\n"
                f"        self.assertEqual(settle_subtotal(10000, 9000, {exempt_tier!r}), 10000 - {cap_bps})\n\n"
                "    def test_2_service_negative_discount(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            settle_subtotal(5000, -10, 'STANDARD')\n\n"
                "    def test_3_service_bps_math(self):\n"
                f"        self.assertEqual(settle_subtotal(10000, 500, 'STANDARD'), 9500 + (9500 * {std_tax_bps}) // 10000)\n\n"
                "    def test_4_controller_empty_or_negative(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            generate_invoice({'items': [], 'tier': 'STANDARD', 'discount_bps': 0})\n\n"
                "    def test_5_e2e_immutability_and_tier_norm(self):\n"
                "        items = [4000, 1000]\n"
                f"        out = generate_invoice({{'items': items, 'tier': ' {exempt_tier.lower()} ', 'discount_bps': 1000}})\n"
                "        self.assertEqual(items, [4000, 1000])\n"
                "        self.assertEqual(out['total_cents'], 4500)\n"
            )
        specs.append(
            {
                "repo_id": rid,
                "domain": "billing_ledger",
                "coupled_integration": coupled,
                "focal_symbol": "generate_invoice",
                "issue": (
                    "Invoice generation produces wrong discounted/taxed totals, mutates the caller's items list, "
                    "and fails on lowercase/padded tier strings or invalid negative/empty inputs."
                ),
                "files": {
                    "contracts.py": contracts_py,
                    "service.py": service_buggy,
                    "controller.py": controller_buggy,
                    **distractors,
                },
                "ground_truth_fixes": {"service.py": service_fixed, "controller.py": controller_fixed},
                "test_suite": test_py,
            }
        )

    # -------------------------------------------------------------------------
    # DOMAIN 2: API Token-Bucket Rate Limiter (RL-01 .. RL-04)
    # -------------------------------------------------------------------------
    rl_configs = [
        ("RL-01", 2, 100, True),
        ("RL-02", 3, 150, True),
        ("RL-03", 2, 250, False),
        ("RL-04", 4, 80, False),
    ]
    for rid, burst_mult, base_quota, coupled in rl_configs:
        distractors = _make_distractors(
            ("quota", "burst", "bucket", "throttle"),
            f"LEGACY_BURST_FACTOR = {burst_mult + 3}  # Obsolete v1 burst multiplier",
            "token accumulation burst capacity exact-balance zero cost client state",
        )
        contracts_py = (
            '"""Rate-limit token bucket contract specifications."""\n\n'
            "class QuotaContract:\n"
            f'    """[service.py] evaluate_bucket: call validate_cost(cost) (<=0 ValueError), clamp avail = min(max_capacity(), current_tokens + max(0, refill)), allow iff cost <= avail. '
            f'[controller.py] handle_throttle_check: normalize cid = client_id.strip().lower() (empty ValueError), copy state dict (default {base_quota})."""\n\n'
            "    @staticmethod\n"
            "    def max_capacity() -> int:\n"
            f'        """Returns maximum burst token capacity = {base_quota * burst_mult}."""\n'
            f"        return {base_quota * burst_mult}\n\n"
            "    @staticmethod\n"
            "    def validate_cost(cost: int) -> int:\n"
            '        """Raises ValueError if cost <= 0, otherwise returns cost."""\n'
            "        if cost <= 0:\n"
            "            raise ValueError('cost must be positive')\n"
            "        return cost\n"
        )
        service_buggy = (
            "from contracts import QuotaContract\n\n"
            "def evaluate_bucket(current_tokens: int, refill: int, cost: int) -> tuple[bool, int]:\n"
            '    """Updates bucket balance and evaluates request admission."""\n'
            "    _cap = QuotaContract.max_capacity()\n"
            "    avail = current_tokens + max(0, refill)\n"
            "    if cost < avail:\n"
            "        return True, avail - cost\n"
            "    return False, min(_cap, avail)\n"
        )
        service_fixed = (
            "from contracts import QuotaContract\n\n"
            "def evaluate_bucket(current_tokens: int, refill: int, cost: int) -> tuple[bool, int]:\n"
            '    """Updates bucket balance and evaluates request admission."""\n'
            "    QuotaContract.validate_cost(cost)\n"
            "    avail = min(QuotaContract.max_capacity(), current_tokens + max(0, refill))\n"
            "    if cost <= avail:\n"
            "        return True, avail - cost\n"
            "    return False, avail\n"
        )
        controller_buggy = (
            "from service import evaluate_bucket\n\n"
            "def handle_throttle_check(state: dict, client_id: str, cost: int, refill: int = 0) -> dict:\n"
            '    """Evaluates rate-limit admission for client_id against state."""\n'
            f"    cur = state.get(client_id, {base_quota})\n"
            "    allowed, rem = evaluate_bucket(cur, refill, cost)\n"
            "    state[client_id] = rem\n"
            "    return {'client_id': client_id, 'allowed': allowed, 'remaining': rem, 'state': state}\n"
        )
        controller_fixed = (
            "from service import evaluate_bucket\n\n"
            "def handle_throttle_check(state: dict, client_id: str, cost: int, refill: int = 0) -> dict:\n"
            '    """Normalizes client_id (strip/lower, non-empty) and returns updated copy without mutating state."""\n'
            "    cid = client_id.strip().lower()\n"
            "    if not cid:\n"
            "        raise ValueError('empty client_id')\n"
            "    new_state = dict(state)\n"
            f"    cur = new_state.get(cid, {base_quota})\n"
            "    allowed, rem = evaluate_bucket(cur, refill, cost)\n"
            "    new_state[cid] = rem\n"
            "    return {'client_id': cid, 'allowed': allowed, 'remaining': rem, 'state': new_state}\n"
        )
        max_cap = base_quota * burst_mult
        if coupled:
            test_py = (
                "import unittest\n"
                "from controller import handle_throttle_check\n"
                "from service import evaluate_bucket\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_exact_boundary_and_clamp(self):\n"
                f"        ok, rem = evaluate_bucket({base_quota}, {max_cap}, {max_cap})\n"
                "        self.assertTrue(ok)\n"
                "        self.assertEqual(rem, 0)\n\n"
                "    def test_2_controller_rejects_blank_client(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            handle_throttle_check({}, '   ', 10, 0)\n\n"
                "    def test_3_e2e_burst_clamp_and_immutable_state(self):\n"
                f"        orig = {{'alice': {base_quota}}}\n"
                f"        res = handle_throttle_check(orig, '  ALICE ', {max_cap}, refill={max_cap * 2})\n"
                f"        self.assertEqual(orig, {{'alice': {base_quota}}})\n"
                "        self.assertTrue(res['allowed'])\n"
                "        self.assertEqual(res['remaining'], 0)\n\n"
                "    def test_4_e2e_over_cap_rejected(self):\n"
                f"        orig = {{'bob': {base_quota}}}\n"
                f"        res = handle_throttle_check(orig, ' Bob ', {max_cap + 1}, refill={max_cap * 2})\n"
                f"        self.assertEqual(orig, {{'bob': {base_quota}}})\n"
                "        self.assertFalse(res['allowed'])\n"
                f"        self.assertEqual(res['remaining'], {max_cap})\n\n"
                "    def test_5_e2e_zero_cost_raises(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            handle_throttle_check({}, ' ServiceA ', 0, refill=10)\n"
            )
        else:
            test_py = (
                "import unittest\n"
                "from controller import handle_throttle_check\n"
                "from service import evaluate_bucket\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_boundary(self):\n"
                "        self.assertEqual(evaluate_bucket(50, 0, 50), (True, 0))\n\n"
                "    def test_2_service_cap_clamp(self):\n"
                f"        self.assertEqual(evaluate_bucket({base_quota}, 9999, 10), (True, {max_cap - 10}))\n\n"
                "    def test_3_service_nonpositive_cost(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            evaluate_bucket(50, 0, -5)\n\n"
                "    def test_4_controller_blank_client(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            handle_throttle_check({}, '  ', 5, 0)\n\n"
                "    def test_5_e2e_state_copy_and_case_norm(self):\n"
                f"        st = {{'api_user': {base_quota}}}\n"
                f"        res = handle_throttle_check(st, ' API_USER ', {base_quota}, refill=0)\n"
                f"        self.assertEqual(st, {{'api_user': {base_quota}}})\n"
                "        self.assertTrue(res['allowed'])\n"
                "        self.assertEqual(res['remaining'], 0)\n"
            )
        specs.append(
            {
                "repo_id": rid,
                "domain": "rate_limiter",
                "coupled_integration": coupled,
                "focal_symbol": "handle_throttle_check",
                "issue": (
                    "Throttle checks allow unbounded token accumulation above max burst capacity, reject exact-balance "
                    "requests, accept zero/negative costs, mutate input state dicts, and fail on whitespace/uppercase IDs."
                ),
                "files": {
                    "contracts.py": contracts_py,
                    "service.py": service_buggy,
                    "controller.py": controller_buggy,
                    **distractors,
                },
                "ground_truth_fixes": {"service.py": service_fixed, "controller.py": controller_fixed},
                "test_suite": test_py,
            }
        )

    # -------------------------------------------------------------------------
    # DOMAIN 3: Session & Role Authorization Guard (AU-01 .. AU-04)
    # -------------------------------------------------------------------------
    au_configs = [
        ("AU-01", 3600, True),
        ("AU-02", 1800, True),
        ("AU-03", 7200, True),
        ("AU-04", 900, False),
    ]
    for rid, max_ttl, coupled in au_configs:
        distractors = _make_distractors(
            ("session", "token", "role", "scope"),
            f"LEGACY_SESSION_TTL = {max_ttl * 4}  # Obsolete v1 TTL window",
            "matching-rank roles scope overlap session TTL timestamps revoked",
        )
        contracts_py = (
            '"""Authentication session TTL and role-hierarchy contract."""\n\n'
            "class SessionContract:\n"
            f'    """[service.py] verify_access: require is_age_valid(issued_at, now) (0 <= age <= {max_ttl}), role_rank(user) >= role_rank(required), and needed.issubset(scopes). '
            '[controller.py] authorize_request: strip().upper() roles, strip().lower() scopes, session.get(\'revoked\') -> authorized=False."""\n\n'
            "    @staticmethod\n"
            "    def is_age_valid(issued_at: int, now: int) -> bool:\n"
            f'        """Returns True iff 0 <= (now - issued_at) <= {max_ttl}."""\n'
            "        age = now - issued_at\n"
            f"        return 0 <= age <= {max_ttl}\n\n"
            "    @staticmethod\n"
            "    def role_rank(role: str) -> int:\n"
            '        """Returns 1 for VIEWER, 2 for EDITOR, 3 for ADMIN; else raises ValueError."""\n'
            "        ranks = {'VIEWER': 1, 'EDITOR': 2, 'ADMIN': 3}\n"
            "        if role not in ranks:\n"
            "            raise ValueError(f'unknown role: {role}')\n"
            "        return ranks[role]\n"
        )
        service_buggy = (
            "from contracts import SessionContract\n\n"
            "def verify_access(issued_at: int, now: int, user_role: str, required_role: str, scopes: set[str], needed: set[str]) -> bool:\n"
            '    """Evaluates session timestamp, role hierarchy, and requested permission scopes."""\n'
            "    if (now - issued_at) > 999999:\n"
            "        return False\n"
            "    if SessionContract.role_rank(user_role) <= SessionContract.role_rank(required_role):\n"
            "        return False\n"
            "    return bool(scopes & needed) if needed else True\n"
        )
        service_fixed = (
            "from contracts import SessionContract\n\n"
            "def verify_access(issued_at: int, now: int, user_role: str, required_role: str, scopes: set[str], needed: set[str]) -> bool:\n"
            '    """Evaluates session timestamp, role hierarchy, and requested permission scopes."""\n'
            "    if not SessionContract.is_age_valid(issued_at, now):\n"
            "        return False\n"
            "    if SessionContract.role_rank(user_role) < SessionContract.role_rank(required_role):\n"
            "        return False\n"
            "    return needed.issubset(scopes)\n"
        )
        controller_buggy = (
            "from service import verify_access\n\n"
            "def authorize_request(session: dict, required_role: str, required_scopes: list[str], now: int) -> dict:\n"
            '    """Checks incoming session authorization decision."""\n'
            "    u_role = session['role']\n"
            "    req_role = required_role\n"
            "    ok = verify_access(session['issued_at'], now, u_role, req_role, set(session['scopes']), set(required_scopes))\n"
            "    return {'authorized': ok, 'role': u_role}\n"
        )
        controller_fixed = (
            "from service import verify_access\n\n"
            "def authorize_request(session: dict, required_role: str, required_scopes: list[str], now: int) -> dict:\n"
            '    """Rejects revoked sessions, normalizes roles/scopes (strip), without mutating session."""\n'
            "    u_role = session['role'].strip().upper()\n"
            "    req_role = required_role.strip().upper()\n"
            "    if session.get('revoked', False):\n"
            "        SessionContract_check = (u_role, req_role)\n"
            "        return {'authorized': False, 'role': u_role}\n"
            "    u_scopes = {s.strip().lower() for s in session.get('scopes', []) if s.strip()}\n"
            "    n_scopes = {s.strip().lower() for s in required_scopes if s.strip()}\n"
            "    ok = verify_access(session['issued_at'], now, u_role, req_role, u_scopes, n_scopes)\n"
            "    return {'authorized': ok, 'role': u_role}\n"
        )
        if coupled:
            test_py = (
                "import unittest\n"
                "from controller import authorize_request\n"
                "from service import verify_access\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_ttl_and_equal_role(self):\n"
                f"        self.assertTrue(verify_access(1000, {1000 + max_ttl}, 'EDITOR', 'EDITOR', {{'read', 'write'}}, {{'read', 'write'}}))\n"
                f"        self.assertFalse(verify_access(1000, {1000 + max_ttl + 1}, 'ADMIN', 'VIEWER', {{'read'}}, {{'read'}}))\n\n"
                "    def test_2_controller_revoked_blocked(self):\n"
                "        s = {'issued_at': 1000, 'role': 'ADMIN', 'scopes': ['read'], 'revoked': True}\n"
                "        self.assertFalse(authorize_request(s, 'VIEWER', ['read'], 1010)['authorized'])\n\n"
                "    def test_3_e2e_normalized_equal_role_and_full_subset(self):\n"
                "        s = {'issued_at': 1000, 'role': ' editor ', 'scopes': [' Read ', ' Write '], 'revoked': False}\n"
                f"        res = authorize_request(s, ' editor ', ['read', 'write'], {1000 + max_ttl})\n"
                "        self.assertTrue(res['authorized'])\n\n"
                "    def test_4_e2e_partial_scope_denied(self):\n"
                "        s = {'issued_at': 1000, 'role': ' admin ', 'scopes': ['read'], 'revoked': False}\n"
                "        res = authorize_request(s, ' editor ', ['read', 'delete'], 1050)\n"
                "        self.assertFalse(res['authorized'])\n\n"
                "    def test_5_e2e_future_token_denied(self):\n"
                "        s = {'issued_at': 2000, 'role': ' admin ', 'scopes': ['read'], 'revoked': False}\n"
                "        res = authorize_request(s, ' viewer ', ['read'], 1999)\n"
                "        self.assertFalse(res['authorized'])\n"
            )
        else:
            test_py = (
                "import unittest\n"
                "from controller import authorize_request\n"
                "from service import verify_access\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_equal_role(self):\n"
                "        self.assertTrue(verify_access(100, 200, 'EDITOR', 'EDITOR', {'a', 'b'}, {'a'}))\n\n"
                "    def test_2_service_subset_check(self):\n"
                "        self.assertFalse(verify_access(100, 200, 'ADMIN', 'VIEWER', {'a'}, {'a', 'b'}))\n\n"
                "    def test_3_service_ttl_bounds(self):\n"
                f"        self.assertFalse(verify_access(100, {100 + max_ttl + 5}, 'ADMIN', 'VIEWER', {{'a'}}, {{'a'}}))\n\n"
                "    def test_4_controller_revoked(self):\n"
                "        s = {'issued_at': 100, 'role': 'ADMIN', 'scopes': ['a'], 'revoked': True}\n"
                "        self.assertFalse(authorize_request(s, 'VIEWER', ['a'], 120)['authorized'])\n\n"
                "    def test_5_e2e_case_normalized(self):\n"
                "        s = {'issued_at': 100, 'role': ' editor ', 'scopes': [' Read '], 'revoked': False}\n"
                "        self.assertTrue(authorize_request(s, ' EDITOR ', ['read'], 150)['authorized'])\n"
            )
        specs.append(
            {
                "repo_id": rid,
                "domain": "auth_session",
                "coupled_integration": coupled,
                "focal_symbol": "authorize_request",
                "issue": (
                    "Session authorization fails for matching-rank roles, permits partial scope overlap instead of "
                    "requiring all requested scopes, ignores TTL/future timestamps and revoked flags, and fails on case/whitespace."
                ),
                "files": {
                    "contracts.py": contracts_py,
                    "service.py": service_buggy,
                    "controller.py": controller_buggy,
                    **distractors,
                },
                "ground_truth_fixes": {"service.py": service_fixed, "controller.py": controller_fixed},
                "test_suite": test_py,
            }
        )

    # -------------------------------------------------------------------------
    # DOMAIN 4: Warehouse Inventory Reservation (IV-01 .. IV-04)
    # -------------------------------------------------------------------------
    iv_configs = [
        ("IV-01", 5, True),
        ("IV-02", 10, True),
        ("IV-03", 3, False),
        ("IV-04", 8, False),
    ]
    for rid, safety_floor, coupled in iv_configs:
        distractors = _make_distractors(
            ("stock", "sku", "reserve", "warehouse"),
            f"LEGACY_SAFETY_FLOOR = 0  # Obsolete v1 zero-floor policy (replaced by {safety_floor})",
            "safety-stock floor zero negative quantities inventory dict padded SKUs",
        )
        contracts_py = (
            '"""Warehouse safety-stock reservation contract."""\n\n'
            "class StockContract:\n"
            f'    """[service.py] try_reserve_sku: call validate_qty(qty) (<=0 ValueError); reserve iff (on_hand - qty) >= safety_floor() ({safety_floor}). '
            '[controller.py] process_reservation: norm_sku = sku.strip().upper(), missing SKU in inventory raises KeyError, copy inventory dict (update only if reserved)."""\n\n'
            "    @staticmethod\n"
            "    def safety_floor() -> int:\n"
            f'        """Minimum remaining units required after reservation = {safety_floor}."""\n'
            f"        return {safety_floor}\n\n"
            "    @staticmethod\n"
            "    def validate_qty(qty: int) -> int:\n"
            '        """Raises ValueError if qty <= 0; else returns qty."""\n'
            "        if qty <= 0:\n"
            "            raise ValueError('qty must be > 0')\n"
            "        return qty\n"
        )
        service_buggy = (
            "from contracts import StockContract\n\n"
            "def try_reserve_sku(on_hand: int, qty: int) -> tuple[bool, int]:\n"
            '    """Attempts to deduct qty units from available SKU stock."""\n'
            "    floor = StockContract.safety_floor()\n"
            "    if on_hand >= floor and on_hand - qty >= 0:\n"
            "        return True, on_hand - qty\n"
            "    return False, on_hand\n"
        )
        service_fixed = (
            "from contracts import StockContract\n\n"
            "def try_reserve_sku(on_hand: int, qty: int) -> tuple[bool, int]:\n"
            '    """Attempts to deduct qty units from available SKU stock."""\n'
            "    StockContract.validate_qty(qty)\n"
            "    if on_hand - qty >= StockContract.safety_floor():\n"
            "        return True, on_hand - qty\n"
            "    return False, on_hand\n"
        )
        controller_buggy = (
            "from service import try_reserve_sku\n\n"
            "def process_reservation(inventory: dict[str, int], sku: str, qty: int) -> dict:\n"
            '    """Processes warehouse SKU reservation request."""\n'
            "    ok, rem = try_reserve_sku(inventory[sku], qty)\n"
            "    inventory[sku] = rem\n"
            "    return {'sku': sku, 'reserved': ok, 'remaining': rem, 'inventory': inventory}\n"
        )
        controller_fixed = (
            "from service import try_reserve_sku\n\n"
            "def process_reservation(inventory: dict[str, int], sku: str, qty: int) -> dict:\n"
            '    """Normalizes sku (strip/upper), checks existence without mutating caller inventory dict."""\n'
            "    norm_sku = sku.strip().upper()\n"
            "    if not norm_sku or norm_sku not in inventory:\n"
            "        raise KeyError(f'unknown sku: {norm_sku}')\n"
            "    snap = dict(inventory)\n"
            "    ok, rem = try_reserve_sku(snap[norm_sku], qty)\n"
            "    if ok:\n"
            "        snap[norm_sku] = rem\n"
            "    return {'sku': norm_sku, 'reserved': ok, 'remaining': rem, 'inventory': snap}\n"
        )
        if coupled:
            test_py = (
                "import unittest\n"
                "from controller import process_reservation\n"
                "from service import try_reserve_sku\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_safety_floor_enforced(self):\n"
                f"        self.assertEqual(try_reserve_sku({safety_floor + 4}, 5), (False, {safety_floor + 4}))\n"
                f"        self.assertEqual(try_reserve_sku({safety_floor + 5}, 5), (True, {safety_floor}))\n\n"
                "    def test_2_controller_unknown_sku_raises_keyerror(self):\n"
                "        with self.assertRaises(KeyError):\n"
                "            process_reservation({'SKU-1': 50}, ' sku-missing ', 2)\n\n"
                "    def test_3_e2e_normalized_sku_and_immutability(self):\n"
                f"        inv = {{'SKU-A': {safety_floor + 10}}}\n"
                "        res = process_reservation(inv, ' sku-a ', 10)\n"
                f"        self.assertEqual(inv, {{'SKU-A': {safety_floor + 10}}})\n"
                "        self.assertTrue(res['reserved'])\n"
                f"        self.assertEqual(res['remaining'], {safety_floor})\n\n"
                "    def test_4_e2e_safety_floor_blocks_overdraw(self):\n"
                f"        inv = {{'SKU-B': {safety_floor + 3}}}\n"
                "        res = process_reservation(inv, ' sku-b ', 4)\n"
                f"        self.assertEqual(inv, {{'SKU-B': {safety_floor + 3}}})\n"
                "        self.assertFalse(res['reserved'])\n"
                f"        self.assertEqual(res['remaining'], {safety_floor + 3})\n\n"
                "    def test_5_e2e_nonpositive_qty_raises(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            process_reservation({'SKU-A': 50}, ' sku-a ', 0)\n"
            )
        else:
            test_py = (
                "import unittest\n"
                "from controller import process_reservation\n"
                "from service import try_reserve_sku\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_floor(self):\n"
                f"        self.assertEqual(try_reserve_sku({safety_floor + 2}, 3), (False, {safety_floor + 2}))\n\n"
                "    def test_2_service_exact_floor(self):\n"
                f"        self.assertEqual(try_reserve_sku({safety_floor + 3}, 3), (True, {safety_floor}))\n\n"
                "    def test_3_service_zero_qty(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            try_reserve_sku(20, 0)\n\n"
                "    def test_4_controller_missing_sku(self):\n"
                "        with self.assertRaises(KeyError):\n"
                "            process_reservation({'SKU-X': 30}, 'SKU-Y', 1)\n\n"
                "    def test_5_e2e_copy_and_norm(self):\n"
                f"        inv = {{'SKU-X': {safety_floor + 6}}}\n"
                "        out = process_reservation(inv, ' sku-x ', 6)\n"
                f"        self.assertEqual(inv, {{'SKU-X': {safety_floor + 6}}})\n"
                "        self.assertTrue(out['reserved'])\n"
            )
        specs.append(
            {
                "repo_id": rid,
                "domain": "inventory_order",
                "coupled_integration": coupled,
                "focal_symbol": "process_reservation",
                "issue": (
                    "Stock reservations breach the warehouse safety-stock floor, accept zero/negative quantities, "
                    "mutate the caller's inventory dictionary, and fail on lowercase/padded SKUs."
                ),
                "files": {
                    "contracts.py": contracts_py,
                    "service.py": service_buggy,
                    "controller.py": controller_buggy,
                    **distractors,
                },
                "ground_truth_fixes": {"service.py": service_fixed, "controller.py": controller_fixed},
                "test_suite": test_py,
            }
        )

    # -------------------------------------------------------------------------
    # DOMAIN 5: Telemetry Latency Alert Pipeline (TP-01 .. TP-04)
    # -------------------------------------------------------------------------
    tp_configs = [
        ("TP-01", 250, 500, True),
        ("TP-02", 200, 450, True),
        ("TP-03", 300, 600, True),
        ("TP-04", 150, 400, False),
    ]
    for rid, warn_ms, crit_ms, coupled in tp_configs:
        distractors = _make_distractors(
            ("latency", "alert", "telemetry", "severity"),
            f"LEGACY_CRIT_MS = {crit_ms * 2}  # Obsolete v1 latency threshold",
            "empty sample windows WARN CRITICAL boundaries negative latency service names",
        )
        contracts_py = (
            '"""Telemetry latency severity classification contract."""\n\n'
            "class TelemetryContract:\n"
            f'    """[service.py] summarize_window: any sample < 0 raises ValueError; empty [] returns (\'OK\', 0); else peak = max(samples_ms) and classify_latency(peak) (>={crit_ms} CRITICAL, >={warn_ms} WARN, else OK). '
            '[controller.py] ingest_telemetry: svc = payload[\'service\'].strip().lower() (empty ValueError), copy samples_ms list."""\n\n'
            "    @staticmethod\n"
            "    def classify_latency(max_ms: int) -> str:\n"
            f'        """Raises ValueError if max_ms < 0; returns CRITICAL (>={crit_ms}), WARN (>={warn_ms}), else OK."""\n'
            "        if max_ms < 0:\n"
            "            raise ValueError('negative latency')\n"
            f"        if max_ms >= {crit_ms}:\n"
            "            return 'CRITICAL'\n"
            f"        if max_ms >= {warn_ms}:\n"
            "            return 'WARN'\n"
            "        return 'OK'\n"
        )
        service_buggy = (
            "from contracts import TelemetryContract\n\n"
            "def summarize_window(samples_ms: list[int]) -> tuple[str, int]:\n"
            '    """Summarizes latency window into (severity, peak_ms)."""\n'
            "    peak = max(samples_ms)\n"
            f"    if peak > {crit_ms}:\n"
            "        return TelemetryContract.classify_latency(peak), peak\n"
            "    return 'OK', peak\n"
        )
        service_fixed = (
            "from contracts import TelemetryContract\n\n"
            "def summarize_window(samples_ms: list[int]) -> tuple[str, int]:\n"
            '    """Summarizes latency window into (severity, peak_ms)."""\n'
            "    if any(x < 0 for x in samples_ms):\n"
            "        raise ValueError('negative sample')\n"
            "    if not samples_ms:\n"
            "        return 'OK', 0\n"
            "    peak = max(samples_ms)\n"
            "    return TelemetryContract.classify_latency(peak), peak\n"
        )
        controller_buggy = (
            "from service import summarize_window\n\n"
            "def ingest_telemetry(payload: dict) -> dict:\n"
            '    """Ingests service latency window payload and returns alert summary."""\n'
            "    svc = payload['service']\n"
            "    samples = payload['samples_ms']\n"
            "    samples.append(0)\n"
            "    sev, peak = summarize_window(samples)\n"
            "    return {'service': svc, 'severity': sev, 'max_ms': peak}\n"
        )
        controller_fixed = (
            "from service import summarize_window\n\n"
            "def ingest_telemetry(payload: dict) -> dict:\n"
            '    """Normalizes service_name (strip/lower, non-empty) without mutating caller payload."""\n'
            "    svc = payload['service'].strip().lower()\n"
            "    if not svc:\n"
            "        raise ValueError('empty service')\n"
            "    samples = list(payload['samples_ms'])\n"
            "    sev, peak = summarize_window(samples)\n"
            "    return {'service': svc, 'severity': sev, 'max_ms': peak}\n"
        )
        if coupled:
            test_py = (
                "import unittest\n"
                "from controller import ingest_telemetry\n"
                "from service import summarize_window\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_empty_and_warn_boundary(self):\n"
                "        self.assertEqual(summarize_window([]), ('OK', 0))\n"
                f"        self.assertEqual(summarize_window([10, {warn_ms}]), ('WARN', {warn_ms}))\n\n"
                "    def test_2_controller_rejects_empty_service(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            ingest_telemetry({'service': '   ', 'samples_ms': [10]})\n\n"
                "    def test_3_e2e_warn_and_immutable_samples(self):\n"
                f"        arr = [20, {warn_ms}]\n"
                "        res = ingest_telemetry({'service': ' Auth-API ', 'samples_ms': arr})\n"
                f"        self.assertEqual(arr, [20, {warn_ms}])\n"
                "        self.assertEqual(res, {'service': 'auth-api', 'severity': 'WARN', 'max_ms': " + str(warn_ms) + "})\n\n"
                "    def test_4_e2e_critical_exact_boundary(self):\n"
                f"        arr = [{crit_ms}]\n"
                "        res = ingest_telemetry({'service': ' Billing ', 'samples_ms': arr})\n"
                f"        self.assertEqual(arr, [{crit_ms}])\n"
                f"        self.assertEqual(res['severity'], 'CRITICAL')\n\n"
                "    def test_5_e2e_negative_sample_rejected(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            ingest_telemetry({'service': ' gateway ', 'samples_ms': [-5, 100]})\n"
            )
        else:
            test_py = (
                "import unittest\n"
                "from controller import ingest_telemetry\n"
                "from service import summarize_window\n\n"
                "class TestRepo(unittest.TestCase):\n"
                "    def test_1_service_empty(self):\n"
                "        self.assertEqual(summarize_window([]), ('OK', 0))\n\n"
                "    def test_2_service_warn_and_crit(self):\n"
                f"        self.assertEqual(summarize_window([{warn_ms}]), ('WARN', {warn_ms}))\n"
                f"        self.assertEqual(summarize_window([{crit_ms}]), ('CRITICAL', {crit_ms}))\n\n"
                "    def test_3_service_negative(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            summarize_window([-1, 50])\n\n"
                "    def test_4_controller_blank_service(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            ingest_telemetry({'service': '  ', 'samples_ms': [10]})\n\n"
                "    def test_5_e2e_immutable_and_normalized(self):\n"
                "        arr = [40]\n"
                "        out = ingest_telemetry({'service': ' Edge-Svc ', 'samples_ms': arr})\n"
                "        self.assertEqual(arr, [40])\n"
                "        self.assertEqual(out['service'], 'edge-svc')\n"
            )
        specs.append(
            {
                "repo_id": rid,
                "domain": "telemetry_pipeline",
                "coupled_integration": coupled,
                "focal_symbol": "ingest_telemetry",
                "issue": (
                    "Telemetry ingestion crashes on empty sample windows, misclassifies WARN and exact CRITICAL "
                    "boundaries, ignores negative latency samples, mutates caller sample lists, and fails to normalize service names."
                ),
                "files": {
                    "contracts.py": contracts_py,
                    "service.py": service_buggy,
                    "controller.py": controller_buggy,
                    **distractors,
                },
                "ground_truth_fixes": {"service.py": service_fixed, "controller.py": controller_fixed},
                "test_suite": test_py,
            }
        )

    return specs


def evaluate_repo_sandbox(files: Dict[str, str], test_suite_code: str) -> Tuple[int, int, str, Dict[str, str]]:
    """
    Materializes `files` + held-out `_eval_tests.py` in an isolated temporary directory,
    checks AST syntax/import resolution (`unresolved_symbols`), runs `unittest` in a subprocess
    (timeout 5s), and returns `(test_failures, unresolved_symbols, failure_summary, file_hashes)`.
    """
    file_hashes = {
        k: hashlib.sha256(v.encode("utf-8")).hexdigest()
        for k, v in sorted(files.items())
    }
    unresolved = 0
    syntax_msgs: List[str] = []
    for fname, src in sorted(files.items()):
        try:
            ast.parse(src, filename=fname)
        except SyntaxError as exc:
            unresolved += 1
            syntax_msgs.append(f"SyntaxError in {fname} line {exc.lineno}: {exc.msg}")

    if unresolved > 0:
        return 5, unresolved, "; ".join(syntax_msgs), file_hashes

    with tempfile.TemporaryDirectory() as tmp:
        tpath = Path(tmp)
        for fname, src in files.items():
            (tpath / fname).write_text(src, encoding="utf-8")
        (tpath / "_eval_tests.py").write_text(test_suite_code, encoding="utf-8")

        runner = (
            "import json, sys, unittest\n"
            "sys.path.insert(0, '.')\n"
            "try:\n"
            "    import _eval_tests\n"
            "except Exception as e:\n"
            "    print(json.dumps({'failures': 5, 'unresolved': 1, 'summary': f'Import/Load error: {type(e).__name__}: {e}'}))\n"
            "    sys.exit(0)\n"
            "suite = unittest.defaultTestLoader.loadTestsFromModule(_eval_tests)\n"
            "res = unittest.TestResult()\n"
            "suite.run(res)\n"
            "bad = len(res.failures) + len(res.errors)\n"
            "msgs = []\n"
            "for tc, tb in (res.failures + res.errors)[:5]:\n"
            "    lines = [ln.strip() for ln in tb.strip().splitlines() if ln.strip()]\n"
            "    exc_ln = lines[-1] if lines else 'failed'\n"
            "    for cand_ln in reversed(lines):\n"
            "        if cand_ln.startswith(('AssertionError', 'ValueError', 'KeyError', 'TypeError', 'IndexError', 'ZeroDivisionError', 'AttributeError', 'RuntimeError')):\n"
            "            exc_ln = cand_ln\n"
            "            break\n"
            "    msgs.append(f'{tc._testMethodName}: {exc_ln[:140]}')\n"
            "print(json.dumps({'failures': bad, 'unresolved': 0, 'summary': ' | '.join(msgs)}))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                cwd=str(tpath),
                capture_output=True,
                text=True,
                timeout=6,
            )
            out_line = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1]
            parsed = json.loads(out_line)
            return int(parsed["failures"]), int(parsed["unresolved"]), str(parsed["summary"]), file_hashes
        except Exception as exc:
            return 5, 1, f"Execution error: {exc}", file_hashes


def parse_single_file_edit(llm_output: str) -> Optional[Tuple[str, str]]:
    """
    Extracts `(target_file, replacement_code)` from LLM output.
    Only permits edits to `controller.py` or `service.py` (protects contracts & held-out tests).
    """
    target: Optional[str] = None
    m_file = re.search(r"FILE\s*\**\s*:\s*[`*'\"\s]*([a-zA-Z0-9_./-]+\.py)", llm_output, re.IGNORECASE)
    if m_file:
        cand = Path(m_file.group(1).strip()).name
        if cand in EDITABLE_FILES:
            target = cand

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", llm_output, flags=re.DOTALL | re.IGNORECASE)
    code = blocks[0].strip() if blocks else ""
    if not code:
        # Fallback if model emitted raw code after FILE: header
        if m_file:
            tail = llm_output[m_file.end() :].strip()
            if "def " in tail:
                code = tail.strip("` \n")

    if not code:
        return None

    if target is None:
        for cand in EDITABLE_FILES:
            if cand in llm_output:
                target = cand
                break
    if target is None:
        if "from contracts import" in code:
            target = "service.py"
        elif "from service import" in code:
            target = "controller.py"

    if target not in EDITABLE_FILES:
        return None
    return target, code + "\n"


def build_agent_prompt(
    spec: Dict[str, Any],
    retrieved_context: str,
    test_failures: int,
    failure_summary: str,
    rollback_feedback: str = "",
    edited_files_history: Optional[List[str]] = None,
) -> str:
    fb_block = f"\n{rollback_feedback}\n" if rollback_feedback else ""
    unedited = [f for f in EDITABLE_FILES if f not in set(edited_files_history or [])]
    if edited_files_history and unedited:
        hist_line = (
            f"Files Modified in Prior Accepted Turns: {', '.join(edited_files_history)} "
            f"| Remaining Unmodified Editable File: `{unedited[0]}` (inspect and repair `{unedited[0]}` next if its bugs remain).\n"
        )
    elif edited_files_history:
        hist_line = f"Files Modified in Prior Accepted Turns: {', '.join(edited_files_history)}\n"
    else:
        hist_line = ""
    return (
        "You are an autonomous Python software engineering agent repairing a multi-file repository.\n"
        "You may modify ONLY ONE file per turn: either `service.py` or `controller.py` (never `contracts.py`, which is already implemented and read-only).\n"
        f"Issue Report: {spec['issue']}\n"
        f"Current Test Failures ({test_failures}/5 failing): {failure_summary}\n"
        f"{hist_line}"
        f"{fb_block}\n"
        "Retrieved Repository Slice:\n"
        f"{retrieved_context}\n\n"
        "Do NOT include prose explanations or docstrings. "
        "Start line 1 with `FILE: service.py` or `FILE: controller.py` "
        "followed immediately by a single ```python ... ``` block containing the complete updated file, "
        "and stop immediately after the closing triple backticks."
    )


def run_self_check() -> Dict[str, Any]:
    specs = build_20_repo_specs()
    assert len(specs) == 20, f"Expected 20 repos, got {len(specs)}"
    bm25_full_recall = 0
    tcgs_full_recall = 0
    bm25_tokens: List[int] = []
    tcgs_tokens: List[int] = []

    for sp in specs:
        t0, u0, sum0, _ = evaluate_repo_sandbox(sp["files"], sp["test_suite"])
        assert t0 in (3, 4, 5) and u0 == 0, f"{sp['repo_id']} unexpected init T={t0}, U={u0}: {sum0}"

        # Verify ground-truth fix solves 5/5 tests
        fixed_files = dict(sp["files"])
        fixed_files.update(sp["ground_truth_fixes"])
        tf, uf, sumf, _ = evaluate_repo_sandbox(fixed_files, sp["test_suite"])
        assert tf == 0 and uf == 0, f"{sp['repo_id']} ground-truth failed T={tf}, U={uf}: {sumf}"

        # Check retrieval recall of causal chain {'controller.py', 'service.py', 'contracts.py'}
        with tempfile.TemporaryDirectory() as tmp:
            rdir = Path(tmp)
            for fn, src in sp["files"].items():
                (rdir / fn).write_text(src, encoding="utf-8")
            _, bm_files, bm_tok = retrieve_bm25_context(rdir, f"{sp['focal_symbol']} {sp['issue']}", SLICE_TOKEN_BUDGET)
            _, tc_files, tc_tok = retrieve_tcgs_k2_context(rdir, sp["focal_symbol"], SLICE_TOKEN_BUDGET)
            causal = {"controller.py", "service.py", "contracts.py"}
            if causal.issubset(bm_files):
                bm25_full_recall += 1
            if causal.issubset(tc_files):
                tcgs_full_recall += 1
            bm25_tokens.append(bm_tok)
            tcgs_tokens.append(tc_tok)

    return {
        "repos_verified": len(specs),
        "total_unit_tests": len(specs) * 5,
        "bm25_2hop_causal_chain_recall_pct": round(100.0 * bm25_full_recall / len(specs), 1),
        "tcgs_k2_2hop_causal_chain_recall_pct": round(100.0 * tcgs_full_recall / len(specs), 1),
        "bm25_mean_slice_tokens": int(round(sum(bm25_tokens) / len(bm25_tokens))),
        "tcgs_k2_mean_slice_tokens": int(round(sum(tcgs_tokens) / len(tcgs_tokens))),
    }


def run_single_model_gpu_worker(
    model_dir: str,
    model_label: str,
    out_json: Path,
    seed: int = 20261112,
    device_map: str = "cuda:0",
    repo_start: int = 0,
    repo_end: int = 20,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    class CodeBlockStopper(StoppingCriteria):
        def __init__(self, tok: Any, prompt_len: int) -> None:
            super().__init__()
            self.tok = tok
            self.prompt_len = prompt_len

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            if int(input_ids.shape[-1]) - self.prompt_len < 16:
                return False
            txt = self.tok.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
            return bool(re.search(r"```(?:python)?\s*\n.+?```", txt, flags=re.DOTALL | re.IGNORECASE))

    t_start = time.time()
    gpu_cnt = torch.cuda.device_count()
    cc_major = torch.cuda.get_device_capability(0)[0]
    dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_cnt)]
    gpu_desc = f"{gpu_cnt}x {gpu_names[0]}" if (device_map == "auto" and gpu_cnt > 1) else gpu_names[0]

    t_load0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    max_mem = {0: "13GiB", 1: "13GiB", "cpu": "24GiB"} if (device_map == "auto" and gpu_cnt >= 2) else None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=device_map,
        max_memory=max_mem,
    )
    model.eval()
    first_dev = next(model.parameters()).device

    # Immediate 4-token smoke verification before starting full suite
    warm_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write `FILE: service.py` only."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    warm_inp = {k: v.to(first_dev) for k, v in tokenizer(warm_chat, return_tensors="pt").items()}
    with torch.no_grad():
        _ = model.generate(**warm_inp, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    load_seconds = round(time.time() - t_load0, 2)
    print(
        f"[{model_label}] Loaded & smoke-verified in {load_seconds}s | device_map={device_map} | "
        f"gpu={gpu_desc} | repos=[{repo_start}:{repo_end}]",
        flush=True,
    )

    specs = build_20_repo_specs()
    repo_indices = list(range(max(0, repo_start), min(len(specs), repo_end)))
    configs = [
        {"id": "bm25_ungoverned", "name": f"{model_label} + Okapi BM25 (Ungoverned ReAct)", "retrieval": "bm25", "governor": "none"},
        {"id": "tcgs_ungoverned", "name": f"{model_label} + TCGS k=2 (Ungoverned ReAct)", "retrieval": "tcgs_k2", "governor": "none"},
        {"id": "tcgs_fixed_dclv", "name": f"{model_label} + TCGS k=2 + Fixed DCLV (lambda_0=0.75)", "retrieval": "tcgs_k2", "governor": "fixed_dclv"},
        {"id": "wayfinder_full", "name": f"{model_label} + Wayfinder Full (TCGS k=2 + Adaptive DCLV + QF_RDL)", "retrieval": "tcgs_k2", "governor": "adaptive_dclv"},
    ]

    config_results: List[Dict[str, Any]] = []
    sample_generations: List[Dict[str, Any]] = []

    for cfg in configs:
        cfg_t0 = time.time()
        per_repo: List[Dict[str, Any]] = []
        resolved_count = 0
        cyclic_thrash_count = 0
        accepted_steps_total = 0
        smt_witness_verified_steps = 0
        rollbacks_total = 0
        llm_calls_total = 0
        accepted_tokens_sum = 0
        billed_tokens_sum = 0

        for repo_idx in repo_indices:
            sp = specs[repo_idx]
            working_files = dict(sp["files"])
            cur_t, cur_u, cur_summary, cur_hashes = evaluate_repo_sandbox(working_files, sp["test_suite"])

            dclv: Optional[DiscreteLyapunovValidator] = None
            if cfg["governor"] == "fixed_dclv":
                dclv = DiscreteLyapunovValidator(
                    contraction_factor=0.75,
                    max_turns=MAX_LLM_CALLS_PER_REPO,
                    allow_neutral_refactor_steps=0,
                    baseline_files=cur_hashes,
                    adaptive_lattice_envelope=False,
                    max_blast_radius_files=4,
                    max_prompt_tokens=3500,
                )
                dclv.register_turn(0, cur_hashes, cur_t, cur_u, prompt_tokens=0)
            elif cfg["governor"] == "adaptive_dclv":
                dclv = DiscreteLyapunovValidator(
                    contraction_factor=0.75,
                    max_turns=MAX_LLM_CALLS_PER_REPO,
                    allow_neutral_refactor_steps=1,
                    baseline_files=cur_hashes,
                    adaptive_lattice_envelope=True,
                    max_blast_radius_files=4,
                    max_prompt_tokens=3500,
                )
                dclv.register_turn(0, cur_hashes, cur_t, cur_u, prompt_tokens=0)

            committed_seen_hashes: Set[str] = {
                DiscreteLyapunovValidator.compute_state_hash(cur_hashes, cur_t, cur_u)
            }
            had_cycle = False
            accepted_turns_repo = 0
            accepted_tokens_repo = 0
            billed_tokens_repo = 0
            rollbacks_repo = 0
            consecutive_rejects = 0
            rollback_feedback = ""
            calls_used = 0
            edited_files_history: List[str] = []

            for call_idx in range(1, MAX_LLM_CALLS_PER_REPO + 1):
                if cur_t == 0 and cur_u == 0:
                    break

                with tempfile.TemporaryDirectory() as tmp:
                    rdir = Path(tmp)
                    for fn, src in working_files.items():
                        (rdir / fn).write_text(src, encoding="utf-8")
                    if cfg["retrieval"] == "bm25":
                        ctx_str, _, slice_tok = retrieve_bm25_context(
                            rdir, f"{sp['focal_symbol']} {sp['issue']}", SLICE_TOKEN_BUDGET
                        )
                    else:
                        ctx_str, _, slice_tok = retrieve_tcgs_k2_context(
                            rdir, sp["focal_symbol"], SLICE_TOKEN_BUDGET
                        )

                prompt_text = build_agent_prompt(
                    sp,
                    ctx_str,
                    cur_t,
                    cur_summary,
                    rollback_feedback=rollback_feedback,
                    edited_files_history=edited_files_history,
                )
                msgs = [{"role": "user", "content": prompt_text}]
                try:
                    chat_str = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except TypeError:
                    chat_str = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                inputs = {k: v.to(first_dev) for k, v in tokenizer(chat_str, return_tensors="pt").items()}
                in_toks = int(inputs["input_ids"].shape[-1])
                stop_crit = StoppingCriteriaList([CodeBlockStopper(tokenizer, in_toks)])

                with torch.no_grad():
                    if consecutive_rejects == 0:
                        out_ids = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            stopping_criteria=stop_crit,
                        )
                    else:
                        step_seed = seed + repo_idx * 100 + call_idx * 13 + consecutive_rejects
                        torch.manual_seed(step_seed)
                        torch.cuda.manual_seed_all(step_seed)
                        temp = 0.35 if consecutive_rejects == 1 else 0.55
                        out_ids = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=temp,
                            top_p=0.90,
                            stopping_criteria=stop_crit,
                        )

                gen_tokens_tensor = out_ids[0][in_toks:]
                out_toks = int(gen_tokens_tensor.shape[-1])
                raw_reply = tokenizer.decode(gen_tokens_tensor, skip_special_tokens=True)

                call_billed = in_toks + out_toks
                billed_tokens_repo += call_billed
                calls_used += 1
                llm_calls_total += 1

                parsed_edit = parse_single_file_edit(raw_reply)
                if parsed_edit is None:
                    if dclv is not None:
                        rollbacks_repo += 1
                        rollbacks_total += 1
                        consecutive_rejects += 1
                        rollback_feedback = (
                            "PREVIOUS CANDIDATE REJECTED: `contracts.py` and tests are read-only. "
                            "Start line 1 with `FILE: service.py` or `FILE: controller.py` followed by a single ```python ... ``` code block."
                        )
                    else:
                        had_cycle = True
                    continue

                target_file, new_code = parsed_edit
                cand_files = dict(working_files)
                cand_files[target_file] = new_code
                cand_t, cand_u, cand_summary, cand_hashes = evaluate_repo_sandbox(cand_files, sp["test_suite"])
                cand_sig = DiscreteLyapunovValidator.compute_state_hash(cand_hashes, cand_t, cand_u)

                if dclv is None:
                    if (cand_t > 0 or cand_u > 0) and cand_sig in committed_seen_hashes:
                        had_cycle = True
                    committed_seen_hashes.add(cand_sig)
                    working_files = cand_files
                    cur_t, cur_u, cur_summary, cur_hashes = cand_t, cand_u, cand_summary, cand_hashes
                    accepted_turns_repo += 1
                    accepted_tokens_repo += call_billed
                    edited_files_history.append(target_file)
                else:
                    ok, reason = dclv.register_turn(
                        accepted_turns_repo + 1,
                        cand_hashes,
                        cand_t,
                        cand_u,
                        prompt_tokens=slice_tok,
                    )
                    if ok:
                        if (cand_t > 0 or cand_u > 0) and cand_sig in committed_seen_hashes:
                            had_cycle = True
                        committed_seen_hashes.add(cand_sig)
                        working_files = cand_files
                        cur_t, cur_u, cur_summary, cur_hashes = cand_t, cand_u, cand_summary, cand_hashes
                        accepted_turns_repo += 1
                        accepted_steps_total += 1
                        accepted_tokens_repo += call_billed
                        edited_files_history.append(target_file)
                        if dclv.smt_proofs and dclv.smt_proofs[-1].get("witness_independently_verified"):
                            smt_witness_verified_steps += 1
                        consecutive_rejects = 0
                        rollback_feedback = ""
                        if len(sample_generations) < 3 and cfg["id"] == "wayfinder_full" and cur_t == 0:
                            sample_generations.append(
                                {
                                    "model": model_label,
                                    "repo_id": sp["repo_id"],
                                    "edited_file": target_file,
                                    "input_tokens": in_toks,
                                    "output_tokens": out_toks,
                                    "generated_patch_excerpt": new_code[:420],
                                }
                            )
                    else:
                        rollbacks_repo += 1
                        rollbacks_total += 1
                        consecutive_rejects += 1
                        other_file = "service.py" if target_file == "controller.py" else "controller.py"
                        if "duplicate cryptographic state hash" in reason.lower():
                            rollback_feedback = (
                                f"PREVIOUS CANDIDATE REJECTED (duplicate state on `{target_file}`): "
                                f"`{target_file}` produced an already-tested repository state ({cand_t}/5 failing: {cand_summary}). "
                                f"Do NOT repeat the same patch; if `{target_file}` is already fixed, repair `{other_file}` next."
                            )
                        else:
                            rollback_feedback = (
                                f"PREVIOUS CANDIDATE ON `{target_file}` ROLLED BACK BY LYAPUNOV/QF_RDL GOVERNOR:\n"
                                f"- Governor verdict: {reason}\n"
                                f"- Candidate failures ({cand_t}/5): {cand_summary}\n"
                                f"Repair the root cause cleanly (in `{target_file}` or `{other_file}`) so failing tests strictly decrease under all contract rules."
                            )

            is_resolved = int(cur_t == 0 and cur_u == 0)
            resolved_count += is_resolved
            if had_cycle:
                cyclic_thrash_count += 1
            accepted_tokens_sum += accepted_tokens_repo
            billed_tokens_sum += billed_tokens_repo

            per_repo.append(
                {
                    "repo_index": repo_idx,
                    "repo_id": sp["repo_id"],
                    "domain": sp["domain"],
                    "coupled_integration": sp["coupled_integration"],
                    "resolved": bool(is_resolved),
                    "final_test_failures": cur_t,
                    "final_unresolved_symbols": cur_u,
                    "llm_calls_used": calls_used,
                    "accepted_turns": accepted_turns_repo,
                    "rollbacks": rollbacks_repo,
                    "accepted_tokens": accepted_tokens_repo,
                    "total_billed_tokens": billed_tokens_repo,
                    "cyclic_thrash": had_cycle,
                }
            )

        n_repos = len(repo_indices)
        smt_pct = (
            round(100.0 * smt_witness_verified_steps / accepted_steps_total, 1)
            if accepted_steps_total > 0
            else None
        )
        cfg_summary = {
            "config_id": cfg["id"],
            "configuration": cfg["name"],
            "model": model_label,
            "n_repositories": n_repos,
            "resolved_count": resolved_count,
            "resolve_rate_pct": round(100.0 * resolved_count / n_repos, 1),
            "cyclic_thrash_count": cyclic_thrash_count,
            "cyclic_thrash_rate_pct": round(100.0 * cyclic_thrash_count / n_repos, 1),
            "mean_llm_calls": round(llm_calls_total / n_repos, 2),
            "mean_accepted_turns": round(sum(r["accepted_turns"] for r in per_repo) / n_repos, 2),
            "mean_accepted_tokens": int(round(accepted_tokens_sum / n_repos)),
            "mean_total_billed_tokens": int(round(billed_tokens_sum / n_repos)),
            "total_rollbacks": rollbacks_total,
            "accepted_steps_total": accepted_steps_total,
            "smt_witness_verified_steps": smt_witness_verified_steps,
            "smt_witness_verified_pct": smt_pct,
            "wall_clock_seconds": round(time.time() - cfg_t0, 2),
            "per_repo_outcomes": per_repo,
        }
        config_results.append(cfg_summary)
        print(
            f"[{model_label} {repo_start}:{repo_end}] {cfg['id']}: resolved={resolved_count}/{n_repos} "
            f"({cfg_summary['resolve_rate_pct']}%), thrash={cfg_summary['cyclic_thrash_rate_pct']}%, "
            f"billed_tok={cfg_summary['mean_total_billed_tokens']}, rollbacks={rollbacks_total}, "
            f"dt={cfg_summary['wall_clock_seconds']}s",
            flush=True,
        )

    worker_out = {
        "model_label": model_label,
        "model_dir": model_dir,
        "gpu_device": gpu_desc,
        "gpu_models": gpu_names,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "total_worker_seconds": round(time.time() - t_start, 2),
        "configurations": config_results,
        "sample_verified_generations": sample_generations,
    }
    out_json.write_text(json.dumps(worker_out, indent=2), encoding="utf-8")


def merge_model_shards(shards: List[Dict[str, Any]], expected_repos: int = 20) -> Dict[str, Any]:
    if len(shards) == 1:
        return shards[0]
    first = shards[0]
    model_label = first["model_label"]
    merged_configs: List[Dict[str, Any]] = []
    for cfg_pos, cfg0 in enumerate(first["configurations"]):
        cfg_id = cfg0["config_id"]
        outcomes_by_idx: Dict[int, Dict[str, Any]] = {}
        accepted_steps_total = 0
        smt_witness_verified_steps = 0
        wall_sum = 0.0
        for sh in shards:
            c_sh = sh["configurations"][cfg_pos]
            assert c_sh["config_id"] == cfg_id
            accepted_steps_total += int(c_sh["accepted_steps_total"])
            smt_witness_verified_steps += int(c_sh["smt_witness_verified_steps"])
            wall_sum += float(c_sh["wall_clock_seconds"])
            for pos_in_shard, rec in enumerate(c_sh["per_repo_outcomes"]):
                ridx = int(rec.get("repo_index", pos_in_shard))
                outcomes_by_idx[ridx] = rec

        per_repo = [outcomes_by_idx[k] for k in sorted(outcomes_by_idx.keys())]
        n_repos = len(per_repo)
        assert n_repos == expected_repos, f"Expected {expected_repos} merged repos for {model_label}/{cfg_id}, got {n_repos}"

        resolved_count = sum(1 for r in per_repo if r["resolved"])
        cyclic_thrash_count = sum(1 for r in per_repo if r["cyclic_thrash"])
        llm_calls_total = sum(int(r["llm_calls_used"]) for r in per_repo)
        accepted_turns_sum = sum(int(r["accepted_turns"]) for r in per_repo)
        accepted_tokens_sum = sum(int(r["accepted_tokens"]) for r in per_repo)
        billed_tokens_sum = sum(int(r["total_billed_tokens"]) for r in per_repo)
        rollbacks_total = sum(int(r["rollbacks"]) for r in per_repo)
        smt_pct = (
            round(100.0 * smt_witness_verified_steps / accepted_steps_total, 1)
            if accepted_steps_total > 0
            else None
        )
        merged_configs.append(
            {
                "config_id": cfg_id,
                "configuration": cfg0["configuration"],
                "model": model_label,
                "n_repositories": n_repos,
                "resolved_count": resolved_count,
                "resolve_rate_pct": round(100.0 * resolved_count / n_repos, 1),
                "cyclic_thrash_count": cyclic_thrash_count,
                "cyclic_thrash_rate_pct": round(100.0 * cyclic_thrash_count / n_repos, 1),
                "mean_llm_calls": round(llm_calls_total / n_repos, 2),
                "mean_accepted_turns": round(accepted_turns_sum / n_repos, 2),
                "mean_accepted_tokens": int(round(accepted_tokens_sum / n_repos)),
                "mean_total_billed_tokens": int(round(billed_tokens_sum / n_repos)),
                "total_rollbacks": rollbacks_total,
                "accepted_steps_total": accepted_steps_total,
                "smt_witness_verified_steps": smt_witness_verified_steps,
                "smt_witness_verified_pct": smt_pct,
                "wall_clock_seconds": round(wall_sum, 2),
                "per_repo_outcomes": per_repo,
            }
        )

    samples: List[Dict[str, Any]] = []
    for sh in shards:
        for s in sh.get("sample_verified_generations", []):
            if len(samples) < 3:
                samples.append(s)

    return {
        "model_label": model_label,
        "model_dir": first["model_dir"],
        "gpu_device": first["gpu_device"],
        "gpu_models": first.get("gpu_models", [first["gpu_device"]]),
        "torch_version": first.get("torch_version", ""),
        "transformers_version": first.get("transformers_version", ""),
        "dtype": first["dtype"],
        "load_seconds": first["load_seconds"],
        "total_worker_seconds": round(max(float(sh["total_worker_seconds"]) for sh in shards), 2),
        "configurations": merged_configs,
        "sample_verified_generations": samples,
    }


def _paired_mcnemar_exact_p(b_wins: int, c_losses: int) -> float:
    """Exact two-sided binomial (McNemar) p-value on discordant pairs (b, c)."""
    n = b_wins + c_losses
    if n == 0:
        return 1.0
    k = min(b_wins, c_losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2.0 * tail), 6)


def assemble_combined_telemetry(
    d_e2b: Dict[str, Any],
    d_e4b: Dict[str, Any],
    self_check_stats: Dict[str, Any],
    wall_seconds: float,
) -> Dict[str, Any]:
    paired_comparisons: Dict[str, Any] = {}
    for base_id in ("bm25_ungoverned", "tcgs_ungoverned", "tcgs_fixed_dclv"):
        wf_wins = 0
        base_wins = 0
        both_pass = 0
        both_fail = 0
        for mdata in (d_e2b, d_e4b):
            by_id = {c["config_id"]: c["per_repo_outcomes"] for c in mdata["configurations"]}
            for r_wf, r_bs in zip(by_id["wayfinder_full"], by_id[base_id]):
                assert r_wf["repo_id"] == r_bs["repo_id"]
                w, b = r_wf["resolved"], r_bs["resolved"]
                if w and not b:
                    wf_wins += 1
                elif b and not w:
                    base_wins += 1
                elif w and b:
                    both_pass += 1
                else:
                    both_fail += 1
        paired_comparisons[f"wayfinder_full_vs_{base_id}"] = {
            "n_paired_evaluations": 40,
            "wayfinder_only_wins": wf_wins,
            "baseline_only_wins": base_wins,
            "both_resolved": both_pass,
            "both_unresolved": both_fail,
            "mcnemar_two_sided_exact_p": _paired_mcnemar_exact_p(wf_wins, base_wins),
        }

    gpu_models = d_e4b.get("gpu_models") or d_e2b.get("gpu_models") or ["Tesla T4", "Tesla T4"]
    combined = {
        "benchmark_name": "Wayfinder v3.1 Live On-GPU Google Gemma 4 Multi-File SWE Benchmark",
        "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_and_runtime": {
            "gpu_count": len(gpu_models),
            "gpu_models": gpu_models,
            "torch_version": d_e4b.get("torch_version") or d_e2b.get("torch_version") or "unknown",
            "transformers_version": d_e4b.get("transformers_version") or d_e2b.get("transformers_version") or "unknown",
            "slice_token_budget": SLICE_TOKEN_BUDGET,
            "max_llm_calls_per_repo": MAX_LLM_CALLS_PER_REPO,
            "max_new_tokens_per_call": MAX_NEW_TOKENS,
            "total_wall_clock_seconds": round(wall_seconds, 2),
        },
        "suite_retrieval_diagnostics": self_check_stats,
        "models": {
            "gemma_4_e2b_it": d_e2b,
            "gemma_4_e4b_it": d_e4b,
        },
        "paired_statistical_tests_n40": paired_comparisons,
    }
    canon = json.dumps(
        {k: v for k, v in combined.items() if k != "execution_timestamp_utc"},
        sort_keys=True,
    ).encode("utf-8")
    combined["sha256_digest"] = hashlib.sha256(canon).hexdigest()
    return combined


def _discover_gemma4_dirs() -> Tuple[str, str]:
    e2b_dir = ""
    e4b_dir = ""
    for root, _, files in os.walk("/kaggle/input"):
        if "config.json" in files:
            if "e2b" in root.lower():
                e2b_dir = root
            elif "e4b" in root.lower():
                e4b_dir = root
    return e2b_dir, e4b_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker-model-dir", type=str, default="")
    parser.add_argument("--worker-label", type=str, default="")
    parser.add_argument("--worker-out", type=str, default="")
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--repo-start", type=int, default=0)
    parser.add_argument("--repo-end", type=int, default=20)
    parser.add_argument("--only-model", type=str, default="all", choices=["all", "e2b", "e4b"])
    parser.add_argument("--shard-out", type=str, default="")
    parser.add_argument("--merge-e2b", type=str, default="")
    parser.add_argument("--merge-e4b", type=str, default="")
    parser.add_argument("--output", type=str, default="live_gemma4_gpu_telemetry.json")
    args = parser.parse_args()

    if args.self_check:
        res = run_self_check()
        print(json.dumps(res, indent=2))
        return

    if args.merge_e2b and args.merge_e4b:
        self_check_stats = run_self_check()
        e2b_shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.merge_e2b.split(",") if p.strip()]
        e4b_shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.merge_e4b.split(",") if p.strip()]
        d_e2b = merge_model_shards(e2b_shards)
        d_e4b = merge_model_shards(e4b_shards)
        wall_s = max(float(d_e2b["total_worker_seconds"]), float(d_e4b["total_worker_seconds"]))
        combined = assemble_combined_telemetry(d_e2b, d_e4b, self_check_stats, wall_s)
        Path(args.output).write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print("MERGED_GPU_TELEMETRY_COMPLETE:", args.output, "SHA256=", combined["sha256_digest"], flush=True)
        return

    if args.worker_model_dir:
        run_single_model_gpu_worker(
            args.worker_model_dir,
            args.worker_label,
            Path(args.worker_out),
            device_map=args.device_map,
            repo_start=args.repo_start,
            repo_end=args.repo_end,
        )
        return

    import torch

    self_check_stats = run_self_check()
    print("SELF_CHECK_OK:", json.dumps(self_check_stats), flush=True)

    e2b_dir, e4b_dir = _discover_gemma4_dirs()
    gpu_count = torch.cuda.device_count()
    print(f"FOUND GPUs: {gpu_count} | E2B: {e2b_dir} | E4B: {e4b_dir}", flush=True)

    t0 = time.time()

    if args.only_model == "e2b":
        if not e2b_dir:
            raise RuntimeError("Missing mounted gemma-4-e2b-it")
        out_target = Path(args.shard_out or "worker_e2b.json")
        span = args.repo_end - args.repo_start
        if gpu_count >= 2 and span >= 2:
            mid = args.repo_start + (span // 2)
            s0 = Path("worker_e2b_s0.json")
            s1 = Path("worker_e2b_s1.json")
            env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false")
            env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TOKENIZERS_PARALLELISM="false")
            p0 = subprocess.Popen(
                [sys.executable, __file__, "--worker-model-dir", e2b_dir, "--worker-label", "Gemma 4 E2B-it", "--worker-out", str(s0), "--device-map", "cuda:0", "--repo-start", str(args.repo_start), "--repo-end", str(mid)],
                env=env0,
            )
            p1 = subprocess.Popen(
                [sys.executable, __file__, "--worker-model-dir", e2b_dir, "--worker-label", "Gemma 4 E2B-it", "--worker-out", str(s1), "--device-map", "cuda:0", "--repo-start", str(mid), "--repo-end", str(args.repo_end)],
                env=env1,
            )
            rc0, rc1 = p0.wait(), p1.wait()
            if rc0 != 0 or rc1 != 0:
                raise RuntimeError(f"E2B shard worker failure: rc0={rc0}, rc1={rc1}")
            merged_e2b = merge_model_shards([json.loads(s0.read_text("utf-8")), json.loads(s1.read_text("utf-8"))], expected_repos=span)
            out_target.write_text(json.dumps(merged_e2b, indent=2), encoding="utf-8")
        else:
            run_single_model_gpu_worker(e2b_dir, "Gemma 4 E2B-it", out_target, device_map="cuda:0", repo_start=args.repo_start, repo_end=args.repo_end)
        print("E2B_COMPLETE:", str(out_target), flush=True)
        return

    if args.only_model == "e4b":
        if not e4b_dir:
            raise RuntimeError("Missing mounted gemma-4-e4b-it")
        out_target = Path(args.shard_out or "worker_e4b.json")
        dmap = "auto" if gpu_count >= 2 else "cuda:0"
        run_single_model_gpu_worker(
            e4b_dir,
            "Gemma 4 E4B-it",
            out_target,
            device_map=dmap,
            repo_start=args.repo_start,
            repo_end=args.repo_end,
        )
        print("E4B_COMPLETE:", str(out_target), flush=True)
        return

    if not e2b_dir or not e4b_dir:
        raise RuntimeError(f"Expected mounted gemma-4-e2b-it and gemma-4-e4b-it; found e2b={e2b_dir!r}, e4b={e4b_dir!r}")

    out_e2b = Path("worker_e2b.json")
    out_e4b = Path("worker_e4b.json")
    if gpu_count >= 2:
        s0 = Path("worker_e2b_s0.json")
        s1 = Path("worker_e2b_s1.json")
        env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false")
        env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="1", TOKENIZERS_PARALLELISM="false")
        p0 = subprocess.Popen(
            [sys.executable, __file__, "--worker-model-dir", e2b_dir, "--worker-label", "Gemma 4 E2B-it", "--worker-out", str(s0), "--device-map", "cuda:0", "--repo-start", "0", "--repo-end", "10"],
            env=env0,
        )
        p1 = subprocess.Popen(
            [sys.executable, __file__, "--worker-model-dir", e2b_dir, "--worker-label", "Gemma 4 E2B-it", "--worker-out", str(s1), "--device-map", "cuda:0", "--repo-start", "10", "--repo-end", "20"],
            env=env1,
        )
        rc0, rc1 = p0.wait(), p1.wait()
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError(f"E2B shard worker failure: rc0={rc0}, rc1={rc1}")
        d_e2b = merge_model_shards([json.loads(s0.read_text("utf-8")), json.loads(s1.read_text("utf-8"))])
        out_e2b.write_text(json.dumps(d_e2b, indent=2), encoding="utf-8")
        run_single_model_gpu_worker(e4b_dir, "Gemma 4 E4B-it", out_e4b, device_map="auto", repo_start=0, repo_end=20)
        d_e4b = json.loads(out_e4b.read_text("utf-8"))
    else:
        run_single_model_gpu_worker(e2b_dir, "Gemma 4 E2B-it", out_e2b, device_map="cuda:0")
        run_single_model_gpu_worker(e4b_dir, "Gemma 4 E4B-it", out_e4b, device_map="cuda:0")
        d_e2b = json.loads(out_e2b.read_text("utf-8"))
        d_e4b = json.loads(out_e4b.read_text("utf-8"))

    combined = assemble_combined_telemetry(d_e2b, d_e4b, self_check_stats, time.time() - t0)
    Path(args.output).write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("LIVE_GPU_BENCHMARK_COMPLETE:", args.output, "SHA256=", combined["sha256_digest"], flush=True)


if __name__ == "__main__":
    main()
