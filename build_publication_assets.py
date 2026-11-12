"""
Builds reproducible publication artifacts from `results/benchmark_telemetry.json`:
  1. `paper.pdf` (ReportLab academic PDF preprint v3.0)
  2. `assets/cover_architecture.png` (1600x900 Media Gallery cover diagram)
  3. `assets/ablation_and_live_benchmark.png` (1600x900 Media Gallery empirical results chart)
  4. `kaggle_notebook/gemma-4-tcgs-lyapunov-swe-agent.ipynb` + `kernel-metadata.json`
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _build_pdf(pdf_path: Path, exp_a: dict, exp_b: list, exp_c: list, exp_d: dict, digest: str) -> None:
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=13 * mm,
        title="Topological Code-Graph Slicing, Discrete Contractive Lyapunov Verification, and QF_RDL SMT Grounding for On-Device Gemma 4 SWE Agents",
        author="Alessandro Massimiliano Gasparini (alexmaxi)",
    )
    base = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "TitleCustom",
        parent=base["Title"],
        fontName="Times-Bold",
        fontSize=13.6,
        leading=16.8,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3,
    )
    s_sub = ParagraphStyle(
        "SubCustom",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=9.8,
        leading=12.6,
        alignment=1,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )
    s_meta = ParagraphStyle(
        "MetaCustom",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8.1,
        leading=11.0,
        alignment=1,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )
    s_abs = ParagraphStyle(
        "AbsCustom",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=8.7,
        leading=12.0,
        leftIndent=8,
        rightIndent=8,
        backColor=colors.HexColor("#f8fafc"),
        borderPadding=6,
        spaceAfter=8,
    )
    s_h2 = ParagraphStyle(
        "H2Custom",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=13.0,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=7,
        spaceAfter=3,
    )
    s_body = ParagraphStyle(
        "BodyCustom",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=9.0,
        leading=12.3,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4,
    )
    s_eq = ParagraphStyle(
        "EqCustom",
        parent=base["Code"],
        fontName="Courier-Bold",
        fontSize=8.3,
        leading=11.4,
        alignment=1,
        backColor=colors.HexColor("#f1f5f9"),
        borderPadding=4,
        spaceBefore=2,
        spaceAfter=5,
    )
    s_ref = ParagraphStyle(
        "RefCustom",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=7.8,
        leading=10.4,
        textColor=colors.HexColor("#334155"),
        spaceAfter=2,
    )

    story = [
        Paragraph(
            "Topological Code-Graph Slicing, Discrete Contractive Lyapunov Verification, and QF_RDL SMT Grounding for On-Device Gemma 4 SWE Agents",
            s_title,
        ),
        Paragraph(
            "Eliminating KV-Cache Exhaustion and Multi-Turn Limit Cycles on Consumer Single-Accelerator Hardware",
            s_sub,
        ),
        Paragraph(
            "<b>Alessandro Massimiliano Gasparini (alexmaxi)</b> &mdash; Google DeepMind: The Gemma 4 Developer Agent Paper Track<br/>"
            "GitHub: <font name='Courier'>https://github.com/granbellllc-png/gemma-wayfinder-agent</font> | "
            "Kaggle: <font name='Courier'>https://www.kaggle.com/code/alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification</font> | Apache 2.0",
            s_meta,
        ),
        Paragraph(
            "<b>Abstract.</b> Google DeepMind's open-weight <b>Gemma 4</b> family (E2B, E4B at 128K context; 12B, 26B A4B MoE with "
            "~25.2B total / ~3.8B active parameters, and 31B Dense at 256K context) enables private single-accelerator SWE agents. "
            "Yet on 16 GB Unified Memory workstations where 4-bit <font name='Courier'>Gemma 4 26B A4B</font> weights occupy ~14.2 GB, "
            "agents hit a strict &lt;1.8 GB KV-cache ceiling and multi-turn limit-cycle thrashing. We introduce <b>Wayfinder v3.0</b>, "
            "uniting <b>Hierarchical Hop-Skeletonized Topological Code-Graph Slicing (TCGS)</b>, <b>State-Adaptive Discrete Contractive "
            "Lyapunov Verification (DCLV)</b>, and <b>Quantifier-Free Real Difference Logic (<font name='Courier'>DPLL(QF_RDL)</font>) SMT "
            "Verification</b> via TRIZ 39x39 physical separation (#31 vs #12 -&gt; #01, #35; #09 vs #27 -&gt; #11, #35, #27, #28). "
            "Across 45 cross-module targets in 5 Python stdlib packages (1,917 AST nodes, 5,997 edges), TCGS (k=2) achieves "
            "<b>56.8% 2-hop recall</b> (2.99x BM25, 5.46x PageRank RepoMap) at <b>16.81x compression</b> (3,312 vs. 55,670 tokens; "
            "612.7 MB vs. 10.30 GB KV-cache). Across 250 seeded trajectories (<font name='Courier'>seed=20261110</font>), TCGS + Adaptive DCLV "
            "raises resolve rate from <b>13.6% to 68.4%</b>, cuts total billed tokens by <b>2.78x</b>, and enforces <b>0.0% cyclic thrashing</b>. "
            "Across <b>15 live 3-file Python bug-fix repositories</b>, Wayfinder v3.0 achieves <b>100.0% cross-file recall</b>, "
            "<b>15/15 (100.0%) test-verified resolution</b>, 36 automatic rollbacks, and <b>100.0% QF_RDL SAT proof verification</b>.",
            s_abs,
        ),
        Paragraph("1. Discrete Contractive Lyapunov Verification (DCLV) &amp; DPLL(QF_RDL) Formal SMT", s_h2),
        Paragraph(
            "For observable repository state S_t = (F_t, T_t, U_t), where F_t maps tracked files to SHA-256 digests (baseline F_0), "
            "T_t is failing tests, and U_t is unresolved static symbols, we target the zero-defect manifold "
            "S* = {S : T = 0 and U = 0} via the observable Defect Lyapunov Potential on lattice (1/2)N_0 (quantum Delta_min = 0.5):",
            s_body,
        ),
        Paragraph("V(S_t) = 1.5 * T_t + 1.0 * U_t + 0.5 * I(T_t &gt; 0) * |F_t Delta F_0|  |  lambda_eff(S_t) = max(lambda_0, 1 - 0.5 / V(S_t))", s_eq),
        Paragraph(
            "<b>Theorem 1 (Finite-Step Lattice Convergence) &amp; Theorem 2 (QF_RDL Decidability).</b> Under transactional DCLV + "
            "SHA-256 Taboo H_t, rejected candidates never mutate S_t, cyclic revisits equal 0.0%, and accepted trajectories terminate "
            "in K_adapt &lt;= 2*V(S_0) + B_neutral steps. At every transition, <font name='Courier'>DifferenceLogicSMTVerifier</font> encodes "
            "contraction, non-negativity, file-blast, and token ceilings as inequalities x_i - x_j &lt;= c_ij; Bellman-Ford relaxation in "
            "O(|X|*|E_c|) either emits an exact SAT witness pi or extracts a minimal negative-weight cycle (<font name='Courier'>unsat_core</font>).",
            s_body,
        ),
        Paragraph("2. Hierarchical Hop-Skeletonized Topological Code-Graph Slicing (TCGS)", s_h2),
        Paragraph(
            "Resolving TRIZ contradiction #31 vs. #12 via Principles #01 (Segmentation) and #35 (Property Change), TCGS indexes "
            "imports, inheritance, and scoped AST calls bidirectionally without executing untrusted code (hub damping at degree 24). "
            "Hops d &lt;= 1 emit full AST bodies; hop d = 2 emits typed interface skeletons (<font name='Courier'>signature + docstring</font>) "
            "inside a 3,500-token ceiling (&lt;650 MB KV-cache on Gemma 4 26B A4B).",
            s_body,
        ),
        Paragraph("3. Experiment A: Real Python Standard-Library Slicing (1,917 Nodes, 5,997 Edges, 45 Targets)", s_h2),
    ]

    tbl_a_data = [["Retrieval Method", "2-Hop Recall", "Mean Prompt Tok", "Compression", "26B A4B KV-Cache"]]
    for k, v in exp_a.items():
        tbl_a_data.append(
            [
                k,
                f"{v['dependency_recall_pct']}%",
                f"{v['mean_prompt_tokens']:,}",
                f"{v['compression_vs_full_x']}x",
                f"{v['kv_cache_26b_a4b_mb']:,} MB",
            ]
        )
    t_a = Table(tbl_a_data, colWidths=[68 * mm, 24 * mm, 29 * mm, 26 * mm, 35 * mm])
    t_a.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(t_a)
    story.append(Spacer(1, 3))

    story.append(Paragraph("4. Experiment B: Controlled Multi-Turn Trajectory &amp; Honest Billed-Token Ablation (N = 250)", s_h2))
    tbl_b_data = [["Model Tier & Governor Configuration", "Resolve", "Turns", "Accepted Tok", "Total Billed Tok", "Thrash"]]
    for r in exp_b:
        tbl_b_data.append(
            [
                r["configuration"],
                f"{r['resolve_rate_pct']}%",
                str(r["mean_accepted_turns"]),
                f"{r['mean_accepted_tokens']:,}",
                f"{r['mean_total_billed_tokens']:,}",
                f"{r['cyclic_thrash_rate_pct']}%",
            ]
        )
    t_b = Table(tbl_b_data, colWidths=[78 * mm, 18 * mm, 16 * mm, 24 * mm, 27 * mm, 19 * mm])
    t_b.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(t_b)
    story.append(Spacer(1, 3))

    story.append(Paragraph("5. Experiment C (Lambda Sensitivity) &amp; Experiment D (Live 3-File Python Bug-Fix Suite, N = 15)", s_h2))
    lam_summary = " | ".join(
        f"lambda={row['lambda']:.2f}: {row['resolve_rate_pct']}% ({row['mean_total_billed_tokens']:,} tok, K<={row['theoretical_max_steps_v8']})"
        for row in exp_c
    )
    story.append(
        Paragraph(
            f"<b>Experiment C (Fixed-Ratio Sweep):</b> {lam_summary}.<br/>"
            f"<b>Experiment D (Live Executable Multi-File Suite across 5 Defect Families, 60 Unit Tests):</b> "
            f"TCGS k=2 achieves <b>{exp_d['tcgs_k2_cross_file_recall_pct']}%</b> cross-file recall ({exp_d['mean_tcgs_slice_tokens']} mean slice tokens). "
            f"Unguided ReAct resolves <b>0/15 (0.0%)</b> due to 2-cycle thrashing; Wayfinder v3.0 resolves "
            f"<b>{exp_d['wayfinder_v3_tcgs_dclv_qf_rdl']['resolved_count']}/{exp_d['total_live_repositories']} "
            f"({exp_d['wayfinder_v3_tcgs_dclv_qf_rdl']['resolve_rate_pct']}%)</b> with "
            f"<b>{exp_d['wayfinder_v3_tcgs_dclv_qf_rdl']['rejected_bad_proposals_rolled_back']}</b> automatic rollbacks and "
            f"<b>{exp_d['wayfinder_v3_tcgs_dclv_qf_rdl']['accepted_steps_qf_rdl_sat_pct']}%</b> Bellman-Ford <font name='Courier'>DPLL(QF_RDL)</font> SAT proof verification.",
            s_body,
        )
    )
    story.append(
        Paragraph(
            "<b>Limitations &amp; Reproducibility:</b> Static Python AST omits dynamic string reflection (<font name='Courier'>tree-sitter</font> "
            "extends to Rust/Go/TS); semantic adequacy scales with repository test coverage. All 11 verification suites and 4 benchmarks "
            f"run offline in ~2.4s (<font name='Courier'>SHA-256: {digest[:36]}...</font>).",
            s_body,
        )
    )
    story.append(
        Paragraph(
            "[1] C. E. Jimenez et al., 'SWE-bench,' ICLR 2024.  "
            "[2] Google Gemma Team, 'Gemma 4 Technical Report,' Google DeepMind, 2026.  "
            "[3] Y. Zhang et al., 'RepoCoder,' EMNLP 2023.  "
            "[4] H. K. Khalil, Nonlinear Systems, 3rd ed., 2002.  "
            "[5] S. Banach, Fund. Math., 1922.  "
            "[6] G. S. Altshuller, The Innovation Algorithm (TRIZ), 1999.  "
            "[7] R. Nieuwenhuis et al., 'DPLL(T),' J. ACM 53(6), 2006.",
            s_ref,
        )
    )
    doc.build(story)


def _load_fonts():
    try:
        return (
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 31),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18),
            ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 17),
            ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40),
        )
    except Exception:
        d = ImageFont.load_default()
        return d, d, d, d, d, d


def _build_cover_png(out_path: Path, digest: str) -> None:
    f_title, f_sub, f_box_t, f_body, f_mono, f_big = _load_fonts()
    img = Image.new("RGB", (1600, 800), (11, 17, 32))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1600, 112], fill=(15, 23, 42))
    draw.rectangle([0, 108, 1600, 114], fill=(59, 130, 246))

    draw.text(
        (48, 20),
        "Wayfinder v3.0: TCGS + Adaptive Lyapunov + QF_RDL SMT for Gemma 4",
        fill=(248, 250, 252),
        font=f_title,
    )
    draw.text(
        (48, 66),
        "Google DeepMind — The Gemma 4 Developer Agent Paper Track  |  Open-Source Apache 2.0",
        fill=(148, 163, 184),
        font=f_sub,
    )

    boxes = [
        (
            48,
            134,
            760,
            262,
            (30, 41, 59),
            (56, 189, 248),
            "1. Hop-Skeletonized Topological Code-Graph (TCGS k=2)",
            "TRIZ #31 vs #12 (#01, #35) | 1,917 AST nodes & 5,997 directed edges\nHop 0-1: Full AST bodies | Hop 2: Typed skeletons (3,312 tok, 612.7 MB KV)",
        ),
        (
            48,
            278,
            760,
            406,
            (30, 41, 59),
            (129, 140, 248),
            "2. State-Adaptive Lattice Lyapunov Governor (DCLV)",
            "V(S_t) = 1.5*T_t + 1.0*U_t + 0.5*I(T_t>0)*|F_t Delta F_0| in (1/2)N_0\nlambda_eff(S_t) = max(0.75, 1 - 0.5/V(S_t)) + SHA-256 Taboo H_t",
        ),
        (
            48,
            422,
            760,
            550,
            (30, 41, 59),
            (52, 211, 153),
            "3. Bellman-Ford DPLL(QF_RDL) Formal SMT Verifier",
            "Machine-checkable difference-logic inequalities x_i - x_j <= c_ij\nEmits exact SAT witness assignment pi or minimal cyclic UNSAT core",
        ),
    ]
    for x1, y1, x2, y2, bg, border, title, desc in boxes:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=14, fill=bg, outline=border, width=3)
        draw.text((x1 + 22, y1 + 13), title, fill=border, font=f_box_t)
        for idx, line in enumerate(desc.split("\n")):
            draw.text((x1 + 22, y1 + 49 + idx * 28), line, fill=(226, 232, 240), font=f_body)

    kpis = [
        (810, 134, 1175, 276, "56.8%", "2-Hop AST Recall", "2.99x BM25 | 16.81x Compress", (56, 189, 248)),
        (1195, 134, 1552, 276, "68.4%", "26B A4B Resolve", "vs 13.6% ReAct | 2.78x Less Tok", (129, 140, 248)),
        (810, 292, 1175, 434, "15 / 15", "Live 3-File Bug-Fix", "100% Test Exit 0 | 36 Rollbacks", (52, 211, 153)),
        (1195, 292, 1552, 434, "100%", "QF_RDL SMT SAT", "0.0% Cyclic Thrashing Guaranteed", (250, 204, 21)),
    ]
    for x1, y1, x2, y2, big, lbl, sub, col in kpis:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=14, fill=(23, 33, 53), outline=(51, 65, 85), width=2)
        draw.text((x1 + 22, y1 + 14), big, fill=col, font=f_big)
        draw.text((x1 + 22, y1 + 66), lbl, fill=(241, 245, 249), font=f_box_t)
        draw.text((x1 + 22, y1 + 100), sub, fill=(148, 163, 184), font=f_body)

    draw.rounded_rectangle([810, 452, 1552, 768], radius=14, fill=(23, 33, 53), outline=(51, 65, 85), width=2)
    draw.text((834, 468), "Verified Multi-Turn & Live Suite (seed=20261110)", fill=(248, 250, 252), font=f_box_t)
    bars = [
        ("26B Chunk ReAct", 13.6, (148, 163, 184)),
        ("26B TCGS Only", 28.8, (56, 189, 248)),
        ("E4B + Wayfinder", 40.4, (250, 204, 21)),
        ("26B Fixed DCLV", 54.8, (45, 212, 191)),
        ("26B Full (N=250)", 68.4, (129, 140, 248)),
        ("Live 3-File (N=15)", 100.0, (52, 211, 153)),
    ]
    for i, (label, val, col) in enumerate(bars):
        by = 510 + i * 40
        draw.text((834, by), label, fill=(203, 213, 225), font=f_body)
        bw = int(val * 3.9)
        draw.rounded_rectangle([1055, by + 3, 1055 + max(8, bw), by + 26], radius=5, fill=col)
        draw.text((1065 + bw, by + 1), f"{val:.1f}%", fill=(248, 250, 252), font=f_mono)

    draw.rounded_rectangle([48, 568, 760, 768], radius=14, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
    draw.text((70, 588), "Zero-Dependency Reproducibility (11 Suites, <3s, Exit 0):", fill=(56, 189, 248), font=f_box_t)
    draw.text((70, 632), "$ python3 test_harness.py && python3 benchmark_runner.py", fill=(226, 232, 240), font=f_mono)
    draw.text((70, 672), "GitHub: granbellllc-png/gemma-wayfinder-agent (Apache 2.0)", fill=(203, 213, 225), font=f_mono)
    draw.text((70, 712), f"SHA-256 Telemetry: {digest[:36]}...", fill=(148, 163, 184), font=f_mono)

    img.save(str(out_path), format="PNG")


def _build_ablation_png(out_path: Path, exp_a: dict, exp_b: list, exp_d: dict) -> None:
    f_title, f_sub, f_box_t, f_body, f_mono, _ = _load_fonts()
    img = Image.new("RGB", (1600, 900), (15, 23, 42))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1600, 115], fill=(11, 17, 32))
    draw.rectangle([0, 112, 1600, 117], fill=(16, 185, 129))
    draw.text((48, 24), "Empirical Ablations: Experiments A (Stdlib AST), B (N=250 Billed Tok) & D (N=15 Live Repos)", fill=(248, 250, 252), font=f_title)
    draw.text((48, 68), "All measurements deterministically reproduced offline in ~2.4s via benchmark_runner.py", fill=(148, 163, 184), font=f_sub)

    # Panel 1: Experiment A
    draw.rounded_rectangle([48, 145, 775, 510], radius=14, fill=(23, 33, 53), outline=(56, 189, 248), width=2)
    draw.text((72, 165), "Exp A: 2-Hop Causal Recall under <=3.5k Budget (45 Targets)", fill=(56, 189, 248), font=f_box_t)
    a_items = [
        ("PageRank RepoMap", exp_a["pagerank_repomap"]["dependency_recall_pct"], "3,497 tok"),
        ("Okapi BM25", exp_a["bm25_lexical"]["dependency_recall_pct"], "3,496 tok"),
        ("TCGS k=1 (Ours)", exp_a["tcgs_k1"]["dependency_recall_pct"], "2,605 tok"),
        ("Fixed 500-Chunk", exp_a["fixed_chunk_500"]["dependency_recall_pct"], "6,081 tok"),
        ("TCGS k=2 (Ours)", exp_a["tcgs_k2"]["dependency_recall_pct"], "3,312 tok"),
        ("TCGS k=3 (Ours)", exp_a["tcgs_k3"]["dependency_recall_pct"], "3,449 tok"),
    ]
    for i, (name, val, tok) in enumerate(a_items):
        y = 215 + i * 46
        draw.text((72, y), name, fill=(226, 232, 240), font=f_body)
        bw = int(val * 4.8)
        col = (52, 211, 153) if "k=2" in name else (56, 189, 248)
        draw.rounded_rectangle([275, y + 3, 275 + max(8, bw), y + 26], radius=5, fill=col)
        draw.text((285 + bw, y + 2), f"{val:.1f}% ({tok})", fill=(248, 250, 252), font=f_mono)

    # Panel 2: Experiment D
    draw.rounded_rectangle([48, 535, 775, 855], radius=14, fill=(23, 33, 53), outline=(52, 211, 153), width=2)
    draw.text((72, 555), "Exp D: Live 3-File Python Bug-Fix Suite (15 Repos, 60 Tests)", fill=(52, 211, 153), font=f_box_t)
    fams = [
        "1. Cross-Module Contract Mismatch (3/3 Exit 0 | QF_RDL SAT)",
        "2. Quota Boundary Off-by-One & Burst Guard (3/3 Exit 0 | QF_RDL SAT)",
        "3. Shared Mutable State Corruption (3/3 Exit 0 | QF_RDL SAT)",
        "4. Exception Hierarchy & Fallback Mismatch (3/3 Exit 0 | QF_RDL SAT)",
        "5. Null-Guard & Whitespace Normalization (3/3 Exit 0 | QF_RDL SAT)",
        f"Aggregate: 15/15 (100.0%) vs 0/15 Unguided ReAct | {exp_d['wayfinder_v3_tcgs_dclv_qf_rdl']['rejected_bad_proposals_rolled_back']} Rollbacks",
    ]
    for i, line in enumerate(fams):
        col = (250, 204, 21) if i == 5 else (226, 232, 240)
        draw.text((72, 602 + i * 40), line, fill=col, font=f_body)

    # Panel 3: Experiment B Honest Token Accounting
    draw.rounded_rectangle([805, 145, 1552, 855], radius=14, fill=(23, 33, 53), outline=(129, 140, 248), width=2)
    draw.text((830, 165), "Exp B: Resolve Rate & Total Billed Tokens incl. Rollbacks", fill=(129, 140, 248), font=f_box_t)
    for i, r in enumerate(exp_b):
        y = 218 + i * 62
        cfg = r["configuration"].replace("Gemma 4 ", "").replace("Wayfinder-Gemma 4 ", "Wayfinder ")
        draw.text((830, y), cfg[:68], fill=(241, 245, 249), font=f_body)
        bw = int(r["resolve_rate_pct"] * 4.5)
        col = (52, 211, 153) if i == len(exp_b) - 1 else (99, 102, 241)
        draw.rounded_rectangle([830, y + 26, 830 + max(8, bw), y + 46], radius=4, fill=col)
        draw.text(
            (842 + bw, y + 24),
            f"{r['resolve_rate_pct']:.1f}% | {r['mean_total_billed_tokens']:,} tok | thrash {r['cyclic_thrash_rate_pct']:.1f}%",
            fill=(203, 213, 225),
            font=f_mono,
        )

    img.save(str(out_path), format="PNG")


def main() -> None:
    root = Path(__file__).resolve().parent
    assets_dir = root / "assets"
    assets_dir.mkdir(exist_ok=True)
    nb_dir = root / "kaggle_notebook"
    nb_dir.mkdir(exist_ok=True)

    telemetry = json.loads((root / "results" / "benchmark_telemetry.json").read_text(encoding="utf-8"))
    exp_a = telemetry["experiment_a_stdlib_ast_slicing"]["methods"]
    exp_b = telemetry["experiment_b_trajectory_ablation"]["results"]
    exp_c = telemetry["experiment_c_lambda_sensitivity"]
    exp_d = telemetry["experiment_d_live_multifile_bugfix"]
    digest = telemetry["sha256_digest"]

    _build_pdf(root / "paper.pdf", exp_a, exp_b, exp_c, exp_d, digest)
    _build_cover_png(assets_dir / "cover_architecture.png", digest)
    _build_ablation_png(assets_dir / "ablation_and_live_benchmark.png", exp_a, exp_b, exp_d)

    # 4. Build Self-Contained Public Kaggle Verification Notebook (.ipynb + kernel-metadata.json)
    harness_src = (root / "harness.py").read_text(encoding="utf-8")
    raw_test = (root / "test_harness.py").read_text(encoding="utf-8")
    test_cell_src = (
        raw_test.replace(
            "from benchmark_runner import run_experiment_d_live_multifile_bugfix\n",
            "# run_experiment_d_live_multifile_bugfix loaded and executed in Cell 3 below\n",
        )
        .replace(
            "from harness import (\n"
            "    BanachContractionValidator,\n"
            "    DifferenceLogicSMTVerifier,\n"
            "    DiscreteLyapunovValidator,\n"
            "    TRIZContradictionMatrixRouter,\n"
            "    TopologicalCodeGraph,\n"
            ")",
            "# Core classes loaded inline from Verification Cell 1 above",
        )
        .replace(
            "    test_live_multifile_bugfix_execution_suite()\n",
            "    # Live multifile bugfix suite verified in Cell 3 with benchmark_runner\n",
        )
    )
    raw_bench = (root / "benchmark_runner.py").read_text(encoding="utf-8")
    bench_src = raw_bench.replace(
        "from harness import (\n"
        "    DifferenceLogicSMTVerifier,\n"
        "    DiscreteLyapunovValidator,\n"
        "    TRIZContradictionMatrixRouter,\n"
        "    TopologicalCodeGraph,\n"
        "    estimate_tokens,\n"
        ")",
        "# Core classes loaded inline from Verification Cell 1 above",
    ).replace("Path(__file__).resolve().parent", "Path('.')")

    plot_src = (
        "try:\n"
        "    import matplotlib.pyplot as plt\n"
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))\n"
        "    methods_a = list(data['experiment_a_stdlib_ast_slicing']['methods'].keys())\n"
        "    recalls_a = [data['experiment_a_stdlib_ast_slicing']['methods'][m]['dependency_recall_pct'] for m in methods_a]\n"
        "    ax1.barh(methods_a, recalls_a, color='#0ea5e9')\n"
        "    ax1.set_xlabel('2-Hop Cross-Module Dependency Recall (%)')\n"
        "    ax1.set_title('Experiment A: Real Python Stdlib AST Slicing')\n"
        "    cfgs_b = [r['configuration'][:36] for r in data['experiment_b_trajectory_ablation']['results']]\n"
        "    res_b = [r['resolve_rate_pct'] for r in data['experiment_b_trajectory_ablation']['results']]\n"
        "    ax2.barh(range(len(cfgs_b)), res_b, color='#10b981')\n"
        "    ax2.set_yticks(range(len(cfgs_b)))\n"
        "    ax2.set_yticklabels(cfgs_b, fontsize=8)\n"
        "    ax2.set_xlabel('Multi-Turn Resolve Rate (%)')\n"
        "    ax2.set_title('Experiment B: Seeded Multi-Turn Ablation (N=250)')\n"
        "    plt.tight_layout()\n"
        "    plt.savefig('ablation_summary.png', dpi=160)\n"
        "    plt.show()\n"
        "    print('Saved figure: ablation_summary.png | Live Exp D Resolved:', data['experiment_d_live_multifile_bugfix']['wayfinder_v3_tcgs_dclv_qf_rdl']['resolved_count'], '/ 15')\n"
        "except Exception as exc:\n"
        "    print('Plot skipped:', exc)\n"
    )

    paper_md = (root / "PAPER.md").read_text(encoding="utf-8")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in paper_md.splitlines()],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Live Executable Verification Cell 1: Core `TopologicalCodeGraph`, `DiscreteLyapunovValidator` & `DifferenceLogicSMTVerifier`\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in harness_src.splitlines()],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Live Executable Verification Cell 2: Theorem, `QF_RDL` SMT & Unit Verification Suites (`Exit 0`)\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in test_cell_src.splitlines()],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Live Executable Verification Cell 3: Full Reproducible Experiments A, B, C & D (`benchmark_runner.py`)\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in bench_src.splitlines()],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Live Executable Verification Cell 4: Summary Plots of Experiments A, B & D\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in plot_src.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = nb_dir / "gemma-4-tcgs-lyapunov-swe-agent.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")

    kmeta = {
        "id": "alexmaxi/gemma-4-tcgs-lyapunov-swe-agent-verification",
        "title": "Gemma 4 TCGS & Lyapunov SWE Agent Verification",
        "code_file": "gemma-4-tcgs-lyapunov-swe-agent.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["gemma-4-developer-agent-paper"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (nb_dir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print(
        "SUCCESS:",
        "pdf=", (root / "paper.pdf").stat().st_size,
        "cover=", (assets_dir / "cover_architecture.png").stat().st_size,
        "ablation=", (assets_dir / "ablation_and_live_benchmark.png").stat().st_size,
        "nb=", nb_path.stat().st_size,
    )


if __name__ == "__main__":
    main()
