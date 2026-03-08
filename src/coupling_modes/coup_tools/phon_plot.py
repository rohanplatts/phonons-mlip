from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .phon_classes import ComparisonOutput, ndarray_realFloats
from .phon_analysis import top_clusters_by_weight


 
def render_report(
    out: ComparisonOutput,
    threshold: float,
    freq_cluster_tol: float,
    freq_window: float,
    *,
    alpha: float,
    weight_kind: str,
) -> str:
    """create comparison report.
    """

    # Render a plain-text report with per-q summaries
    c = out.dft_cache
    lines: List[str] = []

    def h(s: str) -> None:
        # Section header helper
        lines.append(s)
        lines.append("-" * len(s))

    def fmt_q(q: ndarray_realFloats) -> str:
        # Consistent q-position formatting
        return f"[{q[0]: .6f}, {q[1]: .6f}, {q[2]: .6f}]"

    # High-level report header uses dQ 
    h("Mode–coupling comparison report")
    lines.append(f"DFT: {c.dft_path}")
    lines.append(f"N atoms: {int(c.masses.shape[0])}    N modes: {int(c.dq_flat.size)}")
    # Settings printed here should include the knobs that affect final ranking.
    lines.append(
        "Settings: "
        f"threshold={threshold:.3f}, "
        f"freq_cluster_tol={freq_cluster_tol:.3f}, "
        f"freq_window={freq_window:.3f}, "
        f"alpha={float(alpha):.3f}, "
        f"weight_kind={str(weight_kind)}"
    )
    lines.append(f"q-points used: {len(c.q_indices)}")
    for i, qi in enumerate(c.q_indices):
        lines.append(f"  q[{i}] index={qi}  pos={fmt_q(c.q_positions[i])}")
    lines.append("")

    h("DFT Coupling modes  (per mode) ")
    for i, qi in enumerate(c.q_indices):
        leg = c.AvgProjPowX_by_q[i]
        warn = leg.get("sum_p_warning", None)
        lines.append(f"q[{i}] index={qi}  sum(p)={leg['sum_p']:.6f}" + (f"  WARNING: {warn}" if warn else ""))
        lines.append(f"  selected modes: k={len(leg['selected_indices'])}  cumsum_last={leg['selected_cumsum_last']:.6f}")
        lines.append("  Coupling modes: mode | freq | p):")
        for (m, f, p) in leg["top_contrib_preview"]:
            lines.append(f"    {m:4d}  {f: .6f}  {p: .6e}")
        lines.append("")

    # DFT clusters ranked by their weights in dq
    h("DFT Coupling clusters (eigenspace)")
    for i, qi in enumerate(c.q_indices):
        w = c.w_dft_by_q[i]
        rel = top_clusters_by_weight(w, threshold=threshold)
        lines.append(f"q[{i}] index={qi}  pos={fmt_q(c.q_positions[i])}")
        lines.append("  cid  size  fmin        fmax        w_dft")
        for cid in rel:
            cl = c.clusters_by_q[i][cid]
            fmin, fmax = c.cluster_ranges_by_q[i][cid]
            lines.append(f"  {cid:3d}  {len(cl):4d}  {fmin: .6f}  {fmax: .6f}  {w[cid]: .6f}")
        lines.append("")

    # Per-ML model comparisons vs DFT
    h("MLIP comparisons")
    for ml_path, res in out.results_per_ml.items():
        lines.append(f"Model: {ml_path}")
        leg_sum = res["AvgProjPowX"]["summary"]
        r_sum = res["research"]["summary"]
        g_sum = res.get("gses", {}).get("summary", None)
        if g_sum is not None:
            lines.append(
                f"  dQ score: mean={g_sum['Score_mean']:.6f}  min={g_sum['Score_min']:.6f}  "
                f"E_freq={g_sum['E_freq_mean']:.6f}  E_freq_rel={g_sum['E_freq_rel_mean']:.6f}  E_vec={g_sum['E_vec_mean']:.6f}"
            )
        lines.append(
            f"  AvgProjPowX coupling modes subspace scores X: mean={leg_sum['X_mean']:.6f}  min={leg_sum['X_min']:.6f}  max={leg_sum['X_max']:.6f}"
        )
        lines.append(
            f"  Coupling cluster (eigenspaces) subspace scores: L1w_mean={r_sum['L1_weights_mean']:.6f}  "
            f"wθ_mean={r_sum['weighted_mean_theta_deg_mean']:.6f}  "
            f"w(1-σ^2)_mean={r_sum['weighted_1_minus_sigma2_mean']:.6f}"
        )

        if "gses" in res:
            lines.append("  Per-q dQ score (summary):")
            for qrow in res["gses"]["per_q"]:
                lines.append(
                    f"    q={fmt_q(np.asarray(qrow['q_position']))}  "
                    f"Score={float(qrow['Score']):.6f}  "
                    f"E_freq={float(qrow['E_freq']):.6f}  "
                    f"E_freq_rel={float(qrow['E_freq_rel']):.6f}  "
                    f"E_vec={float(qrow['E_vec']):.6f}"
                )

        lines.append("  Per-q AvgProjPowX angles (summary):")
        for qrow in res["AvgProjPowX"]["per_q"]:
            sig = np.asarray(qrow["sigma"], dtype=float)
            theta = np.asarray(qrow["theta_deg"], dtype=float)
            sig_min = float(np.min(sig)) if sig.size else float("nan")
            sig_mean = float(np.mean(sig)) if sig.size else float("nan")
            th_mean = float(np.mean(theta)) if theta.size else float("nan")
            th_max = float(np.max(theta)) if theta.size else float("nan")
            warn_d = qrow.get("dft_sum_p_warning", None)
            warn_m = qrow.get("ml_sum_p_warning", None)
            wtxt = ""
            if warn_d or warn_m:
                wtxt = "  WARN:" + (" DFT" if warn_d else "") + (" ML" if warn_m else "")
            lines.append(
                f"    q={fmt_q(np.asarray(qrow['q_position']))}  X={qrow['X']:.6f}  "
                f"k_dft={qrow['k_dft']}  k_ml={qrow['k_ml']}  "
                f"σ_min={sig_min:.6f}  σ_mean={sig_mean:.6f}  "
                f"θ_mean={th_mean:.3f}°  θ_max={th_max:.3f}°{wtxt}"
            )

        lines.append("  Per-q coupling cluster stats:")
        for qrow in res["research"]["per_q"]:
            summ = qrow["summary"]
            lines.append(
                f"    q={fmt_q(np.asarray(qrow['q_position']))}  "
                f"L1w={summ['L1_weights_relevant']:.6f}  "
                f"wθ={summ['weighted_mean_theta_deg_relevant']:.6f}  "
                f"w(1-σ^2)={summ['weighted_1_minus_sigma2_relevant']:.6f}"
            )
            lines.append("      cid  size  fmin        fmax        w_dft     w_ml      θ_mean    θ_max     θ_min")
            for cl in qrow["clusters_relevant"]:
                sig = np.asarray(cl["sigma"], dtype=float)
                sig_min = float(np.min(sig)) if sig.size else float("nan")
                fmin, fmax = cl["freq_range_dft"]
                lines.append(
                    f"      {cl['cluster_id']:3d}  {cl['size_dft']:4d}  {fmin: .6f}  {fmax: .6f}  "
                    f"{cl['w_dft']: .6f}  {cl['w_ml_window']: .6f}  "
                    f"{cl['theta_mean_deg']:7.3f}°  {cl['theta_max_deg']:7.3f}°  {np.cos(sig_min): .6f}"
                )
        lines.append("")
    # Final ranking table dQ Score_mean; fallback: AvgProjPowX X_mean).
    ranking_rows: List[Tuple[float, float, str]] = []
    for ml_path, res in out.results_per_ml.items():
        g_sum = res.get("gses", {}).get("summary", {})
        leg_sum = res.get("AvgProjPowX", {}).get("summary", {})
        score = float(g_sum.get("Score_mean", float("nan")))
        xmean = float(leg_sum.get("X_mean", float("nan")))
        ranking_rows.append((score, -xmean, ml_path))

    # Sort by Score_mean ascending (lower is better); tie-break by higher X_mean
    ranking_rows.sort(key=lambda t: (np.inf if not np.isfinite(t[0]) else t[0], t[1], t[2]))

    h("FINAL RANKING (lower dQ Score is better)")
    lines.append("rank  Score_mean    E_freq      E_vec       E_freq_rel   X_mean      model")
    for r, (score, neg_xmean, ml_path) in enumerate(ranking_rows, start=1):
        # Print a short model identifier rather than the full band.yaml path.
        # Expected layout: .../results/<model_name>/.../band.yaml
        mp = Path(str(ml_path))
        model_name = mp.parent.parent.parent.parent.name if mp.name == "band.yaml" else mp.name
        res = out.results_per_ml[ml_path]
        g_sum = res.get("gses", {}).get("summary", {})
        leg_sum = res.get("AvgProjPowX", {}).get("summary", {})
        lines.append(
            f"{r:4d}  "
            f"{float(g_sum.get('Score_mean', float('nan'))):10.6f}  "
            f"{float(g_sum.get('E_freq_mean', float('nan'))):9.6f}  "
            f"{float(g_sum.get('E_vec_mean', float('nan'))):9.6f}  "
            f"{float(g_sum.get('E_freq_rel_mean', float('nan'))):11.6f}  "
            f"{float(leg_sum.get('X_mean', float('nan'))):9.6f}  "
            f"{model_name}"
        )
    lines.append("")


    return "\n".join(lines)
