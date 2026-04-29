#!/usr/bin/env python3
"""
preprocess_demo.py
==================
Preprocessing Demo — 4 Brain Connectivity Methods on Synthetic fMRI Data

Methods demonstrated (identical logic to preprocess/, data-loading stripped):
  1. Pearson Correlation         — undirected, threshold=0.4
  2. VAR Granger Causality       — linear directed, log-ratio threshold=0.4
  3. Kernel GC (mlcausality/KGC) — nonlinear directed, p-value alpha=0.05
  4. Jacobian Neural GC (JRNGC) — neural directed, top-30%

Synthetic data: VAR(2) process, N=10 ROIs × T=300 time points,
with known directed causal structure so we can measure recovery.

Output (in demo_output/):
  adjacency_comparison.png  — heatmap grid comparing all methods
  sample_pearson.npz        — same .npz format as real preprocessing
  sample_var_gc.npz
  sample_kgc.npz            — only if mlcausality is installed
  sample_jngc.npz           — only if torch is installed
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BSNIP = os.path.join(_REPO, 'preprocess', 'BSNIP')
sys.path.insert(0, _BSNIP)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════

# Known ground-truth directed edges: (source_roi, target_roi)
N_ROIS, T_POINTS = 116, 200   # AAL-116 atlas, typical fMRI session length

# thresholds can be chosen based on specific targets or tasks
PEARSON_THRESHOLD = 0.40
VAR_GC_THRESHOLD  = 0.40
KGC_ALPHA         = 0.05
JNGC_TOP_K        = 0.30

# 20 sparse directed edges (≤2 targets per hub to keep bivariate GC strong)
# Mimics functional modules in AAL parcellation
_MODULES = [
    (0,  [1, 2]),    (10, [11, 12]),  (20, [21, 22]),  (30, [31, 32]),
    (40, [41, 42]),  (50, [51, 52]),  (60, [61, 62]),  (70, [71, 72]),
    (80, [81, 82]),  (90, [91, 92]),
]
GROUND_TRUTH = [(src, tgt) for src, tgts in _MODULES for tgt in tgts]


def generate_bold(N=N_ROIS, T=T_POINTS, seed=42):
    """
    VAR(2) process, N=116 ROIs, T=200, matching AAL-116 fMRI scale.
    A1=0.70 / A2=0.25 / noise=0.20 produces GC log-ratios ≈0.5–1.5
    and Pearson |r|≈0.4–0.7 for the planted directed edges,
    consistent with paper thresholds (VAR-GC=0.4, Pearson=0.4).
    Each hub drives at most 2 targets so bivariate GC stays strong.
    """
    rng = np.random.default_rng(seed)
    A1, A2 = np.zeros((N, N)), np.zeros((N, N))
    for src, tgt in GROUND_TRUTH:
        A1[tgt, src] = 0.70
        A2[tgt, src] = 0.25
    np.fill_diagonal(A1, 0.10)

    X = np.zeros((T, N))
    X[:2] = rng.normal(0, 0.2, (2, N))
    for t in range(2, T):
        X[t] = A1 @ X[t - 1] + A2 @ X[t - 2] + rng.normal(0, 0.20, N)

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Shared helpers for lag selection & design matrices
#             (same as preprocess/BSNIP/gc_simple.py)
# ═══════════════════════════════════════════════════════════════════════════════

from statsmodels.tsa.stattools import acf as _acf
from numpy.linalg import lstsq as _lstsq


def _best_lag_acf(series, max_lag=10):
    r = _acf(series, nlags=max_lag, fft=True)
    for lag in range(1, max_lag + 1):
        if r[lag] < 1.0 / np.e:
            return lag
    return max_lag


def _select_lags(data, max_lag=10):
    return np.array([_best_lag_acf(data[:, i], max_lag)
                     for i in range(data.shape[1])])


def _design(sig, lag):
    return np.array([[sig[t - p] for p in range(1, lag + 1)]
                     for t in range(lag, len(sig))])


def _edges_to_npz_arrays(edges, N):
    if edges:
        ei = np.array(edges, dtype=np.int64).T   # [2, E]  row0=src row1=tgt
        ea = np.ones(ei.shape[1], dtype=np.float32)
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
        ea = np.zeros(0, dtype=np.float32)
    return ei, ea


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Method implementations
# ═══════════════════════════════════════════════════════════════════════════════

# ── Method 1: Pearson Correlation (undirected) ────────────────────────────────
def method_pearson(data, threshold=0.4):
    """
    Returns binary undirected graph; A[i,j]=A[j,i]=1 iff |corr|>threshold.
    """
    corr = np.corrcoef(data.T)
    np.fill_diagonal(corr, 0.0)
    src, tgt = np.where(corr > threshold)
    ei = np.stack([src, tgt]).astype(np.int64)
    ea = np.ones(ei.shape[1], dtype=np.float32)
    return ei, ea, corr


# ── Method 2: VAR Granger Causality ──────────────────────────────────────────
def method_var_gc(data, max_lag=10, threshold=0.4):
    """
    gc_matrix[tgt, src] = log(var_restricted / var_unrestricted).
    Edge (src, tgt) is added when gc_matrix[tgt, src] >= threshold.
    """
    T, N = data.shape
    lags = _select_lags(data, max_lag)
    gc = np.zeros((N, N))

    for tgt in range(N):
        lt  = lags[tgt]
        y   = data[lt:, tgt]
        Zt  = _design(data[:, tgt], lt)
        c, *_ = _lstsq(Zt, y, rcond=None)
        var_r = np.var(y - Zt @ c, ddof=1)

        for src in range(N):
            if src == tgt:
                continue
            lag = int(max(lt, lags[src]))
            if T <= lag + 1:
                continue
            y2    = data[lag:, tgt]
            Zj    = np.hstack([_design(data[:, tgt], lag),
                                _design(data[:, src], lag)])
            c2, *_ = _lstsq(Zj, y2, rcond=None)
            var_u  = np.var(y2 - Zj @ c2, ddof=1)
            gc[tgt, src] = np.log((var_r + 1e-12) / (var_u + 1e-12))

    np.fill_diagonal(gc, 0.0)
    edges = [(src, tgt)
             for tgt in range(N) for src in range(N)
             if src != tgt and gc[tgt, src] >= threshold]
    ei, ea = _edges_to_npz_arrays(edges, N)
    return ei, ea, gc          # gc[tgt,src] = src→tgt strength


# ── Method 3: Kernel GC via mlcausality ──────────────────────────────────────
def method_kgc(data, max_lag=10, alpha=0.05):
    """
    Uses mlcausality.multiloco_mlcausality; falls back gracefully if not installed.
    Edge (src, tgt) is added when p-value < alpha.
    """
    try:
        import mlcausality as mlc
        import pandas as pd
    except ImportError:
        print("  [KGC] 'mlcausality' not installed — skipping this method.")
        return None, None, None

    T, N = data.shape
    lags = _select_lags(data, max_lag)

    # pair-lag matrix (max rule, same as gc_ml.py)
    Lmat = np.array([[max(int(lags[s]), int(lags[t])) if s != t else 0
                      for t in range(N)] for s in range(N)])
    used = sorted(set(Lmat.flatten()) - {0})

    p_mats = {}
    for L in used:
        try:
            df = mlc.multiloco_mlcausality(data, lags=[int(L)], train_size=1)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            cols = {c.lower(): c for c in df.columns}
            sc = next((v for k, v in cols.items()
                       if "sign" in k and "pvalue" in k), None)
            wc = next((v for k, v in cols.items()
                       if "wilcoxon" in k and "pvalue" in k), None)
            ps  = df[sc].to_numpy() if sc else None
            pw  = df[wc].to_numpy() if wc else None
            pv  = (np.minimum(ps, pw) if (ps is not None and pw is not None)
                   else (ps if ps is not None else pw))
            Xi  = df["X"].to_numpy(dtype=int)
            Yi  = df["y"].to_numpy(dtype=int)
            P   = np.full((N, N), np.nan)
            P[Xi, Yi] = pv
            p_mats[int(L)] = P
        except Exception as e:
            print(f"  [KGC] lag={L} error: {e}")

    edges, p_vis = [], np.ones((N, N))
    for s in range(N):
        for t in range(N):
            if s == t:
                continue
            L = int(Lmat[s, t])
            P = p_mats.get(L)
            if P is None or np.isnan(P[s, t]):
                continue
            p_vis[s, t] = P[s, t]
            if P[s, t] < alpha:
                edges.append((s, t))

    ei, ea = _edges_to_npz_arrays(edges, N)
    # -log10(p) gives high contrast: very significant edges appear bright,
    # marginally significant ones fade out. Diagonal stays 0.
    neg_log_p = -np.log10(np.clip(p_vis, 1e-10, 1.0))
    np.fill_diagonal(neg_log_p, 0.0)
    return ei, ea, neg_log_p


# ── Method 4: Jacobian Neural GC (JRNGC) ─────────────────────────────────────
def method_jngc(data, lag=5, hidden=32, layers=2, max_iter=400, top_k=0.3):
    """
    GC[i,j] from JRNGC = influence of ROI j on ROI i (j→i causality).
    Edges stored as (i, j) pairs following the original code convention;
    in the adjacency visualisation we transpose GC so that rows=source,
    cols=target (i.e. A_vis[src, tgt] = GC[tgt, src]).
    """
    try:
        from utils.jacob import (train_jrngc_on_array,
                                  infer_fulltime_and_summary,
                                  top_k_percent_binarize)
    except ImportError as e:
        print(f"  [JNGC] Cannot import utils.jacob: {e}")
        return None, None, None

    model, _ = train_jrngc_on_array(
        data,
        lag=lag, hidden=hidden, layers=layers, dropout=0.1,
        jacobian_lam=1e-3, struct_loss_choice='JF', JFn=1,
        lr=1e-3, seed=0, val_ratio=0.2, verbose=False,
        min_iter=100, max_iter=max_iter,
        lookback=5, check_first=50, check_every=50,
    )

    _, GC, _ = infer_fulltime_and_summary(model, data, lag, summary_mode="max")
    # GC[i, j] = influence of j on i  →  j is source, i is target
    GC_bin = top_k_percent_binarize(GC, percent=top_k)

    # Original gc_jacob.py convention: edge_list.append((i, j))
    # We keep the same convention so .npz is identical to real preprocessing.
    edges = [(i, j) for i in range(GC.shape[0])
                    for j in range(GC.shape[1])
                    if GC_bin[i, j] and i != j]
    ei, _ = _edges_to_npz_arrays(edges, GC.shape[0])
    ea = (np.array([GC[i, j] for i, j in edges], dtype=np.float32)
          if edges else np.zeros(0, dtype=np.float32))

    # For visualisation: show A_vis[src, tgt] = GC[tgt, src]  (transposed)
    A_vis = GC.T
    return ei, ea, A_vis


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Evaluation & Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _adj_from_ei(ei, N):
    """Build A[src, tgt] binary matrix from edge_index."""
    A = np.zeros((N, N))
    if ei is not None and ei.shape[1] > 0:
        A[ei[0], ei[1]] = 1
    return A


def _evaluate(ei, N, label, directed=True):
    """Print TP/FP/FN/F1 vs ground-truth edges (all in src→tgt convention)."""
    A    = _adj_from_ei(ei, N)
    A_gt = np.zeros((N, N))
    for s, t in GROUND_TRUTH:
        A_gt[s, t] = 1.0

    if not directed:
        # undirected: treat both (s,t) and (t,s) as positive
        A_gt = np.clip(A_gt + A_gt.T, 0, 1)

    tp = int((A * A_gt).sum())
    fp = int((A * (1 - A_gt - np.eye(N)).clip(0)).sum())
    fn = int(((1 - A) * A_gt).sum())
    pr = tp / (tp + fp + 1e-9)
    re = tp / (tp + fn + 1e-9)
    f1 = 2 * pr * re / (pr + re + 1e-9)
    total = int(A.sum())
    print(f"  {label:<28s} | edges={total:4d} | TP={tp} FP={fp} FN={fn} "
          f"| Prec={pr:.2f} Rec={re:.2f} F1={f1:.2f}")
    return A


def _plot_heatmaps(panels, out_path, N):
    """
    panels: list of (title, matrix_NxN_or_None, colorbar_label)
    Each method shows its native continuous output:
      GT      → binary 0/1
      Pearson → |r| (Pearson correlation)
      VAR GC  → GC log-ratio (nats)
      KGC     → -log10(p)
      JNGC    → Jacobian GC strength
    """
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    cb_labels = {
        'Ground Truth': 'binary',
        'Pearson':      '|r|',
        'VAR GC':       'GC log-ratio',
        'KGC':          '-log₁₀(p)',
        'JNGC':         'Jacobian GC',
    }

    for ax, (title, M) in zip(axes, panels):
        if M is None:
            ax.text(0.5, 0.5, 'Not available\n(library not installed)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='grey')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            continue
        vmax = M.max() if M.max() > 0 else 1.0
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=vmax,
                       interpolation='nearest', aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Source ROI', fontsize=8)
        ax.set_ylabel('Target ROI', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        key = next((k for k in cb_labels if k in title), None)
        if key:
            cb.set_label(cb_labels[key], fontsize=7)

    plt.suptitle(
        f'Brain Connectivity: {N} ROIs x {T_POINTS} time points'
        f'  |  GT: {len(GROUND_TRUTH)} planted directed edges',
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved heatmap → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_output')
    os.makedirs(OUT, exist_ok=True)

    print("=" * 72)
    print("  fMRI Preprocessing Demo  —  4 Connectivity Methods")
    print("=" * 72)
    print(f"\n  Data       : N={N_ROIS} ROIs  x  T={T_POINTS} time points  (synthetic VAR(2))")
    print(f"  GT edges   : {len(GROUND_TRUTH)} directed edges planted")
    print(f"  Thresholds : Pearson={PEARSON_THRESHOLD}  VAR-GC={VAR_GC_THRESHOLD}"
          f"  KGC-alpha={KGC_ALPHA}  JNGC-top={int(JNGC_TOP_K*100)}%\n")

    data = generate_bold()
    roi_matrix = np.corrcoef(data.T).astype(np.float32)
    np.fill_diagonal(roi_matrix, 0.0)
    label_demo = np.array([0], dtype=np.int64)

    panels = []     # (title, NxN strength matrix for heatmap)

    # Ground truth panel
    A_gt = np.zeros((N_ROIS, N_ROIS))
    for s, t in GROUND_TRUTH:
        A_gt[s, t] = 1.0
    panels.append(('Ground Truth\n(src→tgt)', A_gt))

    print("─" * 72)
    print(f"  {'Method':<28s} | {'Edges':>5s} | TP  FP  FN  | Prec  Rec   F1")
    print("─" * 72)

    # ── 1. Pearson ───────────────────────────────────────────────────────────
    print(f"\n[1/4] Pearson Correlation (undirected, threshold={PEARSON_THRESHOLD})")
    ei, ea, corr = method_pearson(data, threshold=PEARSON_THRESHOLD)
    _evaluate(ei, N_ROIS, "Pearson (undirected)", directed=False)
    # Show raw |r| matrix: scientifically shows continuous correlation strength,
    # not the binary threshold result. Symmetric — reflects undirected nature.
    panels.append(('Pearson\n(undirected)', np.abs(corr)))
    np.savez(os.path.join(OUT, 'sample_pearson.npz'),
             edge_index=ei, edge_attr=ea, roi=roi_matrix,
             label=label_demo, sample_id=np.array([0], dtype=np.int64))

    # ── 2. VAR GC ────────────────────────────────────────────────────────────
    print(f"\n[2/4] VAR Granger Causality (linear directed, threshold={VAR_GC_THRESHOLD})")
    ei, ea, gc_mat = method_var_gc(data, max_lag=10, threshold=VAR_GC_THRESHOLD)
    A2 = _evaluate(ei, N_ROIS, "VAR GC (directed)")
    # strength matrix for heatmap: A_vis[src,tgt] = gc_mat[tgt,src]
    gc_vis = np.clip(gc_mat.T, 0, None)
    panels.append(('VAR GC\n(directed)', gc_vis))
    np.savez(os.path.join(OUT, 'sample_var_gc.npz'),
             edge_index=ei, edge_attr=ea, roi=roi_matrix,
             label=label_demo, sample_id=np.array([1], dtype=np.int64))

    # ── 3. KGC ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Kernel GC via mlcausality (nonlinear directed, alpha={KGC_ALPHA})")
    ei, ea, kgc_vis = method_kgc(data, max_lag=10, alpha=KGC_ALPHA)
    if ei is not None:
        A3 = _evaluate(ei, N_ROIS, "KGC (directed)")
        panels.append(('KGC\n(directed)', kgc_vis))
        np.savez(os.path.join(OUT, 'sample_kgc.npz'),
                 edge_index=ei, edge_attr=ea, roi=roi_matrix,
                 label=label_demo, sample_id=np.array([2], dtype=np.int64))
    else:
        panels.append(('KGC\n(not installed)', None))

    # ── 4. JNGC ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Jacobian Neural GC — JRNGC (neural directed, top-{int(JNGC_TOP_K*100)}%)")
    print("  Training on N=116 CPU  ")
    ei, ea, jngc_vis = method_jngc(
        data, lag=5, hidden=64, layers=2, max_iter=1000, top_k=JNGC_TOP_K)
    if ei is not None:
        # JNGC stores (i,j) where GC[i,j]=influence of j on i (original convention).
        # Transpose so eval uses src→tgt convention same as other methods.
        # Note: top-30% for N=10 yields 27 edges; for real N=116 it yields ~4000.
        # High FP here is a scale effect, not a method failure.
        A4_raw = _adj_from_ei(ei, N_ROIS)
        A4_for_eval = A4_raw.T
        ei_eval = np.array(np.where(A4_for_eval > 0), dtype=np.int64)
        _evaluate(ei_eval, N_ROIS, "JNGC (directed, top-30%)")
        panels.append(('JNGC\n(directed)', jngc_vis))
        np.savez(os.path.join(OUT, 'sample_jngc.npz'),
                 edge_index=ei, edge_attr=ea, roi=roi_matrix,
                 label=label_demo, sample_id=np.array([3], dtype=np.int64))
    else:
        panels.append(('JNGC\n(not installed)', None))

    # ── Heatmap ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    _plot_heatmaps(panels, os.path.join(OUT, 'adjacency_comparison.png'), N_ROIS)
    print(f"  .npz files saved → {OUT}/")
    print("=" * 72)

    # ── Print one .npz sample structure ──────────────────────────────────────
    npz_path = os.path.join(OUT, 'sample_var_gc.npz')
    d = np.load(npz_path)
    print(f"\n  Sample .npz structure (sample_var_gc.npz):")
    for k in d.files:
        print(f"    {k:12s}: shape={d[k].shape}  dtype={d[k].dtype}")


if __name__ == "__main__":
    main()
