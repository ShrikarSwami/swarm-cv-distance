"""Shared metrics for the ML swarm-reconstruction track (frozen on acceptance).

One implementation, both tracks consume it identically. `ml_evaluate` and
`geometric_evaluate` are thin wrappers over the shared core `evaluate` so that
identical inputs always produce byte-identical dicts (frozen test
`test_metric_path_identity`).

Contract: `docs/superpowers/specs/ml_contracts.md` §1. Definitions are pinned in
§1.4; this module implements exactly those, and nothing else.

Interpretation notes (documented here because §1.4 leaves these points open):
- **Tau-restricted Hungarian** (`precision`/`recall`/`f1`, `n_matched`): pairs
  farther than `tau` are disallowed by setting their cost to a large finite
  penalty `BIG = 1 + d_max * min(K, N)`, so `linear_sum_assignment` first
  minimises the number of disallowed pairs (maximises in-threshold matches) and
  then minimises total matched distance. `np.inf` cannot be used directly:
  scipy raises "cost matrix is infeasible" when a row has no allowed entry.
- **median_err_m**: median of the unrestricted min-cost Hungarian matched-pair
  distances -- the geometric track's frozen `_median_matched_error` convention.
  NaN when there are no matched pairs (K == 0 or N == 0 after the NaN drop).
- **chamfer_m**: NaN when either point set is empty.
- **AP@tau**: greedy matching in descending confidence order (nearest unmatched
  truth within `tau`), integrated with the canonical VOC-style precision
  interpolation. `confidence=None` is a single operating point: `ap =
  precision@tau`.

Everything is numpy/scipy only; no torch, no bpy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

DEFAULT_TAUS = (0.5, 1.0, 2.0, 5.0)  # metres; frozen (spec §6)

__all__ = ["DEFAULT_TAUS", "evaluate", "ml_evaluate", "geometric_evaluate",
           "ap_at_tau"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(pred, confidence=None):
    """Cast to float64 and drop `pred` rows containing any non-finite value.

    Returns (pred_eff, conf_eff). `confidence` is aligned to `pred` BEFORE the
    drop, so the surviving confidence values correspond exactly to the
    surviving rows (contract §1.2).
    """
    pred = np.asarray(pred, dtype=np.float64)
    finite = np.isfinite(pred).all(axis=1)
    pred_eff = pred[finite]
    conf_eff = None
    if confidence is not None:
        conf_eff = np.asarray(confidence, dtype=np.float64)[finite]
    return pred_eff, conf_eff


def _hungarian_match(pred, true, tau):
    """Hungarian min-cost assignment restricted to pairs with distance <= tau.

    Disallowed pairs (distance > tau) get a large finite penalty, so the solver
    first maximises the number of in-threshold matches, then minimises total
    matched distance. Returns the number of in-threshold matched pairs.
    """
    K, N = pred.shape[0], true.shape[0]
    if K == 0 or N == 0:
        return 0
    D = cdist(pred, true)  # (K, N) pairwise Euclidean metres, all finite
    d_max = float(D.max())
    # Any assignment using a disallowed pair costs >= BIG, while any fully
    # allowed assignment costs <= d_max * min(K, N) < BIG, so minimising the
    # count of disallowed pairs dominates. Exact whenever d_max * min(K, N) is
    # finite (all realistic world-metre data).
    BIG = 1.0 + d_max * min(K, N)
    cost = np.where(D <= tau, D, BIG)
    rows, cols = linear_sum_assignment(cost)
    return int((D[rows, cols] <= tau).sum())


def _median_matched_error(pred, true):
    """Median of unrestricted min-cost Hungarian matched-pair distances.

    Matches the geometric track's frozen reference `_median_matched_error`
    (tests/test_predictions_ml.py). NaN when there are no matched pairs.
    """
    K, N = pred.shape[0], true.shape[0]
    if K == 0 or N == 0:
        return float("nan")
    D = cdist(pred, true)
    rows, cols = linear_sum_assignment(D)
    return float(np.median(D[rows, cols]))


def _chamfer(pred, true):
    """Symmetric two-sided chamfer over non-dropped rows (contract §1.4)."""
    K, N = pred.shape[0], true.shape[0]
    if K == 0 or N == 0:
        return float("nan")
    D = cdist(pred, true)  # (K, N)
    true_to_pred = D.min(axis=0).mean()  # (1/N) sum_true min_pred ||t - p||
    pred_to_true = D.min(axis=1).mean()  # (1/K) sum_pred min_true ||p - t||
    return float(true_to_pred + pred_to_true)


def _voc_ap(recall, precision):
    """Canonical VOC-style PR integration with backward precision interpolation.

    Precision at each recall level is the max precision at any recall >= it; the
    AP is the area under that interpolated curve (VOC 2010+ convention).
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ap_at_tau(pred, true, tau, confidence=None):
    """Confidence-aware average precision at one tau (spec §6, contract §1.4).

    `confidence=None` -> a single operating point: `ap = precision@tau` after
    matching all predictions. Otherwise predictions are greedily matched in
    descending confidence order (nearest unmatched truth within `tau`), and the
    resulting TP/FP sequence is integrated VOC-style over all predictions.
    Non-finite `pred` rows are dropped before matching (never matched).
    """
    pred_eff, conf_eff = _clean(pred, confidence)
    K, N = pred_eff.shape[0], true.shape[0]
    tau = float(tau)
    if K == 0 or N == 0:
        return 0.0
    if confidence is None:
        n_matched = _hungarian_match(pred_eff, true, tau)
        return float(n_matched / K)
    D = cdist(pred_eff, true)  # (K, N)
    order = np.argsort(-conf_eff, kind="stable")  # descending confidence
    used = np.zeros(N, dtype=bool)
    tp = np.zeros(K, dtype=bool)
    for i, pi in enumerate(order):
        dists = D[pi]
        cand = np.flatnonzero((dists <= tau) & ~used)
        if cand.size == 0:
            tp[i] = False  # false positive
        else:
            j = cand[int(np.argmin(dists[cand]))]  # nearest unmatched truth
            used[j] = True
            tp[i] = True
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(~tp)
    recall = cum_tp / float(N)
    precision = cum_tp / (cum_tp + cum_fp)
    return _voc_ap(recall, precision)


def evaluate(pred, true, taus=DEFAULT_TAUS, confidence=None):
    """Shared core -- the single code path both tracks execute.

    Returns the frozen dict schema (contract §1.3). All values are Python
    float/int, JSON-serializable. `pred` rows with any non-finite value are
    dropped before matching and do not count toward `n_pred`; `confidence` is
    aligned to `pred` before that drop.
    """
    pred_eff, conf_eff = _clean(pred, confidence)
    true_arr = np.asarray(true, dtype=np.float64)
    n_pred = int(pred_eff.shape[0])
    n_true = int(true_arr.shape[0])

    per_tau = {}
    aps = []
    for tau in taus:
        tau = float(tau)
        n_matched = _hungarian_match(pred_eff, true_arr, tau)
        precision = float(n_matched / n_pred) if n_pred > 0 else 0.0
        recall = float(n_matched / n_true) if n_true > 0 else 0.0
        if precision + recall > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0
        ap = float(ap_at_tau(pred_eff, true_arr, tau, confidence=conf_eff))
        per_tau[tau] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "n_matched": int(n_matched),
        }
        aps.append(ap)

    mAP = float(np.mean(aps)) if aps else float("nan")
    median_err = _median_matched_error(pred_eff, true_arr)
    chamfer = _chamfer(pred_eff, true_arr)

    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "per_tau": per_tau,
        "mAP": mAP,
        "median_err_m": median_err,
        "chamfer_m": chamfer,
        "count_err": int(n_pred - n_true),
    }


def ml_evaluate(pred, true, taus=DEFAULT_TAUS, confidence=None):
    """ML-track entry point. Delegates to the shared core `evaluate`."""
    return evaluate(pred, true, taus=taus, confidence=confidence)


def geometric_evaluate(pred, true, taus=DEFAULT_TAUS, confidence=None):
    """Geometric-track entry point. Delegates to the shared core `evaluate`."""
    return evaluate(pred, true, taus=taus, confidence=confidence)
