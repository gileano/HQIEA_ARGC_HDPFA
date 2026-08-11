"""
Pareto-front quality indicators and cross-algorithm statistical tests.

hypervolume / IGD / IGD+ / spacing come straight from pymoo. "Spread" (Deb's Delta)
is not in pymoo, so it is implemented here directly. Wilcoxon signed-rank (paired,
two algorithms on one instance) and Friedman (many algorithms across instances) come
from scipy.stats, matching the statistical protocol in the plan.
"""
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.spacing import SpacingIndicator
from scipy import stats


def spread(F):
    """Deb's Delta spread indicator: extent + evenness of consecutive-distance spacing."""
    if len(F) < 2:
        return float("nan")
    F = np.asarray(F)
    n_obj = F.shape[1]
    d_extreme = 0.0
    for m in range(n_obj):
        order = np.argsort(F[:, m])
        d_extreme += np.linalg.norm(F[order[0]] - F[order[-1]])

    dists = []
    for i in range(len(F)):
        others = np.delete(F, i, axis=0)
        dists.append(np.min(np.linalg.norm(others - F[i], axis=1)))
    dists = np.array(dists)
    d_mean = dists.mean()
    d_sum = np.sum(np.abs(dists - d_mean))
    return float((d_extreme + d_sum) / (d_extreme + len(F) * d_mean))


def all_indicators(F, ref_point, pareto_front=None):
    F = np.asarray(F)
    out = {
        "hypervolume": HV(ref_point=ref_point)(F),
        "spacing": SpacingIndicator()(F),
        "spread": spread(F),
        "n_solutions": len(F),
    }
    if pareto_front is not None:
        out["igd"] = IGD(pareto_front)(F)
        out["igd_plus"] = IGDPlus(pareto_front)(F)
    return out


def wilcoxon_test(scores_a, scores_b):
    """Paired Wilcoxon signed-rank test, e.g. per-run hypervolume of alg A vs alg B."""
    stat, p = stats.wilcoxon(scores_a, scores_b)
    return {"statistic": float(stat), "p_value": float(p)}


def friedman_test(score_matrix):
    """score_matrix: shape (n_runs_or_instances, n_algorithms)."""
    stat, p = stats.friedmanchisquare(*[score_matrix[:, j] for j in range(score_matrix.shape[1])])
    return {"statistic": float(stat), "p_value": float(p)}
