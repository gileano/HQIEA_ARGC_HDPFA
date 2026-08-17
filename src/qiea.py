"""
Qubit-encoded, decomposition-guided QIEA with adaptive rotation-gate control.

Representation: each individual is a vector of real qubit angles theta in [0, pi/2],
one per decision variable, implicitly encoding amplitudes (alpha, beta) = (cos theta,
sin theta) with |alpha|^2 + |beta|^2 = 1 by construction (a real-amplitude restriction
of the general qubit state, sufficient since only relative ordering/interpolation of
amplitudes is used -- same simplification implicitly made by the rotation gate in the
first paper's Eq. 3-4).

Two decode modes share this representation:
  - "continuous": x_i = xl_i + sin(theta_i)^2 * (xu_i - xl_i)   -- for ZDT/DTLZ/WFG.
  - "permutation": tour = argsort(theta)                        -- for the CVRP.
    Sorting real numbers cannot produce duplicates, so this decode needs NO repair step.
    That directly targets the scalability failure mode reported in the first paper
    ("the potential problem of the QIEA on a high number of cities is the repair step
    ... the algorithm loses its diversity on high number of cities") -- the binary/
    Gray-code measurement + duplicate-repair pipeline is replaced outright.

Many-objective scalability: MOEA/D-style decomposition. A fixed set of H uniformly
spread weight vectors (Das-Dennis simplex lattice) partitions the objective space into
H Tchebycheff subproblems, one individual per subproblem, each looking only at a small
neighborhood of nearby weight vectors -- this is the paper1.txt-requested "hybridize
with a decomposition scheme (MOEA/D-style) ... for scalability to many objectives".

Adaptive rotation gate: each qubit is rotated toward the corresponding qubit of the best
neighbor under its own subproblem's scalarizing function, by a step size that decays
over generations (exploration -> exploitation) but is boosted back up whenever the
population's angular diversity collapses below a threshold (stagnation escape), together
with a temporary bump in mutation probability -- the "angle schedule informed by
convergence diversity" requested in the plan.

CAUTION -- this stagnation-escape branch (the "ARGC" in the project/algorithm name) was
found to be almost completely inert in every result reported so far, and actively
harmful when forced to fire more often. A diagnostic (diag_qiea_stagnation.py) showed
the boost triggers on 0% of generations on A-n32-k5/E-n101-k8 and ~1% on route2_199 at
the default diversity_stagnation_tol=1e-3, across 80-generation runs -- the windowed
diversity range that tol is compared against (diag_qiea_diversity_trace.py) has a floor
around 0.0017-0.003 on all three instances, i.e. tol is ~2-3x stricter than the metric
ever reaches. Raising tol so the mechanism actually engages (tune_qiea_stagnation_scaling.py,
7-point grid x 3 instances x 5 seeds) made hypervolume DECREASE monotonically with every
single step up, on all three instances independently -- the current near-total inactivity
is accidentally optimal, not a bug. A 15-seed confirm of the mildest deviation (tol=0.002,
2x default) was directionally worse on all three instances (-1.9% to -4.1%) though not
individually significant there; the full-grid monotonic pattern replicating identically
across three independent instances is the stronger evidence. Net effect: whatever this
mechanism was intended to do, forcing it to trigger more is not a viable lever for closing
QIEA's gap to RVEA/MOEA-D. diversity_stagnation_tol, diversity_window, and the boost
multipliers (rotation_boost_multiplier=3.0, mutation_boost_multiplier=5.0, now exposed
as constructor params) are therefore left at their existing defaults.

Follow-up (tune_qiea_boost_magnitude.py/_confirm.py) tried to separate two
explanations for the above -- (1) triggering the escape at all is harmful regardless
of response size, vs (2) triggering is fine but the current 3x/5x response is simply
too disruptive -- by fixing tol=0.003 (confirmed to reliably engage the mechanism) and
sweeping the boost multipliers themselves from a no-op (1.0, 1.0) up to the default
(3.0, 5.0). A first 5-seed screen suggested a clean answer (even the mildest tested
boost, 1.1x/1.2x, underperformed the no-op on all three instances), but a 15-seed
confirm of exactly that comparison did NOT hold up: only A-n32-k5 stayed significantly
worse (-5.1%, p=0.015 -- consistent with this instance's known extreme sensitivity to
any hyperparameter drift, see the theta_max clamping incident below), E-n101-k8 was a
non-significant -1.1%, and route2_199's direction flipped to a non-significant +4.5%.
So magnitude vs. frequency could not be cleanly separated -- the mild-boost-is-bad
signal from the small-seed screen was itself mostly noise, the same lesson as the
tol confirm's small-deviation case above. No default changed; this remains a genuinely
open, instance-dependent question rather than a resolved one.

Rotation step scales with instance size (permutation decode only): theta_max/theta_min
used to be fixed absolute radians, tuned once on a 31-city instance. But the permutation
decode is tour = argsort(theta), and the average gap between adjacent sorted theta values
is ~(pi/2)/n_var -- so a fixed absolute step is a shrinking multiple of that gap as n_var
grows, and becomes far more disruptive relative to local tour structure on larger
instances (confirmed by hypervolume regressions on E-n101-k8/route2_199 that a fixed
theta_min tuned on a 32-node instance did not fix). theta_min/theta_max therefore default
to a multiple of that gap when not given explicitly, so late-run exploitation steps stay
proportionally fine-grained regardless of n_var. Validated (paired Wilcoxon, matched
population): +15-18% hypervolume on route2_199 (n_var=198, p<0.05), a smaller
non-significant gain on E-n101-k8 (n_var=100); both bounds are clamped (min(0.1, ...),
min(0.35, ...)) so the formula reduces EXACTLY to the old fixed defaults at n_var<=31
(A-n32-k5, where they were originally tuned) -- an earlier unclamped version let
theta_max drift 0.35->0.3547 there, which was enough noise to regress QIEA's ranking
against NSGA-II/SPEA2 in a 5-seed run_experiment.py comparison. Continuous decode
(ZDT/DTLZ/WFG) keeps the old fixed defaults -- there is no argsort step, so this
mechanism doesn't apply.

Mutation rate scales with instance size (permutation decode only): quantum_mutation
flips each gene independently with probability mutation_prob, so the expected number
of mutated genes per individual is mutation_prob * n_var. A fixed absolute
mutation_prob=0.05 is therefore ~1.5 expected flips on a 31-city tour but ~10 on a
198-city tour -- the same "fixed absolute parameter, more disruptive as n_var grows"
mechanism as the rotation-step finding above, and it turned out to be a much bigger
effect: an OFAT screen on E-n101-k8/route2_199 found mutation_prob=0.01 alone worth
+49-70% hypervolume there (paired Wilcoxon, matched population, p<0.01), while the
same value regresses A-n32-k5. mutation_prob therefore defaults to min(0.05, 1.55 /
n_var) when not given explicitly -- clamped so it reduces EXACTLY to the old 0.05 at
n_var<=31 (A-n32-k5), same clamping rationale as theta_min/theta_max: this
representation was shown sensitive to even ~1% drift in a scale-family parameter at
n_var=31. neighborhood_size was screened alongside mutation_prob and looked like it
helped too, but an isolated confirm run showed that was an OFAT interaction artifact
of the screen, not a real effect -- with mutation_prob fixed, a larger
neighborhood_size gave 0% change on E-n101-k8 but +30% on route2_199, an
instance-specific result with no clean size-based rule, so it is NOT included in any
default here (would violate the population/hyperparameter-fairness rule that keeps
QIEA comparable to the baselines -- see tune_qiea_matched.py). Continuous decode
keeps the old fixed 0.05 -- no evidence this mechanism applies there either.

A dedicated follow-up grid sweep (tune_qiea_neighborhood_scaling.py /
tune_qiea_neighborhood_confirm.py) confirmed neighborhood_size has no size-tied
formula at all, not even one linked to n_var non-monotonically: with mutation_prob/
theta bounds held at their auto-scaled defaults, A-n32-k5's optimum stayed at the
default 10 (larger values monotonically hurt), route2_199's optimum was a modest
nb=20 (+34.8% HV, Wilcoxon p=0.018, 15 seeds -- a real, reproduced effect matching
the earlier confirm run), and E-n101-k8's screened optimum was nb=126=H (i.e. fully
unrestricted mating across all subproblems, +20.7% HV at 15 seeds but p=0.169, not
significant despite the large point estimate). Since H is identical across all three
instances here (n_partitions=5 is fixed for baseline fairness), this rules out both
"scales with n_var" and "scales with H" as an explanation -- route2_199 has the
largest n_var but wants the smallest neighborhood boost, the opposite of what an
n_var-scaling formula would predict. The effect looks tied to per-instance Pareto
front geometry, not any measurable size parameter, so neighborhood_size is left at
its fixed default (10) rather than shipping an unjustified formula.

Plateau-gated diversity reinjection (restart_patience/restart_fraction): a
generation-trace diagnostic (diag_qiea_generation_trace.py, see paper1.txt section
14) found QIEA hits a hard hypervolume plateau early in a 500-generation run and
then NEVER improves again -- e.g. frozen bit-for-bit from generation 60 onward on
A-n32-k5 -- while every pymoo baseline keeps climbing steadily to generation 500.
Cause: the population is fixed at H individuals (one per MOEA/D subproblem); once
every individual converges to its neighborhood's best guide, the diversity-
stagnation escape above almost never fires (it was already shown near-dead in
practice), so nothing reinjects novelty and the archive stops growing regardless
of how many generations remain. Fix: track generations-since-the-archive-last-grew,
and once that exceeds restart_patience, reseed a restart_fraction of the H
subproblems to fresh random theta (elitist -- untouched individuals keep their
converged state; see _do_restart). A two-stage grid screen (tune_qiea_restart.py,
tune_qiea_restart_finegrid.py; patience in 5-200, fraction in 0.1-0.5, 5 seeds,
A-n32-k5/E-n101-k8/route2_199) found EVERY tested config beat the restart-disabled
control on every instance -- the opposite of the diversity_stagnation_tol result
above, where triggering MORE was uniformly worse. The per-instance-optimal patience
did not agree (10/5/15 respectively -- no clean size-based formula, same situation
as neighborhood_size), so restart_patience=10/restart_fraction=0.5 was chosen as
the single most robust candidate (top-3 by rank on every instance screened, never
an outlier) rather than any per-instance value, for the same baseline-fairness
reason as every other default in this file. Confirmed with 20 seeds across ALL
SEVEN CVRP instances (tune_qiea_restart_confirm.py): +27.7% to +66.0% hypervolume,
Wilcoxon p < 0.005 on every single instance -- by far the largest and most general
effect found across this entire tuning investigation (sections 8-14 combined).
Shipped as the new default (previously restart_patience=None, i.e. disabled).
Continuous decode (ZDT/DTLZ/WFG) has not been checked for an analogous plateau --
open, see paper1.txt section 15.
"""
import numpy as np


def tchebycheff(f, w, z_ideal, eps=1e-6):
    return np.max(np.maximum(w, eps) * np.abs(f - z_ideal))


class QIEA:
    @staticmethod
    def _default_theta_bounds(decode, n_var):
        """Gap-relative rotation-step defaults for permutation decode; see module docstring."""
        if decode != "permutation":
            return 0.1, 0.35
        gap = (np.pi / 2) / n_var
        return min(0.1, 2.0 * gap), min(0.35, 7.0 * gap)

    @staticmethod
    def _default_mutation_prob(decode, n_var):
        """Gap-relative mutation-rate default for permutation decode; see module docstring."""
        if decode != "permutation":
            return 0.05
        return min(0.05, 1.55 / n_var)

    def __init__(
        self,
        problem,
        decode="permutation",
        n_partitions=None,
        neighborhood_size=10,
        theta_max=None,
        theta_min=None,
        mutation_prob=None,
        diversity_window=10,
        diversity_stagnation_tol=1e-3,
        rotation_boost_multiplier=3.0,
        mutation_boost_multiplier=5.0,
        restart_patience=10,
        restart_fraction=0.5,
        seed=0,
    ):
        self.problem = problem
        self.decode_mode = decode
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self.xl = np.asarray(problem.xl, dtype=float) if problem.xl is not None else np.zeros(self.n_var)
        self.xu = np.asarray(problem.xu, dtype=float) if problem.xu is not None else np.full(self.n_var, self.n_var - 1)
        self.rng = np.random.default_rng(seed)

        if n_partitions is None:
            n_partitions = {2: 99, 3: 13, 4: 7, 5: 5, 6: 4}.get(self.n_obj, 3)
        from pymoo.util.ref_dirs import get_reference_directions

        self.weights = get_reference_directions("das-dennis", self.n_obj, n_partitions=n_partitions)
        self.H = len(self.weights)
        self.T = min(neighborhood_size, self.H)
        dists = np.linalg.norm(self.weights[:, None, :] - self.weights[None, :, :], axis=-1)
        self.neighbors = np.argsort(dists, axis=1)[:, : self.T]

        if theta_max is None or theta_min is None:
            auto_min, auto_max = self._default_theta_bounds(decode, self.n_var)
            theta_min = auto_min if theta_min is None else theta_min
            theta_max = auto_max if theta_max is None else theta_max
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.mutation_prob = mutation_prob if mutation_prob is not None else self._default_mutation_prob(decode, self.n_var)
        self.diversity_window = diversity_window
        self.diversity_stagnation_tol = diversity_stagnation_tol
        self.rotation_boost_multiplier = rotation_boost_multiplier
        self.mutation_boost_multiplier = mutation_boost_multiplier
        self.restart_patience = restart_patience
        self.restart_fraction = restart_fraction

        self.theta = self.rng.uniform(0.0, np.pi / 2, size=(self.H, self.n_var))
        self.F = None
        self.z_ideal = np.full(self.n_obj, np.inf)
        self.history_diversity = []
        self.archive_X, self.archive_F = [], []
        self._best_archive_size = 0
        self._gens_since_growth = 0
        self.n_restarts = 0

    # -- decode / evaluate --------------------------------------------------
    def decode(self, theta):
        if self.decode_mode == "continuous":
            return self.xl + np.sin(theta) ** 2 * (self.xu - self.xl)
        return np.argsort(theta)

    def evaluate_population(self, theta_pop):
        X = np.array([self.decode(t) for t in theta_pop])
        out = {}
        self.problem._evaluate(X, out)
        return X, out["F"]

    # -- operators ------------------------------------------------------------
    def diversity(self):
        centroid = self.theta.mean(axis=0)
        return float(np.mean(np.abs(self.theta - centroid)))

    def rotation_angle(self, gen, max_gen):
        progress = gen / max(max_gen, 1)
        base = self.theta_max - (self.theta_max - self.theta_min) * progress
        if len(self.history_diversity) >= self.diversity_window:
            recent = self.history_diversity[-self.diversity_window :]
            if (max(recent) - min(recent)) < self.diversity_stagnation_tol:
                return min(self.theta_max, base * self.rotation_boost_multiplier), True
        return base, False

    def rotate_toward(self, theta, guide_theta, step):
        direction = np.sign(guide_theta - theta)
        return np.clip(theta + step * direction, 0.0, np.pi / 2)

    def quantum_crossover(self, theta_a, theta_b):
        mask = self.rng.random(self.n_var) < 0.5
        child = np.where(mask, theta_a, theta_b)
        return child

    def quantum_mutation(self, theta, prob):
        mask = self.rng.random(self.n_var) < prob
        theta = theta.copy()
        theta[mask] = np.pi / 2 - theta[mask]  # continuous analog of a qubit flip
        return theta

    # -- main loop --------------------------------------------------------------
    def run(self, n_gen, verbose=False):
        _, F = self.evaluate_population(self.theta)
        self.F = F
        self.z_ideal = np.minimum(self.z_ideal, F.min(axis=0))
        self._update_archive(self.theta, F)

        for gen in range(n_gen):
            step, boosted = self.rotation_angle(gen, n_gen)
            mut_p = self.mutation_prob * (self.mutation_boost_multiplier if boosted else 1.0)

            new_theta = np.empty_like(self.theta)
            for i in range(self.H):
                nbrs = self.neighbors[i]
                scal = [tchebycheff(self.F[j], self.weights[i], self.z_ideal) for j in nbrs]
                guide = nbrs[int(np.argmin(scal))]

                mate = nbrs[self.rng.integers(len(nbrs))]
                child = self.quantum_crossover(self.theta[i], self.theta[mate])
                child = self.rotate_toward(child, self.theta[guide], step)
                child = self.quantum_mutation(child, mut_p)
                new_theta[i] = child

            _, new_F = self.evaluate_population(new_theta)
            self.z_ideal = np.minimum(self.z_ideal, new_F.min(axis=0))

            for i in range(self.H):
                for j in self.neighbors[i]:
                    if tchebycheff(new_F[i], self.weights[j], self.z_ideal) <= tchebycheff(
                        self.F[j], self.weights[j], self.z_ideal
                    ):
                        self.theta[j] = new_theta[i]
                        self.F[j] = new_F[i]

            self.history_diversity.append(self.diversity())
            self._update_archive(self.theta, self.F)

            if len(self.archive_F) > self._best_archive_size:
                self._best_archive_size = len(self.archive_F)
                self._gens_since_growth = 0
            else:
                self._gens_since_growth += 1
            if self.restart_patience is not None and self._gens_since_growth >= self.restart_patience:
                self._do_restart()
                self._gens_since_growth = 0

            if verbose and gen % max(1, n_gen // 10) == 0:
                print(f"gen {gen:4d}  step={step:.4f}  boosted={boosted}  |archive|={len(self.archive_F)}")

        return self._pareto_result()

    def _do_restart(self):
        """Plateau-gated diversity reinjection: reseed a fraction of subproblems to
        fresh random theta when the archive has stopped growing for restart_patience
        generations -- elitist (untouched individuals keep their converged state), see
        paper1.txt section 14g/15 for the rationale (the fixed-tol diversity-stagnation
        escape essentially never fires once truly converged, per section 12's finding)."""
        k = max(1, int(round(self.restart_fraction * self.H)))
        idx = self.rng.choice(self.H, size=k, replace=False)
        self.theta[idx] = self.rng.uniform(0.0, np.pi / 2, size=(k, self.n_var))
        _, F_new = self.evaluate_population(self.theta[idx])
        self.F[idx] = F_new
        self.z_ideal = np.minimum(self.z_ideal, F_new.min(axis=0))
        self.n_restarts += 1

    def _update_archive(self, theta_pop, F):
        X = np.array([self.decode(t) for t in theta_pop])
        self.archive_X.extend(list(X))
        self.archive_F.extend(list(F))
        self._prune_archive()

    def _prune_archive(self, cap=500):
        F = np.array(self.archive_F)
        _, unique_idx = np.unique(F, axis=0, return_index=True)
        self.archive_X = [self.archive_X[i] for i in unique_idx]
        self.archive_F = [self.archive_F[i] for i in unique_idx]

        F = np.array(self.archive_F)
        keep = _non_dominated_mask(F)
        self.archive_X = [x for x, k in zip(self.archive_X, keep) if k]
        self.archive_F = [f for f, k in zip(self.archive_F, keep) if k]
        if len(self.archive_F) > cap:
            idx = self.rng.choice(len(self.archive_F), size=cap, replace=False)
            self.archive_X = [self.archive_X[i] for i in idx]
            self.archive_F = [self.archive_F[i] for i in idx]

    def _pareto_result(self):
        class Result:
            pass

        res = Result()
        res.X = np.array(self.archive_X)
        res.F = np.array(self.archive_F)
        return res


def _non_dominated_mask(F, chunk=500):
    """Vectorized non-dominated filter; chunked over axis 0 to bound memory (O(chunk*n*m))."""
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = F[start:end, None, :]  # (b, 1, m)
        le = np.all(block >= F[None, :, :], axis=2)  # (b, n): F[j] <= block[i]
        lt = np.any(block > F[None, :, :], axis=2)  # F[j] < block[i] somewhere
        dom_by_any = le & lt
        idx = np.arange(start, end)
        dom_by_any[np.arange(end - start), idx] = False  # exclude self
        dominated[start:end] = np.any(dom_by_any, axis=1)
    return ~dominated


if __name__ == "__main__":
    from pathlib import Path

    from problem import CVRPInstance, CVRPProblem

    inst = CVRPInstance.from_file(Path(__file__).resolve().parent.parent / "data" / "processed" / "route2_199.json")
    problem = CVRPProblem(inst)
    algo = QIEA(problem, decode="permutation", n_partitions=4, seed=1)
    res = algo.run(n_gen=60, verbose=True)
    print("final archive size:", len(res.F))
    print(res.F[: min(5, len(res.F))])
