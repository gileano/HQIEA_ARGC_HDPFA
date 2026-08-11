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
"""
import numpy as np


def tchebycheff(f, w, z_ideal, eps=1e-6):
    return np.max(np.maximum(w, eps) * np.abs(f - z_ideal))


class QIEA:
    def __init__(
        self,
        problem,
        decode="permutation",
        n_partitions=None,
        neighborhood_size=10,
        theta_max=0.35,
        theta_min=0.02,
        mutation_prob=0.05,
        diversity_window=10,
        diversity_stagnation_tol=1e-3,
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

        self.theta_max = theta_max
        self.theta_min = theta_min
        self.mutation_prob = mutation_prob
        self.diversity_window = diversity_window
        self.diversity_stagnation_tol = diversity_stagnation_tol

        self.theta = self.rng.uniform(0.0, np.pi / 2, size=(self.H, self.n_var))
        self.F = None
        self.z_ideal = np.full(self.n_obj, np.inf)
        self.history_diversity = []
        self.archive_X, self.archive_F = [], []

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
                return min(self.theta_max, base * 3.0), True
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
            mut_p = self.mutation_prob * (5.0 if boosted else 1.0)

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
            if verbose and gen % max(1, n_gen // 10) == 0:
                print(f"gen {gen:4d}  step={step:.4f}  boosted={boosted}  |archive|={len(self.archive_F)}")

        return self._pareto_result()

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
