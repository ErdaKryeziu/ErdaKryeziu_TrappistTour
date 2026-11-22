"""Mjete per optimizimin multi-objektiv (Dita 3+).

1. TrappistRK: problemi i "mbeshtjelle" per algoritmet e pygmo-s.
   - Sekuenca kodohet me "random keys": 7 numra ne [0, 1]; renditja e tyre
     (argsort) jep permutacionin. Keshtu cdo vektor jep sekuence te vlefshme
     dhe kufizimi eq eshte gjithmone 0.
   - Kufizimi ineq trajtohet me penalizim: zgjidhja e pavlefshme merr
     objektiva shume te medhenj, pra cdo zgjidhje e vlefshme e dominon.
2. Arkivi: ruan cdo zgjidhje te vlefshme qe bie brenda kutise
   (DV < 2500, T < 4000), qe te mos humbet asnje zgjidhje e mire.
3. hypervolume(): metrika zyrtare e problemit (si combine_scores te UDP-ja).
"""
import numpy as np
import pygmo as pg

import problem as P

PEN = 1e8          # objektiv per zgjidhje te pavlefshme
REF = np.array([P.DV_REF, P.T_REF])

# Arkivi global (ploteson TrappistRK.fitness gjate optimizimit)
ARCHIVE_F = []     # [(dv, t), ...] vetem te vlefshmet brenda kutise
ARCHIVE_X = []     # vektoret perkates (34 numra, formati zyrtar)
STATS = {"evals": 0, "feasible": 0, "best_dv": np.inf}


def reset_archive():
    ARCHIVE_F.clear()
    ARCHIVE_X.clear()
    STATS.update(evals=0, feasible=0, best_dv=np.inf)


def decode(z):
    """Vektori i algoritmit (27 + 7 random keys) -> vektori zyrtar (27 + permutacion)."""
    z = np.asarray(z, dtype=float)
    x = np.empty(P.N_DIM)
    x[:P.N_CONT] = z[:P.N_CONT]
    x[P.N_CONT:] = np.argsort(z[P.N_CONT:], kind="stable")
    return x


def tight_bounds():
    """Kufij te ngushtuar me njohuri te domenit (hapesire kerkimi me e vogel).
    Arsyeja: planetet e TRAPPIST-1 kane periudha 1.5-19 dite, ndaj legjet
    ndermjet planeteve jane te shkurtra; vetem hyrja nga 10 AU zgjat shume.
    Kufijte jane nenbashkesi e kufijve zyrtare, pra cdo zgjidhje mbetet e lejuar.
      T0 (hyrja ne sistem)  : [1000, 2000] dite  (zyrtar: [5, 2000])
      T e legjeve te tjera  : [5, 400] dite      (zyrtar: [5, 2000])
      rp/rP (lartesia flyby): [1.1, 10]          (zyrtar: [1.1, 100])
    """
    lb = P.LB[:P.N_CONT].copy()
    ub = P.UB[:P.N_CONT].copy()
    lb[2], ub[2] = 1000.0, 2000.0
    for k in range(6):
        ub[4 + 4 * k] = 10.0      # rp/rP
        ub[6 + 4 * k] = 400.0     # T e legut
    return lb, ub


class TrappistRK:
    """Problem pa kufizime (per pygmo): 34 numra te vazhdueshem, 2 objektiva.
    tight=True perdor kufijte e ngushtuar (shih tight_bounds)."""

    def __init__(self, tight=False):
        self.tight = tight

    def get_bounds(self):
        if self.tight:
            lbc, ubc = tight_bounds()
        else:
            lbc, ubc = P.LB[:P.N_CONT], P.UB[:P.N_CONT]
        lb = np.concatenate([lbc, np.zeros(P.N_INT)])
        ub = np.concatenate([ubc, np.ones(P.N_INT)])
        return (lb, ub)

    def get_nobj(self):
        return 2

    def get_name(self):
        return "Trappist Tour (random keys + penalty)"

    def fitness(self, z):
        x = decode(z)
        dv, t, eq, ineq, status = P.evaluate(x)
        STATS["evals"] += 1
        if P.is_feasible(eq, ineq, status):
            STATS["feasible"] += 1
            STATS["best_dv"] = min(STATS["best_dv"], dv)
            if dv < P.DV_REF and t < P.T_REF:
                ARCHIVE_F.append((dv, t))
                ARCHIVE_X.append(x)
            return [dv, t]
        # e pavlefshme: sa me e madhe shkelja, aq me keq (ndihmon algoritmin)
        v = PEN + min(max(ineq, 0.0), 1e14) / 1e6
        return [v, v]


def encode(x):
    """E kunderta e decode: vektori zyrtar -> vektori i algoritmit.
    Planeti ne pozicionin k merr celesin (k + 0.5) / 7, keshtu argsort jep
    perseri te njejten sekuence."""
    x = np.asarray(x, dtype=float)
    z = np.empty(P.N_DIM)
    z[:P.N_CONT] = x[:P.N_CONT]
    seq = x[P.N_CONT:].astype(int)
    keys = np.empty(P.N_INT)
    for pos, planet in enumerate(seq):
        keys[planet] = (pos + 0.5) / P.N_INT
    z[P.N_CONT:] = keys
    return z


def warm_start_population(prob, size, seed, sigma=0.02):
    """Popullata fillestare: zgjidhja shembull e ESA-s + (size-1) variante te saj
    me zhurme gausiane (sigma = pjese e gjeresise se kufirit) ne pjesen e
    vazhdueshme. Sekuenca e planeteve mbetet ajo e shembullit."""
    import pygmo as pg
    rng = np.random.default_rng(seed)
    lb, ub = (np.asarray(b, dtype=float) for b in prob.get_bounds())
    z0 = np.clip(encode(P.udp.example()), lb, ub)
    pop = pg.population(prob, size=0, seed=seed)
    pop.push_back(z0)
    for _ in range(size - 1):
        z = z0.copy()
        z[:P.N_CONT] += rng.normal(0.0, sigma, P.N_CONT) * (ub - lb)[:P.N_CONT]
        pop.push_back(np.clip(z, lb, ub))
    return pop


def prune_archive(limit=5000):
    """Kur arkivi rritet shume, mban vetem pikat jo te dominuara
    (hypervolume nuk ndryshon, sepse varet vetem prej tyre)."""
    if len(ARCHIVE_F) <= limit:
        return
    F = np.asarray(ARCHIVE_F, dtype=float)
    keep = pg.non_dominated_front_2d(F)
    newF = [ARCHIVE_F[i] for i in keep]
    newX = [ARCHIVE_X[i] for i in keep]
    ARCHIVE_F[:] = newF
    ARCHIVE_X[:] = newX


def pareto_front(points):
    """Kthen pikat jo te dominuara (2 objektiva), te renditura sipas DV."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    idx = pg.non_dominated_front_2d(pts)
    front = pts[idx]
    return front[np.argsort(front[:, 0])]


def hypervolume(points, ref=REF):
    """Hypervolume me pike reference (2500, 4000). Llogariten vetem pikat
    qe jane rreptesisht brenda kutise (si te combine_scores zyrtare)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    pts = pts[(pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])]
    if len(pts) == 0:
        return 0.0
    return float(pg.hypervolume(pts).compute(ref))
