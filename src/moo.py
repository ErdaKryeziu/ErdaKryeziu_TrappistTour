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


class TrappistRK:
    """Problem pa kufizime (per pygmo): 34 numra te vazhdueshem, 2 objektiva."""

    def get_bounds(self):
        lb = np.concatenate([P.LB[:P.N_CONT], np.zeros(P.N_INT)])
        ub = np.concatenate([P.UB[:P.N_CONT], np.ones(P.N_INT)])
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
