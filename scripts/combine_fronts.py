"""Bashkon frontet e te gjitha ekzekutimeve, zgjedh 100 zgjidhjet qe japin
hypervolume me te madh (kufiri i ESA-s: max 100 zgjidhje per dergese)
dhe vizaton krahasimin e algoritmeve.

Perdorimi:  python scripts/combine_fronts.py
Nuk kerkon pykep/pygmo (vetem numpy + matplotlib), punon edhe ne Windows.

Dalja (ne results/):
  final_front_100.csv       100 zgjidhjet e zgjedhura (dv, t, x0..x33)
  final_comparison.png      konvergjenca dhe frontet NSGA-II vs MOEA/D
  final_summary.csv         HV per cdo ekzekutim + rezultati i kombinuar
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R_DV, R_T = 2500.0, 4000.0
K = 100

# Renditja e SpOC 2022 (optimize.esa.int, HV pa shenjen minus)
LEADERBOARD = [5870436.818, 5731151.887, 5601395.897, 5503790.681, 5502957.03,
               5421736.524, 5210459.303, 4200526.721, 4161632.839, 3440253.65,
               3346163.076, 3057624.391, 3040767.195, 2403468.26, 2311987.584,
               2298801.02, 1844967.153, 1759975.005, 795321.727, 0.0]

RUNS = {
    "NSGA-II": ["nsga2_front_warm_seed1.csv", "nsga2_front_warm_seed2.csv", "nsga2_front_warm_seed3.csv"],
    "MOEA/D": ["moead_warm_seed1_front.csv", "moead_warm_seed2_front.csv", "moead_warm_seed3_front.csv"],
}
PROGRESS = {
    "NSGA-II": ["nsga2_progress_warm_seed%d.csv" % s for s in (1, 2, 3)],
    "MOEA/D": ["moead_warm_seed%d_progress.csv" % s for s in (1, 2, 3)],
}


def nondominated(F, X=None):
    """Pikat jo te dominuara brenda kutise, te renditura sipas DV rrites."""
    m = (F[:, 0] < R_DV) & (F[:, 1] < R_T)
    F = F[m]
    X = X[m] if X is not None else None
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best_t = [], np.inf
    for i in order:
        if F[i, 1] < best_t:
            keep.append(i)
            best_t = F[i, 1]
    return F[keep], (X[keep] if X is not None else None)


def hv(F):
    F, _ = nondominated(F)
    if len(F) == 0:
        return 0.0
    t_prev = np.concatenate([[R_T], F[:-1, 1]])
    return float(np.sum((R_DV - F[:, 0]) * (t_prev - F[:, 1])))


def best_k(F, X, k=K):
    """Programim dinamik: k pikat qe maksimizojne hypervolume (2 objektiva).
    D[j][i] = HV maksimal me j+1 pika ku pika e fundit (DV me i madh) eshte i."""
    F, X = nondominated(F, X)
    n = len(F)
    if n <= k:
        return F, X
    a = R_DV - F[:, 0]          # gjeresia ne DV
    t = F[:, 1]                 # T zbret kur DV rritet
    D = a * (R_T - t)
    parents = []
    lower = np.tril(np.ones((n, n), dtype=bool), -1)   # p < i
    for _ in range(1, k):
        M = D[None, :] + a[:, None] * (t[None, :] - t[:, None])
        M[~lower] = -np.inf
        arg = M.argmax(axis=1)
        D = M[np.arange(n), arg]
        parents.append(arg)
    i = int(np.argmax(D))
    sel = [i]
    for arg in reversed(parents):
        i = int(arg[i])
        sel.append(i)
    sel = sorted(sel)
    return F[sel], X[sel]


def load_front(name):
    A = np.genfromtxt(os.path.join(RES, name), delimiter=",", skip_header=1)
    A = A.reshape(-1, A.shape[-1]) if A.size else np.empty((0, 36))
    return A[:, :2], A[:, 2:]


def rank_of(score):
    return 1 + sum(s > score for s in LEADERBOARD)


rows, allF, allX, per_algo = [], [], [], {}
for algo, files in RUNS.items():
    per_algo[algo] = []
    for f in files:
        F, X = load_front(f)
        Fk, _ = best_k(F, X)
        h = hv(Fk)
        per_algo[algo].append(h)
        rows.append((algo, f, len(nondominated(F)[0]), round(hv(F)), round(h)))
        allF.append(F)
        allX.append(X)
        print(f"{algo:8s} {f:32s} HV(100) = {h:12.0f}  -> vendi {rank_of(h)}/20")

# Fronti i CMA-ES me epsilon-kufizim (nese ekziston)
cma_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(RES, "cmaes_w*_front.csv")))
for f in cma_files:
    Fc, Xc = load_front(f)
    h = hv(best_k(Fc, Xc)[0])
    rows.append(("CMA-ES eps", f, len(nondominated(Fc)[0]), round(hv(Fc)), round(h)))
    allF.append(Fc)
    allX.append(Xc)
    print(f"CMA-ES   {f:32s} HV(100) = {h:12.0f}  -> vendi {rank_of(h)}/20")

F = np.vstack(allF)
X = np.vstack(allX)
Fk, Xk = best_k(F, X)
h_comb = hv(Fk)
print(f"\nI KOMBINUAR ({len(allF)} frontet, 100 zgjidhje): HV = {h_comb:.0f} -> vendi {rank_of(h_comb)}/20")
for algo, v in per_algo.items():
    print(f"{algo:8s} mesatarja = {np.mean(v):.0f} +/- {np.std(v, ddof=1):.0f}")

with open(os.path.join(RES, "final_front_100.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dv", "t"] + [f"x{i}" for i in range(X.shape[1])])
    for f_, x_ in zip(Fk, Xk):
        w.writerow(list(f_) + list(x_))
with open(os.path.join(RES, "final_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["algoritmi", "file", "pika_jo_te_dominuara", "hv_te_gjitha", "hv_100"])
    w.writerows(rows)
    w.writerow(["KOMBINUAR", f"{len(allF)} frontet", len(nondominated(F)[0]), round(hv(F)), round(h_comb)])

# ---- grafiku i krahasimit ----
colors = {"NSGA-II": "tab:green", "MOEA/D": "tab:blue"}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for algo, files in PROGRESS.items():
    for j, f in enumerate(files):
        P = np.genfromtxt(os.path.join(RES, f), delimiter=",", names=True)
        m = P["evals"] >= 1e4
        ax[0].plot(P["evals"][m] / 1e6, P["hv"][m] / 1e6, color=colors[algo], alpha=0.8,
                   label=algo if j == 0 else None)
ax[0].axhspan(LEADERBOARD[2] / 1e6, LEADERBOARD[0] / 1e6, color="gold", alpha=0.3,
              label=f"top 3 SpOC 2022 ({LEADERBOARD[2] / 1e6:.2f}-{LEADERBOARD[0] / 1e6:.2f} M)")
ax[0].axhline(h_comb / 1e6, color="red", ls="--", lw=1.2, label=f"fronti final (100 pika) = {h_comb / 1e6:.2f} M")
ax[0].set_xscale("log")
ax[0].set_xlim(0.01, 40)
ax[0].set_ylim(0, 6.2)
ax[0].set_xlabel("vleresime [milion, shkalle log]")
ax[0].set_ylabel("hypervolume [milion]")
ax[0].set_title("Konvergjenca (3 seed-e per algoritem, warm start)")
ax[0].legend(loc="upper left", fontsize=8)

for algo, files in RUNS.items():
    Fa = np.vstack([load_front(f)[0] for f in files])
    Fa, _ = nondominated(Fa)
    ax[1].step(Fa[:, 1], Fa[:, 0], where="post", color=colors[algo], alpha=0.7, label=f"{algo} (3 seed-e)")
if cma_files:
    Fc = np.vstack([load_front(f)[0] for f in cma_files])
    Fc, _ = nondominated(Fc)
    ax[1].step(Fc[:, 1], Fc[:, 0], where="post", color="tab:orange", alpha=0.8, label="CMA-ES me eps-kufizim")
ax[1].scatter(Fk[:, 1], Fk[:, 0], s=10, c="red", zorder=3, label="100 zgjidhjet finale")
ax[1].scatter([3469.83], [2378.99], marker="*", s=200, c="purple", label="shembulli ESA")
ax[1].set_xlim(0, R_T)
ax[1].set_ylim(0, R_DV)
ax[1].set_xlabel("T [dite]")
ax[1].set_ylabel("DV [m/s]")
ax[1].set_title("Pareto frontet brenda kutise se hypervolume")
ax[1].legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RES, "final_comparison.png"), dpi=130)
print("Grafiku: results/final_comparison.png")
