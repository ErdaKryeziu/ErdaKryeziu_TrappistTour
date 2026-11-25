"""CMA-ES me epsilon-kufizim per Trappist Tour.

Ideja: Pareto frontin e gjejme pike per pike. Per cdo kufi kohe T_max
zgjidhim problemin me nje objektiv
        min  DV    me kusht   T <= T_max  dhe  ineq <= 0
me CMA-ES (Covariance Matrix Adaptation Evolution Strategy). Duke ndryshuar
T_max fitojme zgjidhje te shperndara pergjate frontit, sidomos ne zonen e
misioneve te shkurtra qe NSGA-II dhe MOEA/D nuk e arriten.

Pika e nisjes per cdo T_max merret nga arkivi (fronti final i 6 ekzekutimeve
+ cdo zgjidhje e re). Nese nuk ka zgjidhje me T <= T_max, merret zgjidhja me
T me te vogel dhe kohezgjatjet e legjeve shkurtohen ne menyre proporcionale.

Perdorimi:
    python scripts/eps_cmaes.py PUNETORI NR_PUNETOREVE SEED MAX_ORE
    p.sh. 4 procese paralele: PUNETORI = 0, 1, 2, 3 dhe NR_PUNETOREVE = 4
Rezultatet: results/cmaes_w<PUNETORI>_front.csv, _progress.csv, log ne stdout.
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")

import problem as P
import moo

WORKER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MAX_SEC = float(sys.argv[4]) * 3600 if len(sys.argv) > 4 else 600.0
rng = np.random.default_rng(1000 * SEED + WORKER)

TARGETS_ALL = list(range(1300, 2500, 100)) + [2600, 2800, 3000, 3300]
TARGETS = TARGETS_ALL[WORKER::N_WORKERS]
RUN_EVALS = 40000          # buxheti per nje ekzekutim CMA-ES
PATIENCE = 6000            # ndalim nese s'ka permiresim per kaq vleresime
T_IDX = [2] + [6 + 4 * k for k in range(6)]   # pozicionet e T0 dhe T e legjeve
SLOPE = 10.0               # penalizim: 10 m/s per cdo dite mbi T_max

LB, UB = (np.asarray(b, dtype=float) for b in moo.TrappistRK().get_bounds())
W = UB - LB


# ---------------- CMA-ES (versioni standard i Hansen-it) ----------------
class CMAES:
    def __init__(self, x0, sigma0, rng, popsize=None):
        n = len(x0)
        self.n, self.rng = n, rng
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.invsqrtC = np.eye(n)
        self.count = 0
        self.eigen_at = 0

    def ask(self):
        Z = self.rng.standard_normal((self.lam, self.n))
        return self.m + self.sigma * (Z * self.D) @ self.B.T

    def tell(self, X, f):
        order = np.argsort(f)[: self.mu]
        old = self.m
        Xs = X[order]
        self.m = self.w @ Xs
        y = (self.m - old) / self.sigma
        self.count += self.lam
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ y)
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.count / self.lam))
                / self.chiN) < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        A = (Xs - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (A.T * self.w) @ A)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = min(self.sigma, 1.0)
        if self.count - self.eigen_at > self.lam / (self.c1 + self.cmu) / self.n / 10:
            self.eigen_at = self.count
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            d, B = np.linalg.eigh(self.C)
            d = np.maximum(d, 1e-20)
            self.D, self.B = np.sqrt(d), B
            self.invsqrtC = B @ np.diag(1 / self.D) @ B.T


# ---------------- funksioni objektiv ----------------
def record(x, dv, t):
    if dv < P.DV_REF and t < P.T_REF:
        moo.ARCHIVE_F.append((dv, t))
        moo.ARCHIVE_X.append(np.array(x, dtype=float))


def objective(u, t_max):
    """u ne [0,1]^34 (hapesire e normalizuar) -> vlera skalare per CMA-ES."""
    uc = np.clip(u, 0.0, 1.0)
    pen_box = 1e3 * float(np.sum((u - uc) ** 2))
    x = moo.decode(LB + uc * W)
    dv, t, eq, ineq, status = P.evaluate(x)
    moo.STATS["evals"] += 1
    if status != "ok":
        return 1e12 + pen_box
    if ineq > 0:
        return 1e7 + min(ineq, 1e13) / 1e3 + pen_box
    moo.STATS["feasible"] += 1
    record(x, dv, t)
    return dv + SLOPE * max(0.0, t - t_max) + pen_box


def start_point(t_max):
    """Zgjidhja e nisjes (ne formatin zyrtar) per kufirin t_max."""
    F = np.asarray(moo.ARCHIVE_F, dtype=float)
    ok = np.where(F[:, 1] <= t_max)[0]
    if len(ok):
        if rng.random() < 0.6:
            j = ok[np.argmin(F[ok, 0])]          # DV me i vogel nen t_max
        else:
            j = rng.choice(ok)                   # diversitet
        return np.array(moo.ARCHIVE_X[j], dtype=float)
    j = int(np.argmin(F[:, 1]))                  # me i shpejti qe kemi
    x = np.array(moo.ARCHIVE_X[j], dtype=float)
    T = x[T_IDX]
    goal = 0.97 * t_max
    legs = T[1:]
    if T[0] + 5 * len(legs) < goal:
        legs = np.maximum(5.0, legs * (goal - T[0]) / legs.sum())
    else:
        legs = np.full(len(legs), 5.0)
        T[0] = max(5.0, goal - legs.sum())
    T[1:] = legs
    x[T_IDX] = T
    return x


def run_cmaes(t_max):
    x0 = start_point(t_max)
    u0 = (moo.encode(x0) - LB) / W
    u0 = np.clip(u0, 0.0, 1.0)
    sigma0 = rng.choice([0.005, 0.02, 0.05])
    es = CMAES(u0, sigma0, rng)
    best, best_at, used = np.inf, 0, 0
    while used < RUN_EVALS and used - best_at < PATIENCE and es.sigma > 1e-9:
        X = es.ask()
        f = np.array([objective(u, t_max) for u in X])
        es.tell(X, f)
        used += len(X)
        if f.min() < best - 1e-6:
            best, best_at = f.min(), used
    return best, used, sigma0


# ---------------- ngarkimi i frontit fillestar ----------------
moo.reset_archive()
start_file = os.path.join(RESULTS, "final_front_100.csv")
A = np.genfromtxt(start_file, delimiter=",", skip_header=1)
for row in A:
    moo.ARCHIVE_F.append((row[0], row[1]))
    moo.ARCHIVE_X.append(row[2:])
hv0 = moo.hypervolume(moo.ARCHIVE_F)
print(f"Punetori {WORKER}/{N_WORKERS}, seed {SEED}, T_max: {TARGETS}")
print(f"Fronti fillestar: {len(A)} pika, HV = {hv0:.0f}", flush=True)

tag = f"cmaes_w{WORKER}"
log = []


def save():
    front = moo.pareto_front(moo.ARCHIVE_F)
    F = np.asarray(moo.ARCHIVE_F)
    with open(os.path.join(RESULTS, f"{tag}_front.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dv", "t"] + [f"x{i}" for i in range(P.N_DIM)])
        for p in front:
            j = int(np.where((F[:, 0] == p[0]) & (F[:, 1] == p[1]))[0][0])
            w.writerow([p[0], p[1]] + list(moo.ARCHIVE_X[j]))
    with open(os.path.join(RESULTS, f"{tag}_progress.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "t_max", "evals", "hv", "best_obj", "sigma0", "seconds"])
        w.writerows(log)


t0 = time.time()
last_save = t0
k = 0
while time.time() - t0 < MAX_SEC:
    t_max = TARGETS[k % len(TARGETS)]
    best, used, s0 = run_cmaes(t_max)
    k += 1
    moo.prune_archive(2000)
    hv = moo.hypervolume(moo.ARCHIVE_F)
    log.append((k, t_max, moo.STATS["evals"], hv, best, s0, time.time() - t0))
    print(f"run {k:4d} | T_max {t_max:5d} | DV me i mire {best:10.1f} | sigma0 {s0:.3f} | "
          f"vleresime {moo.STATS['evals']:9d} | HV {hv:10.0f} | {(time.time() - t0) / 60:6.1f} min",
          flush=True)
    if time.time() - last_save > 15 * 60:
        last_save = time.time()
        save()

save()
front = moo.pareto_front(moo.ARCHIVE_F)
print(f"\n=== PERMBLEDHJE CMA-ES (punetori {WORKER}) ===")
print(f"Ekzekutime CMA-ES : {k}")
print(f"Vleresime         : {moo.STATS['evals']}")
print(f"T me i vogel      : {front[:, 1].min():.1f} dite")
print(f"HV fillestar      : {hv0:.0f}")
print(f"HV final          : {moo.hypervolume(front):.0f}")
