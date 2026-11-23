"""Ekzekuton nje algoritem multi-objektiv te pygmo-s ne Trappist Tour.

Perdorimi:
    python scripts/run_moo.py ALGO NISJA SEED MAX_VLERESIME [MAX_ORE] [POP]

    ALGO          : nsga2 | moead | nspso
    NISJA         : warm (nga zgjidhja shembull e ESA-s) | full (e rastesishme)
    SEED          : numri i rastesise, p.sh. 1
    MAX_VLERESIME : sa vleresime te funksionit lejohen (buxheti), p.sh. 10000000
    MAX_ORE       : kufi kohe ne ore (0 = pa kufi), parazgjedhje 0
    POP           : madhesia e popullates, parazgjedhje 100

Shembuj:
    ... run_moo.py moead warm 42 200000          (prove ~1 minute)
    ... run_moo.py nspso warm 1 10000000 6       (10 milion vleresime, max 6 ore)

Te gjitha algoritmet marrin te njejten popullate fillestare dhe te njejtin
buxhet vleresimesh, qe krahasimi te jete i drejte.
Rezultatet: results/<algo>_<nisja>_seed<SEED>_{progress,front}.csv dhe .png
(ruhen edhe gjate punes, cdo 15 minuta).
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygmo as pg

import problem as P
import moo

if len(sys.argv) < 5:
    sys.exit(__doc__)
ALGO = sys.argv[1]
MODE = sys.argv[2]
SEED = int(sys.argv[3])
MAX_EVALS = int(float(sys.argv[4]))
MAX_H = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
POP = int(sys.argv[6]) if len(sys.argv) > 6 else 100
MAX_SEC = MAX_H * 3600 if MAX_H > 0 else float("inf")
CHUNK = 10                      # gjenerata ndermjet dy matjeve
SAVE_EVERY = 15 * 60
PRINT_EVERY = 60

if ALGO not in ("nsga2", "moead", "nspso"):
    sys.exit("ALGO duhet te jete nsga2, moead ose nspso")
if MODE not in ("warm", "full"):
    sys.exit("NISJA duhet te jete warm ose full")
if ALGO == "nsga2" and (POP < 8 or POP % 4 != 0):
    sys.exit("Per nsga2, POP duhet te jete shumefish i 4 dhe >= 8")
tag = f"{ALGO}_{MODE}_seed{SEED}"


def make_algorithm():
    """Algoritmet me parametrat e parazgjedhur te pygmo-s (te dokumentuar ne raport)."""
    if ALGO == "nsga2":
        uda = pg.nsga2(gen=CHUNK, cr=0.95, eta_c=10.0, m=0.01, eta_m=50.0, seed=SEED)
    elif ALGO == "moead":
        uda = pg.moead(gen=CHUNK, weight_generation="grid", decomposition="tchebycheff",
                       neighbours=20, CR=1.0, F=0.5, eta_m=20.0, realb=0.9, limit=2,
                       preserve_diversity=True, seed=SEED)
    else:  # nspso; memory=True qe shpejtesite te ruhen ndermjet thirrjeve
        uda = pg.nspso(gen=CHUNK, omega=0.6, c1=0.01, c2=0.5, chi=0.5, v_coeff=0.5,
                       leader_selection_range=2, diversity_mechanism="crowding distance",
                       memory=True, seed=SEED)
    return pg.algorithm(uda)


# ---- Kontrolli i metrikes ----
dv_ex, t_ex, _, _, _ = P.evaluate(P.udp.example())
hv_ex = moo.hypervolume([(dv_ex, t_ex)])
print(f"Kontroll: shembulli i ESA-s DV={dv_ex:.1f}, T={t_ex:.1f} -> HV={hv_ex:.0f}")


def save(log, final=False):
    front = moo.pareto_front(moo.ARCHIVE_F)
    hv = moo.hypervolume(front)
    with open(os.path.join(RESULTS, f"{tag}_progress.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gen", "evals", "hv", "best_feasible_dv", "n_in_box",
                    "pop_feasible_frac", "seconds"])
        w.writerows(log)
    with open(os.path.join(RESULTS, f"{tag}_front.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dv", "t"] + [f"x{i}" for i in range(P.N_DIM)])
        F = np.asarray(moo.ARCHIVE_F).reshape(-1, 2)
        for p in front:
            j = int(np.where((F[:, 0] == p[0]) & (F[:, 1] == p[1]))[0][0])
            w.writerow([p[0], p[1]] + list(moo.ARCHIVE_X[j]))

    L = np.array(log)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(L[:, 1], L[:, 2], color="green")
    ax[0].axhline(hv_ex, color="purple", ls=":", label="shembulli ESA")
    ax[0].set_xlabel("vleresime")
    ax[0].set_ylabel("hypervolume")
    ax[0].set_title(f"{ALGO.upper()}: hypervolume")
    ax[0].legend()
    finite = np.isfinite(L[:, 3])
    ax[1].plot(L[finite, 1], L[finite, 3], color="steelblue")
    ax[1].axhline(P.DV_REF, color="red", ls="--", label="DV = 2500")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("vleresime")
    ax[1].set_ylabel("DV me i mire i vlefshem [m/s]")
    ax[1].set_title("Konvergjenca e DV")
    ax[1].legend()
    if len(front):
        ax[2].step(front[:, 1], front[:, 0], where="post", color="green", alpha=0.5)
        ax[2].scatter(front[:, 1], front[:, 0], s=15, c="green", label=f"Pareto front {ALGO.upper()}")
    ax[2].scatter([t_ex], [dv_ex], marker="*", s=200, c="purple", label="shembulli ESA")
    ax[2].set_xlim(0, P.T_REF)
    ax[2].set_ylim(0, P.DV_REF)
    ax[2].set_xlabel("T [dite]")
    ax[2].set_ylabel("DV [m/s]")
    ax[2].set_title(f"Kutia (HV = {hv:.0f})")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"{tag}.png"), dpi=130)
    plt.close(fig)
    return front, hv


# ---- Optimizimi ----
limit = "pa kufi" if MAX_SEC == float("inf") else f"{MAX_H:.1f} ore"
print(f"\n{ALGO.upper()}: nisja={MODE}, pop={POP}, seed={SEED}, "
      f"buxheti={MAX_EVALS} vleresime, kufiri i kohes={limit}", flush=True)
moo.reset_archive()
prob = pg.problem(moo.TrappistRK())
if MODE == "warm":
    pop = moo.warm_start_population(prob, POP, SEED)
else:
    pop = pg.population(prob, size=POP, seed=SEED)
print(f"Popullata fillestare: {len(pop)} zgjidhje, HV = {moo.hypervolume(moo.ARCHIVE_F):.0f}", flush=True)
algo = make_algorithm()

log = []
t0 = time.time()
last_save = last_print = t0
done = 0
while moo.STATS["evals"] < MAX_EVALS and time.time() - t0 < MAX_SEC:
    pop = algo.evolve(pop)
    done += CHUNK
    moo.prune_archive()
    f = pop.get_f()
    feas_pop = (f[:, 0] < moo.PEN).mean()
    hv = moo.hypervolume(moo.ARCHIVE_F)
    now = time.time()
    row = (done, moo.STATS["evals"], hv, moo.STATS["best_dv"],
           len(moo.ARCHIVE_F), feas_pop, now - t0)
    log.append(row)
    if now - last_print > PRINT_EVERY:
        last_print = now
        print(f"gen {done:7d} | vleresime {row[1]:9d} | HV {hv:12.0f} | "
              f"DV me i mire {row[3]:10.1f} | pop e vlefshme {100 * feas_pop:5.1f}% | "
              f"{row[6] / 60:6.1f} min", flush=True)
    if now - last_save > SAVE_EVERY:
        last_save = now
        save(log)

elapsed = time.time() - t0
front, hv = save(log)

print(f"\n=== PERMBLEDHJE {ALGO.upper()} ({MODE}, seed {SEED}) ===")
print(f"Gjenerata              : {done}")
print(f"Vleresime gjithsej     : {moo.STATS['evals']}")
print(f"Koha                   : {elapsed:.1f} s ({elapsed / 3600:.2f} ore)")
print(f"Te vlefshme            : {moo.STATS['feasible']} ({100 * moo.STATS['feasible'] / moo.STATS['evals']:.1f}%)")
print(f"DV me i mire (vlefshme): {moo.STATS['best_dv']:.1f} m/s")
print(f"Pareto front (ne kuti) : {len(front)} pika")
print(f"HYPERVOLUME            : {hv:.0f}")
print(f"Krahasim: shembulli i ESA-s HV = {hv_ex:.0f}")
