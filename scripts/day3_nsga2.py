"""Dita 3: NSGA-II (pygmo) ne Trappist Tour + hypervolume.

Perdorimi:  python scripts/day3_nsga2.py [GEN] [POP] [SEED] [MAX_ORE] [KUFIJTE]
            parazgjedhje: GEN=500, POP=100, SEED=42, MAX_ORE=0 (pa kufi), KUFIJTE=full
            KUFIJTE: full  = kufijte zyrtare, popullata fillestare e rastesishme
                     tight = kufijte e ngushtuar (shih moo.tight_bounds)
                     warm  = kufijte zyrtare + nisje e ngrohte nga zgjidhja
                             shembull e ESA-s (shih moo.warm_start_population)
Shembuj:    ... day3_nsga2.py 50 100 42                    (prove e shkurter)
            ... day3_nsga2.py 2000 100 42 0 tight          (kufij te ngushtuar)
            ... day3_nsga2.py 2000 100 42 0 warm           (nisje e ngrohte)
            ... day3_nsga2.py 100000000 100 42 7 warm      (ekzekutim 7-oresh)
Rezultatet ruhen ne results/ me prapashtesen _<KUFIJTE>_seed<SEED> dhe ruhen
edhe gjate punes (cdo 15 min), qe te mos humbin nese ndalet sesioni.
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

GEN = int(sys.argv[1]) if len(sys.argv) > 1 else 500
POP = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
MAX_H = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
MAX_SEC = MAX_H * 3600 if MAX_H > 0 else float("inf")
MODE = sys.argv[5] if len(sys.argv) > 5 else "full"
if MODE not in ("full", "tight", "warm"):
    sys.exit("KUFIJTE duhet te jete 'full', 'tight' ose 'warm'")
CHUNK = 10                      # sa gjenerata ndermjet dy matjeve
SAVE_EVERY = 15 * 60            # ruajtje e ndermjetme (sekonda)
if POP < 8 or POP % 4 != 0:
    sys.exit("POP duhet te jete shumefish i 4 dhe >= 8 (kerkese e NSGA-II ne pygmo)")
tag = f"{MODE}_seed{SEED}"

# ---- 1. Kontroll i metrikes me zgjidhjen shembull te ESA-s ----
dv_ex, t_ex, _, _, _ = P.evaluate(P.udp.example())
hv_ex = moo.hypervolume([(dv_ex, t_ex)])
print(f"Kontroll: zgjidhja shembull e ESA-s DV={dv_ex:.1f}, T={t_ex:.1f} -> HV={hv_ex:.0f}")
print(f"          (duhet: (2500-{dv_ex:.1f}) x (4000-{t_ex:.1f}) = {(2500-dv_ex)*(4000-t_ex):.0f})")


def save(log):
    """Ruan progresin, Pareto frontin dhe grafikun ne results/."""
    front = moo.pareto_front(moo.ARCHIVE_F)
    hv = moo.hypervolume(front)
    with open(os.path.join(RESULTS, f"nsga2_progress_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gen", "evals", "hv", "best_feasible_dv", "n_in_box",
                    "pop_feasible_frac", "seconds"])
        w.writerows(log)
    with open(os.path.join(RESULTS, f"nsga2_front_{tag}.csv"), "w", newline="") as fh:
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
    ax[0].set_title("Hypervolume gjate optimizimit")
    ax[0].legend()

    ax[1].plot(L[:, 1], L[:, 3], color="steelblue")
    ax[1].axhline(P.DV_REF, color="red", ls="--", label="DV = 2500")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("vleresime")
    ax[1].set_ylabel("DV me i mire i vlefshem [m/s]")
    ax[1].set_title("Konvergjenca e DV")
    ax[1].legend()

    if len(front):
        ax[2].step(front[:, 1], front[:, 0], where="post", color="green", alpha=0.5)
        ax[2].scatter(front[:, 1], front[:, 0], s=15, c="green", label="Pareto front NSGA-II")
    ax[2].scatter([t_ex], [dv_ex], marker="*", s=200, c="purple", label="shembulli ESA")
    ax[2].set_xlim(0, P.T_REF)
    ax[2].set_ylim(0, P.DV_REF)
    ax[2].set_xlabel("T [dite]")
    ax[2].set_ylabel("DV [m/s]")
    ax[2].set_title(f"Kutia e hypervolume (HV = {hv:.0f})")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"day3_nsga2_{tag}.png"), dpi=130)
    plt.close(fig)
    return front, hv


# ---- 2. NSGA-II ----
limit = "pa kufi" if MAX_SEC == float("inf") else f"{MAX_SEC / 3600:.1f} ore"
print(f"\nNSGA-II: pop={POP}, gen={GEN}, seed={SEED}, kufijte: {MODE}, kufiri i kohes: {limit}", flush=True)
moo.reset_archive()
prob = pg.problem(moo.TrappistRK(tight=(MODE == "tight")))
if MODE == "warm":
    pop = moo.warm_start_population(prob, POP, SEED)
else:
    pop = pg.population(prob, size=POP, seed=SEED)
print(f"Popullata fillestare: {len(pop)} zgjidhje, HV fillestar = {moo.hypervolume(moo.ARCHIVE_F):.0f}", flush=True)
algo = pg.algorithm(pg.nsga2(gen=CHUNK, seed=SEED))

log = []
t0 = time.time()
last_save = t0
last_print = t0
done = 0
while done < GEN and time.time() - t0 < MAX_SEC:
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
    if done % 50 == 0 or done >= GEN or now - last_print > 60:
        last_print = now
        print(f"gen {done:7d} | vleresime {row[1]:9d} | HV {hv:12.0f} | "
              f"DV me i mire {row[3]:12.1f} | ne kuti {row[4]:5d} | "
              f"pop e vlefshme {100 * feas_pop:5.1f}% | {row[6]:8.1f} s", flush=True)
    if now - last_save > SAVE_EVERY:
        last_save = now
        save(log)
        print("   (rezultatet e ndermjetme u ruajten)", flush=True)

elapsed = time.time() - t0
front, hv = save(log)

# ---- 3. Permbledhja ----
print("\n=== PERMBLEDHJE NSGA-II ===")
print(f"Gjenerata              : {done}")
print(f"Vleresime gjithsej     : {moo.STATS['evals']}")
print(f"Koha                   : {elapsed:.1f} s ({elapsed / 3600:.2f} ore)")
print(f"Te vlefshme            : {moo.STATS['feasible']} ({100 * moo.STATS['feasible'] / moo.STATS['evals']:.1f}%)")
print(f"DV me i mire (vlefshme): {moo.STATS['best_dv']:.1f} m/s")
print(f"Pareto front (ne kuti) : {len(front)} pika")
print(f"HYPERVOLUME            : {hv:.0f}")
print(f"Krahasim: random search HV = 0, zgjidhja shembull e ESA-s HV = {hv_ex:.0f}")
print(f"\nGrafiku u ruajt: {os.path.join(RESULTS, f'day3_nsga2_{tag}.png')}")
