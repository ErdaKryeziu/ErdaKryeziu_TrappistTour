"""Dita 2: vleresojme N zgjidhje te rastesishme (me sekuence te vlefshme)
dhe ruajme rezultatet + grafiket ne results/."""
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
matplotlib.use("Agg")           # pa dritare, vetem ruajtje ne file
import matplotlib.pyplot as plt

import problem as P

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 42
rng = np.random.default_rng(SEED)

print(f"Dimensionet: {P.N_DIM} (te vazhdueshme: {P.N_CONT}, te plota: {P.N_INT})")
print(f"Po vleresoj {N} zgjidhje te rastesishme (seed={SEED})...")

X = np.empty((N, P.N_DIM))
rows = []
t0 = time.time()
for i in range(N):
    x = P.random_valid_vector(rng)
    dv, t, eq, ineq, status = P.evaluate(x)
    X[i] = x
    rows.append((dv, t, eq, ineq, status, P.is_feasible(eq, ineq, status)))
elapsed = time.time() - t0
print(f"Koha: {elapsed:.2f} s ({1000 * elapsed / N:.3f} ms per vleresim)")

# ruajme rezultatet
np.save(os.path.join(RESULTS, "random_samples_x.npy"), X)
with open(os.path.join(RESULTS, "random_samples.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dv", "t", "eq", "ineq", "status", "feasible"])
    w.writerows(rows)

dv = np.array([r[0] for r in rows])
t = np.array([r[1] for r in rows])
status = np.array([r[4] for r in rows])
feas = np.array([r[5] for r in rows])
ok = status == "ok"
inbox = feas & (dv < P.DV_REF) & (t < P.T_REF)

print("\n=== PERMBLEDHJE ===")
print(f"status 'ok'      : {ok.sum()} / {N}")
print(f"death penalty    : {(status == 'penalty').sum()}")
print(f"perjashtime      : {(status == 'error').sum()}")
print(f"te vlefshme (ineq <= 0): {feas.sum()} / {N}")
print(f"te vlefshme dhe brenda kutise (DV<{P.DV_REF:.0f}, T<{P.T_REF:.0f}): {inbox.sum()}")
if ok.any():
    print(f"DV (status ok): min={dv[ok].min():.1f}  mediana={np.median(dv[ok]):.1f}")
    print(f"T  (status ok): min={t[ok].min():.1f}  mediana={np.median(t[ok]):.1f}  max={t[ok].max():.1f}")
if feas.any():
    j = np.argmin(np.where(feas, dv, np.inf))
    print(f"DV me i miri nder te vlefshmet: {dv[j]:.1f} m/s (T={t[j]:.1f} d)")

# grafiket
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hist(np.log10(dv[ok]), bins=40, color="steelblue")
ax[0].axvline(np.log10(P.DV_REF), color="red", ls="--", label="DV = 2500")
ax[0].set_xlabel("log10(DV [m/s])")
ax[0].set_ylabel("numri i zgjidhjeve")
ax[0].set_title("Shperndarja e DV")
ax[0].legend()

ax[1].hist(t[ok], bins=40, color="darkorange")
ax[1].axvline(P.T_REF, color="red", ls="--", label="T = 4000")
ax[1].set_xlabel("T [dite]")
ax[1].set_title("Shperndarja e kohes")
ax[1].legend()

ax[2].scatter(t[ok & ~feas], dv[ok & ~feas], s=8, c="gray", label="jo e vlefshme")
if feas.any():
    ax[2].scatter(t[feas], dv[feas], s=12, c="green", label="e vlefshme")
ax[2].axvline(P.T_REF, color="red", ls="--")
ax[2].axhline(P.DV_REF, color="red", ls="--")
ax[2].set_yscale("log")
ax[2].set_xlabel("T [dite]")
ax[2].set_ylabel("DV [m/s]")
ax[2].set_title("DV kundrejt T")
ax[2].legend()
plt.tight_layout()
out = os.path.join(RESULTS, "day2_random_sampling.png")
plt.savefig(out, dpi=130)
print(f"\nGrafiku u ruajt: {out}")
