"""Funksione ndihmese rreth UDP-se zyrtare (trappist_udp.py).
UDP-ja zyrtare NUK ndryshohet; ketu vetem e perdorim."""
import contextlib
import io

import numpy as np

from trappist_udp import udp

# Kufijtet e vektorit te vendimit (34 dimensione)
LB, UB = (np.array(b, dtype=float) for b in udp.get_bounds())
N_DIM = len(LB)            # 34
N_INT = udp.get_nix()      # 7  (sekuenca e planeteve)
N_CONT = N_DIM - N_INT     # 27 (pjesa e vazhdueshme)

# Pika e references e hypervolume (nga combine_scores te UDP-ja zyrtare)
DV_REF, T_REF = 2500.0, 4000.0

# Vlere penalizuese kur llogaritja deshton numerikisht
PENALTY = 1e10


def random_valid_vector(rng):
    """Vektor i rastesishem: pjesa e vazhdueshme uniforme brenda kufijve,
    sekuenca = permutacion i rastesishem i 0..6 (pra sekuence e vlefshme)."""
    x = np.empty(N_DIM)
    x[:N_CONT] = rng.uniform(LB[:N_CONT], UB[:N_CONT])
    x[N_CONT:] = rng.permutation(N_INT)
    return x


def evaluate(x):
    """Vlereson nje vektor. Kthen (dv, t, eq, ineq, status).
    status: 'ok' | 'penalty' (UDP-ja aplikoi death penalty) | 'error' (perjashtim)."""
    with contextlib.redirect_stdout(io.StringIO()):   # heshtim printimet e UDP-se
        try:
            f = udp.fitness([float(v) for v in x])
        except Exception:
            return (PENALTY, PENALTY, 1.0, PENALTY, "error")
    if len(f) != 4:
        # Ne rastin e death penalty, UDP-ja kthen tuple me 6 elemente
        return (PENALTY, PENALTY, 1.0, PENALTY, "penalty")
    dv, t, eq, ineq = f
    return (float(dv), float(t), float(eq), float(ineq), "ok")


def is_feasible(eq, ineq, status="ok"):
    """Zgjidhja eshte e vlefshme nese sekuenca eshte permutacion (eq == 0)
    dhe kufizimi i largesise nga ylli plotesohet (ineq <= 0)."""
    return status == "ok" and eq == 0 and ineq <= 0
