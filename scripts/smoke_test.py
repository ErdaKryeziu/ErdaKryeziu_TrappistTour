import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pygmo as pg
from trappist_udp import udp

lb, ub = udp.get_bounds()
print("dim", len(lb), "nobj", udp.get_nobj(), "nix", udp.get_nix(),
      "nec", udp.get_nec(), "nic", udp.get_nic())
x = udp.example()
print("fitness(example):", udp.fitness(x))
udp.pretty(x)
prob = pg.problem(udp)
print("random vector:", prob.fitness(pg.random_decision_vector(prob)))