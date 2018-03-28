import gen_alg as ga
from model import Creature
import numpy as np
import random
import data_loader

n = 12

p1 = Creature.random(n)
p2 = Creature.random(n)

# children_pmx = ga.reproduce_pmx(p1, p2)
# children_classic = ga.reproduce(p1, p2)
#
# print(p1)
# print(p2)
# print()
# print(children_pmx[0])
# print(children_pmx[1])
# print()
# print(children_classic[0])
# print(children_classic[1])

n, distances, flows = data_loader.DataLoader.load('../data/had12.dat')

cost = ga.calculate_cost_vec(p1, distances, flows, n)

print(p1.genotype)
print(cost)