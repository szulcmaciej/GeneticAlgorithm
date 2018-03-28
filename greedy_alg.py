import numpy as np
from data_loader import DataLoader
from model import Creature
import time

DATA_FILE = 'data/had20.dat'

def run(data_filename):
    n, distances, flows = DataLoader.load(data_filename)

    c = Creature(n)
    c.genotype = []

    for i in range(n):
        c.genotype.append(0)
    present_genes = []
    for i in range(n):
        min_cost = 1000000000
        best_gene = -1
        changed_c = Creature(n)
        missing_genes = [x for x in list(range(n)) if x not in present_genes]
        for j in missing_genes:
            changed_c = Creature(n)
            changed_c.genotype = c.genotype
            changed_c.genotype[i] = j
            if changed_c.calculate_cost_vec(distances, flows) < min_cost:
                min_cost = changed_c.calculate_cost_vec(distances, flows)
                best_gene = j
        present_genes.append(best_gene)
        c.genotype[i] = best_gene

        print(c.genotype)


    return c, c.calculate_cost_vec(distances, flows)


c, cost = run(DATA_FILE)

print(cost)