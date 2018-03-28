import numpy as np
import random


class Creature:

    def __init__(self, n):
        self.genotype = []

    def __str__(self):
        return '[' + ' '.join(map(lambda x: str(x), self.genotype)) + ']'

    def __eq__(self, other):
        return self.genotype == other.genotype

    def __gt__(self, other):
        return self.genotype[0] > other.genotype[0]

    def calculate_cost_vec(self, distances, flows):
        np_genotype = np.array(self.genotype)

        flows_sorted = flows[np_genotype]
        flows_sorted = np.transpose(flows_sorted)
        flows_sorted = flows_sorted[np_genotype]
        flows_sorted = np.transpose(flows_sorted)

        cost = distances.flatten().transpose().__matmul__(flows_sorted.flatten())
        return cost

    @classmethod
    def random(cls, n):
        c = Creature(n)
        c.genotype = list(range(n))
        random.shuffle(c.genotype)
        return c

