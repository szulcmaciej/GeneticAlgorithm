import numpy as np
from data_loader import DataLoader
from model import Creature
import time


def run(data_filename, time_limit):
    start_time = time.time()
    elapsed_time = 0

    # load data
    n, distances, flows = DataLoader.load(data_filename)

    best_creature = Creature.random(n)
    best_cost = best_creature.calculate_cost_vec(distances, flows)
    best_cost_array = []

    while elapsed_time < time_limit:
        c = Creature.random(n)
        cost = c.calculate_cost_vec(distances, flows)
        if cost < best_cost:
            best_cost = cost
            best_creature = c
        best_cost_array.append(best_cost)
        elapsed_time = time.time() - start_time

    return best_cost, best_creature, best_cost_array
