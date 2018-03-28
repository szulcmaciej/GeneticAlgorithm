from model import Creature
from data_loader import DataLoader
import numpy as np
import gen_alg as ga
import matplotlib.pyplot as plt
import cProfile
import time
import random_alg

TEST_ITERATIONS = 50

DATA_FILE = '../data/had20.dat'
TIME_LIMIT = 5

min_costs_all = []
best_cost_all = []
min_costs_array = []

for i in range(TEST_ITERATIONS):
    print('TEST ITERATION ' + str(i))
    best_cost, best_creature, min_costs_array = random_alg.run(DATA_FILE, TIME_LIMIT)

    print('best_cost: ' + str(best_cost))
    min_costs_all.append(min_costs_array)
    best_cost_all.append(best_cost)
    plt.plot(min_costs_array, '0.9')

min_length = min(map(lambda x: len(x), min_costs_all))
min_costs_all = list(map(lambda x: x[0:min_length], min_costs_all))

min_costs_all = np.asarray(min_costs_all)

min_costs_avg = np.average(min_costs_all, axis=0)
best_cost_avg = np.average(best_cost_all, axis=0)

plt.plot(min_costs_avg)

plt.xlabel('creature number')
plt.ylabel('best cost')

plt.suptitle('Random algorithm, time limit: ' + str(TIME_LIMIT) + 's')
plt.title('Best cost avg: ' + str(best_cost_avg))

plt.savefig('images/20_random')
plt.show()