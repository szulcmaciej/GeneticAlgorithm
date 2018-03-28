from model import Creature
from data_loader import DataLoader
import numpy as np
import gen_alg as ga
import matplotlib.pyplot as plt
import cProfile
import time


# # test wielokrotny z uśrednieniem

TEST_ITERATIONS = 5

TIME_LIMIT = 1

DATA_FILE = '../data/had12.dat'
POPULATION = 500
MUTATION_PROBABILITY = 0.05
CROSS_PROBABILITY = 1

SELECTION_METHOD = 'tournament'
# SELECTION_METHOD = 'roulette'
# SELECTION_METHOD = 'best_half'
MUTATION_METHOD = 'swap'
# MUTATION_METHOD = 'insert'
# MUTATION_METHOD = 'both'
TOURNAMENT_SIZE = 25


min_costs_all = []
avg_costs_all = []
best_cost_all = []

start_time = time.time()


for i in range(TEST_ITERATIONS):
    print()
    print('TEST ITERATION ' + str(i + 1))
    min_costs, avg_costs, best, best_cost = ga.run_vec_time(DATA_FILE, POPULATION, MUTATION_PROBABILITY, TIME_LIMIT, CROSS_PROBABILITY, MUTATION_METHOD, SELECTION_METHOD, TOURNAMENT_SIZE)
    min_costs_all.append(min_costs)
    avg_costs_all.append(avg_costs)
    best_cost_all.append(best_cost)

    plt.plot(min_costs, '0.9')

finish_time = time.time()
elapsed_time = finish_time - start_time

min_length = min(map(lambda x: len(x), min_costs_all))

min_costs_all = list(map(lambda x: x[0:min_length], min_costs_all))
avg_costs_all = list(map(lambda  x: x[0:min_length], avg_costs_all))


min_costs_all = np.asarray(min_costs_all)
avg_costs_all = np.asarray(avg_costs_all)
# best_cost_all = np.asarray(best_cost_all)

min_costs_avg = np.average(min_costs_all, axis=0)
avg_costs_avg = np.average(avg_costs_all, axis=0)
best_cost_avg = np.average(best_cost_all, axis=0)

print()
print()
print('Best cost avg: ' + str(best_cost_avg))
print('Elapsed time: ' + '{0:.2f}'.format(elapsed_time))


plt.plot(min_costs_avg, label='Najlepszy osobnik')
plt.plot(avg_costs_avg, label='Średni osobnik')

plt.legend()
# plt.title(' Time limit: ' + str(TIME_LIMIT) + 's Population: ' + str(POPULATION) + ' Mutation: ' + str(MUTATION_PROBABILITY))
plt.title('Best cost avg: ' + str(best_cost_avg))
plt.suptitle('File: ' + DATA_FILE)
plt.xlabel('Iterations')
plt.ylabel('Cost')

# plt.text(len(min_costs_avg) / 2 * 1.3, best_cost_avg * 1.07, 'Best cost avg: ' + str(best_cost_avg),)

# plt.savefig('images/18')
# plt.savefig('images/12_pop' + str(POPULATION))
plt.show()

