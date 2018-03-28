from model import Creature
from data_loader import DataLoader
import numpy as np
import gen_alg as ga
import matplotlib.pyplot as plt
import time


# # test wielokrotny z uśrednieniem

TEST_ITERATIONS = 5

DATA_FILE = '../data/had12.dat'
ITERATIONS = 10
POPULATION = 1000
MUTATION_PROBABILITY = 0.025
CROSS_PROBABILITY = 1

SELECTION_METHOD = 'tournament'
# SELECTION_METHOD = 'roulette'
# SELECTION_METHOD = 'best_half'
MUTATION_METHOD = 'swap'
TOURNAMENT_SIZE = 200

min_costs_all = []
avg_costs_all = []
best_cost_all = []

exec_times = []

start_time = time.time()


for i in range(TEST_ITERATIONS):
    iter_start_time = time.time()

    print()
    print('TEST ITERATION ' + str(i + 1))
    # min_costs, avg_costs, best, best_cost = ga.run_vec_time(DATA_FILE, POPULATION, MUTATION_PROBABILITY, TIME_LIMIT)
    min_costs, avg_costs, best, best_cost = ga.run_vec(DATA_FILE, POPULATION, MUTATION_PROBABILITY, ITERATIONS, CROSS_PROBABILITY, MUTATION_METHOD, SELECTION_METHOD, TOURNAMENT_SIZE)
    min_costs_all.append(min_costs)
    avg_costs_all.append(avg_costs)
    best_cost_all.append(best_cost)

    plt.plot(min_costs, '0.9')

    exec_times.append(time.time() - iter_start_time)

finish_time = time.time()
elapsed_time = finish_time - start_time

min_costs_all = np.asarray(min_costs_all)
avg_costs_all = np.asarray(avg_costs_all)
# best_cost_all = np.asarray(best_cost_all)


min_costs_avg = np.average(min_costs_all, axis=0)
avg_costs_avg = np.average(avg_costs_all, axis=0)
best_cost_avg = np.average(best_cost_all, axis=0)
exec_time_avg = np.average(exec_times)

print()
print()
print('Best cost avg: ' + str(best_cost_avg))
print('Execution time avg: ' + '{0:.2f}'.format(exec_time_avg))
print('Elapsed time: ' + '{0:.2f}'.format(elapsed_time))


plt.plot(min_costs_avg, label='min_cost')
plt.plot(avg_costs_avg, label='avg_cost')

plt.legend()
# plt.title('Test iters: ' + str(TEST_ITERATIONS) + ' Time limit: ' + str(TIME_LIMIT) + ' Population: ' + str(POPULATION) + ' Mutation: ' + str(MUTATION_PROBABILITY))
plt.title('Test iters: ' + str(TEST_ITERATIONS) + ' Algorithm iters: ' + str(ITERATIONS) + ' Population: ' + str(POPULATION) + ' Mutation: ' + str(MUTATION_PROBABILITY))
plt.xlabel('Iterations')
plt.ylabel('Cost')

plt.text(ITERATIONS / 2 * 1.3, best_cost_avg * 1.07, 'Best cost avg: ' + str(best_cost_avg),)

plt.show()


# # pojedynczy test
# start = time.time()
#
# min_costs, avg_costs, best, best_cost = ga.run('data/had12.dat', 1000, 0.005, 50)
#
# finish = time.time()
#
# elapsed_time = finish - start
#
# print(elapsed_time)
# # print(min_costs)
# print(best)
# print(best_cost)
#
# plt.plot(min_costs)
# plt.plot(avg_costs)
# plt.show()



# def calculate_cost(creature, distances, flows, n):
#     cost = 0
#     for i in range(n):
#         for j in range(n):
#             cost += distances[i, j] * flows[creature.genotype[i], creature.genotype[j]]
#     return cost
#
# def calculate_cost_vec(creature, distances, flows, n):
#     flows_sorted = flows[c1.genotype]
#     flows_sorted = np.transpose(flows_sorted)
#     flows_sorted = flows_sorted[c1.genotype]
#     flows_sorted = np.transpose(flows_sorted)
#
#     cost = 0
#     for i in range(n):
#         for j in range(n):
#             cost += distances[i, j] * flows_sorted[i, j]
#     return cost



# n, distances, flows = DataLoader.load('data/had12.dat')
# c1 = Creature(12)
# c1.genotype = np.array([3, 10, 11, 2, 12, 5, 6, 7, 8, 1, 4, 9]) - 1
# # cost = ga.calculate_cost(c1, distances, flows, n)
# # print(cost)
#
# # flows_sorted = flows[c1.genotype.argsort()]
# # flows_sorted = np.transpose(flows_sorted)
# # flows_sorted = flows_sorted[c1.genotype.argsort()]
# # flows_sorted = np.transpose(flows_sorted)
#
# # print(flows)
# # print(flows_sorted)
#
# cost = calculate_cost(c1, distances, flows, n)
# cost_vec = calculate_cost_vec(c1, distances, flows, n)
#
# print('cost: ' + str(cost))
# print('cost_vec: ' + str(cost_vec))