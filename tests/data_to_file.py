import gen_alg as ga
import time
import numpy as np
import csv

TEST_ITERATIONS = 2

DATA_FILE = '../data/had12.dat'

ITERATIONS = [10, 30, 100, 300, 1000]
POPULATIONS = [10, 30, 100, 300, 1000]
MUTATION_PROBS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
CROSS_PROBS = [0.75, 0.9, 1]

# ITERATIONS = [10, 30, 100, 300]
# POPULATIONS = [10, 30, 100, 300, 1000]
# MUTATION_PROBS = [0.1]
# CROSS_PROBS = [1]

csvfile = open('results.csv', 'w', newline='')
results = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

for i in ITERATIONS:
    for p in POPULATIONS:
        for m in MUTATION_PROBS:
            for c in CROSS_PROBS:
                min_costs_all = []
                avg_costs_all = []
                best_cost_all = []

                exec_times = []

                start_time = time.time()

                for test_iteration in range(TEST_ITERATIONS):
                    iter_start_time = time.time()

                    print()
                    print('TEST ITERATION ' + str(test_iteration + 1))
                    min_costs, avg_costs, best, best_cost = ga.run_vec(DATA_FILE, p, m, i)
                    min_costs_all.append(min_costs)
                    avg_costs_all.append(avg_costs)
                    best_cost_all.append(best_cost)

                    exec_times.append(time.time() - iter_start_time)

                finish_time = time.time()
                elapsed_time = finish_time - start_time

                min_costs_all = np.asarray(min_costs_all)
                avg_costs_all = np.asarray(avg_costs_all)
                best_cost_all = np.asarray(best_cost_all)

                min_costs_avg = np.average(min_costs_all, axis=0)
                avg_costs_avg = np.average(avg_costs_all, axis=0)
                best_cost_avg = np.average(best_cost_all, axis=0)
                exec_time_avg = np.average(exec_times)

                results.writerow([i, p, m, c, best_cost_avg, exec_time_avg])
