import numpy as np
from data_loader import DataLoader
from model import Creature
import time
import math


def create_random_population(population_size, n):
    population = []
    for i in range(population_size):
        population.append(Creature.random(n))
    return population


def calculate_costs(population, distances, flows, n):
    costs = []
    for creature in population:
        cost = 0
        for i in range(n):
            for j in range(n):
                cost += distances[i, j] * flows[creature.genotype[i], creature.genotype[j]]
        costs.append(cost)
    return costs


def calculate_costs_vec(population, distances, flows, n):
    costs = []
    for creature in population:
        # sorting flows matrix for matrix multiplication
        flows_sorted = flows[creature.genotype]
        flows_sorted = np.transpose(flows_sorted)
        flows_sorted = flows_sorted[creature.genotype]
        flows_sorted = np.transpose(flows_sorted)

        cost = distances.flatten().transpose().__matmul__(flows_sorted.flatten())
        costs.append(cost)
    return costs


def calculate_cost_vec(creature, distances, flows, n):
    flows_sorted = flows[creature.genotype]
    flows_sorted = np.transpose(flows_sorted)
    flows_sorted = flows_sorted[creature.genotype]
    flows_sorted = np.transpose(flows_sorted)

    cost = distances.flatten().transpose().__matmul__(flows_sorted.flatten())
    return cost


def calculate_cost(creature, distances, flows, n):
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += distances[i, j] * flows[creature.genotype[i], creature.genotype[j]]
    return cost


def mutate(population, mutation_prob, method='both'):
    if method == 'swap':
        mutate_swap(population, mutation_prob)
    elif method == 'insert':
        mutate_insert(population, mutation_prob)
    elif method == 'both':
        prob = mutation_prob / 2
        mutate_swap(population, prob)
        mutate_insert(population, prob)


def mutate_swap(population, mutation_prob):
    n = len(population[0].genotype)

    for c in population:
        for index in range(len(c.genotype)):
            if np.random.rand() < mutation_prob:
                index2 = np.random.randint(0, n)
                while index2 == index:
                    index2 = np.random.randint(0, n)

                temp = c.genotype[index]
                c.genotype[index] = c.genotype[index2]
                c.genotype[index2] = temp
                # print('m')


def mutate_insert(population, mutation_prob):
    n = len(population[0].genotype)

    for c in population:
        for index in range(len(c.genotype)):
            if np.random.rand() < mutation_prob:
                index2 = np.random.randint(0, n)
                while index2 == index:
                    index2 = np.random.randint(0, n)

                #insert
                c.genotype.insert(index2, c.genotype.pop(index))
                # print('m')


def selection(population, costs, method='tournament', tournament_size=2):
    selected = []
    best = None
    if method == 'tournament':
        selected, best = tournament_selection(population, costs, selected_number=len(population)//2, tournament_size=tournament_size)
    elif method == 'best_half':
        selected, best = selection_best_half(population, costs)
    elif method == 'roulette':
        selected, best = selection_roulette(population, costs, selected_number=len(population)//2)

    return selected, best


def selection_best_half(population, costs):
    # take half of population with the lowest costs
    ordered = [x for _, x in sorted(zip(costs, population), key=lambda x_y: x_y[0])]
    selected = ordered[0:len(population)//2]
    best = ordered[0]

    return selected, best


def tournament_selection(population, costs, selected_number, tournament_size):
    ordered = [x for _, x in sorted(zip(costs, population), key=lambda x_y: x_y[0])]
    best = ordered[0]
    selected = []

    for i in range(selected_number):
        tournament_pool = []
        for j in range(tournament_size):
            r = np.random.randint(0, len(ordered))
            tournament_pool.append(r)

        winner_index = min(tournament_pool)
        selected.append(ordered[winner_index])

    return selected, best


def selection_roulette(population, costs, selected_number):
    ordered = [x for _, x in sorted(zip(costs, population), key=lambda x_y: x_y[0])]
    best = ordered[0]

    # max_cost = max(costs)
    # min_cost = min(costs)
    #
    # # costs_normalized are in range [0,1] where lowest cost is 0 and highest 1
    # costs_normalized = [(x - min_cost) / (max_cost - min_cost) for x in costs]
    # probs = [1 - cost for cost in costs_normalized]

    min_cost = min(costs)
    max_cost = max(costs)
    # costs_normalized are in range [0,1] where lowest cost is 0 and highest 1
    costs_normalized = [(x - min_cost) / (max_cost - min_cost) for x in costs]
    costs_reversed = [1 - x for x in costs_normalized]

    sum_costs_reversed = sum(costs_reversed)
    probs = [(cost / sum_costs_reversed) for cost in costs_reversed]

    # print()
    # print('costs')
    # print(costs)
    # print('costs normalized:')
    # print(costs_normalized)
    # print('costs_reversed')
    # print(costs_reversed)
    # print('probs')
    # print(probs)
    # print()

    selected = []
    while len(selected) < selected_number:
        mini_sum = 0
        i = 0
        rand = np.random.rand()
        while mini_sum < rand:
            mini_sum += probs[i]
            i += 1
        index_to_add = max(0, i - 1)

        selected.append(population[index_to_add])

    return selected, best


def reproduce_population(selected, cross_prob):
    population_size = len(selected)
    pairs = list(zip(range(population_size), np.random.permutation(range(population_size))))
    next_generation = []
    for pair in pairs:
        children = reproduce_pmx(selected[pair[0]], selected[pair[1]], cross_prob)
        # children = reproduce(selected[pair[0]], selected[pair[1]])
        # children = reproduce(selected[pair[0]], selected[pair[1]], 2)
        next_generation += children

    return next_generation


def repair_children(child1, child2):
    indices_to_repair = get_indices_to_repair(child1) + get_indices_to_repair(child2)

    while len(indices_to_repair) > 0:
        # set random index from to_repair
        rand = np.random.randint(0, len(indices_to_repair))
        rand_index_to_repair = indices_to_repair[rand]

        # swap
        temp = child1.genotype[rand_index_to_repair]
        child1.genotype[rand_index_to_repair] = child2.genotype[rand_index_to_repair]
        child2.genotype[rand_index_to_repair] = temp

        indices_to_repair = get_indices_to_repair(child1) + get_indices_to_repair(child2)
        # print(indices_to_repair)
        # print(child1)
        # print(child2)

    return [child1, child2]


def get_indices_to_repair(creature):
    '''
    Checks if creature contains repeated genes and returns first error index
    :param creature:
    :return: index of first repeated gene or -1 if creature is valid
    '''

    index = 0
    genes_checked = []

    numbers_repeated = []

    while index < len(creature.genotype):
        if creature.genotype[index] in genes_checked:
            numbers_repeated.append(creature.genotype[index])

        genes_checked.append(creature.genotype[index])
        index += 1

    indices_to_repair = []

    for i in range(len(creature.genotype)):
        if creature.genotype[i] in numbers_repeated:
            indices_to_repair.append(i)

    return indices_to_repair


def cross_pmx(parent1, parent2):
    n = len(parent1.genotype)

    i = np.random.randint(0, n-2)
    j = np.random.randint(i + 1, n)

    p1 = parent1.genotype
    p2 = parent2.genotype
    child_genotype = [None for gene in p1]

    child_genotype[i:j] = p1[i:j]
    distribute = set(p2[i:j]) - set(child_genotype[i:j])

    for x in distribute:
        value = x
        while True:
            value = p1[p2.index(value)]
            value_index2 = p2.index(value)
            if value_index2 not in range(i, j):
                child_genotype[value_index2] = x
                break

    child_genotype = [p2[i] if (child_genotype[i] is None) else child_genotype[i]
             for i in range(len(child_genotype))]

    child = Creature(n)
    child.genotype = child_genotype

    return child


def reproduce_pmx(parent1, parent2, cross_prob):
    if np.random.rand() < cross_prob:
        children = [cross_pmx(parent1, parent2), cross_pmx(parent2, parent1)]
    else:
        children = [parent1, parent2]
    return children


def reproduce(parent1, parent2):
    n = len(parent1.genotype)
    split_index = np.random.randint(0, n)
    child1 = Creature(n)
    child2 = Creature(n)

    child1.genotype = list(parent1.genotype)
    child1.genotype[0:split_index] = parent2.genotype[0:split_index]

    child2.genotype = list(parent2.genotype)
    child2.genotype[0:split_index] = parent1.genotype[0:split_index]

    children = repair_children(child1, child2)

    return children


def repair(creature):
    n = creature.genotype.size
    present_genes = []
    for i in range(n):
        if creature.genotype[i] in present_genes:
            creature.genotype[i] = -1
        else:
            present_genes.append(creature.genotype[i])
    all_genes = range(n)
    missing_genes = [x for x in all_genes if x not in present_genes]
    np.random.shuffle(missing_genes)

    for i in range(n):
        if creature.genotype[i] == -1:
            creature.genotype[i] = missing_genes.pop()


# iteration-limited algorithm
def run_vec(data_filename, population_size, mutation_prob, iterations, cross_prob=1, mut_method='swap', sel_method='tournament', tournament_size=5):
    # load data
    n, distances, flows = DataLoader.load(data_filename)

    # set initial population
    # population = []
    # for i in range(population_size):
    #     creature = Creature(n)
    #     creature.genotype = np.random.permutation(n)
    #     population.append(creature)
    population = create_random_population(population_size, n)

    next_generation = population
    selected = []
    min_costs = []
    avg_costs = []
    best_cost = math.inf
    best_creature = []

    for iteration in range(iterations):
        population = next_generation

        costs = calculate_costs_vec(population, distances, flows, n)

        # print costs
        # print(costs)

        # set selection rules
        selected, best = selection(population, costs, sel_method, tournament_size)

        # set gene crossing rules
        next_generation = reproduce_population(selected, cross_prob)

        # set mutation rules
        mutate(next_generation, mutation_prob, mut_method)

        # add best from current generation to next generation
        to_throw_out = np.random.randint(0, len(next_generation))
        next_generation.pop(to_throw_out)
        next_generation.append(best)

        # save min and avg cost
        min_cost = min(costs)
        min_costs.append(min(costs))
        avg_costs.append(np.average(costs))

        if min_cost < best_cost:
            best_cost = min_cost
            best_creature_index = costs.index(min_cost)
            best_creature = population[best_creature_index]

        # print to follow progress
        # print('iteration ' + str(iteration) + ': ' + str(min(costs)) + '     population: ' + str(len(population)))
        print('{:<17}'.format('\riteration ' + str(iteration + 1) + ': ') + str(min(costs)), end='')
    print(end='\r')

    return min_costs, avg_costs, best_creature, best_cost


# time-limited algorithm
def run_vec_time(data_filename, population_size, mutation_prob, time_limit_seconds, cross_prob=1, mut_method='both', sel_method='best_half', tournament_size=2):
    start_time = time.time()
    last_iteration_finish = time.time()
    elapsed_time = 0

    # load data
    n, distances, flows = DataLoader.load(data_filename)

    # set initial population
    population = create_random_population(population_size, n)

    next_generation = population
    selected = []
    min_costs = []
    avg_costs = []
    best_cost = math.inf
    best_creature = []

    while elapsed_time < time_limit_seconds:
        population = next_generation

        # costs = calculate_costs(population, distances, flows, n)
        costs = calculate_costs_vec(population, distances, flows, n)

        # print costs
        # print(costs)

        # set selection rules
        selected, best = selection(population, costs, sel_method, tournament_size)

        # set gene crossing rules
        next_generation = reproduce_population(selected, cross_prob)

        # set mutation rules
        mutate(next_generation, mutation_prob, mut_method)

        # # add best from current generation to next generation
        # to_throw_out = np.random.randint(0, len(next_generation))
        # next_generation.pop(to_throw_out)
        # next_generation.append(best)

        # save min and avg cost
        min_cost = min(costs)
        min_costs.append(min(costs))
        avg_costs.append(np.average(costs))

        if min_cost < best_cost:
            best_cost = min_cost
            best_creature_index = costs.index(min_cost)
            best_creature = population[best_creature_index]

        # add best from current generation to next generation
        to_throw_out = np.random.randint(0, len(next_generation))
        next_generation.pop(to_throw_out)
        next_generation.append(best_creature)

        # print to follow progress
        # print('iteration ' + str(iteration) + ': ' + str(min(costs)) + '     population: ' + str(len(population)))
        print('{:<17}'.format('\rTime: ' + '{0:.2f}'.format(elapsed_time) + ' s') + 'Cost: ' + str(min(costs)), end='')

        last_iteration_finish = time.time()
        elapsed_time = last_iteration_finish - start_time

    print(end='\r')

    return min_costs, avg_costs, best_creature, best_cost


# old, slower version without cost vectorizing
def run(data_filename, population_size, mutation_prob, iterations):
    # load data
    n, distances, flows = DataLoader.load(data_filename)

    # set initial population
    population = []
    for i in range(population_size):
        creature = Creature(n)
        creature.genotype = np.random.permutation(n)
        population.append(creature)

    # print population
    # [print(line) for line in (list(map(lambda x: str(x), population)))]

    # set stop condition

    next_generation = population
    selected = []
    min_costs = []
    avg_costs = []

    for iteration in range(iterations):
        population = next_generation

        # evaluate creatures in population
        costs = calculate_costs(population, distances, flows, n)
        # costs = calculate_costs_vec(population, distances, flows, n)

        # print costs
        # print(costs)

        # set selection rules
        selected = selection(population, costs)
        # selected = tournament_selection(population, costs, int(len(population) / 2), )

        ordered = [x for _, x in sorted(zip(costs, population), key=lambda x_y: x_y[0])]

        # set gene crossing rules
        next_generation = reproduce_population(selected)

        # set mutation rules
        mutate(next_generation, mutation_prob)

        # add 10 best from current generation
        number_of_best_to_save = 1
        for i in range(number_of_best_to_save):
            next_generation.pop()
        next_generation += ordered[0:number_of_best_to_save]

        # save min and avg cost
        min_costs.append(min(costs))
        avg_costs.append(np.average(costs))

        # print to follow progress
        #print('iteration ' + str(iteration + 1) + ': ' + str(min(costs)) + '     population: ' + str(len(population)))
        print('{:<17}'.format('iteration ' + str(iteration + 1) + ': ') + str(min(costs)))

    return min_costs, avg_costs, selected[0], calculate_cost(selected[0], distances, flows, n)
