import numpy as np
import gen_alg as ga
import cProfile

# cProfile.run('ga.run_vec(\'../data/had12.dat\', 200, 0.2, 200)', sort='tottime')
# cProfile.run('ga.run_vec(\'../data/had20.dat\', 200, 0.25, 200)', sort='tottime')
cProfile.run('ga.run_vec_time(\'../data/had20.dat\', 800, 0.05, 10, 1, \'insert\', \'tournament\', 10)', sort='tottime')
# ga.run_vec_time('../data/had20.dat', 1000, 0.02, 10, 1, 'swap', 'tournament', 50)