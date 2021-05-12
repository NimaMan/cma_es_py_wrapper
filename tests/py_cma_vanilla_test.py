
import sys
sys.path.append("./build")

import es
import cma
import numpy as np 

import time


# objective function.
def nfitfunc(x):
    val = 0.0
    n = len(x) 
    for i in range(0,n):
        val += x[i]*x[i]
    return val


def cpp_cma(f=nfitfunc):
    problem_dimension = 100
    x0 = np.array([1]*problem_dimension)
    population_size = 50 # lambda is a reserved keyword in python, using lambda_ instead.
    seed = 0 # 0 for seed auto-generated within the lib.
    sigma = 0.1

    esopt = es.cma( x0=x0, sigma=sigma, population_size=population_size, seed=seed) 
    t = time.process_time()
    for _ in range(1000):
        params = esopt.ask()
        fvals = []
        for p in params: 
            fvals.append(f(p))
        esopt.tell(fvals)
    
    elapsed_time = time.process_time() - t
    return fvals, elapsed_time

def py_cma(f=nfitfunc):
    problem_dimension = 100
    x0 = np.array([1]*problem_dimension)
    population_size = 50 # lambda is a reserved keyword in python, using lambda_ instead.
    seed = 0 # 0 for seed auto-generated within the lib.
    sigma = 0.1

    esopt = cma.CMAEvolutionStrategy(x0, sigma, 
       {"popsize": population_size,
        },
        )
    t = time.process_time()
    for _ in range(1000):
        params = esopt.ask()
        fvals = []
        for p in params: 
            fvals.append(f(p))
        esopt.tell(params, fvals)
    
    elapsed_time = time.process_time() - t
    return fvals, elapsed_time

def test_high_dimensional_ask_tell_communication_time_cpp(problem_dimension = 10000):
    x0 = np.array([1]*problem_dimension)
    population_size = 50 # lambda is a reserved keyword in python, using lambda_ instead.
    seed = 0 # 0 for seed auto-generated within the lib.
    sigma = 0.1

    esopt = es.cma(x0=x0, sigma=sigma, population_size=population_size, seed=seed) 
    #esopt = es.cma(dim=problem_dimension, x0=x0, sigma=sigma, population_size=population_size, seed=seed) 
    t = time.process_time()
    for _ in range(1):
        esopt.ask()
        print("ask done")
        mat = esopt.get_matrix()

        fvals = np.random.random(population_size)
        esopt.tell(fvals)
    
    elapsed_time = time.process_time() - t
    return elapsed_time

def test_high_dimensional_ask_tell_communication_time_py(problem_dimension = 10000):
    x0 = np.array([1]*problem_dimension)
    population_size = 50 # lambda is a reserved keyword in python, using lambda_ instead.
    seed = 0 # 0 for seed auto-generated within the lib.
    sigma = 0.1

    esopt = cma.CMAEvolutionStrategy(x0, sigma, 
       {"popsize": population_size,
        },
        )
    t = time.process_time()
    for _ in range(1):
        params = esopt.ask()
        fvals = np.random.random(population_size)
        esopt.tell(params, fvals)
    
    elapsed_time = time.process_time() - t
    return elapsed_time


print("CPP process time: ", cpp_cma()[1])
print("Py process time: ", py_cma()[1])

#print(test_high_dimensional_ask_tell_communication_time_cpp())
#print(test_high_dimensional_ask_tell_communication_time_py())