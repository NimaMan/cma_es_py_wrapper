
import sys
sys.path.append("./build")

import es
import cma
import numpy as np 

import time


problem_dimension = 10000
x0 = [10]*problem_dimension
population_size = 50 # lambda is a reserved keyword in python, using lambda_ instead.
seed = 0 # 0 for seed auto-generated within the lib.
sigma = 0.1

# objective function.
def nfitfunc(x):
    val = 0.0
    n = len(x) 
    for i in range(0,n):
        val += x[i]*x[i]
    return val


def cpp_cma(f=nfitfunc):
    t = time.process_time()

    esopt = es.cma(x0=x0, sigma=sigma, population_size=population_size, seed=seed) 
    for _ in range(1000):
        params = esopt.ask()
        fvals = []
        for p in params: 
            fvals.append(f(p))
        esopt.tell(fvals)
    
    elapsed_time = time.process_time() - t
    return fvals, elapsed_time

def py_cma(f=nfitfunc):
    t = time.process_time()
    esopt = cma.CMAEvolutionStrategy(x0, sigma, 
       {"popsize": population_size,
        },
        )
    for _ in range(1000):
        params = esopt.ask()
        fvals = []
        for p in params: 
            fvals.append(f(p))
        esopt.tell(params, fvals)
    
    elapsed_time = time.process_time() - t
    return fvals, elapsed_time



print(cpp_cma()[1])
print(py_cma()[1])