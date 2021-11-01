# Python wrapper around the libcma (C++ implementation of the CMA-ES) 

## USage

```python

def nfitfunc(x):
    val = 0.0
    n = len(x) 
    for i in range(0,n):
        val += x[i]*x[i]
    return val

if __name__ == "__main__":
    problem_dimension = 100
    number_of_optimization_episodes = 1000
    x0 = np.array([1]*problem_dimension)
    population_size = 50 
    seed = 0
    sigma = 0.1

    esopt = es.cma( x0=x0, sigma=sigma, population_size=population_size, seed=seed) 
    for _ in range(number_of_optimization_episodes):
        params = esopt.ask()
        fvals = []
        for p in params: 
            fvals.append(f(p))
        esopt.tell(fvals)
    
    
```

