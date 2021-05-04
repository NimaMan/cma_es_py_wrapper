


# input parameters for a 10-D problem
x = [10]*10
lambda_ = 10 # lambda is a reserved keyword in python, using lambda_ instead.
seed = 0 # 0 for seed auto-generated within the lib.
sigma = 0.1

# objective function.
def nfitfunc(x,n):
    val = 0.0
    for i in range(0,n):
        val += x[i]*x[i]
    return val
