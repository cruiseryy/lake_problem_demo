'''
This script is a copy of this one https://github.com/Project-Platypus/Rhodium/blob/master/examples/Basic/dps_example.py
by Dave Hadka, with minor changes for training purposes by Antonia Hadjimichael.
'''

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *
from time import time

  
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

# Construct the lake problem
def lake_problem(pollution_limit,
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100): # monte carlo sampling of natural inflows
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(pollution_limit)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(pollution_limit)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    inertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return (max_P, utility, inertia, reliability)

model = Model(lake_problem)

# Define all parameters to the model that we will be studying
model.parameters = [Parameter("pollution_limit"),
                    Parameter("b"),
                    Parameter("q"),
                    Parameter("mean"),
                    Parameter("stdev"),
                    Parameter("delta")]

# Define the model outputs
model.responses = [Response("max_P", Response.MINIMIZE),
                   Response("utility", Response.MAXIMIZE),
                   Response("inertia", Response.MAXIMIZE),
                   Response("reliability", Response.MAXIMIZE)]

# Define any constraints (can reference any parameter or response by name)
model.constraints = [Constraint("reliability >= 0.95")]

# Some parameters are levers that we control via our policy
model.levers = [RealLever("pollution_limit", 0.0, 0.1, length=100)]

# Some parameters are exogeneous uncertainties, and we want to better
# understand how these uncertainties impact our model and decision making
# process
model.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]


output = optimize(model, "NSGAII", 2000)


#Save optimization results 
output.save("output.csv") 
output.as_dataframe()[list(model.responses.keys())].to_csv('output_objectives.csv')

# The same sets of SOWs are used to evaluate DPS and IT policies
# SOWs are generated using latain hypercube sampling via
# SOWs = sample_lhs(model, 1000)
# SOWs.save("SOWs.csv") 
SOWs=load("SOWs.csv")[1]
evaluate = timer_func(evaluate)
reevaluation_it = [evaluate(model, update(SOWs, policy)) for policy in output]

for i in range(len(reevaluation_it)):
    reevaluation_it[i].save("reevaluation_it_"+str(i)+".csv")

robustness_it = np.zeros(len(output))

for i in range(len(robustness_it)):
    robustness_it[i]=np.mean([1 if SOW['reliability']>=0.95 and SOW['utility']>=0.2 else 0 for SOW in reevaluation_it[i]])
    
np.savetxt("robustness_it.txt",robustness_it)

# parallel coordinate plot
# parallel_coordinates(model, output, colormap="rainbow", zorder="reliability", brush=Brush("reliability > 0.2"))     
# plt.show()

# obj pair scatter plots for comparing conflicting objs
# pairs(model, output)
# plt.show()

# Try this for the PRIM analysis to identify feasible parameter space for robust policy performance
# results = evaluate(model, update(SOWs, output[2]))
# metric = ["Success" if rr['reliability']>=0.95 and rr['utility']>=0.2 else 'Failure' for rr in results]
# p = Prim(results, metric, include=model.uncertainties.keys(), coi="Success")
# box = p.find_box()
# box.show_details()
# plt.show()

# try this for the Sobol analysis (variance-based sensitivity analysis using sobol sequences)
# policy = output.find_max("reliability")
# sobol_results = sa(model, "reliability", policy=policy, method="sobol", nsamples=1000)
# fig = sobol_results.plot_sobol(threshold=0.01)
# PAUSE = 1
