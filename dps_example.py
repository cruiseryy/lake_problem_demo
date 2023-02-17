'''
This script is a copy of this one https://github.com/Project-Platypus/Rhodium/blob/master/examples/Basic/dps_example.py
by Dave Hadka, with minor changes for training purposes. 
'''

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
from rhodium import *
# from platypus import wrappers
import ast

# Example using direct policy search (DPS) following the approach of [1]:
#
# [1] Quinn, J. D., P. M. Reed, and K. Keller (2017).  "Direct policy search for
#     robust multi-objective management of deeply uncertain socio-ecological
#     tipping points."  Environmental Modelling & Software, 92:125-141.

# Create a lever for storing the cubic radial basis functions used by this DPS.
# This lever stores one or more cubic radial basis functions defined by a center,
# radius, and weight.  We could have also created a RealLever for each value, but
# creating a class in this manner aids re-usability.

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

class CubicDPSLever(Lever):
    
    def __init__(self, name, length = 1, c_bounds = (-2, 2), r_bounds = (0.000001, 2)):
        super(CubicDPSLever, self).__init__(name)
        self.length = length
        self.c_bounds = c_bounds
        self.r_bounds = r_bounds
        
    # converts from Rhodium levers to Platypus decision variables; a single lever
    # can map to one or more decision variables.
    def to_variables(self):
        result = []
        
        for _ in range(self.length):
            result += [Real(self.c_bounds[0], self.c_bounds[1])] # the center
            result += [Real(self.r_bounds[0], self.r_bounds[1])] # the radius
            result += [Real(0.000001, 1)]                               # the weight
        
        return result
    
    # convert the value of the decision variables from Platypus back into Rhodium;
    # here we create a complex dictionary object storing the radial basis function
    # parameters.
    def from_variables(self, variables):
        policy = {}
        policy["length"] = self.length
        policy["rbfs"] = []
        
        # extract the parameters for each radial basis function
        for i in range(self.length):
            policy["rbfs"] += [{
                "center" : variables[i*3+0],
                "radius" : variables[i*3+1],
                "weight" : variables[i*3+2] }]
            
        # normalize the weights
        weight_sum = sum([p["weight"] for p in policy["rbfs"]])
        
        for i in range(self.length):
            policy["rbfs"][i]["weight"] /= weight_sum

        return policy
  
# A function for evaluating our cubic DPS.  This is based on equation (12)
# from [1].
def evaluateCubicDPS(policy, current_value):
    value = 0
    
    for i in range(policy["length"]):
        rbf = policy["rbfs"][i]
        value += rbf["weight"] * abs((current_value - rbf["center"]) / rbf["radius"])**3
        
    value = min(max(value, 0.01), 0.1)
    return value      

# Construct the lake problem
def lake_problem_dps(policy,  # the DPS policy
         b = 0.42,        # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,         # recycling exponent
         mean = 0.02,     # mean of natural inflows
         stdev = 0.001,   # standard deviation of natural inflows
         alpha = 0.4,     # utility from pollution
         delta = 0.98,    # future utility discount rate
         nsamples = 100,  # monte carlo sampling of natural inflows
         steps = 100):    # the number of time steps (e.g., days)
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    X = np.zeros((steps,))
    decisions = np.zeros((steps,))
    average_daily_P = np.zeros((steps,))
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = steps)
        
        for t in range(1,steps):
            decisions[t-1] = evaluateCubicDPS(policy, X[t-1])
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
        
        reliability += np.sum(X < Pcrit)/float(nsamples*steps)

      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(steps)))
    inertia = np.sum(np.diff(decisions) > -0.02)/float(steps-1)
    
    return (max_P, utility, inertia, reliability)

dps_model = Model(lake_problem_dps)

dps_model.parameters = [Parameter("policy"),
                    Parameter("b"),
                    Parameter("q"),
                    Parameter("mean"),
                    Parameter("stdev"),
                    Parameter("delta")]

dps_model.responses = [Response("max_P", Response.MINIMIZE),
                   Response("utility", Response.MAXIMIZE),
                   Response("inertia", Response.MAXIMIZE),
                   Response("reliability", Response.MAXIMIZE)]

# Use our new DPS lever
dps_model.levers = [CubicDPSLever("policy", length=3)]

# Define any constraints (can reference any parameter or response by name)
dps_model.constraints = [Constraint("reliability >= 0.95")]

dps_model.uncertainties = [UniformUncertainty("b", 0.1, 0.45),
                       UniformUncertainty("q", 2.0, 4.5),
                       UniformUncertainty("mean", 0.01, 0.05),
                       UniformUncertainty("stdev", 0.001, 0.005),
                       UniformUncertainty("delta", 0.93, 0.99)]

dps_output = optimize(dps_model, "NSGAII", 10000, epsilons=[0.01, 0.01, 0.0001, 0.0001])
dps_output.save("dps_output.csv") 
dps_output.as_dataframe()[list(dps_model.responses.keys())].to_csv('dps_output_objectives.csv')

#dps_output=load("dps_output.csv")[1]
#for i in range(len(dps_output)):
#    dps_output[i]['policy']=ast.literal_eval(dps_output[i]['policy'])
SOWs=load("SOWs.csv")[1]
evaluate = timer_func(evaluate)
reevaluation_dps = [evaluate(dps_model, update(SOWs, policy)) for policy in dps_output]

for i in range(len(reevaluation_dps)):
    reevaluation_dps[i].save("reevaluation_dps_"+str(i)+".csv")

robustness_dps = np.zeros(len(dps_output))

for i in range(len(robustness_dps)):
    robustness_dps[i]=np.mean([1 if SOW['reliability']>=0.95 and SOW['utility']>=0.2 else 0 for SOW in reevaluation_dps[i]])
    
np.savetxt("robustness_dps.txt",robustness_dps)


    
