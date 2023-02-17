from rhodium import *
import numpy as np
from scipy.optimize import brentq as root
import math
import ast
import matplotlib.pyplot as plt


dps_output=load("dps_output.csv")[1]
for i in range(len(dps_output)):
    dps_output[i]['policy']=ast.literal_eval(dps_output[i]['policy'])

policy_profit =dps_output.find_max('utility')
policy_reliability =dps_output.find_max('reliability')

def evaluateCubicDPS(policy, current_value):
    value = 0
    
    for i in range(policy["length"]):
        rbf = policy["rbfs"][i]
        value += rbf["weight"] * abs((current_value - rbf["center"]) / rbf["radius"])**3
        
    value = min(max(value, 0.01), 0.1)
    return value  


q = 2.0
b = 0.42
Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)

lake_state = np.arange(0,2.5,0.01)
Y1 = np.zeros(len(lake_state))
Y2 = np.zeros(len(lake_state))
for i in range(len(lake_state)):
    Y1[i] = evaluateCubicDPS(policy_reliability['policy'], lake_state[i])
    Y2[i] = evaluateCubicDPS(policy_profit['policy'], lake_state[i])
    
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1,1,1)
line1, = ax.plot(lake_state, Y1, c="#08519c",linewidth=2) # DPS best reliability
line2, = ax.plot(lake_state, Y2, c="#006d2c",linewidth=2) # DPS best benefits
line5, = ax.plot([Pcrit,Pcrit],[0,0.1],c='#a50f15',linewidth=2) # critical P threshold
ax.set_xlim(0,1)
ax.set_ylim(0,0.15)
ax.set_ylabel('Anthropogenic P Release, $a_t$',fontsize=16)
ax.set_xlabel('Lake P Concentration, $X_t$',fontsize=16)
ax.set_yticks(np.arange(0,0.16,0.04))
ax.tick_params(axis='both',labelsize=14)
plt.savefig('policy_comparison.png')
plt.close()

