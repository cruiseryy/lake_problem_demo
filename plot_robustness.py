import matplotlib.pyplot as plt
import numpy as np
# from j3 import J3
from rhodium import *


robustness_dps = np.loadtxt("robustness_dps.txt")*100
robustness_IT = np.loadtxt("robustness_it.txt")*100

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1,1,1)
ax.fill_between(range(len(robustness_dps)),np.sort(robustness_dps)[::-1],color='#08519c')
ax.fill_between(range(len(robustness_IT)),np.sort(robustness_IT)[::-1],color='#a50f15')
ax.set_ylim([0,100])
ax.set_ylabel('Percent of Sampled SOWs in which Criteria are Met',fontsize=16)
ax.set_xlabel('Solution # (sorted by rank)',fontsize=16)
ax.tick_params(axis='both',labelsize=14)
plt.savefig('robustness_comparison.png')
plt.close()

dps_output=load("dps_output.csv")[1]
output=load("output.csv")[1]

for i in range(len(output)):
    output[i]['strategy']=1
    output[i]['robustness']=robustness_IT[i]
for i in range(len(dps_output)):
    dps_output[i]['strategy']=0
    dps_output[i]['robustness']=robustness_dps[i]

merged = DataSet(output+dps_output)

# J3(merged.as_dataframe(list(['max_P', 'utility', 'inertia', 'reliability', 'strategy', 'robustness'])))

DPSpolicy =np.argmax(robustness_dps)
ITpolicy =np.argmax(robustness_IT)

DPSpolicySOWs = load("reevaluation_dps_"+str(DPSpolicy)+".csv")[1]
ITpolicySOWs = load("reevaluation_it_"+str(ITpolicy)+".csv")[1]

SOWs=load("SOWs.csv")[1]

SOW_bqd = np.zeros([len(SOWs), 3])
for i in range(len(SOWs)):
    SOW_bqd[i,0]=SOWs[i]['b']
    SOW_bqd[i,1]=SOWs[i]['q']
    SOW_bqd[i,2]=SOWs[i]['delta']


# determine in which SOWs the most robust IT and DPS solutions fails
successes = [k for k in range(len(DPSpolicySOWs)) if DPSpolicySOWs[k]['reliability']>=0.95 and DPSpolicySOWs[k]['utility']>=0.2]
failures = [k for k in range(len(DPSpolicySOWs)) if DPSpolicySOWs[k]['reliability']<0.95 or DPSpolicySOWs[k]['utility']<0.2]
DPS_success = SOW_bqd[successes,:]
DPS_fail = SOW_bqd[failures,:]

successes = [k for k in range(len(ITpolicySOWs)) if ITpolicySOWs[k]['reliability']>=0.95 and ITpolicySOWs[k]['utility']>=0.2]
failures = [k for k in range(len(ITpolicySOWs)) if ITpolicySOWs[k]['reliability']<0.95 or ITpolicySOWs[k]['utility']<0.2]
IT_success = SOW_bqd[successes,:]
IT_fail = SOW_bqd[failures,:]

fig = plt.figure()
ax = fig.add_subplot(2,2,1)
successPts = ax.scatter(DPS_success[:,0],DPS_success[:,1],facecolor='#006d2c',edgecolor='none')
failPts = ax.scatter(DPS_fail[:,0],DPS_fail[:,1],facecolor='0.25',edgecolor='none',alpha=0.5)
ax.set_xlim(0.1,0.45)
ax.set_xticks(np.arange(0.1,0.5,0.1))
ax.set_yticks(np.arange(2.0,5.0,1.0))
ax.set_ylim(2.0,4.5)
ax.tick_params(axis='both',labelsize=14)
ax.set_ylabel('q',fontsize=16,rotation='horizontal')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.95])
ax.set_title('a) DPS: Effect of b and q',loc='left',fontsize=12)

ax = fig.add_subplot(2,2,2)
ax.scatter(DPS_success[:,2],DPS_success[:,1],facecolor='#006d2c',edgecolor='none')
ax.scatter(DPS_fail[:,2],DPS_fail[:,1],facecolor='0.25',edgecolor='none',alpha=0.5)
ax.set_xlim(0.93,0.99)
ax.set_xticks(np.arange(0.93,0.99,0.02))
ax.set_yticks(np.arange(2.0,5.0,1.0))
ax.set_ylim(2.0,4.5)
ax.tick_params(axis='both',labelsize=14)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.95])
ax.set_title('b) DPS: Effect of $\delta$',loc='left',fontsize=12)

ax = fig.add_subplot(2,2,3)
ax.scatter(IT_success[:,0],IT_success[:,1],facecolor='#006d2c',edgecolor='none')
ax.scatter(IT_fail[:,0],IT_fail[:,1],facecolor='0.25',edgecolor='none',alpha=0.5)
ax.set_xlim(0.1,0.45)
ax.set_xticks(np.arange(0.1,0.5,0.1))
ax.set_yticks(np.arange(2.0,5.0,1.0))
ax.set_ylim(2.0,4.5)
ax.tick_params(axis='both',labelsize=14)
ax.set_xlabel('b',fontsize=16)
ax.set_ylabel('q',fontsize=16,rotation='horizontal')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.95])
ax.set_title('c) Intertemporal: Effect of b and q',loc='left',fontsize=12)

ax = fig.add_subplot(2,2,4)
ax.scatter(IT_success[:,2],IT_success[:,1],facecolor='#006d2c',edgecolor='none')
ax.scatter(IT_fail[:,2],IT_fail[:,1],facecolor='0.25',edgecolor='none',alpha=0.5)
ax.set_xlim(0.93,0.99)
ax.set_xticks(np.arange(0.93,0.99,0.02))
ax.set_yticks(np.arange(2.0,5.0,1.0))
ax.set_ylim(2.0,4.5)
ax.tick_params(axis='both',labelsize=14)
ax.set_xlabel(r'$\delta$',fontsize=16)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.95])
ax.set_title('d) Intertemporal: Effect of $\delta$',loc='left',fontsize=12)

fig.suptitle('Parameter Combinations Leading to Failure',fontsize=16)
plt.figlegend([successPts, failPts],['Meets Criteria','Fails to Meet Criteria'],loc='lower center',ncol=2)
fig.set_size_inches([8.7375, 7.2])
plt.savefig('failure_parameters.png')
plt.close()