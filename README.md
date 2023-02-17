# lake_problem_demo
This is a demo for solving multi-objective optimization using MORDM (Kasprzyk et al., 2013). A simplified lake problem is solved using MOEA+DPS in python (using Platypus &amp;Rhodium developed by David Hadka). For further details, please refer to Julianne Quinn's orinigal paper in 2017 (tipping point). Some changes are made to the scripts by Antonia Hadjimichael for training purposes.

Rhodium is required to run the demo scripts and can be installed via 
```
pip install rhodium
```
and further resources about Rhodium can be found here https://github.com/Project-Platypus/Rhodium

To optimize the system using a open loop intertemporal strategy, run ```example.py```.
To optimize the system using a close loop DPS strategy, run ```dps_example.py```.

For visualization, ```plot_plocies.py``` is to plot a DPS policy (decision as a function of system states); ```plot_robustness.py``` is for comparing robustness of DPS and intertemporal policies under altered SOWs; and ```obj_plot.py``` is for comparing DPS and intertemporal policies in the objective space.

Relevant references include but are not limited to:

[1] Kasprzyk, J. R., Nataraj, S., Reed, P. M., & Lempert, R. J. (2013). Many objective robust decision making for complex environmental systems undergoing change. Environmental Modelling & Software, 42, 55-71.

[2] Quinn, J. D., P. M. Reed, and K. Keller (2017).  "Direct policy search for robust multi-objective management of deeply uncertain socio-ecological tipping points."  Environmental Modelling & Software, 92:125-141.

[3] Hadjimichael, A., Gold, D., Hadka, D., & Reed, P. (2020). Rhodium: Python library for many-objective robust decision making and exploratory modeling. Journal of Open Research Software, 8.
