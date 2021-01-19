# An Agent-based Model to Evaluate Interventions on Online Dating Platforms to Decrease Racial Homogamy
This repository contains an agent-based simulation for an online diating platform. It tracks the effects of different platform interventions on the number of out-group relationships.

For detiled information on the model please see the paper titled "An Agent-based Model to Evaluate Interventions on OnlineDating Platforms to Decrease Racial Homogamy" (- FAccT 2021).

## Programming language
The simulation is coded in Python - version 3.8.7.

## Files
Before running the code, please make sure all the required libraries are installed. For each *.py file, the used libraries are imported at the top of the code.

In this repository, you will find the following files:

### Variables.csv
A file with all the relevant parameters, some example values, and a short explanation of what each parameter controls. Note, that there are more parameters than described in the paper. For simplicity, within the paper, we kept some of them (e.g. the types of agents) fixed.

Variable files of this type can be used to run different simulations. You can either change this one, or create a new following the structure of Variables.csv.

### Agent.py
This defines classes for attributes, orientation, and agents. The decision function used by the agents is also defined at the end of the file, within the Agent class.

### My_platfrom.py
This defines classes for relationships, platform, and platform statistics. See the 'iterate' procedure within Relationship and Platform to see how each evolves from one iteration to the next.

### Sim_platform.py
Contains classes for simulation and simulation statistics.

### Run_one.py
This can be run in bash to perform run one parameter configuration. For example, if you want to run the configuration in Variables10.csv, you can run the following command:
```bash
python3 run_one.py Variables10.csv
```
The script creates a *.pkl file with the simulation results for an object of the type SingleSimulation. For the example above it will be named Stats10.pkl.

### Ve.py
This is an example file for running a virtual experiment. This is not part of the core simulation, but could be helpful to do the following:
1) have an example for an automated procedure to generate VariablesXXXXX.csv files
2) parallelising running of the simulation to make use of multi-core machines
3) make .csv files for the .pkl results

### Data_analysis.py
The file contains code for processing simulation-results .csv files (resulted from .pkl files, as done in ve.py), and making data visualisations. Again, this is not part of the core simulation, but could be useful as an example for visualising the data.
