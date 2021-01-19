
import sim_platform as sim
import pickle
import sys
import csv

# The variable file name (v.f.n.) is of the form VariablesXXXXX.csv
# The code below extracts the nubmer XXXXX from the v.f.n. (--> no)
file_name = sys.argv[1]
no = file_name[9:]
no = no[:-4]


def get_list_parameters(file_name):
    '''We read this variables file and keep it in a dictionary of parameters.
    {parameter_name: parameter_value, ...}'''
    global list_parameters
    with open(file_name, mode='r') as infile:
        reader = csv.reader(infile)
        list_parameters = {rows[0]: rows[1] for rows in reader}

    return list_parameters


# get parameters from variable file
list_parameters = get_list_parameters(file_name)
# ininitialise simulation for these parameters
s = sim.SingleSimulation(list_parameters)
# run the created simulation (without printing intermediate steps)
s.get_stats(False)

# save simulation results in .pkl file with the same no
with open('Stats' + no + '.pkl', 'wb') as output:
    pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)
