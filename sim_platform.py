import my_platform
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import ast


class SimulationStatistics:
    '''
    This class keeps tha statistics for one run/seed of the experiment
      (i.e. one population sample).
    Each will consist of a vector that catches the metric at each iteration.
    '''

    def __init__(self, no_iterations):
        self.no_iterations = no_iterations

        # number of long term relationships at each iteration
        self.no_long_term = np.zeros(no_iterations)

        # the average number of relationships that get to a first offline date
        self.no_first_date = {"total": 0, "out-group": 0}
        # the average number of relationships having a second date
        self.no_second_date = {"total": 0, "out-group": 0}

        # the number of outgroup long-term relationships
        self.no_outgroup_long_term = np.zeros(no_iterations)
        self.percentage_outgroup_long_term = np.zeros(no_iterations)

        # the number of online/offline (out-gr) relationships at each iteration
        self.no_online = np.zeros(no_iterations)
        self.no_outgroup_online = np.zeros(no_iterations)
        self.no_offline = np.zeros(no_iterations)
        self.no_outgroup_offline = np.zeros(no_iterations)

        # why did the agents exit (when exiting)
        self.exit_reason = {"Too many bad recommendations": 0,
                            "Too many failed relationships": 0, "Long term relationship": 0}
        # where did agents mostly spend their time
        self.time_by_phase = {"searching": 0, "online_rel": 0, "offline_rel": 0}
        self.percentage_of_time_by_phase = {"searching": 0, "online_rel": 0, "offline_rel": 0}
        # why do users form long-term rel
        self.no_outgroup_long_term_by_type = {'low': 0, 'medium': 0, 'high': 0}

    def divide_by(self, no_samples):
        '''Divides all the numbers by the no_samples.
        Useful for taking average statistics
        '''

        self.no_long_term /= no_samples

        self.no_first_date["total"] /= no_samples
        self.no_first_date["out-group"] /= no_samples
        self.no_second_date["total"] /= no_samples
        self.no_second_date["out-group"] /= no_samples

        self.no_outgroup_long_term /= no_samples
        self.percentage_outgroup_long_term /= no_samples

        self.no_online /= no_samples
        self.no_outgroup_online /= no_samples
        self.no_offline /= no_samples
        self.no_outgroup_offline /= no_samples

        for i in self.exit_reason.keys():
            self.exit_reason[i] /= no_samples
        for i in self.time_by_phase.keys():
            self.time_by_phase[i] /= no_samples
            self.percentage_of_time_by_phase[i] /= no_samples
        for i in self.no_outgroup_long_term_by_type.keys():
            self.no_outgroup_long_term_by_type[i] /= no_samples

    def add_values(self, other_stats):
        ''' Adds the values of another statistics componentwise.'''

        self.no_long_term += other_stats.no_long_term

        self.no_first_date["total"] += other_stats.no_first_date["total"]
        self.no_first_date["out-group"] += other_stats.no_first_date["out-group"]
        self.no_second_date["total"] += other_stats.no_second_date["total"]
        self.no_second_date["out-group"] += other_stats.no_second_date["out-group"]

        self.no_outgroup_long_term += other_stats.no_outgroup_long_term
        self.percentage_outgroup_long_term += other_stats.percentage_outgroup_long_term

        self.no_online += other_stats.no_online
        self.no_outgroup_online += other_stats.no_outgroup_online
        self.no_offline += other_stats.no_offline
        self.no_outgroup_offline += other_stats.no_outgroup_offline

        for i in self.exit_reason.keys():
            self.exit_reason[i] += other_stats.exit_reason[i]
        for i in self.time_by_phase.keys():
            self.time_by_phase[i] += other_stats.time_by_phase[i]
            self.percentage_of_time_by_phase[i] += other_stats.percentage_of_time_by_phase[i]
        for i in self.no_outgroup_long_term_by_type.keys():
            self.no_outgroup_long_term_by_type[i] += other_stats.no_outgroup_long_term_by_type[i]

    def get_percentage_outgroup_long_term(self):
        '''Finds the percentage of long-term relationships that are out-group,
        in each iteration.
        '''

        per_outgroup_lt = np.zeros(self.no_iterations)

        for i in range(self.no_iterations):
            if self.no_long_term[i] != 0:
                per_outgroup_lt[i] = self.no_outgroup_long_term[i]/self.no_long_term[i]

        return per_outgroup_lt

    def plot_percentage_outgroup_long_term(self, label=-1, col=-1, l_style='-'):
        ''' Plots the percentage of long term out-group relationships
        with/without label, with/without given colours'''

        if label == -1 and col == -1:
            plt.plot(self.percentage_outgroup_long_term, linestyle=l_style)
        elif col == -1:
            plt.plot(self.percentage_outgroup_long_term, label=label, linestyle=l_style)
        elif label == -1:
            plt.plot(self.percentage_outgroup_long_term, color=col, linestyle=l_style)
        else:
            plt.plot(self.percentage_outgroup_long_term, label=label, color=col, linestyle=l_style)


class SingleSimulation:
    ''' This class contains parameters and corresponding statistics.'''

    def __init__(self, list_parameters):
        self.parameters = list_parameters
        no_iterations = int(list_parameters['no_iterations'])
        self.statistics = SimulationStatistics(no_iterations)

    def add_values(self, other):
        self.statistics.add_values(other.statistics)

    def divide_by(self, no_samples):
        self.statistics.divide_by(no_samples)

    def get_different_parameters(self, other_sim):
        '''Returns a list with the names of the parameters that are different
        between the current simulation and another simulation (other_sim).
        '''

        differences = []
        for param_name in self.parameters.keys():
            if self.parameters[param_name] != other_sim.parameters[param_name]:
                differences += [param_name]

        return differences

    def get_stats(self, show_progress=True):
        '''Runs the simulation with the given parameters.'''

        # read simulation parameters
        no_iterations = int(self.parameters['no_iterations'])
        random_seed = int(self.parameters['random_seed'])
        initial_population_size = int(self.parameters['initial_population_size'])
        no_new_agents_per_iteration = int(self.parameters['no_new_agents_per_iteration'])
        norm_out_bias_value = float(self.parameters['norm_out_bias_value'])
        sum_searchable = float(self.parameters['sum_searchable'])
        # 0/1 depending on whetehr intervention 3 (on on-platfrom norms)
        #   is or not in place
        norm_intervention = int(self.parameters['norm_intervention'])

        # set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # generate a new platform
        p = my_platform.Platform(self.parameters)
        p.generate(initial_population_size, norm_out_bias_value, sum_searchable, norm_intervention)

        # p.statistics.write()
        for i in range(no_iterations):
            if show_progress:
                print('Seed ', random_seed, '--- iteration ', i)
                print(' no single agents - ', len(p.agents))

            # iterate the platform
            p.iterate(no_new_agents_per_iteration, no_bias=norm_intervention)

            # update the statistics
            self.statistics.no_long_term[i] = p.statistics.no_long_term
            self.statistics.no_outgroup_long_term[i] = p.statistics.no_outgroup_long_term
            if self.statistics.no_long_term[i] != 0:
                self.statistics.percentage_outgroup_long_term[i] = self.statistics.no_outgroup_long_term[
                    i]/self.statistics.no_long_term[i]
            self.statistics.no_online[i] = p.statistics.no_online
            self.statistics.no_offline[i] = p.statistics.no_offline
            self.statistics.no_outgroup_online[i] = p.statistics.no_outgroup_online
            self.statistics.no_outgroup_offline[i] = p.statistics.no_outgroup_offline

        # update the statistics on the survival rate of first dates
        self.statistics.no_first_date["total"] += p.statistics.no_first_date["total"]
        self.statistics.no_first_date["out-group"] += p.statistics.no_first_date["out-group"]
        self.statistics.no_second_date["total"] += p.statistics.surviving_first_date["total"]
        self.statistics.no_second_date["out-group"] += p.statistics.surviving_first_date["out-group"]

        self.statistics.exit_reason = p.statistics.exit_reason
        no_exit = sum([p.statistics.exit_reason[i] for i in p.statistics.exit_reason.keys()])
        # where did agents mostly spend their time
        self.statistics.time_by_phase = p.statistics.time_by_phase
        self.statistics.percentage_of_time_by_phase = {}
        for i in p.statistics.percentage_of_time_by_phase.keys():
            self.statistics.percentage_of_time_by_phase[i] = p.statistics.percentage_of_time_by_phase[i]/no_exit
        # why do users form long-term rel
        self.statistics.no_outgroup_long_term_by_type = p.statistics.no_outgroup_long_term_by_type


class Simulations:
    '''This class contains multiple simulations and makes the relevant plot.
    '''

    def __init__(self, list_simulations):
        # a given list of variables of type simulation
        self.list_simulations = list_simulations
        self.names_parameters = list_simulations[0].parameters.keys()

    def group_by_non_seed(self):
        '''Goes through the lsit of simulations and groups up everything
        having all the parameters but the seed identical'''

        new_list = []
        considered = []
        for i in range(len(self.list_simulations)):
            # check that the current simulation was not already bundeled
            if i not in considered:
                considered += [i]
                current_sim = self.list_simulations[i]
                # check for other simulations
                no_samples = 1
                new_stats = deepcopy(current_sim)
                for j in range(i+1, len(self.list_simulations)):
                    considered_sim = self.list_simulations[j]
                    if (j not in considered) and (current_sim.get_different_parameters(considered_sim) in [[], ["random_seed"]]):
                        # remove j
                        considered += [j]
                        no_samples += 1
                        new_stats.add_values(considered_sim)
                # average over the number of samples
                new_stats.divide_by(no_samples)
                # add new aggregated statistics to the new list
                new_list.append(deepcopy(new_stats))

        self.list_simulations = deepcopy(new_list)

    def identify_changing_parameters(self):
        '''Looks at the parameters of each simulation in the list and returns
        a dictionary of the ones that are different and their values
        {param_name: [value1, value2, ...]}'''

        changing_params = {}
        for param_name in self.names_parameters:
            param_value = self.list_simulations[0].parameters[param_name]
            no_distinct_found = True

            for sim in self.list_simulations:
                new_param_value = sim.parameters[param_name]
                if param_value != new_param_value:
                    if no_distinct_found:
                        changing_params[param_name] = [param_value, new_param_value]
                        no_distinct_found = False
                    elif new_param_value not in changing_params[param_name]:
                        changing_params[param_name] += [new_param_value]

        return changing_params

    def remove_duplicates(self, progress=False):
        '''Removes simulations that are the same'''
        to_remove = []
        no_sims = len(self.list_simulations)
        for i in range(no_sims):
            if progress and i % 10000 == 0:
                print(i)
            s = self.list_simulations[i]
            for j in range(i+1, no_sims):
                s2 = self.list_simulations[j]
                if s.parameters == s2.parameters and s2 not in to_remove:
                    to_remove.append(s2)
        for s in to_remove:
            self.list_simulations.remove(s)

    def sd(self):
        '''Gets the variance of simulations from the list'''
        no_sim = len(self.list_simulations)
        no_iter = int(self.list_simulations[0].parameters['no_iterations'])

        # find the expected value
        exp = np.zeros(no_iter)
        for s in self.list_simulations:
            # print(s.statistics.percentage_outgroup_long_term[-1])
            exp += s.statistics.percentage_outgroup_long_term
        exp /= no_sim

        # find the variance
        var = 0
        for s in self.list_simulations:
            var += (s.statistics.percentage_outgroup_long_term - exp)**2
        var /= no_sim

        return np.sqrt(var)

    def filter(self, param_name, list_param_values):
        '''
        For the value of param_name levaes only the values in the list.
        param_name = string with the name of the parameter we want to filter on
        list_param_values = list with remaining values
        '''

        to_remove = []
        for sim in self.list_simulations:
            if sim.parameters[param_name] not in list_param_values:
                to_remove.append(sim)

        for sim in to_remove:
            self.list_simulations.remove(sim)

    def filter_multiple(self):
        '''Sees the changing parameters and asks wich ones to keep.
        This requires keyboard input.
        '''

        changing_params = self.identify_changing_parameters()
        for param_name in changing_params.keys():
            print("The parameter ", param_name,
                  " is changing with values ", changing_params[param_name])
            print("Do you want to filter these values? [Y/N]")
            filter = input()
            if filter == "Y":
                print("Please list the values you want to remain:")
                list_remain = ast.literal_eval(input())
                self.filter(param_name, list_remain)

    def plot_stats(self, group_by_intn=False):
        '''Plots the relevant statistics depending on the changing parameters
        '''

        cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        line_styles = ["-", "--", "-.", ":"]

        def plot_one(simulation, col=-1, line_style="-", add_label=True):
            if add_label:
                label = ''
                for p in changing_params:
                    if p != "norm_intervention":
                        label += p + "=" + simulation.parameters[p] + " | "
            else:
                label = -1
            simulation.statistics.plot_percentage_outgroup_long_term(label, col, line_style)

        # identify the changing parameters
        changing_params = list(self.identify_changing_parameters().keys())

        if not group_by_intn:
            # plot each statistic and add the relevant variable on the legend
            for simulation in self.list_simulations:
                plot_one(simulation, add_label=True)
        else:
            col = 0
            # plot stats with intervention 0-1 in paris, same col
            for sim in self.list_simulations:
                if sim.parameters["norm_intervention"] == '1':
                    col = (col+1) % len(cols)
                    plot_one(sim, cols[col])
                    for sim_without_intn in self.list_simulations:
                        if sim.get_different_parameters(sim_without_intn) == ["norm_intervention"]:
                            plot_one(sim_without_intn, cols[col], "--", False)

        plt.xlabel('Time')
        plt.ylabel('Percentage of out-group long-term relationships')
        plt.legend()
        plt.show()
