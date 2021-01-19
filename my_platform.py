import copy
import numpy as np
import random
import agent

# Defines platforms and relationships

class Relationship:
    def __init__(self, agents, knowledge=[]):
        list_parameters = agents[0].list_parameters

        self.agents = copy.deepcopy(agents)  # list of 2 agents in rel
        self.time = 0  # no of iterations spent so far in rel
        # info(Attributes) accumulated so far by the two agents
        self.knowledge = copy.deepcopy(knowledge)
        self.time_to_offline = int(list_parameters['time_to_offline'])
        self.time_to_long_term = int(list_parameters['time_to_long_term'])
        # probability of learning searchable/experiential attributes
        self.prob_learn_searchable = float(list_parameters['prob_learn_searchable'])
        self.prob_learn_experiential = float(list_parameters['prob_learn_experiential'])
        # set the initial knowledge to be the profile-observable attributes
        # knowledge is a list that has two positions
        #   0 - the attributes of 1 known by agent 0
        #   1 - the attributes of 0 known by agent 1
        if knowledge == []:
            self.knowledge = [agents[1].get_profile_visible_attributes(
            ), agents[0].get_profile_visible_attributes()]
        # the probability the two agents interact when offline
        self.prob_interact_offline = float(list_parameters['prob_interact_offline'])
        # number of offline dates the 2 agents had
        self.no_offline_dates = 0

    def is_out_group(self):
        '''
        Find if the current relationship has partners belonging to different
        bias groups.
        '''

        is_bias = False
        for i in self.agents[0].attributes.get_bias_indices():
            # if the two agents are in diffeent bias groups w.r.t. index i
            if self.agents[0].attributes.values[i] != self.agents[1].attributes.values[i]:
                is_bias = True

        return is_bias

    def is_offline(self):
        '''Retruns True iff the relationship is offline.
        '''
        return self.time >= self.time_to_offline

    def is_first_date(self):
        return self.no_offline_dates == 1

    def interact(self):
        '''
        Updates the knowledge and the observed sample of the agent.
        '''

        # update knoledge about the other, for each agent
        # update searchable first
        for i in self.agents[0].attributes.get_searchable_indices():
            for a in range(2):
                other_a = 1 - a
                # if a doesn't know attribute i of other_a, learn with prob_s
                if self.knowledge[a].values[i] == -1:
                    if np.random.random() < self.prob_learn_searchable:
                        self.knowledge[a].values[i] = self.agents[other_a].attributes.values[i]

        # update experiential if offline
        if self.is_offline():
            for i in self.agents[0].attributes.get_experiential_indices():
                for a in range(2):
                    other_a = 1 - a
                    # if a doesn't know attribute i of other_a, learn with prob_s
                    if self.knowledge[a].values[i] == -1:
                        if np.random.random() < self.prob_learn_searchable:
                            self.knowledge[a].values[i] = self.agents[other_a].attributes.values[i]

        # update the sample of the agents
        self.agents[0].update_sample(self.knowledge[0])
        self.agents[1].update_sample(self.knowledge[1])

    def both_want_to_continue(self):
        '''
        The two agents decide whether tey continue or not.
        Both answer whether to continue & both must answer 1 for the
         relationship to continue.
        '''

        # when offline time gets resetted for decision purposes
        t = (self.no_offline_dates-1) if self.is_offline() else self.time

        d1 = self.agents[0].answer(self.knowledge[0], time=t, first=self.is_first_date())
        d2 = self.agents[1].answer(self.knowledge[1], time=t, first=self.is_first_date())

        # each decision is 0 or 1
        # for the relationship to continue both need to decide for 1
        # so, continuation is the product of decisions
        return d1*d2

    def iterate(self):
        '''
        The relationship is iterated once.
        1. agents interact (update knoledge and sample)
        2. agents decide if they continue
           (update preferences based on the decision)
        3. if continue (increase rel time)
        4. if not (decrease tollerance of agents)

        returns:
         - status = whether rel continues, is long_term, or interrupted
         - remove = 0/1 agent ids that need to be removed from the system
        '''
        # increase the contor for the time spent in rel
        rel = "offline_rel" if self.is_offline() else "online_rel"
        self.agents[0].no_iterations[rel] += 1
        self.agents[1].no_iterations[rel] += 1

        interact = True
        first_interaction = False
        if self.is_offline():
            if np.random.random() > self.prob_interact_offline:
                interact = False
            else:
                self.no_offline_dates += 1
                first_interaction = self.is_first_date()

        if interact:
            # Step 1 - perform interaction
            self.interact()

            # Step 2 - decide if continue
            decision = self.both_want_to_continue()

            # Step 2 - update preferences
            self.agents[0].update_preferences_after_interaction(self.knowledge[0], decision)
            self.agents[1].update_preferences_after_interaction(self.knowledge[1], decision)
        else:
            decision = 1

        # Steps 3 & 4
        remove = []  # agents that are removed
        status = 'continues'
        if decision:
            self.time += 1
            if self.time > self.time_to_long_term:
                remove = [1, 2]
                status = 'long_term'
        else:
            status = 'interrupted'
            for a in range(2):
                self.agents[a].failure_tolerance -= 1
                if self.agents[a].failure_tolerance < 0:
                    remove += [a]

        return (first_interaction, status, remove)


class PlatformStatistics:
    def __init__(self, no_online=0, no_offline=0, no_long_term=0,
                 no_outgroup_online=0, no_outgroup_offline=0,
                 no_outgroup_long_term=0,
                 no_first_date={"total": 0, "out-group": 0},
                 surviving_first_date={"total": 0, "out-group": 0}):
        self.no_online = no_online
        self.no_offline = no_offline
        self.no_long_term = no_long_term
        self.no_outgroup_online = no_outgroup_online
        self.no_outgroup_offline = no_outgroup_offline
        self.no_outgroup_long_term = no_outgroup_long_term
        # number of relationships entering the first meeting
        # [total, out-group]
        self.no_first_date = no_first_date
        # number of relationships survivint the first meeting
        # [total, out-group]
        self.surviving_first_date = surviving_first_date

        # why did agents exit the platform
        self.exit_reason = {"Too many bad recommendations": 0,
                            "Too many failed relationships": 0, "Long term relationship": 0}
        # where did agents mostly spend their time
        self.time_by_phase = {"searching": 0, "online_rel": 0, "offline_rel": 0}
        self.percentage_of_time_by_phase = {"searching": 0, "online_rel": 0, "offline_rel": 0}

        # why do users form long-term rel
        self.no_outgroup_long_term_by_type = {'low': 0, 'medium': 0, 'high': 0}

    def write(self):
        '''Writes current platform values for the no of relationships'''
        print('No online', self.no_online)
        print('No offline', self.no_offline)
        print('No long term', self.no_long_term)


class Platform:
    def __init__(self, list_parameters, norm_on=[], norm_out=[],
                 active_rel=[], agents={},
                 pending_messages={}, max_id_agent=-1,
                 statistics=PlatformStatistics()):
        self.list_parameters = list_parameters
        self.norm_on = copy.deepcopy(norm_on)
        self.norm_out = copy.deepcopy(norm_out)
        self.active_rel = copy.deepcopy(active_rel)
        # dictionary id_agent: agent; only for the single agents
        self.agents = copy.deepcopy(agents)
        self.pending_messages = copy.deepcopy(pending_messages)
        self.max_id_agent = max_id_agent  # the number of agents added so far
        self.statistics = copy.deepcopy(statistics)

        # get the unbiased contingency from beginning
        def form_lists_contingency():
            x = agent.Agent(0, list_parameters)
            cont_all = x.generate_unbiased_contingency()

            # group for bias group 0
            no_comb = len(cont_all)
            cont = [np.zeros(no_comb) for i in range(2)]
            for i in range(no_comb):
                bias_gr = 0 if i < no_comb/2 else 1
                cont[bias_gr][i] = cont_all[i]
            # normalise
            for i in range(2):
                s = sum(cont[i])
                cont[i] /= s

            return [cont_all, list(cont[0]), list(cont[1])]

        self.cont = form_lists_contingency()

    def add_one_agent(self):
        '''Adds one more agents to the dictionary of agents.'''
        self.max_id_agent += 1
        self.agents[self.max_id_agent] = agent.Agent(self.max_id_agent, self.list_parameters)
        self.agents[self.max_id_agent].generate(self.norm_out, self.cont)
        self.pending_messages[self.max_id_agent] = []

    def generate_agents(self, no_agents):
        '''Generates no_agents agents'''

        for i in range(no_agents):
            self.add_one_agent()

    def generate_norm_on(self, no_bias=False):
        '''Generates an initial norm on the platform.
        For now, it's just the mean of the normalised preferences of the agents
           for the searchable attributes.
        Always generate the norm after genreating the agents.
        if no_bias is True, then set the bias norm to 0
        '''

        no_agents = len(self.agents)

        attributes_ex = agent.Attributes(self.list_parameters)
        bias_indices = attributes_ex.get_bias_indices()
        experiental_indices = attributes_ex.get_experiential_indices()

        # add up the normalised preferences of searchable attributes
        self.norm_on = sum(self.agents[i].preferences /
                           self.agents[i].get_importance_searchable()
                           for i in self.agents.keys())
        self.norm_on /= no_agents
        for i in experiental_indices:
            self.norm_on[i] = 0

        # in case we intervene not to have any bias
        if no_bias:
            sum_bias_attributes = 0
            # set the norm on bias attribute to 0
            for i in bias_indices:
                sum_bias_attributes += self.norm_on[i]
                self.norm_on[i] = 0
            # re-do normalisation - for relative interpretation
            if sum_bias_attributes == 1:
                print("Error: the norm cannot be to only care about bias att.")
            self.norm_on *= (1/(1-sum_bias_attributes))

    def generate_norm_out(self, bias_value, sum_searchable):
        '''
        Generates the norm outside the platform.
        It has exactly bias_value for the protected attributes.
        The values for the other attributes are generated at random.
        '''

        attributes_ex = agent.Attributes(self.list_parameters)
        no_attributes = attributes_ex.no_attributes
        bias_indices = attributes_ex.get_bias_indices()
        searchable_indices = attributes_ex.get_searchable_indices()

        # norm_out_i = random numbers in (0, 1)
        self.norm_out = agent.generate_norms_pref(
            no_attributes, bias_value, sum_searchable, bias_indices, searchable_indices, True)

    def generate(self, no_agents, norm_out_bias_value, sum_searchable, no_bias=False):
        ''' Puts together the three generate methods: out-norm, agents, on-norm.
        '''

        self.generate_norm_out(norm_out_bias_value, sum_searchable)
        self.generate_agents(no_agents)
        self.generate_norm_on(no_bias)

    def recommend(self, id_agent):
        '''
        Gives a list of IDs of users in a specific order.
        For now, recommand at renadom from the set of agents that are not in a
         relationship, did not already sent a message to the agent,
         were not already in a relationship with the agent,
         and do fit the type of interests of our agent
        id_agent = the agent for which we show the lsit of recommandations
        '''

        # for filtering purposes - get the attribute of max imporance
        our_ag = self.agents[id_agent]
        # find the index of the most important searchable attribute
        max_attribute = -1
        max_val = -1
        if 'non_bias' not in our_ag.filter:
            type_filter = our_ag.filter
            for i in our_ag.attributes.get_searchable_indices():
                if max_val < our_ag.preferences[i]:
                    max_val = our_ag.preferences[i]
                    max_attribute = i
        else:
            type_filter = our_ag.filter[9:]
            for i in our_ag.attributes.get_searchable_indices():
                if (i not in our_ag.attributes.get_bias_indices()) and max_val < our_ag.preferences[i]:
                    max_val = our_ag.preferences[i]
                    max_attribute = i

        is_matching = (max_attribute in our_ag.attributes.get_matching_indices())

        # Choose the agents that are not in rel & not involved with current ag
        l = []
        for ag in self.agents.keys():
            other_ag = self.agents[ag]

            condn = True
            # condn = (ag not in self.pending_messages[id_agent]) & (id_agent not in self.pending_messages[ag])
            if condn:
                # the agent must not be the same with the considered one
                if ag != id_agent:
                    # the orientation types match
                    # i.e. the agent recommended is in the list of interests of the current agent
                    if other_ag.orientation.type in our_ag.orientation.interests:
                        # check if our agent wanted to filter the recommandations
                        if type_filter == 'OFF':
                            l += [ag]
                        elif type_filter in ['STRONG', 'WEAK']:
                            # attributes_other = other_ag.get_profile_visible_attributes().values
                            vis_attr_other = other_ag.visible_attributes_indices
                            is_unknown = (max_attribute not in vis_attr_other)
                            is_equal = (not is_unknown) and (
                                other_ag.attributes.values[max_attribute]
                                == our_ag.attributes.values[max_attribute])
                            is_1 = (not is_unknown) and (
                                other_ag.attributes.values[max_attribute] == 1)
                            # both for STRONG and WEAK, add agent if the characteristic is known and good
                            if is_matching and is_equal:
                                l += [ag]
                            if (not is_matching) and is_1:
                                l += [ag]
                            # if WEAK can also be unspecified
                            if is_unknown and type_filter == 'WEAK':
                                l += [ag]

        # Permute the list l
        return np.random.permutation(l)

    def iterate_single_agent(self, id_agent):
        '''
        Does one time step for an agent.
         1. consideres in turn answering messages / texting
         2. when saying YES enter relationship/send message/stop
        prob_check_message = probability of looking at one of the sent texts
        '''
        # increase the contor for no. iterations in searching
        self.agents[id_agent].no_iterations["searching"] += 1

        prob_check_messages = float(self.list_parameters['prob_check_messages'])
        no_steps_searching = int(self.list_parameters['no_steps_searching'])

        replied = False
        list_recommandations = self.recommend(id_agent)
        last_considered_recommandation = -1
        no_considered_profiles = 0
        no_texted_profiles = 0
        while ((not replied) and no_steps_searching):
            if (np.random.random() < prob_check_messages) & (len(self.pending_messages[id_agent]) != 0):
                # the agent consideres anserwing one pending message
                id_considered = self.pending_messages[id_agent][0]
                # can only enter relationship if the other is single
                if id_considered in self.agents.keys():
                    # the agent decides whether to enter the relationship
                    knowledge_about_other = self.agents[id_considered].get_profile_visible_attributes(
                    )
                    # the agent updates their contingency bsed on knoweldge
                    self.agents[id_agent].update_sample(knowledge_about_other)
                    if self.agents[id_agent].answer(knowledge_about_other):
                        # enter relationship
                        self.active_rel += [Relationship([self.agents[id_agent],
                                                          self.agents[id_considered]])]

                        # update the statistics
                        self.statistics.no_online += 1
                        if self.active_rel[-1].is_out_group():
                            self.statistics.no_outgroup_online += 1

                        # the two agents are not single any more
                        self.agents.pop(id_agent)
                        self.agents.pop(id_considered)
                        self.pending_messages.pop(id_agent)
                        self.pending_messages.pop(id_considered)

                        # the agent already made a decision
                        replied = True

                    if not replied:
                        self.pending_messages[id_agent].pop(0)
            else:
                # otw the agent consideres texting one more agent from the list of recommandations
                last_considered_recommandation += 1
                no_considered_profiles += 1
                # if there are no more recommandations in the lsit, search again
                if last_considered_recommandation >= len(list_recommandations):
                    list_recommandations = self.recommend(id_agent)
                    last_considered_recommandation = 0

                    # stop if the list of recommandations is empty (avoid infinite loop)
                    if len(list_recommandations) == 0:
                        break

                # consider texting the next person
                id_considered = list_recommandations[last_considered_recommandation]
                knowledge_about_other = self.agents[id_considered].get_profile_visible_attributes()
                # the agent updates their contingency bsed on
                self.agents[id_agent].update_sample(knowledge_about_other)
                if self.agents[id_agent].answer(knowledge_about_other):
                    no_texted_profiles += 1
                    # if texted before -> don't add - don't keep doubles
                    # if the other has texted -> text them as initial mess (interpret as another agent, same tipology)
                    # condn1 = (id_considered not in self.pending_messages[id_agent])
                    # condn2 = (id_agent not in self.pending_messages[id_considered])
                    # if condn2:

                    # send a message to the agent
                    self.pending_messages[id_considered] += [id_agent]
                    # a decision was made - changed to only reply
                    # replied = True

            no_steps_searching -= 1

        # if they stopped not by deciding what to do
        # -> the tolerance for bad reommandations decreases
        if not replied:
            if no_considered_profiles:
                per_good = no_texted_profiles/no_considered_profiles
            else:
                per_good = 0
            if per_good < 0.5:
                self.agents[id_agent].bad_recommandation_tolerance -= 1
                # remove the agent if they've reached the tollerance for bad recommandations
                if self.agents[id_agent].bad_recommandation_tolerance < 0:
                    # increase the statistics counter for bad rec
                    self.statistics.exit_reason["Too many bad recommendations"] += 1
                    # add the stats on where the agent spent their time
                    ag_no_it = self.agents[id_agent].no_iterations
                    s = sum([ag_no_it[i] for i in ["searching", "online_rel", "offline_rel"]])
                    for i in ["searching", "online_rel", "offline_rel"]:
                        self.statistics.time_by_phase[i] += ag_no_it[i]
                        self.statistics.percentage_of_time_by_phase[i] += ag_no_it[i]/s
                    # remove the agent
                    self.agents.pop(id_agent)

    def one_iteration_arrival_new_agents(self, no_agents):
        '''Adds the new agents that arrive at one iteration.
        For now, add a fixed number of agents.'''

        self.generate_agents(no_agents)

    def check(self):
        '''Debug function. It checks if the percentages are computed correctly.
        '''
        no_exit = sum([self.statistics.exit_reason[i]
                       for i in self.statistics.exit_reason.keys()])
        if no_exit > 0:
            percentage_of_time_by_phase = {}
            for i in self.statistics.percentage_of_time_by_phase.keys():
                percentage_of_time_by_phase[i] = self.statistics.percentage_of_time_by_phase[i]/no_exit
            sum_prob = sum([percentage_of_time_by_phase[i]
                            for i in self.statistics.percentage_of_time_by_phase.keys()])
            if abs(sum_prob - 1) > 0.01:
                print(percentage_of_time_by_phase)
                print(sum_prob, no_exit)
                return False
            else:
                return True
        return True

    def iterate_relationships(self):
        '''Iterates the relationships'''

        rel_to_be_removed = []
        # for each active relationsip
        for r in self.active_rel:

            # iterate relationship and check if the status changed
            offline_before = r.is_offline()
            was_first_date, status, remove = r.iterate()
            offline_after = r.is_offline()
            changed_to_offline = offline_after and not offline_before

            # update the statistics on the survival of first dates
            if was_first_date:
                self.statistics.no_first_date["total"] += 1
                if r.is_out_group():
                    self.statistics.no_first_date["out-group"] += 1
                if status == 'continues':
                    self.statistics.surviving_first_date["total"] += 1
                    if r.is_out_group():
                        self.statistics.surviving_first_date["out-group"] += 1

            if status == 'interrupted':
                # update the statistics to have r removed
                if r.is_offline():
                    self.statistics.no_offline -= 1
                    if r.is_out_group():
                        self.statistics.no_outgroup_offline -= 1
                else:
                    self.statistics.no_online -= 1
                    if r.is_out_group():
                        self.statistics.no_outgroup_online -= 1

                # add back the agents that do not need to be removed
                for i in range(2):
                    if i not in remove:
                        id_agent = r.agents[i].id
                        self.agents[id_agent] = r.agents[i]
                        self.pending_messages[id_agent] = []
                    else:
                        # increase the statistics counter for bad rec
                        self.statistics.exit_reason["Too many failed relationships"] += 1
                        # add the stats on where the agent spent their time
                        ag_no_it = r.agents[i].no_iterations
                        s = sum([ag_no_it[i] for i in ["searching", "online_rel", "offline_rel"]])
                        for i in ["searching", "online_rel", "offline_rel"]:
                            self.statistics.time_by_phase[i] += ag_no_it[i]
                            self.statistics.percentage_of_time_by_phase[i] += ag_no_it[i]/s

                # mark the relatioship to be removed
                rel_to_be_removed += [r]

            elif status == 'long_term':
                # no agents is returend to the single list of agents
                # mark relationship to be removed
                rel_to_be_removed += [r]

                # update statistics: r moves offline -> long term
                # remove from offline and add to long term
                self.statistics.no_offline -= 1
                self.statistics.no_long_term += 1
                if r.is_out_group():
                    self.statistics.no_outgroup_offline -= 1
                    self.statistics.no_outgroup_long_term += 1
                    for ag in r.agents:
                        self.statistics.no_outgroup_long_term_by_type[ag.initial_bias_type] += 1

                # increase the statistics counter for bad rec

                # add the stats on where the agent spent their time
                for j in range(2):
                    self.statistics.exit_reason["Long term relationship"] += 1
                    ag_no_it = r.agents[j].no_iterations
                    s = sum([ag_no_it[i] for i in ["searching", "online_rel", "offline_rel"]])
                    for i in ["searching", "online_rel", "offline_rel"]:
                        self.statistics.time_by_phase[i] += ag_no_it[i]
                        self.statistics.percentage_of_time_by_phase[i] += ag_no_it[i]/s

            else:
                # update statistics
                # check if the relationship changed from offline to online
                # if yes, mirror this change in the statistics
                if changed_to_offline:
                    self.statistics.no_online -= 1
                    self.statistics.no_offline += 1
                    if r.is_out_group():
                        self.statistics.no_outgroup_online -= 1
                        self.statistics.no_outgroup_offline += 1

        # remove the relationships marked as to removed
        for r in rel_to_be_removed:
            self.active_rel.remove(r)

    def update_norm(self, strength, no_bias=False):
        '''The norms are updated to get closer to the aveerage preferences.
        strength = percentage of update (between 0 and 1)
        '''

        # there are single agents, and agents in a relaionship

        no_agents = len(self.agents) + 2 * len(self.active_rel)
        if no_agents != 0:

            if (len(self.agents) != 0):
                one_agent_id = list(self.agents.keys())[0]
                bias_indices = self.agents[one_agent_id].attributes.get_bias_indices()
                searchable_indices = self.agents[one_agent_id].attributes.get_searchable_indices()
                no_attributes = self.agents[one_agent_id].attributes.no_attributes
            else:
                bias_indices = self.active_rel[0].agents[0].attributes.get_bias_indices()
                searchable_indices = self.active_rel[0].agents[0].attributes.get_searchable_indices(
                )
                no_attributes = self.active_rel[0].agents[0].attributes.no_attributes

            # get the average taste
            average_taste = np.zeros(no_attributes)
            for i in self.agents.keys():
                imp_searchable = self.agents[i].get_importance_searchable()
                for j in searchable_indices:
                    average_taste[j] += self.agents[i].preferences[j]/imp_searchable
            for r in self.active_rel:
                imp_searchable_0 = r.agents[0].get_importance_searchable()
                imp_searchable_1 = r.agents[1].get_importance_searchable()
                for j in searchable_indices:
                    average_taste[j] += r.agents[0].preferences[j] / imp_searchable_0
                    average_taste[j] += r.agents[1].preferences[j] / imp_searchable_1
            average_taste /= no_agents
            # self.norm_on = self.norm_on + (average_taste - self.norm_on) * strength
            self.norm_on = strength*average_taste + (1-strength) * self.norm_on
            if no_bias:
                sum_bias_attributes = 0
                # set the norm on bias attribute to 0
                for i in bias_indices:
                    sum_bias_attributes += self.norm_on[i]
                    self.norm_on[i] = 0
                # re-do normalisation - for relative interpretation
                self.norm_on *= (1/(1-sum_bias_attributes))

    def iterate(self, no_new_agents_per_turn, no_bias=False):
        '''
        Performs one iteration of the platform.
        0. new agents arrive
        1. agents that are not in a relationship are ordered at random
        2. they decide which action to take (iterate_single_agent)
           2a. message someone new
           2b. answer to a message and enter an online rel
        3. relationships are iterated (rel.iterate() )
        4. norms are updated from preferences
        5. preferences are updated back from norms
        '''

        strenght_norm_update = float(self.list_parameters['strenght_norm_update'])

        # Step 0
        self.one_iteration_arrival_new_agents(no_new_agents_per_turn)

        # Step 1
        list_agents = list(self.agents.keys())
        list_agents = np.random.permutation(list_agents)

        # Step 2
        for id in list_agents:
            if id in self.agents.keys():
                self.iterate_single_agent(id)

        # Step 3
        self.iterate_relationships()

        # Step 4
        self.update_norm(strenght_norm_update, no_bias)

        # Step 5
        # update preferences for the single agents
        for id in self.agents.keys():
            self.agents[id].update_preferences_from_norms(self.norm_on, False)
        # update preferences for the agents in a relationship
        for r in self.active_rel:
            r.agents[0].update_preferences_from_norms(self.norm_out, True)
            r.agents[1].update_preferences_from_norms(self.norm_out, True)
