import copy
import numpy as np
import random
import math
import ast


# Encoding and decoding functions for the attribute combinations


def encoding(list):
    ''' Encodes the vector of attributes into a number.
    As each attribute is either 0 or 1, the resulting number is the number
       obtained from concatenating all elements of the list, in base 2.
    '''
    return int("".join(str(v) for v in list), 2)


def decoding(decimal_rep, no_attributes):
    '''Transform an encoded list of attributes, back into the list.
    The encoding function is described above.
    '''

    l = [int(v) for v in list('{0:0b}'.format(decimal_rep))]
    # put necessary zeros before
    l = [0]*(no_attributes - len(l)) + l

    return l


def get_value_from_encoded(n, index, length):
    '''
    Gets the i^th value if n was to be decoded as a list of attributes,
    given that the number of attributes is length.
    n - encoded vector of attributes
    index - postition from which you want the value
    length - number of attributes
    '''
    return (n >> (length - 1-index)) % 2


def generate_uniform_simplex(no_entries):
    ''' Generates a vector uniformly in the (no_entries - 1) simplex.
    That is, generates a vector of length no_entries where the sum of its
       elements is 1.
    '''

    v = np.random.rand(no_entries-1)
    v.sort()
    w = np.zeros(no_entries)
    for i in range(no_entries):
        if i == 0:
            w[i] = v[i] - 0
        elif i == no_entries-1:
            w[i] = 1 - v[i-1]
        else:
            w[i] = v[i] - v[i-1]

    return w


def generate_norms_pref(no_entries, bias_value, sum_searchable, bias_indices, searchable_indices, keep_at_bias_value=False):
    '''Generates uniform preferences for given sum of searchable and given
    level of bias. Same for out-platform norms.
    no_entries = # of attributes
    bias_value = minimum value on the bias attributes
    keep_at_bias_value = True iff the values of the bias attributes will be
                                  exactly bias_value (i.e. the minimum)
    sum_searchable(/exp) = the sum of the searchable(/exp) attributes
                     (including the bias ones, if they are searchable(/exp))
    bias(/search.))indices = the respective index values'''

    # is the bias index searchable (True --> yes, False --> experiential)
    bias_searchable = bias_indices[0] in searchable_indices

    # get the experiential indices
    experiential_indices = []
    for i in range(no_entries):
        if i not in searchable_indices:
            experiential_indices.append(i)

    # if the bias indices need to be kept at bias_value,
    # then remove the bias indices from searchable/experiential
    if keep_at_bias_value:
        if bias_searchable:
            for i in bias_indices:
                searchable_indices.remove(i)
        else:
            for i in bias_indices:
                experiential_indices.remove(i)

    no_searchable = len(searchable_indices)
    no_experiential = len(experiential_indices)
    sum_exp = 1-sum_searchable

    # find the sum of searchable & experiential non-bias indices
    if bias_searchable:
        sum_searchable -= len(bias_indices)*bias_value
    else:
        sum_exp -= len(bias_indices)*bias_value

    # set the preferences on the bias indices at bias_value
    pref_bias = np.zeros(no_entries)
    for i in bias_indices:
        pref_bias[i] = bias_value

    # generate the searchable part uniformly
    pref_searchable = np.zeros(no_entries)
    aux = generate_uniform_simplex(no_searchable)*sum_searchable
    index = 0
    for i in searchable_indices:
        pref_searchable[i] = aux[index]
        index += 1

    # generate the experiential part uniformly
    pref_experiental = np.zeros(no_entries)
    aux = generate_uniform_simplex(no_experiential) * sum_exp
    index = 0
    for i in experiential_indices:
        pref_experiental[i] = aux[index]
        index += 1

    # the final preference is formed from the bias, search. and exp. parts
    # if keep_at_bias_value, then the 3 have non-zero values on different
    # positions otherwise, the serch./exp. part overlap on bias components
    return pref_bias + pref_searchable + pref_experiental


class Attributes:

    def __init__(self, list_parameters, values="not_def"):
        self.list_parameters = list_parameters

        # Start with the class variables - use values from file
        self.no_bias = int(list_parameters['no_bias'])
        self.no_matching_searchable = int(list_parameters['no_matching_searchable'])
        self.no_matching_experiential = int(list_parameters['no_matching_experiential'])
        self.no_competing_searchable = int(list_parameters['no_competing_searchable'])
        self.no_competing_experiential = int(list_parameters['no_competing_experiential'])

        # the number of attributes is the sum of the number of each type
        self.no_attributes = self.no_bias + self.no_matching_searchable + \
            self.no_matching_experiential + \
            self.no_competing_searchable + self.no_competing_experiential

        # only 0 or 1 & -1 for unknown
        if values == 'not_def':
            self.values = list(np.zeros(self.no_attributes))
        else:
            self.values = values

        # Set the number of matching attributes - bias attribute included in the right category
        self.no_matching = self.no_matching_searchable + self.no_matching_experiential
        self.no_competing = self.no_competing_searchable + self.no_competing_experiential

        self.is_bias_searchable = int(list_parameters['is_bias_searchable'])
        self.is_bias_matching = int(list_parameters['is_bias_matching'])

        if self.is_bias_matching:
            self.no_matching += self.no_bias
        else:
            self.no_competing += self.no_bias

        # intervention in place - one maching searcable attribute is artificial
        self.artificial_attribute = int(list_parameters['artificial_attribute'])

    def get_bias_indices(self):
        return list(range(self.no_bias))

    def get_matching_indices(self):
        '''Returns a list with the indices of the matching attribute
        '''
        return list(range(self.no_matching)) if self.is_bias_matching else list(range(self.no_bias, self.no_bias + self.no_matching))

    def get_competing_indices(self):
        return list(range(self.no_matching, self.no_matching + self.no_competing)) if self.is_bias_matching else list(range(self.no_bias)) + list(range(self.no_bias + self.no_matching, self.no_matching + self.no_competing))

    def get_searchable_indices(self):
        return list(range(self.no_bias + self.no_matching_searchable)) + list(range(self.no_matching, self.no_matching + self.no_competing_searchable)) if self.is_bias_searchable else list(range(self.no_bias, self.no_bias + self.no_matching_searchable)) + list(range(self.no_matching, self.no_matching + self.no_competing_searchable))

    def get_experiential_indices(self):
        if self.is_bias_searchable:
            return list(range(self.no_bias + self.no_matching_searchable, self.no_matching)) + list(range(self.no_matching + self.no_competing_searchable, self.no_matching + self.no_competing))
        else:
            return list(range(self.no_bias)) + list(range(self.no_bias + self.no_matching_searchable, self.no_matching)) + list(range(self.no_matching + self.no_competing_searchable, self.no_matching + self.no_competing))

    def get_matching(self):
        return [self.values[i] for i in self.get_matching_indices()]

    def get_bias(self):
        return[self.values[i] for i in self.get_bias_indices()]

    def get_competing(self):
        return[self.values[i] for i in self.get_competing_indices()]

    def get_searchable(self):
        return[self.values[i] for i in self.get_searchable_indices()]

    def get_experiential(self):
        return[self.values[i] for i in self.get_experiential_indices()]

    def generate_values_from_contingency(self, cont, bias_group=-1):
        ''' Samples a set of attributes from a given contingency table cont.
        Sets this set as the values for the current object.
        cont = contingency tables for [both, only group0, only group1]
        bias_group = -1 if any, 0 for group0, 1 for group1'''

        no_comb = len(cont[bias_group+1])
        enc = np.random.choice(range(no_comb), 1, p=cont[bias_group+1])
        enc = enc[0]
        self.values = decoding(enc, self.no_attributes)

    def generate_values_from_multivariate(self, bias_group=-1):
        '''This generates a set of attributes for an agent from the multivariate
        normal distribution.
        '''

        beta = float(self.list_parameters['beta'])
        gamma = float(self.list_parameters['gamma'])

        mean = np.zeros(self.no_attributes)
        cov = np.ones((self.no_attributes, self.no_attributes))

        # create the covariance matrix
        bias_indices = self.get_bias_indices()
        for i in bias_indices:
            for j in range(i+1, self.no_attributes):
                cov[i, j] = beta
                cov[j, i] = beta
        for i in range(self.no_attributes):
            for j in range(i+1, self.no_attributes):
                if i not in bias_indices and j not in bias_indices:
                    cov[i, j] = gamma
                    cov[j, i] = gamma

        # change correlation to 0 for the artificial matching searchable
        # attribute, if it exists
        if self.artificial_attribute:
            artificial_index = self.no_bias
            for i in range(self.no_attributes):
                if i != artificial_index:
                    cov[i, artificial_index] = 0
                    cov[artificial_index, i] = 0

        # generate the unrounded values
        self.values = np.random.multivariate_normal(mean, cov)
        # trashold to get 0/1
        def f(x): return 1 if x > 0 else 0
        self.values = [f(x) for x in self.values]

    def add_bias_to_values(self, attributes_other):
        ''' Generates bias values for bias sampling of different groups.
        The present(self) values are changed through the eyes of an agent with
          attributes attributes_others.
        The values of self become negative with a probability p_negativity.
        p_negativity = chance of having negative attributes.
                   0 = faithfull representation, all attributes are the true ones'''

        p_negativity = float(self.list_parameters['p_negativity'])

        # The bias attributes are set to be different
        for j in self.get_bias_indices():
            self.values[j] = 1 - attributes_other.values[j]

        # with probability p_negativity, the matching attributes are set to be different
        for j in self.get_matching_indices():
            set_bad_attribute = random.uniform(0, 1)
            if set_bad_attribute < p_negativity:
                self.values[j] = 1 - attributes_other.values[j]

        # with probability p_negativity, the competing attributes are set to 0
        for j in self.get_competing_indices():
            set_bad_attribute = random.uniform(0, 1)
            if set_bad_attribute < p_negativity:
                if j not in self.get_bias_indices():
                    self.values[j] = 0

    def values_encoding(self):
        ''' Encodes the vector of attributes into a number.
        Same function as the global encoding function.
        '''
        return int("".join(str(int(v)) for v in self.values), 2)

    def list_compatible_combinations(self):
        '''Gives a list with encoded attributes that are comopatible with the
        observed values of 0 and 1.'''

        no_attributes = len(self.values)

        def list_value_combination(must_correspond, checked_until,
                                   list_of_comb):
            '''Recursive function that forms in list_of_comb all the variants of
            replacing -1 with 0 and 1 in must_correspond.

            must_correspond = vector with 0, 1, and -1
            checked_until   = position until which there are no ones
            list_of_comb    = list of combinations found so far'''

            for i in range(checked_until, no_attributes):
                if must_correspond[i] == -1:
                    must_correspond[i] = 0
                    list_value_combination(must_correspond, i, list_of_comb)
                    must_correspond[i] = 1
                    list_value_combination(must_correspond, i, list_of_comb)
                    must_correspond[i] = -1
                    return 0

            def encode(vector_0_1):
                return int("".join(str(int(v)) for v in vector_0_1), 2)

            list_of_comb += [encode(must_correspond)]

            return 0

        list_of_combinations = []
        list_value_combination(self.values, 0, list_of_combinations)

        return list_of_combinations


class Orientation:
    '''
    Defines how many types of agents there are, and which type is interested in
       dating which.
    In the paper, we only reported when having two, correspondig with f, and m,
       and agents of type f (m) interested in those of type m (f),
       but the simulations were also run with other variants
       (e.g. only one type of agents; everybody interested in that type).
    '''

    def __init__(self, list_parameters):

        # a list of all the possible types
        self.types = list_parameters['orientation_types']
        self.types = ast.literal_eval(self.types)

        # a dictionary with the interests of the possible types
        self.all_interests = {}
        for t in self.types:
            self.all_interests[t] = list_parameters['interests_' + str(t)]
            self.all_interests[t] = ast.literal_eval(self.all_interests[t])

        # probability distribution for types
        self.prob_types = list_parameters['prob_types']
        self.prob_types = ast.literal_eval(self.prob_types)

        # the type of the agent (-1 -> undefined)
        self.type = np.random.choice(self.types, p=self.prob_types)
        # the types the agent is interested in
        self.interests = self.all_interests[self.type]


class Agent:
    def __init__(self, id, list_parameters):
        self.list_parameters = list_parameters

        strength_update_sample = float(list_parameters['strength_update_sample'])
        strength_update_pref = float(list_parameters['strength_update_pref'])
        strength_update_norms = float(list_parameters['strength_update_norms'])
        relative_importance_norms = float(list_parameters['relative_importance_norms'])
        visible_attributes_indices = []
        failure_tolerance = int(list_parameters['failure_tolerance'])
        bad_recommandation_tolerance = int(list_parameters['bad_recommandation_tolerance'])

        # agent identifier
        self.id = id
        # option for the filter strategy - options (for the top wheighted attribute A):
        #   - OFF = agent doesn't want any filters
        #   - WEAK = if A unspecified, still shows up
        #   - STRONG = only wants agents who have A specified and "good"
        self.filter = list_parameters['filter']
        # option for the sort starategy - options:
        #   - OFF = no sorting
        #   - BY NO = by the number of attributes specified and "good"
        #   - BY WEIGHT = by the wheighted importance of the specified attributes
        self.sort = list_parameters['sort']
        # orientation type - an integer encoding the orientation of the agent
        # simple starting state - orientation_type = 0/1 depending on gender
        # -1 -> not yet defined
        self.orientation = Orientation(list_parameters)
        # a variable of type Attributes
        self.attributes = Attributes(list_parameters)
        # a list of importance given to each attribute (preference)
        self.preferences = np.array([])
        # a frequency vector of the observed attributes correlations in others
        self.sample = []
        # weight given to 1 new observation when updating the contingency table
        self.strength_update_sample = strength_update_sample
        # weight given to 1 new interaction when updating preferences
        self.strength_update_pref = strength_update_pref
        # degree of influence of norms on preferences
        self.strength_update_norms = strength_update_norms
        # the importance of out-platform norms in initial preference generation
        self.relative_importance_norms = relative_importance_norms
        # list with indices of variables visible on the profile
        self.visible_attributes_indices = visible_attributes_indices
        # how many failed relationships before exiting the platform
        self.failure_tolerance = failure_tolerance
        # how many rounds of unsuccessful recommandations/messages can handle the agent before exiting the platform
        self.bad_recommandation_tolerance = bad_recommandation_tolerance
        # no iterations in each state
        self.no_iterations = {"searching": 0, "online_rel": 0, "offline_rel": 0}
        # agent category based on the level of bias
        self.initial_bias_type = -1
        self.offset = float(list_parameters['offset'])

    def get_contingency(self):
        '''Getting the contingency table from the observed population sample.
        '''
        s = sum(self.sample)
        return [x/s for x in self.sample]

    def generate_sample(self, no_samples_same_bias_group,
                        no_samples_different_bias_group, cont):
        '''Generating a contingency table by sampling from the population.
        '''
        self.sample = list(np.zeros(2**len(self.attributes.values)))
        cur = self.attributes

        for i in range(no_samples_same_bias_group):
            # gnerate a combination of attributes (from the true contingency table) for the same bias group
            a = Attributes(self.list_parameters, list(np.zeros(len(cur.values))))
            a.generate_values_from_contingency(cont, bias_group=cur.values[0])
            # add that observation to the observed sample of the current agent
            self.sample[a.values_encoding()] += 1

        for i in range(no_samples_different_bias_group):
            # generate a combination of attributes, and add bias to it
            a = Attributes(self.list_parameters, list(np.zeros(len(cur.values))))
            a.generate_values_from_contingency(cont, bias_group=1-cur.values[0])
            a.add_bias_to_values(self.attributes)
            # add that observation to sample
            self.sample[a.values_encoding()] += 1

    def generate_unbiased_contingency(self):
        sample_size = int(self.list_parameters['sample_size'])
        self.sample = list(np.zeros(2**len(self.attributes.values)))

        for i in range(sample_size):
            # gnerate a combination of attributes
            a = Attributes(self.list_parameters)
            a.generate_values_from_multivariate()
            # add that observation to the observed sample so far
            self.sample[a.values_encoding()] += 1

        # transform to probabilites
        return self.get_contingency()

    def determine_initial_bias_type(self, norm_out_0, initial_min_bias_pref, sum_searchable, change=True):
        '''Based on the preferences, determines the level of pref-bias'''
        alpha = self.relative_importance_norms
        min = initial_min_bias_pref
        n0 = norm_out_0
        if self.attributes.is_bias_searchable:
            k = len(self.attributes.get_searchable_indices())-1
            total = sum_searchable - initial_min_bias_pref
        else:
            k = len(self.attributes.get_experiential_indices())-1
            total = 1 - initial_min_bias_pref - sum_searchable

        low_boundary = (1-alpha)*min + alpha*n0 + (1-alpha)*total*(1-(2/3)**(1/k))
        high_boundary = (1-alpha)*min + alpha*n0 + (1-alpha)*total*(1-(1/3)**(1/k))
        type_b = -1
        if self.preferences[0] < low_boundary:
            type_b = 'low'
        elif self.preferences[0] < high_boundary:
            type_b = 'medium'
        else:
            type_b = 'high'

        if change:
            self.initial_bias_type = type_b

        return type_b

    def generate_preferences(self, initial_min_bias_pref, sum_searchable, norm_out):
        '''Generates the initial preferences of an agent.
        For the idiosyncratic part
        - generate all attributes in U(0, 1)
        - normalise them such that sum is 1-initial_min_bias_pref*no_attributes
        - add initial_min_bias_pref to all bias attributes
        Then, it combines with norm_out.
        '''

        # the idiosyncratic part for the bias attributes
        # start with the bias attributes - weight bias_level
        no_attributes = self.attributes.no_attributes
        bias_indices = self.attributes.get_bias_indices()
        searchable_indices = self.attributes.get_searchable_indices()

        # generate idiosyncratic part of pref.s
        self.preferences = generate_norms_pref(
            no_attributes, initial_min_bias_pref, sum_searchable, bias_indices, searchable_indices)

        # convex combination with the out_norm
        alpha = self.relative_importance_norms
        self.preferences = alpha*norm_out + (1-alpha)*self.preferences

        self.determine_initial_bias_type(norm_out[0], initial_min_bias_pref, sum_searchable)

    def generate_visible_attributes_indices(self, prob_showing_on_profile):
        '''Generate the indices of attributes that the agent make visible
        on th eprofile.
        prob_showing_on_profile = probability of showing searchable attributes
        '''

        self.visible_attributes_indices = []
        for i in self.attributes.get_searchable_indices():
            if np.random.random() < prob_showing_on_profile:
                self.visible_attributes_indices += [i]

    def generate(self, norm_out, cont):
        '''Combines all the 3 generating procedures above.
        Useful for creating an agent.
        '''
        no_samples_same_bias_group = int(self.list_parameters['no_samples_same_bias_group'])
        no_samples_different_bias_group = int(
            self.list_parameters['no_samples_different_bias_group'])
        initial_min_bias_pref = float(self.list_parameters['initial_min_bias_pref'])
        prob_showing_on_profile = float(self.list_parameters['prob_showing_on_profile'])
        sum_searchable = float(self.list_parameters['sum_searchable'])

        self.attributes.generate_values_from_multivariate()
        self.generate_sample(no_samples_same_bias_group, no_samples_different_bias_group, cont)
        self.generate_preferences(initial_min_bias_pref, sum_searchable, norm_out)
        self.generate_visible_attributes_indices(prob_showing_on_profile)

    def get_profile_visible_attributes(self):
        '''Constructs an Attribute object with the profile observable attributes
        '''

        a = Attributes(self.list_parameters)
        no_attributes = len(a.values)
        for i in range(no_attributes):
            if i not in self.visible_attributes_indices:
                a.values[i] = -1
            else:
                a.values[i] = self.attributes.values[i]

        return a

    def get_importance_searchable(self):
        ''' Sums the preference for the searchable attributes.
        '''

        searchable_indices = self.attributes.get_searchable_indices()
        return sum(self.preferences[j] for j in searchable_indices)

    def update_sample(self, attributes_other):
        '''Update record of observed attribute co-occurrences after an
          interaction with another agent agent.
        Each plausible complete attribute combination is updated in proportion
          to the believed likelihood to observe it like this.
        attributes_other = the list of observed attributes so far in the other
                           agent
        compatible_combinations = list of plausible scenarios on the values for
           the attributes of the other agent
        '''

        # find the list of possible attribute completions
        compatible_combinations = attributes_other.list_compatible_combinations()

        # number of observations of compatible attributes so far
        sum_compatible = sum(self.sample[c] for c in compatible_combinations)
        # if no observations, behave as if one of each
        if sum_compatible == 0:
            sum_compatible = len(compatible_combinations)
            for c in compatible_combinations:
                self.sample[c] = self.strength_update_sample/sum_compatible
        else:
            # update each compatible combination in proportion to observation
            for c in compatible_combinations:
                self.sample[c] = self.sample[c] * (1 + self.strength_update_sample / sum_compatible)

    def update_preferences_after_interaction(self, attributes_other, good_interaction):
        '''
        After each interaction preferences are updated to reflect the quality
          of that interaction.
        attributes_other = attributes observed in the other
        qood_interaction = 0 for good (continued), 1 for bad (ended)
        '''

        # only keep the values
        attributes_other = attributes_other.values

        # fn that gives 1 if the trait is good in the other, 0 if neutral, -1 otherwise
        def attribute_is_good(index):
            ''' index = index of attribute being considered'''
            if attributes_other[index] == -1:
                return 'unchanged'
            elif index in self.attributes.get_matching_indices():
                return 1 if self.attributes.values[index] == attributes_other[index] else 0
            else:
                if self.attributes.values[index] < attributes_other[index]:
                    return 1
                elif self.attributes.values[index] > attributes_other[index]:
                    return 0
                else:
                    return 'unchanged'

        # fn that checks if the attribute fits with the interaction quality
        def attribute_match_interaction(attr_is_good):
            ''' attr_is_good = result of above fn (is the attribute good)'''
            return 1 if good_interaction == attr_is_good else 0

        # compose the update vector - based on the quality of interaction
        # iterate through known attributes
        pref_intern = np.zeros(len(attributes_other))
        pref_unchanged = np.zeros(len(attributes_other))
        sum_pref_intern = 0
        sum_changing_pref = 0
        no_changed = 0
        for i in range(len(attributes_other)):
            is_good = attribute_is_good(i)
            if not is_good == 'unchanged':
                no_changed += 1
                pref_intern[i] = attribute_match_interaction(is_good) * self.preferences[i]
                sum_pref_intern += pref_intern[i]
                sum_changing_pref += self.preferences[i]
            else:
                pref_unchanged[i] = self.preferences[i]

        # normalise the current vector to give the same sum
        # - if no relative change, then no change in preferences
        if not (sum_pref_intern == 0):
            pref_intern *= (sum_changing_pref/sum_pref_intern)
            # take convex combination for the update
            pref_intern += pref_unchanged
            alpha = self.strength_update_pref
            self.preferences = alpha*pref_intern + (1-alpha)*self.preferences


    def update_preferences_from_norms(self, norm, is_offline):
        '''
        After each iteration, the norms influence the preferences of
         individuals.
        norm = on/out-platform norm, depending on the phase of the relationship
        is_offline = T/F depending on the phase of the relationship
        '''

        if is_offline:
            # same as convex combination
            self.preferences += (norm - self.preferences)*self.strength_update_norms
        else:
            # for normalisation, also rescale norm to agent's importance of
            #   searchable attributes
            imp_search = self.get_importance_searchable()
            for i in self.attributes.get_searchable_indices():
                self.preferences[i] += (norm[i]*imp_search -
                                        self.preferences[i])*self.strength_update_norms

    def answer(self, attributes_other, time=0, first=True):
        '''
        Function in accordance to which an agent makes a decision, where:
        attributes_other = list of known attributes of the person that
         is currently considered.
        time = time spent so far in the relationship
         (time = 0 -> just considering messaging/replying)
        attribute_importance = the importance of attributes values compared to
         time
        '''
        if first:
            time += self.offset

        attribute_importance = float(self.list_parameters['attribute_importance'])
        # find the list of possible attribute completions
        compatible_combinations = attributes_other.list_compatible_combinations()

        no_attributes = len(self.attributes.values)
        # number of observations of compatible attributes so far
        sum_compatible = sum(self.sample[c] for c in compatible_combinations)

        # function that gives 1 if matching -1 otherwise
        def matching(a, b): return 1 if a == b else -1

        # compute the weighted desirability gap + matching degree
        # (attribute influence with value between -1 and 1)
        attribute_influence = 0
        for c in compatible_combinations:
            def get_value_c(i): return get_value_from_encoded(c, i, no_attributes)
            attribute_influence_c = 0
            for i in range(no_attributes):
                if i in self.attributes.get_competing_indices():
                    attribute_influence_c += (get_value_c(i) -
                                              self.attributes.values[i])*self.preferences[i]
                else:
                    attribute_influence_c += matching(get_value_c(i),
                                                      self.attributes.values[i]) * self.preferences[i]
            # multiply by the perceived probability of observing c
            attribute_influence += attribute_influence_c * self.sample[c] / sum_compatible
        # normalise
        attribute_influence /= sum(self.preferences)

        # offset and expansion used to test robustness of qualitative resuts
        #    under variations in the decision fucntion
        # throught the paper they were keept at 0 and 1 respectively
        offset = float(self.list_parameters["offset"])
        expansion = float(self.list_parameters["expansion"])

        probability_yes = 1/(1 + math.exp(offset-expansion *
                                          attribute_influence - time/attribute_importance))
        probability_no = 1 - probability_yes

        return np.random.choice([0, 1], p=[probability_no, probability_yes])
