'''
    Helper class that defines a BayesNet.
'''

# Necessary libraries
import math
import copy

# combinatorics
from itertools import combinations

from MF_DiscreteFactors import Factor
from MF_Graph import Graph
import MF_Utils as Utils

class BayesNet():
    '''Helper class that defines a BayesNet object.'''

    def __init__(self, graph=None, outcome_space=None, factor_dict=None):
        self.graph = Graph()
        self.outcome_space = {}
        self.factors = {}
        if graph is not None:
            self.graph = graph
        if outcome_space is not None:
            self.outcome_space = outcome_space
        if factor_dict is not None:
            self.factors = factor_dict

    def learnParameters(self, data):
        '''Learns the factors of the defined graph given training data.'''
        graph_t = self.graph.transpose()
        for node, parents in graph_t.adj_list.items():
            f = Utils.estimate_factor(data, node, parents, self.outcome_space)
            self.factors[node] = f

    def joint(self):
        '''Finds the join distribution of this Bayes Network.'''

        factor_list = list(self.factors.values())

        accumulator = factor_list[0]
        for factor in factor_list[1:]:
            accumulator = accumulator.join(factor)
        return accumulator

    def width(self, order):
        '''
        argument 
        `order`, a list of variable names specifying an elimination order.

        Returns the width of the elimination order
            i.e., the number of variables of the largest factor
        '''
        # Initialize w, a variable that has a width of the elimination order
        w = 0
        # Let's make a list of tuples, where each tuple is a factor domain
        factor_list = [f.domain for f in self.factors.values()]
        # We process the factor in elimination order
        for var in order:
            # This is the domain of the new factor.
            # We use sets as it is handy to eliminate duplicate variables
            new_factor_domain = set()
            # A list to keep track of all the factors we will keep for the next iteration
            # (all factors not containing `var`)
            updated_factors_list = []

            # Lets iterate over all factors
            for f_dom in factor_list:
                # and select the ones that have the variable to be eliminated
                if var in f_dom:
                    # Merge the newFactorDomain list with the selected domain,
                    # since we are combining these ones
                    new_factor_domain.update(f_dom)
                else:
                    # otherwise, we add the factor to the list to be processed in the next iteration
                    updated_factors_list.append(f_dom)

            # Now, we need to remove var from the domain of the new factor.
            # We are simulating a summation
            new_factor_domain.remove(var)   # Remove var from the list new_dom
            # Let's check if we have found a new largest factor
            if len(new_factor_domain) > w:
                w = len(new_factor_domain)
            # add the new combined factor domain to the list
            updated_factors_list.append(new_factor_domain)
            # replace factor list with updated factor list
            # (getting rid of all factors containing var)
            factor_list = updated_factors_list

        return w

    def VE(self, order):
        '''
        argument 
        `order`, a list of variable names specifying an elimination order.

        Returns a single factor, the which remains after eliminating all other factors
        '''

        # Let's make a copy of factors, so we can freely modify it without destroying the original
        # dictionary
        factor_list = list(self.factors.values())
        # We process the factors in elimination order
        for var in order:
            # We create an empty factor as an accumulator
            new_factor = Factor(tuple(), self.outcome_space)
            # A list to keep track of all the factors we will keep for the next step
            updated_factors_list = []

            # Lets iterate over all factors
            for f in factor_list:
                # and select the ones that have the variable to be eliminated
                if var in f.domain:
                    # Merge the newFactorDomain list with the selected domain,
                    # since we are combining these ones
                    new_factor = new_factor*f
                else:
                    # otherwise, we leave the factor for the next iteration
                    updated_factors_list.append(f)

            # Now, we need to remove var from the domain of the new factor.
            # We are simulating a summation
            new_factor = new_factor.marginalize(var)
            # add the new combined factor domain to the list
            updated_factors_list.append(new_factor)
            # replace factorList with the new factor list, ready for the next iteration
            factor_list = updated_factors_list
        # for the final step, we join all remaining factors
        # (usually there will only be one factor remaining)
        return_factor = Factor(tuple(), self.outcome_space)
        for f in factor_list:
            return_factor = return_factor * f
        return return_factor

    def interactionGraph(self):
        '''
        Returns the interaction graph for this network.
        There are two ways to implement this function:
        - Iterate over factors and check which vars are in the same factors
        - Start with the directed graph, make it undirected and moralise it
        '''
        # Initialise an empty graph
        g = Graph()
        for var in self.factors.keys():
            g.add_node(var)
        for factor in self.factors.values():
            # for every pair of vars in the domain
            for var1 in factor.domain:
                for var2 in factor.domain:
                    # check if connection already exists
                    if var1 != var2 and var1 not in g.children(var2):
                        # add an *undirected* connection
                        g.add_edge(var1, var2, directed=False)
        return g

    def minDegree(self):
        ig = self.interactionGraph()
        # Initialize order with empty list. This variable will have the answer in the end of the
        # execution
        order = []
        # While the induced graph has nodes to be eliminated
        while len(ig) > 0:
            # Initialize minDegree with a large number: math.inf
            min_degree = math.inf
            for var in ig:
                # Test if var has a degree smaller than minDegree
                if len(ig.children(var)) < min_degree:
                    # We have found a new candidate to be the next eliminated variable.
                    # Let's save its degree and name
                    min_degree = len(ig.children(var))
                    min_var = var
            # We need to connect the neighbours of minVar, let us start using combinations function
            # to find all pairs of minVar's neighbours
            for var1, var2 in combinations(ig.children(min_var), 2):
                # We need to check if these neighbour are not already connected by an edge
                if var1 not in ig.children(var2):
                    ig.add_edge(var1, var2, directed=False)
            # Insert in order the variable in minVar
            order.append(min_var)
            # Now, we need to remove minVar from the adjacency list of every node
            ig.remove_node(min_var)
        return order

    def query(self, q_vars, **q_evi):
        '''
        A faster VE-based query function
        Returns a factor P(q_vars| q_evi)
        '''
        # backup factors dict
        backup_factors = copy.deepcopy(self.factors)
        # set evidence on all relevant factors
        for key, factor in self.factors.items():
            self.factors[key] = factor.evidence(**q_evi)

        # get minDegree order if order is None
        order = self.minDegree()

        # remove any q_vars from order
        for var in q_vars:
            order.remove(var)

        # run VE
        factor = self.VE(order)

        # marginalise out any vars not in q_vars
        for var in factor.domain:
            if var not in q_vars:
                factor = factor.marginalize(var)

        self.factors = backup_factors

        return factor.normalize()
