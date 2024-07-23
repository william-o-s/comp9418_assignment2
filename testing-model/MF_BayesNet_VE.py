# Necessary libraries
import math
import copy

# combinatorics
from itertools import combinations

from MF_DiscreteFactors import Factor
from MF_Graph import Graph
from MF_Utils import estimate_factor

class BayesNet():
    def __init__(self, graph=None, outcomeSpace=None, factor_dict=None):
        self.graph = Graph()
        self.outcomeSpace = dict()
        self.factors = dict()
        if graph is not None:
            self.graph = graph
        if outcomeSpace is not None:
            self.outcomeSpace = outcomeSpace
        if factor_dict is not None:
            self.factors = factor_dict
            
    def learnParameters(self, data):
        graphT = self.graph.transpose()
        for node, parents in graphT.adj_list.items():
            f = estimate_factor(data, node, parents, self.outcomeSpace)
            self.factors[node] = f
            
    def joint(self):
        factor_list = list(self.factors.values())
        
        accumulator = factor_list[0]
        for factor in factor_list[1:]:
            accumulator = accumulator.join(factor)
        return accumulator
    
    def width(self, order):
        """
        argument 
        `order`, a list of variable names specifying an elimination order.

        Returns the width of the elimination order, i.e., the number of variables of the largest factor
        """   
        # Initialize w, a variable that has a width of the elimination order
        w = 0
        # Let's make a list of tuples, where each tuple is a factor domain
        factorList = [f.domain for f in self.factors.values()]
        # We process the factor in elimination order
        for var in order:
            # This is the domain of the new factor. We use sets as it is handy to eliminate duplicate variables
            newFactorDom = set()
            # A list to keep track of all the factors we will keep for the next iteration (all factors not containing `var`)
            updatedFactorsList = list()            

            # Lets iterate over all factors
            for f_dom in factorList:
                # and select the ones that have the variable to be eliminated
                if var in f_dom:
                    # Merge the newFactorDomain list with the selected domain, since we are combining these ones
                    newFactorDom.update(f_dom)
                else:
                    # otherwise, we add the factor to the list to be processed in the next iteration
                    updatedFactorsList.append(f_dom)

            # Now, we need to remove var from the domain of the new factor. We are simulating a summation
            newFactorDom.remove(var)            # Remove var from the list new_dom by calling the method remove(). 1 line
            # Let's check if we have found a new largest factor
            if len(newFactorDom) > w:
                w = len(newFactorDom)
            # add the new combined factor domain to the list
            updatedFactorsList.append(newFactorDom)
            # replace factor list with updated factor list (getting rid of all factors containing var)
            factorList = updatedFactorsList

        return w

    def VE(self, order):
        """
        argument 
        `order`, a list of variable names specifying an elimination order.

        Returns a single factor, the which remains after eliminating all other factors
        """   

        # Let's make a copy of factors, so we can freely modify it without destroying the original dictionary
        factorList = [f for f in self.factors.values()]
        # We process the factors in elimination order
        for var in order:
            # We create an empty factor as an accumulator
            newFactor = Factor(tuple(), self.outcomeSpace)
            # A list to keep track of all the factors we will keep for the next step
            updatedFactorsList = list()            

            # Lets iterate over all factors
            for f in factorList:
                # and select the ones that have the variable to be eliminated
                if var in f.domain:
                    # Merge the newFactorDomain list with the selected domain, since we are combining these ones
                    newFactor = newFactor*f
                else:
                    # otherwise, we leave the factor for the next iteration
                    updatedFactorsList.append(f)

            # Now, we need to remove var from the domain of the new factor. We are simulating a summation
            newFactor = newFactor.marginalize(var)
            # add the new combined factor domain to the list
            updatedFactorsList.append(newFactor)
            # replace factorList with the new factor list, ready for the next iteration
            factorList = updatedFactorsList
        # for the final step, we join all remaining factors (usually there will only be one factor remaining)
        returnFactor = Factor(tuple(), self.outcomeSpace)
        for f in factorList:
            returnFactor = returnFactor*f
        return returnFactor

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
        # Initialize order with empty list. This variable will have the answer in the end of the execution
        order = []
        # While the induced graph has nodes to be eliminated
        while len(ig) > 0:
            # Initialize minDegree with a large number: math.inf
            minDegree = math.inf
            for var in ig:
                # Test if var has a degree smaller than minDegree
                if len(ig.children(var)) < minDegree:
                    # We have found a new candidate to be the next eliminated variable. Let's save its degree and name
                    minDegree = len(ig.children(var))
                    minVar = var
            # We need to connect the neighbours of minVar, let us start using combinations function to find all pairs of minVar's neighbours
            for var1, var2 in combinations(ig.children(minVar), 2):
                # We need to check if these neighbour are not already connected by an edge
                if var1 not in ig.children(var2):
                    ig.add_edge(var1, var2, directed=False)
            # Insert in order the variable in minVar
            order.append(minVar)
            # Now, we need to remove minVar from the adjacency list of every node
            ig.remove_node(minVar)
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