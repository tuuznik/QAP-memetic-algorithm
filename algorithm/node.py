from typing import List
from random import random, randint, shuffle
from algorithm.solution import Solution
from parameters.algorithm_parameters import AlgorithmParameters
from parameters.qap_parameters import QAPParameters

"""
Represents node object in the ternary tree.
"""

class Node:

    def __init__(self,  algorithm_parameters: AlgorithmParameters, qap_parameters: QAPParameters,
                 init_current: List[int]):
        self.algorithm_parameters = algorithm_parameters
        self.qap_parameters = qap_parameters
        self.size = self.algorithm_parameters.node_size
        #self.pockets = [None for x in range(self.size)]
        self.pockets = []
        self.buffer = Solution(init_current, qap_parameters, algorithm_parameters)
        self.best_solution = None

    def __str__(self):
        #return "Buffer:\n{buffer}\nPocket:\n".format(buffer=self.buffer) + "\n".join(map(str, self.pockets))
        return "Pocket:\n" + "\n".join(map(str, self.pockets))

    def update_node(self):
        """
        Procedure to move the solution from buffer to the node or to remove it, if it is worse than all solutions
        currently stored in node. Additionally, the best solution for node is selected.
        :return:
        """
        if len(self.pockets) < 10:
            self.pockets.append(self.buffer)
            self.buffer = None
        self.best_solution = min(self.pockets, key=lambda s: s.fitness if isinstance(s, Solution) else float('inf'))
        worst_solution = max(self.pockets, key=lambda s: s.fitness if isinstance(s, Solution) else 0)
        if self.buffer:
            if self.buffer.assignment not in [s.assignment for s in self.pockets if isinstance(s, Solution)]:
                if self.buffer.fitness < self.best_solution.fitness:
                    self.pockets[randint(0, self.size-1)] = self.buffer
                    self.best_solution = self.buffer
                elif self.buffer.fitness < worst_solution.fitness:
                    worst_id = self.pockets.index(worst_solution)
                    self.pockets[worst_id] = self.buffer
        self.buffer = None
        self.best_solution = min(self.pockets, key=lambda s: s.fitness if isinstance(s, Solution) else float('inf'))

    def add_to_buffer(self, current: List[int]):
        """
        Adds a new solution to the node in situation when it is not generated as a result of crossover. The solution in
        the buffer is then analyzed by update_node function that decides if it should be kept in generation or removed.
        :param current: Assignment list of a new solution.
        :return:
        """
        self.buffer = Solution(current, qap_parameters=self.qap_parameters, alg_parameters=self.algorithm_parameters)

    def select_parents(self, ni_max: float = 1.5) -> List[Solution]:
        """
        Based on rank selection, function returns list of parents that will be used in crossover. The parameter
        parents_number says what part of feasible parents are taken, it influences directly the number of offspring.
        :param ni_max:
        :return:
        """
        potential_parents = [pocket for pocket in self.pockets if isinstance(pocket, Solution)]
        potential_parents.sort(key=lambda s: s.fitness, reverse=False)
        lam = len(potential_parents)
        if lam == 1:
            return potential_parents
        else:
            probability_table = [1/lam*(ni_max - (2*ni_max - 2)*(i - 1)/(lam - 1)) for i in range(1, lam+1)]
            temp_list = []
            for j in range(lam):
                random_number = random()
                probability_sum = 0
                for k in range(len(probability_table)):
                    probability_sum += probability_table[k]
                    if random_number <= probability_sum:
                        temp_list.append(potential_parents[k])
                        break
            return temp_list[0:round(len(probability_table)*self.algorithm_parameters.parents_number)]
