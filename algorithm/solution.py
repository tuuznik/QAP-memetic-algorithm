from typing import List
from random import random, randint, shuffle
from copy import copy
from parameters.qap_parameters import QAPParameters
from parameters.algorithm_parameters import AlgorithmParameters

"""
CLass representing single solution, a representative from the population.
"""

class Solution:

    def __init__(self, assignment: List[int], qap_parameters: QAPParameters, alg_parameters: AlgorithmParameters):
        self.assignment = assignment
        self.alg_parameters = alg_parameters
        self.qap_parameters = qap_parameters
        self.mutation_ratio = alg_parameters.mutation_ratio
        self.distance = qap_parameters.distance
        self.flow = qap_parameters.flow
        self.fitness = 0
        if len(self.assignment) != len(self.distance):
            raise ValueError("Assignment list must be the same length as distance matrix")
        else:
            self.problem_size = len(self.assignment)
        self.calculate_fitness()

    def __str__(self):
        return str(self.assignment).strip('[]') #+ " Fitness: {fitness}".format(fitness= self.fitness)

    def calculate_fitness(self):
        """
        Calculates the objective function value for particular solution, taking as a parameters two matrices that
        represent distance between locations and flow between facilities.
        """
        self.fitness = 0
        for i in range(self.problem_size):
            for j in range(self.problem_size):
                distance = self.distance[i][j]
                flow = self.flow[self.assignment[i]][self.assignment[j]]
                self.fitness += (distance*flow)

    def gene_mutation(self, modify_mutation: bool) -> int:
        """
        Proceeds the mutation on genes of a chromosome.There is a parametrised probability that determines whether the
        mutation occurs. For each of the genes, this value is compared to randomly chosen number from range (0,1).
        The mutation itself is limited to swap of the genes in the genotype, so the proper assignment is maintained.
        :return:
        """
        mutated = False
        if modify_mutation:
            self.mutation_ratio = self.alg_parameters.mutation_ratio * 2
        for gene_id in range(self.problem_size):
            probability = random()
            if probability <= self.mutation_ratio:
                mutated = True
                swapped_gene_id = gene_id
                while swapped_gene_id == gene_id:
                    swapped_gene_id = randint(0, self.problem_size-1)
                self.assignment[gene_id], self.assignment[swapped_gene_id] = self.assignment[swapped_gene_id], self.assignment[gene_id]
        self.calculate_fitness()
        return mutated

    def local_search(self):
        """
        Implements basic local search that go through neighbors of solution searching for the best one among them.
        Overall there should be n(n-1) possible swaps but in order to decrease the cardinality of neighborhood, only
        part of moves is analyzed.
        :return:
        """
        possible_moves = [[i, j] for i in range(self.problem_size) for j in range(self.problem_size) if i != j]
        shuffle(possible_moves)
        moves_number = round(len(possible_moves)*self.alg_parameters.neighborhood_size)
        possible_moves = possible_moves[0:moves_number]
        for move in possible_moves:
            temp_assignment = copy(self.assignment)
            temp_assignment[move[0]], temp_assignment[move[1]] = temp_assignment[move[1]], temp_assignment[move[0]]
            temp_solution = Solution(assignment=temp_assignment, alg_parameters=self.alg_parameters,
                                     qap_parameters=self.qap_parameters)
            if temp_solution.fitness < self.fitness:
                self.assignment[:self.problem_size] = temp_assignment
            self.calculate_fitness()

