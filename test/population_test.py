from algorithm.population import Population
from parameters.qap_parameters import QAPParameters
from parameters.algorithm_parameters import AlgorithmParameters
from algorithm.node import Node
from typing import List
import unittest
import random


class NodeTest(unittest.TestCase):

    def test_node_init(self):
        distance = [[0, 5, 10], [5, 0, 4], [10, 4, 0]]
        flow = [[0, 2, 3], [2, 0, 5], [3, 5, 0]]
        test_input = [[2, 1, 0], [0, 1, 2], [1, 2, 0], [0, 2, 1], [2, 0, 1], [1, 0, 2], [2, 1, 0], [0, 1, 2], [1, 2, 0],
                      [0, 2, 1], [2, 0, 1], [1, 0, 2], [1, 0, 2]]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters()
        population = Population(qap_parameters=qap_parameters, algorithm_parameters=alg_parameters,
                                population_data=test_input)
        self.assertIsInstance(population, Population, "Should be instance of Node")
        population.update_population()
        self.assertEqual(population.nodes[0].pockets[0].assignment, [2, 0, 1], "Population updated incorrectly")
        # for i in range(4, 13):
        #     population.nodes[i].add_current(test_input[i-4])
        population.update_population()
        self.assertEqual(population.nodes[0].best_solution.fitness, 146, "Population updated incorrectly")

    def test_ox_crossover(self):
        distance = [[0 for i in range(10)] for i in range(10)]
        flow = [[0 for i in range(10)] for i in range(10)]
        solution = [0 for i in range(10)]
        test_input = [solution for x in range(13)]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters(ox_substring_width=4)
        population = Population(qap_parameters=qap_parameters, algorithm_parameters=alg_parameters,
                                population_data=test_input)
        parents = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 4, 6, 8, 0, 1, 3, 5, 9, 7]]
        offspring = population.ox_crossover(parents=parents, start= 4)
        self.assertEqual(offspring, [8, 0, 1, 3, 4, 5, 6, 7, 9, 2], "Order One crossover failed!")