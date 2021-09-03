from algorithm.solution import Solution
from parameters.qap_parameters import QAPParameters
from parameters.algorithm_parameters import AlgorithmParameters
import unittest


class SolutionTest(unittest.TestCase):
    def test_solution(self):
        distance = [[0, 5, 10], [5, 0, 4], [10, 4, 0]]
        flow = [[0, 2, 3], [2, 0, 5], [3, 5, 0]]
        test_input = [1, 0, 2]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters()
        solution = Solution(test_input, qap_parameters, alg_parameters)
        self.assertIsInstance(solution, Solution, "Should be instance of Solution")
        #solution.calculate_fitness()
        self.assertEqual(solution.fitness, 144, "Fitness of this solution should be 144")
        solution.gene_mutation()

    def test_solution_exception(self):
        distance = [[0, 5, 10]]
        flow = [[0, 2, 3], [2, 0, 5], [3, 5, 0]]
        test_input = [1, 0, 2]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters()
        with self.assertRaises(Exception):
            solution = Solution(test_input, qap_parameters, alg_parameters)


if __name__ == '__main__':
    unittest.main()
