import unittest
from algorithm.data_parser import generate_population_data, generate_qap_parameters
from parameters.qap_parameters import QAPParameters


class DataParserTest(unittest.TestCase):

    def test_generate_population_data(self):
        population_data = generate_population_data(solution_size=5)
        self.assertEqual(len(population_data), 13, "The number of solution data is different from 13")
        self.assertEqual(len(population_data[0]), 5, "The number of genes in solution is different from 5")

    def test_generate_qap_parameters(self):
        qap_params = generate_qap_parameters("tai10a.dat")
        self.assertIsInstance(qap_params, QAPParameters, "The received object is not QAPParameters type")
        self.assertEqual(qap_params.problem_size, 16, "The size of problem was set incorrectly.")
        self.assertEqual(len(qap_params.flow), 16, "The size of flow matrix is incorrect")