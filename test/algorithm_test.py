import unittest
from algorithm.main_algorithm import algorithm
from algorithm.data_analyzer import get_data


class NodeTest(unittest.TestCase):

    def test_algorithm_stages(self):
        algorithm()
