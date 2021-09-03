from algorithm.solution import Solution
from parameters.qap_parameters import QAPParameters
from parameters.algorithm_parameters import AlgorithmParameters
from algorithm.node import Node
from typing import List
import unittest


class NodeTest(unittest.TestCase):

    def test_node_init(self):
        distance = [[0, 5, 10], [5, 0, 4], [10, 4, 0]]
        flow = [[0, 2, 3], [2, 0, 5], [3, 5, 0]]
        test_input = [1, 0, 2]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters()
        node = Node(qap_parameters=qap_parameters, algorithm_parameters=alg_parameters, init_current=test_input)
        self.assertIsInstance(node, Node, "Should be instance of Node")
        self.assertIsInstance(node.buffer, Solution, "Current should be instance of Solution")
        node.update_node()
        self.assertIsInstance(node.pockets[0], Solution, "Current should be instance of Solution")

    def test_node_single_update(self):
        distance = [[0, 5, 10], [5, 0, 4], [10, 4, 0]]
        flow = [[0, 2, 3], [2, 0, 5], [3, 5, 0]]
        test_input = [1, 0, 2]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters()
        node = Node(qap_parameters=qap_parameters, algorithm_parameters=alg_parameters, init_current=test_input)
        node.update_node()
        self.assertIsInstance(node.pockets[0], Solution, "Current should be instance of Solution")

    def test_node_update(self):
        distance = [[0, 5, 10], [5, 0, 4], [10, 4, 0]]
        flow = [[0, 2, 3], [2, 0, 5], [3, 5, 0]]
        test_input = [1, 0, 2]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters()
        node = Node(qap_parameters=qap_parameters, algorithm_parameters=alg_parameters, init_current=test_input)
        node.update_node()
        solution_data = [[2, 1, 0], [0, 1, 2], [1, 2, 0], [0, 2, 1], [2, 0, 1]]
        for data in solution_data:
            node.add_to_buffer(data)
            node.update_node()
        self.assertIsInstance(node.pockets[-1], Solution, "Should be a Solution type")
        self.assertIsNone(node.buffer, "Current should be emptied")
        #self.assertEqual(node.pockets[0].assignment, [2, 0, 1], "Node updated incorrectly")

    def test_selection(self):
        distance = [[0, 5, 10], [5, 0, 4], [10, 4, 0]]
        flow = [[0, 2, 3], [2, 0, 5], [3, 5, 0]]
        test_input = [1, 0, 2]
        qap_parameters = QAPParameters(distance=distance, flow=flow)
        alg_parameters = AlgorithmParameters()
        node = Node(qap_parameters=qap_parameters, algorithm_parameters=alg_parameters, init_current=test_input)
        node.update_node()
        solution_data = [[2, 1, 0], [0, 1, 2], [1, 2, 0], [0, 2, 1], [2, 0, 1]]
        for data in solution_data:
            node.add_to_buffer(data)
            node.update_node()
        parent = node.select_parents()
        self.assertIsInstance(parent, Solution, "Selected parent should be Solution type")


if __name__ == '__main__':
    unittest.main()