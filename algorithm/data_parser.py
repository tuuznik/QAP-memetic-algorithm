from typing import List
from random import shuffle
from parameters.algorithm_parameters import AlgorithmParameters
from parameters.qap_parameters import QAPParameters
from algorithm.population import Population


class DataParser:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def generate_qap_parameters(self) -> QAPParameters:
        path_data = r'../data/Input/' + self.file_name + '.dat'
        with open(path_data, 'r') as file:
            problem_size = int(file.readline())
            lines = file.readlines()
            flow_lines = []
            distance_lines = []
            for i in range(len(lines)):
                if i in range(1, 1 + problem_size):
                    flow_lines.append(lines[i])
                elif i in range(2 + problem_size, 2 + 2 * problem_size):
                    distance_lines.append(lines[i])
            return QAPParameters(self.create_matrix(flow_lines),
                                 self.create_matrix(distance_lines))
    @staticmethod
    def create_matrix(lines: List[str]) -> List[List[int]]:
        matrix = []
        for line in lines:
            values_list = list(map(int, line.split()))
            matrix.append(values_list)
        return matrix

    def generate_alg_parameters(self, mutation_ratio):
        path_data = r'../data/Input/' + self.file_name + '_parameters.dat'
        return AlgorithmParameters(local_search_frequency=mutation_ratio)

    def get_optimal_solution(self) -> tuple:
        path_data = r'../data/Solutions/' + self.file_name + '.sln'
        with open(path_data, 'r') as file:
            problem_size, optimal_fitness = tuple(file.readline().split())
            optimal_assignment = list(map(int, file.readline().split()))
            optimal_assignment.extend(list(map(int, file.readline().split())))
            #print(optimal_assignment)
            #print(optimal_fitness)
        return optimal_assignment, optimal_fitness

    def generate_population(self, randomly= False, mutation = 0.05) -> Population:
        alg_parameters = self.generate_alg_parameters(mutation_ratio=mutation)
        #print(alg_parameters.mutation_ratio)
        qap_parameters = self.generate_qap_parameters()
        solution_size = qap_parameters.problem_size
        node_size = alg_parameters.node_size
        if randomly:
            population_data = self.generate_population_data(solution_size)
            population = Population(algorithm_parameters=alg_parameters, qap_parameters=qap_parameters,
                                    population_data=population_data)
            for i in range(node_size):
                new_population_data = self.generate_population_data(solution_size)
                for j in range(13):
                    population.nodes[j].add_to_buffer(new_population_data[j])
                    population.nodes[j].update_node()
        else:
            population_data = self.get_population_data()
            initial_solution = [population_data[i][0] for i in range(13)]
            population = Population(algorithm_parameters=alg_parameters, qap_parameters=qap_parameters,
                                    population_data=initial_solution)
            for i in range(node_size):
                new_population_data = [population_data[j][i] for j in range(13)]
                for j in range(13):
                    population.nodes[j].add_to_buffer(new_population_data[j])
                    population.nodes[j].update_node()
        return population

    def get_population_data(self):
        with open(r'../data/Init_population/' + self.file_name) as file:
            lines = file.readlines()
            nodes = []
            node_id = -1
            for line in lines:
                if line == "Pocket:\n":
                    node_id +=1
                    nodes.append([])
                else:
                    solutions = list(map(int,line.strip('\n').split(',')))
                    nodes[node_id].append(solutions)
        return nodes

    @staticmethod
    def generate_population_data(solution_size: int) -> List[List[int]]:
        population_data = []
        for i in range(13):
            new_solution = [j for j in range(solution_size)]
            shuffle(new_solution)
            population_data.append(new_solution)
        return population_data







