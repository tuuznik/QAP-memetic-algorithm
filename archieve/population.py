from typing import List, Tuple
from random import randint, shuffle
import numpy as np
from parameters.algorithm_parameters import AlgorithmParameters
from parameters.qap_parameters import QAPParameters
from algorithm.node import Node
from algorithm.solution import Solution


class Population:
    def __init__(self, algorithm_parameters: AlgorithmParameters, qap_parameters: QAPParameters,
                 population_data: List[List[int]]):
        self.algorithm_parameters = algorithm_parameters
        self.qap_parameters = qap_parameters
        self.nodes = [Node(self.algorithm_parameters, self.qap_parameters, data) for data in population_data]
        self.best_solution = None
        self.root_node = 0
        self.sub_populations = [[1, [4, 5, 6]], [2, [7, 8, 9]], [3, [10, 11, 12]]]    # to do ogarnąć to

    def __str__(self):
        return "Nodes:\n" + "\n".join(map(str, self.nodes))

    def update_population(self):
        # WYMAGA DOPASOWANIA I POPRAWY
        for node in self.nodes:
            node.update_node()

        for sub_population in self.sub_populations:
            parent = self.nodes[sub_population[0]]
            for j in sub_population[1]:
                child = self.nodes[j]
                if child.best_solution.fitness < parent.best_solution.fitness:
                    self.nodes[j], self.nodes[sub_population[0]] = swap_best_solution(parent, child)

        root = self.nodes[0]
        for i in range(1, 4):
            parent_node = self.nodes[i]
            root.add_to_buffer(parent_node.best_solution.assignment)
            root.update_node()
            # if parent_node.best_solution.fitness > root.best_solution.fitness:
            #     parent_best_id = parent_node.pockets.index(parent_node.best_solution)
            #     for j in range(0, root.size):
            #         if not isinstance(root.pockets[j], Solution):
            #             root.pockets[j] = parent_node.pockets[parent_best_id]
            #             #parent_node.pockets[parent_best_id] = None
            #             root.update_node()
            #             parent_node.update_node()
            #             break
            #     if parent_node.pockets[parent_best_id]:
            #         self.nodes[0], self.nodes[i] = swap_best_solution(root, parent_node)
        self.best_solution = self.nodes[0].best_solution  # najlepsze rozwiązanie z root node

    # def crossover(self):
    #     for sub_population in self.sub_populations:
    #         for j in sub_population[1]:
    #             parents = [self.nodes[sub_population[0]].select_parent(),
    #                        self.nodes[j].select_parent()]
    #             offspring = self.ox_crossover(parents=parents)
    #             self.nodes[j].add_current(buffer=offspring.assignment)
    #
    #     for j in [1, 2, 3]:
    #         parents = [self.nodes[0].select_parent(),
    #                    self.nodes[j].select_parent()]
    #         offspring = self.ox_crossover(parents=parents)
    #         self.nodes[j].add_current(buffer=offspring.assignment)

    def crossover(self):
        temp_sub = self.sub_populations
        #temp_sub.append([0, [1, 2, 3]])
        for sub_population in temp_sub:
            for j in sub_population[1]:
                parents_list = [self.nodes[sub_population[0]].select_parents(), self.nodes[j].select_parents()]
                offspring_list = []

                for i in range(len(parents_list[0])):

                    parents = [parents_list[0][i], parents_list[1][i]]
                    offspring = self.ox_crossover(parents=parents)
                    offspring_list.extend(mutation(offspring))

                result = self.merge(offspring=offspring_list, parents=self.nodes[j].pockets)
                self.nodes[j].pockets[:len(result)] = result

    def ox_crossover(self, parents: List[Solution], start=None) -> List[Solution]:
        substring_width = self.algorithm_parameters.ox_substring_width
        genes_number = self.qap_parameters.problem_size
        #substring_width = randint(0, genes_number-2)
        if substring_width > genes_number:
            raise ValueError
        if start is None:
            start = randint(0, genes_number - (substring_width + 1))
        end = start + substring_width
        parents = [parents[0].assignment, parents[1].assignment]
        offspring = [
            self.ox_procedure(parent1=parents[0], parent2=parents[1], start=start, end=end, genes_number=genes_number),
            self.ox_procedure(parent1=parents[1], parent2=parents[0], start=start, end=end, genes_number=genes_number)]
        # temp_list = parents[0][start:end]
        # all_ordered = parents[1][end:] + parents[1][0:end]
        # unassigned = get_unique(all_ordered=all_ordered, assigned=temp_list)
        # offspring_list = unassigned[(genes_number-end):] + temp_list + unassigned[:(genes_number-end)]
        # offspring = Solution(assignment=offspring_list,alg_parameters=self.algorithm_parameters,
        #                      qap_parameters=self.qap_parameters)
        return offspring

    def ox_procedure(self, parent1, parent2, start, end, genes_number):
        temp_list = parent1[start:end]
        all_ordered = parent2[end:] + parent2[0:end]
        unassigned = get_unique(all_ordered=all_ordered, assigned=temp_list)
        offspring_list = unassigned[(genes_number - end):] + temp_list + unassigned[:(genes_number - end)]
        offspring = Solution(assignment=offspring_list, alg_parameters=self.algorithm_parameters,
                             qap_parameters=self.qap_parameters)
        return offspring

    def merge(self, offspring: List[Solution], parents: List[Solution]) -> List[Solution]:
        #offspring, parents = self.standard_crowding(offspring=offspring, parents=parents)
        merged = offspring + parents
        merged = self.remove_duplicates(merged)
        merged.sort(key=lambda obj: obj.fitness if isinstance(obj, Solution) else 999999999, reverse=False)
        # for i in range(len(merged)):
        #     print(merged[i])
        #print(100*'-')
        merged = merged[:self.algorithm_parameters.node_size]
        # print(merged)
        return merged

    def remove_duplicates(self, solutions: List[Solution]) -> List[Solution]:
        assignment_list = [tuple(s.assignment) for s in solutions if isinstance(s,Solution)]
        unique_list = list(set(assignment_list))
        unique_list = map(list, unique_list)
        return [Solution(assignment=a, qap_parameters=self.qap_parameters, alg_parameters=self.algorithm_parameters)
                for a in unique_list]

    # def standard_crowding(self, offspring: List[Solution], parents: List[Solution]):
    #     for o in offspring:
    #         for p in parents:
    #             if p:
    #                 difference_list = list(np.array(o.assignment) - np.array(p.assignment))
    #                 #print(difference_list)
    #                 similarity_cnt = difference_list.count(0)
    #                 #print(similarity_cnt)
    #                 if similarity_cnt > self.algorithm_parameters.diversity_ratio:
    #                     if o.fitness > p.fitness:
    #                         p_id = parents.index(p)
    #                         parents[p_id] = None
    #                     else:
    #                         o_id = offspring.index(o)
    #                         offspring[o_id] = None
    #                         break
    #     return offspring, parents


def mutation(offspring: List[Solution]):
    mutated = []
    for o in offspring:
        o.gene_mutation()
        mutated.append(o)
    return mutated


def get_unique(all_ordered, assigned):
    return [element for element in all_ordered if element not in assigned]


def swap_best_solution(parent_node: Node, child_node: Node) -> Tuple[Node, Node]:
    child_best_id = child_node.pockets.index(child_node.best_solution)
    parent_best_id = parent_node.pockets.index(parent_node.best_solution)
    child_node.pockets[child_best_id], parent_node.pockets[parent_best_id] = \
        parent_node.pockets[parent_best_id], child_node.pockets[child_best_id]
    child_node.update_node()
    parent_node.update_node()
    return parent_node, child_node


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
        return AlgorithmParameters(ox_substring_width=mutation_ratio)
    def get_optimal_solution(self) -> tuple:
        path_data = r'../data/Solutions/' + self.file_name + '.sln'
        with open(path_data, 'r') as file:
            problem_size, optimal_fitness = tuple(file.readline().split())
            optimal_assignment = list(map(int, file.readline().split()))
            optimal_assignment.extend(list(map(int, file.readline().split())))
            #print(optimal_assignment)
            #print(optimal_fitness)
        return optimal_assignment, optimal_fitness
    def generate_population(self, randomly= False) -> Population:
        alg_parameters = self.generate_alg_parameters(mutation_ratio=mutation)
        print(alg_parameters.mutation_ratio)
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



