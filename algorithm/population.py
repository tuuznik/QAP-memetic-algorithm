from typing import List, Tuple
#import random
from random import randint, random, shuffle
from parameters.algorithm_parameters import AlgorithmParameters
from parameters.qap_parameters import QAPParameters
from algorithm.node import Node
from algorithm.solution import Solution

"""
Represents the entire population that contains of 13 nodes, among which every node stores fixed number of solutions.
"""


class Population:

    def __init__(self, algorithm_parameters: AlgorithmParameters, qap_parameters: QAPParameters,
                 population_data: List[List[int]]):
        self.algorithm_parameters = algorithm_parameters
        self.qap_parameters = qap_parameters
        self.nodes = [Node(self.algorithm_parameters, self.qap_parameters, data) for data in population_data]
        self.best_solution = None
        self.root_node = self.nodes[0]
        self.sub_populations = [[1, [4, 5, 6]], [2, [7, 8, 9]], [3, [10, 11, 12]]]
        self.fitness_memory = [0]
        self.similarity_cnt = 0
        self.iteration = 0
        self.modify_mutation = False

    def __str__(self):
        return "Nodes:\n" + "\n".join(map(str, self.nodes))
    def update_population(self):
        """
        Facilities the update of entire population and the flow between all nodes. Firstly, evey node is updated
        internally, then the comparison is done within every subpopulation.
        :return:
        """
        for node in self.nodes:
            node.update_node()

        for sub_population in self.sub_populations:
            parent = self.nodes[sub_population[0]]
            for j in sub_population[1]:
                child = self.nodes[j]
                if child.best_solution.fitness < parent.best_solution.fitness:
                    self.nodes[j], self.nodes[sub_population[0]] = self.swap_best_solution(parent, child)

        for i in range(1, 4):
            parent_node = self.nodes[i]
            self.root_node.add_to_buffer(parent_node.best_solution.assignment)
            self.root_node.update_node()

        self.best_solution = self.root_node.best_solution
    def create_new_generation(self):
        """
        Procedure that gathers 3 main steps resulting in a new generation:
        1. Crossover
        2. Mutation
        3. Selection of new generation
        All of these are executed for three subgroups of main population.
        :return:
        """
        for sub_population in self.sub_populations:
            for j in sub_population[1]:
                parents_list = [self.nodes[sub_population[0]].select_parents(), self.nodes[j].select_parents()]
                offspring_list = []
                for i in range(len(parents_list[0])):
                    parents = [parents_list[0][i], parents_list[1][i]]
                    offspring = self.crossover(parents=parents)
                    offspring_list.extend(self.mutation(offspring))

                result = self.merge(offspring=offspring_list, parents=self.nodes[j].pockets)
                self.nodes[j].pockets[:len(result)] = result

    def crossover(self, parents: List[Solution], start=None) -> List[Solution]:
        """
        Function responsible for crossover. In this case it is Order Crossover.
        :param parents: list of two parents based on which pair of offspring is generated.
        :param start: optional parameter used for debug purposes that allows setting a point where copying of genes
                      begins.
        :return: List of offspring.
        """
        genes_number = self.qap_parameters.problem_size
        substring_width = round(self.algorithm_parameters.ox_substring_width*genes_number)
        if substring_width > genes_number:
            raise ValueError
        if start is None:
            start = randint(0, genes_number - (substring_width + 1))
        end = start + substring_width
        parents = [parents[0].assignment, parents[1].assignment]
        offspring = self.pmx_procedure(parent1=parents[0], parent2=parents[1], start=start, end=end,
                                       genes_number=genes_number)
        # offspring = [
        #     self.ox_procedure(parent1=parents[0], parent2=parents[1], start=start, end=end, genes_number=genes_number),
        #     self.ox_procedure(parent1=parents[1], parent2=parents[0], start=start, end=end, genes_number=genes_number)]
        return offspring

    def pmx_procedure(self, parent1, parent2, start, end, genes_number):

        offspring1 = [-1] * genes_number
        offspring2 = [-1] * genes_number
        offspring1[start:end] = parent1[start:end]
        offspring2[start:end] = parent2[start:end]

        mapped = offspring1 + offspring2
        looped = [element for element in mapped if mapped.count(element) > 1]
        mapping_pairs_1 = {}
        mapping_pairs_2 = {}
        # for i in range(start, end):
        #     if offspring2[i] in mapping_pairs_2.values():
        #         key = list(mapping_pairs_2.keys())[list(mapping_pairs_2.values()).index(offspring2[i])]
        #         mapping_pairs_2[key] = offspring1[i]
        #     else:
        #         if offspring1[i] in mapping_pairs_2.keys():
        #             mapping_pairs_2[offspring2[i]] = mapping_pairs_2[offspring1[i]]
        #             mapping_pairs_2.pop(offspring1[i])
        #         else:
        #             mapping_pairs_2[offspring2[i]] = offspring1[i]
        for i in range(start, end):
            if offspring1[i] != offspring2[i]:
                mapping_pairs_2[offspring2[i]] = offspring1[i]

        #print(mapping_pairs_2)
        to_delete = []
        for repeat in range(2):
            items = mapping_pairs_2.items()
            keys = list(mapping_pairs_2.keys())
            for key in keys:
                #print(mapping_pairs_2)
                if key in list(mapping_pairs_2.keys()):
                    item = [key, mapping_pairs_2[key]]
                    #print(item)
                    if item[0] != item[1]:
                        if item[0] in mapping_pairs_2.values():
                            key = list(mapping_pairs_2.keys())[list(mapping_pairs_2.values()).index(item[0])]
                            mapping_pairs_2[key] = item[1]
                            to_delete.append(item[0])
                            mapping_pairs_2.pop(item[0])
                        if item[1] in mapping_pairs_2.keys():
                            if item[0] in to_delete:
                                k = list(mapping_pairs_2.keys())[list(mapping_pairs_2.values()).index(item[1])]
                                # print(k)
                                # print(mapping_pairs_2[item[1]])
                                mapping_pairs_2[k] = mapping_pairs_2[item[1]]
                            else:
                                mapping_pairs_2[item[0]] = mapping_pairs_2[item[1]]

                            mapping_pairs_2.pop(item[1])
                    else:
                        mapping_pairs_2.pop(item[0])

                # for k in to_delete:
                #     if k in mapping_pairs_2.keys():
                #         mapping_pairs_2.pop(k)

        # to_delete = []
        # for repeat in range(2):
        #     for item in mapping_pairs_2.items():
        #         if item[0] in mapping_pairs_2.values():
        #             key = list(mapping_pairs_2.keys())[list(mapping_pairs_2.values()).index(item[0])]
        #             mapping_pairs_2[key] = item[1]
        #             to_delete.append(item[0])
        #         if item[1] in mapping_pairs_2.keys():
        #             if item[0] in to_delete:
        #                 key = list(mapping_pairs_2.keys())[list(mapping_pairs_2.values()).index(item[1])]
        #                 mapping_pairs_2[key] = mapping_pairs_2[item[1]]
        #             else:
        #                 mapping_pairs_2[item[0]] = mapping_pairs_2[item[1]]
        #
        #             to_delete.append(item[1])
        #
        #     for k in to_delete:
        #         if k in mapping_pairs_2.keys():
        #             mapping_pairs_2.pop(k)
        mapping_pairs_1 = {y: x for x, y in mapping_pairs_2.items()}
        # print(parent1)
        # print(parent2)
        # #print(mapped)
        # print(offspring1)
        # print(mapping_pairs_1)

        for j in range(genes_number):
            if offspring2[j] < 0:
                if parent1[j] not in mapped:
                    offspring2[j] = parent1[j]
                else:
                    offspring2[j] = mapping_pairs_2[parent1[j]]
            if offspring1[j] < 0:
                if parent2[j] not in mapped:
                    offspring1[j] = parent2[j]
                else:
                    offspring1[j] = mapping_pairs_1[parent2[j]]

        self.iteration += 2
        return [Solution(assignment=offspring2, alg_parameters=self.algorithm_parameters,
                         qap_parameters=self.qap_parameters), Solution(
            assignment=offspring1, alg_parameters=self.algorithm_parameters, qap_parameters=self.qap_parameters)]

    def ox_procedure(self, parent1, parent2, start, end, genes_number):
        """
        Supports the Ordered Crossover function with the main procedure of OX.
        :param parent1: Solution that is one parents.
        :param parent2: Solution that is the second parent.
        :param start: Point in which the copying of genes from one parent begins.
        :param end: Point in which the copying of genes from one parent ends.
        :param genes_number: Number of all genes that create a Solution.
        :return:
        """
        temp_list = parent1[start:end]
        all_ordered = parent2[end:] + parent2[0:end]
        unassigned = self.get_unique(all_ordered=all_ordered, assigned=temp_list)
        offspring_list = unassigned[(genes_number - end):] + temp_list + unassigned[:(genes_number - end)]
        offspring = Solution(assignment=offspring_list, alg_parameters=self.algorithm_parameters,
                             qap_parameters=self.qap_parameters)
        self.iteration += 1
        return offspring
    def merge(self, offspring: List[Solution], parents: List[Solution]) -> List[Solution]:
        """
        Chooses the best solution among list created by merging current generation with newly created offspring and
        complement a new list which is a generation now.
        :param offspring: List of solution obtained by crossover and mutation process.
        :param parents: List of all solution from current generation.
        :return: List presenting the new generation.
        """
        merged = offspring + parents
        merged = self.remove_duplicates(merged)
        merged.sort(key=lambda obj: obj.fitness if isinstance(obj, Solution) else float('inf'), reverse=False)
        merged = merged[:self.algorithm_parameters.node_size]
        return merged
    def remove_duplicates(self, solutions: List[Solution]) -> List[Solution]:
        """
        Method removes duplicates in List of Solution and gives list of unique elements.
        :param solutions: List of Solution that is to be reviewed.
        :return: List of unique elements.
        """
        # unique_list = []
        # unique_assignments = []
        # for s in solutions:
        #     if s.assignment not in unique_assignments:
        #         unique_list.append(s)
        #         unique_assignments.append(s.assignment)
        # return unique_list
        assignment_list = [tuple(s.assignment) for s in solutions if isinstance(s, Solution)]
        unique_list = list(set(assignment_list))
        unique_list = map(list, unique_list)
        return [Solution(assignment=a, qap_parameters=self.qap_parameters, alg_parameters=self.algorithm_parameters)
                for a in unique_list]
    def check_stop_condition(self, generation_id: int, optimal_fitness: int) -> tuple:
        """
        It is called every iteration of algorithm loop and it checks how many times the same best result occurred.
        If the counter outnumber the stop diff parameter, the algorithm stops.
        :param generation_id: number of generations (loop iterations)
        :param optimal_fitness: the best known value of adaptation
        :return: tuple containing info whether to continue the algorithm as well as counter of repetitive result.
        """
        self.fitness_memory.append(self.best_solution.fitness)
        if self.fitness_memory[generation_id] == self.fitness_memory[generation_id - 1]:
            self.similarity_cnt += 1
        else:
            self.similarity_cnt = 0
        # print("debug")
        # print(self.best_solution.fitness)
        # print(optimal_fitness)

        if self.similarity_cnt < self.algorithm_parameters.stop_diff and self.best_solution.fitness != optimal_fitness:
            return False, self.similarity_cnt
        else:
            return True, self.similarity_cnt
    def insert_local_search(self) -> bool:
        """
        During every iteration with a probability determined by local_search_frequency parameter, takes one random solution
        from entire population and do the local search in order to find the best neighbor.
        :return:
        """
        local_prob = random()
        inserted = False
        if local_prob < self.algorithm_parameters.local_search_frequency:
            self.iteration += 1
            #print("LOCAL")
            local_node = randint(0, 12)
            local_solution = randint(0, self.algorithm_parameters.node_size-1)
            self.nodes[local_node].pockets[local_solution].local_search()
            inserted = True
        return inserted
    def mutation(self, offspring: List[Solution]) -> List[Solution]:
        """
        Calls mutation method for modified offspring.
        :param offspring: List of solutions that are to be mutated.
        :return:
        """
        mutated_offspring = []
        if self.similarity_cnt > 40:
            self.modify_mutation = True
        for o in offspring:
            if o.gene_mutation(self.modify_mutation):
                self.iteration += 1
            mutated_offspring.append(o)
        return mutated_offspring
    @staticmethod
    def get_unique(all_ordered: List[int], assigned: List[int]) -> List[int]:
        """
        Part of OX crossover procedure that allows to get list of facilities that are not assigned yet in the offspring
        genotype, sorted in the same order as it was in the second parent.
        :param all_ordered: List of all facilities in the parent's solution order.
        :param assigned: List of facilities that are already assigned to the specific locations.
        :return: List of facilities that are to be assigned.
        """
        return [element for element in all_ordered if element not in assigned]
    @staticmethod
    def swap_best_solution(parent_node: Node, child_node: Node) -> Tuple[Node, Node]:
        """
        Function takes the nodes and swap their best solutions, firstly by finding their position in the node list, then
        changing them in nodes. To make sure that the new best solution is calculated properly, the nodes are updated.
        :param parent_node: One of the parent nodes in subpopulations.
        :param child_node: One of child nodes in subpopulations
        :return: Both nodes with changed solutions.
        """
        child_best_id = child_node.pockets.index(child_node.best_solution)
        parent_best_id = parent_node.pockets.index(parent_node.best_solution)
        child_node.pockets[child_best_id], parent_node.pockets[parent_best_id] = \
            parent_node.pockets[parent_best_id], child_node.pockets[child_best_id]
        child_node.update_node()
        parent_node.update_node()
        return parent_node, child_node






