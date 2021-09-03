class AlgorithmParameters:
    def __init__(self, mutation_ratio: float = 0.05, stop_diff: int = 150, node_size: int = 10,
                 ox_substring_width: float = 0.2, diversity_ratio: int = 10, neighborhood_size: float = 0.2,
                 parents_number: float = 0.2, local_search_frequency: float = 0.15):
        self.mutation_ratio = mutation_ratio
        self.stop_diff = stop_diff
        self.node_size = node_size
        self.ox_substring_width = ox_substring_width
        #self.diversity_ratio = diversity_ratio
        self.neighborhood_size = neighborhood_size
        self.parents_number = parents_number
        self.local_search_frequency = local_search_frequency
