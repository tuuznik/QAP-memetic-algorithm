from algorithm.population import Population
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, name: str, cnt: int):
        self.stats = {'Time': [], 'Iteration loop': [], 'Iteration': [], 'Fitness': [], 'Assignment': [], 'Local search': [],
                      'Similarity counter': []}
        self.name = name
        self.cnt = cnt

    def get_data(self, iteration_loop: int, iteration: int, population: Population, similarity_cnt: int, local_search: bool):
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M:%S %d/%m/%Y")
        self.save_statistics(time=current_time, iteration_loop=iteration_loop, iteration=iteration, population=population, local_search=local_search,
                             similarity_cnt=similarity_cnt)
        with open("../data/Output/{name}".format(name=self.name), 'a+') as file:
            if iteration == 1:
                file.write(97*'-' + '\n')
                file.write("ALGORITHM STATISTICS".center(97))
                file.write('\n'+97 * '-' + '\n')
                file.write('\n' + "Problem size: {size}".format(size=population.qap_parameters.problem_size))
                file.write('\n' + "Node size: {node_size}".format(node_size=population.algorithm_parameters.node_size))
                file.write('\n' + "Mutation ratio: {ratio}".format(ratio=population.algorithm_parameters.mutation_ratio))
                file.write('\n' + "Length of substring - OX Crossover: {length}".
                           format(length=population.algorithm_parameters.ox_substring_width))
                file.write('\n' + "Stop criteria : {stop}\n".format(stop=population.algorithm_parameters.stop_diff))
                file.write('\n'+97 * '-' + '\n')
                file.write('|'+"Time".center(23)+'|'+"Iteration".center(23)+'|'+"Best Fitness".center(23)+'|'+
                           "Similarity counter".center(23)+'|')
                file.write('\n' + 97 * '-' + '\n')
            else:
                file.write('|' + "{date}".format(date=current_time).center(23) + '|' +
                           "{iteration}".format(iteration=iteration_loop).center(23) + '|' +
                           "{fitness}".format(fitness=population.best_solution.fitness).center(23) +
                           '|' + "{cnt}".format(cnt=similarity_cnt).center(23) + '|\n')

    def save_statistics(self, time, iteration_loop: int, iteration: int, population: Population, local_search: bool, similarity_cnt: int):
        self.stats['Time'].append(time)
        self.stats['Iteration loop'].append(iteration_loop)
        self.stats['Iteration'].append(iteration)
        self.stats['Fitness'].append(population.best_solution.fitness)
        self.stats['Assignment'].append(population.best_solution.assignment)
        self.stats['Local search'].append(local_search)
        self.stats['Similarity counter'].append(similarity_cnt)

    def save_to_csv(self, mutation = 0):
        local_search_cnt = self.stats['Local search'].count(True)
        self.stats['Time'].append('-')
        self.stats['Iteration loop'].append('-')
        self.stats['Iteration'].append('-')
        self.stats['Fitness'].append('-')
        self.stats['Assignment'].append('-')
        self.stats['Similarity counter'].append('-')
        self.stats['Local search'].append(local_search_cnt)
        df = pd.DataFrame(data=self.stats)
        # df.to_csv(path_or_buf="../data/Output/" + self.name + '_all/' + self.name + '_' + str(self.cnt)  + ".csv",
        #           index=False)
        #df.to_csv(path_or_buf="../data/Output/mutation/" + self.name + '_' + str(mutation)  + ".csv",
        #          index=False)
        # df.to_csv(path_or_buf="../data/Output/crossover/" + self.name + '_' + str(mutation)  + ".csv",
        #          index=False)
        # df.to_csv(path_or_buf="../data/Output/selection/" + self.name + '_' + str(mutation) + ".csv",
        #           index=False)
        # df.to_csv(path_or_buf="../data/Output/" + self.name + '_all_4/' + self.name + '_' + str(self.cnt)  + ".csv",
        #           index=False)
        df.to_csv(path_or_buf="../data/Output/local/" + self.name + '_' + str(mutation)  + ".csv",
                  index=False)


def make_summary_separate(instance: str):
        stats_sum = {'Iteration loop': [], 'Iteration': [], 'Local search': [], 'Fitness': []}
        for i in range(1,11):
            #df = pd.read_csv("../data/Output/" + instance + "_all/" + instance + "_" + str(i) + ".csv")
            df = pd.read_csv("../data/Output/" + instance + "_all_4/" + instance + "_" + str(i) + ".csv")
            rows = df.shape[0]
            stats_sum['Iteration loop'].append(df.iloc[rows-2,1])
            stats_sum['Iteration'].append(df.iloc[rows-2,2])
            stats_sum['Local search'].append(df.iloc[rows-1,5])
            stats_sum['Fitness'].append(df.iloc[rows-2,3])
            print(stats_sum)
            df_summary = pd.DataFrame(data=stats_sum)
            #df_summary.to_csv(path_or_buf="../data/Output/Summary/" + instance + "_all.csv",  index=False)
            df_summary.to_csv(path_or_buf="../data/Output/Summary_4/" + instance + "_all.csv", index=False)


def gather_all_results():
    master_stats = {"Name": [], "Best": [], "Iteration": [], "Worst": [], "std_results" : [],"Mean": [], "E best": [],
                    "E worst": [], "E mean": [], "std_e": []}
    test_instances = ['bur26a','bur26b', 'bur26c','bur26d','bur26e', 'bur26f', 'bur26g', 'bur26h', 'chr22a', 'esc32c','esc32d','esc32e', 'esc32g', 'esc32h',
                      'kra30a', 'kra30b', 'lipa30a', 'lipa30b', 'lipa40a', 'lipa40b', 'lipa50a', 'lipa50b', 'lipa60a',
                      'lipa60b', 'sko42', 'sko49', 'sko56', 'tho30', 'tho40', 'wil50', 'chr22b', 'esc32a', 'esc32b',
                      'esc32f', 'esc64a', 'ste36a', 'ste36b', 'ste36c']
    for name in test_instances:
        df = pd.read_csv("../data/Output/Summary/" + name + "_all.csv")
        min_fitness = df["Fitness"].min()
        max_fitness = df["Fitness"].max()
        mean_fitness = df["Fitness"].mean()
        std_results = df["Fitness"].std()
        it_best = (df.loc[df['Fitness'] == min_fitness]).iloc[0,1]
        path_data = r'../data/Solutions/' + name + '.sln'
        with open(path_data, 'r') as file:
            _, optimal_fitness = tuple(file.readline().split())
        optimal_fitness = int(optimal_fitness)
        e_list = [((fitness-optimal_fitness)/optimal_fitness) for fitness in df["Fitness"]]
        df["e"] = e_list
        print(df)
        std_e = df["e"].std()
        e_best = (min_fitness-optimal_fitness)/optimal_fitness
        e_worst = (max_fitness-optimal_fitness)/optimal_fitness
        e_mean = (mean_fitness-optimal_fitness)/optimal_fitness
        master_stats["Name"].append(name.upper())
        master_stats["Best"].append(min_fitness)
        master_stats["E best"].append(e_best)
        master_stats["Iteration"].append(it_best)
        master_stats["Worst"].append(max_fitness)
        master_stats["E worst"].append(e_worst)
        master_stats["Mean"].append(mean_fitness)
        master_stats["E mean"].append(e_mean)
        master_stats["std_e"].append(std_e)
        master_stats["std_results"].append(std_results)
    df_stats = pd.DataFrame(data=master_stats)
    df_stats.to_csv(path_or_buf="../data/Output/Summary/" + "master.csv", index=False)
    #print(df_stats)

def gather_parameter_results():
    test_instances = ['bur26e','chr22b', 'esc32b', 'lipa60b', 'wil50']
    for name in test_instances:
        master_stats = {"Name": [], "Best": [], "Iteration": [], "Worst": [], "std_results": [], "Mean": [],
                        "E best": [], "E worst": [], "E mean": [], "std_e": []}
        for i in range (1,5):
            df = pd.read_csv("../data/Output/Summary_" + str(i) + "/" + name + "_all.csv")
            min_fitness = df["Fitness"].min()
            max_fitness = df["Fitness"].max()
            mean_fitness = df["Fitness"].mean()
            std_results = df["Fitness"].std()
            it_best = (df.loc[df['Fitness'] == min_fitness]).iloc[0,1]
            path_data = r'../data/Solutions/' + name + '.sln'
            with open(path_data, 'r') as file:
                _, optimal_fitness = tuple(file.readline().split())
            optimal_fitness = int(optimal_fitness)
            e_list = [((fitness-optimal_fitness)/optimal_fitness) for fitness in df["Fitness"]]
            df["e"] = e_list
            print(df)
            std_e = df["e"].std()
            e_best = (min_fitness-optimal_fitness)/optimal_fitness
            e_worst = (max_fitness-optimal_fitness)/optimal_fitness
            e_mean = (mean_fitness-optimal_fitness)/optimal_fitness
            master_stats["Name"].append(name.upper())
            master_stats["Best"].append(min_fitness)
            master_stats["E best"].append(e_best)
            master_stats["Iteration"].append(it_best)
            master_stats["Worst"].append(max_fitness)
            master_stats["E worst"].append(e_worst)
            master_stats["Mean"].append(mean_fitness)
            master_stats["E mean"].append(e_mean)
            master_stats["std_e"].append(std_e)
            master_stats["std_results"].append(std_results)
        df_stats = pd.DataFrame(data=master_stats)
        df_stats.to_csv(path_or_buf="../data/Output/Summary/summary_" + name + ".csv", index=False)
    #print(df_stats)


def compare_ox_pmx(name: str):
    df_pmx = pd.read_csv("../data/Output/" + name + "_all/" + name + "_1.csv")
    df_ox = pd.read_csv("../data/Output/" + name + "_all/" + name + "_1ox.csv")
    fitness_pmx = (df_pmx['Fitness'].tolist())[:-1]
    fitness_pmx = list(map(int, fitness_pmx))
    print(fitness_pmx)
    iteration_pmx = (df_pmx['Iteration loop'].tolist())[:-1]
    iteration_pmx = list(map(int, iteration_pmx))
    print(iteration_pmx)
    fitness_ox = (df_ox['Fitness'].tolist())[:-1]
    fitness_ox = list(map(int,fitness_ox))
    print(fitness_ox)
    iteration_ox = (df_ox['Iteration loop'].tolist())[:-1]
    iteration_ox = list(map(int,iteration_ox))
    print(iteration_ox)
    # plt.plot(iteration_ox, fitness_ox)
    # plt.show()


def compare_mutation(name, id):
    df = pd.read_csv("../data/Output/local/" + name + '_' + str(id) + ".csv")
    rows = df.shape[0]
    min_fitness = int(df.iloc[rows - 2, 3])
    optimal_fitness = 151426
    e_best = (min_fitness - optimal_fitness) / optimal_fitness
    return id, min_fitness, e_best
