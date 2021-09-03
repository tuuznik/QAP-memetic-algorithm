from algorithm.data_parser import DataParser
from algorithm.data_analyzer import DataAnalyzer, make_summary_separate, gather_all_results, compare_ox_pmx, compare_mutation, gather_parameter_results
import random
import os
import pandas as pd


def algorithm(mutation: float = 0.05, cnt: int = 1, input_data: str = "esc64a"):
    random.seed(cnt)
    data_parser = DataParser(input_data)
    data_analyzer = DataAnalyzer(input_data, cnt)
    #data_parser.get_population_data()
    _, optimal_fitness = data_parser.get_optimal_solution()
    population = data_parser.generate_population(mutation=mutation)
    print(population.algorithm_parameters.local_search_frequency)
    # print(population.algorithm_parameters.neighborhood_size)
    population.update_population()
    generation_id = 0
    stop_condition = False
    #print(population)
    while not stop_condition:
        generation_id += 1
        population.create_new_generation()
        local_search = population.insert_local_search()
        population.update_population()
        stop_condition, similarity_cnt = population.check_stop_condition(generation_id=generation_id, optimal_fitness=optimal_fitness)
        #print(population.best_solution.fitness)
        data_analyzer.get_data(iteration_loop=generation_id, iteration=population.iteration, population=population, similarity_cnt=similarity_cnt,
                               local_search=local_search)
    data_analyzer.save_to_csv(mutation=mutation)


if __name__ == '__main__':
    #gather_parameter_results()
    # test_instances = ['bur26a','bur26b', 'bur26c','bur26d','bur26e', 'bur26f', 'bur26g', 'bur26h', 'chr22a', 'esc32c','esc32d','esc32e', 'esc32g', 'esc32h',
    #                   'kra30a', 'kra30b', 'lipa30a', 'lipa30b', 'lipa40a', 'lipa40b', 'lipa50a', 'lipa50b', 'lipa60a',
    #                   'lipa60b', 'sko42', 'sko49', 'sko56', 'tho30', 'tho40', 'wil50']
    # test_instances = ['chr22b', 'esc32a', 'esc32b', 'esc32f', 'esc64a', 'ste36a', 'ste36b', 'ste36c']
    # test_instances = ['bur26e', 'chr22b', 'esc32b', 'lipa60b', 'wil50']
    # #test_instances = ['lipa60b','wil50']
    # #gather_all_results()
    # for name in test_instances:
    #      print('*'*100)
    #      print(name)
    #      for i in range (1,11):
    #         print(i)
    #         algorithm(cnt = i, input_data=name)
    #      make_summary_separate(name)
    #algorithm(cnt=1, input_data='ste36c')
    # #compare_ox_pmx("lipa60b")
    results = { "E1": [], "E2": [], "E3": [], "E4": [], "E5": []}
    for j in range (1,6):
        for i in range (0,100,10):
                if i != 0:
                    i = i/100
                #print(i)
                #print(i)
                algorithm(cnt=j ,mutation=i, input_data='tho40')
                _, fitness, e = compare_mutation(name='tho40', id=i)
                #col1 = "Fitness" + str(j)
                col2 = "E" + str(j)
                #results[col1].append(fitness)
                results[col2].append(e)
    df_stats = pd.DataFrame(data=results)
    print(df_stats)
    df_stats.to_csv(path_or_buf="../data/Output/local/summary.csv", index=False)




     #'bur26a','bur26b', 'bur26c','bur26d'
    #algorithm(input_data='tho40')






