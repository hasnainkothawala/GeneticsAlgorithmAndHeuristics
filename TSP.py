"""
Author:

First Name: Hasnain
Last Name:  Kothawala


"""

import random
import math
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from Individual import *

myStudentNumber = 123245
random.seed(myStudentNumber)

best_fitness_list_forHyperParams = []

c5_fitness_list_per_iteration = []
c5_fitness_list_per_execution = []


class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, initilization_type, selection_type,
                 crossover_type, mutation_type):

        self.population = []
        self.matingPool = []
        self.best = None
        self.popSize = _popSize
        self.genSize = None
        self.mutationRate = _mutationRate
        self.maxIterations = _maxIterations
        self.iteration = 0
        self.fName = _fName
        self.data = {}
        self.totalFitness = 0
        self.parentCount = 2
        self.data_numpy = None
        self.initilization_type = initilization_type
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.best_fitness_list = []

        self.readInstance()
        # self.initPopulation()
        # self.initPopulationWithHueristics()

    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        

    def euclidean_distance_forMatrix_Cal(self, x_coordinate, y_coordinate):

        weight = math.sqrt((x_coordinate[0] - y_coordinate[0]) ** 2 + (x_coordinate[1] - y_coordinate[1]) ** 2)
        return weight

    def parse_in(self):
        
        self.data_numpy = pd.read_csv(self.fName, sep=' ', names=range(1, 3), skiprows=1)

        size_ofmatrix = self.genSize
        weight_matrix = np.zeros((size_ofmatrix + 1, size_ofmatrix + 1))

        for x in range(1, size_ofmatrix + 1):
            
            x_coordinate = list(self.data_numpy.loc[x][:])
            for y in range(1, size_ofmatrix + 1):
                y_coordinate = list(self.data_numpy.loc[y][:])
                
                weight = self.euclidean_distance_forMatrix_Cal(x_coordinate, y_coordinate)
                
                weight_matrix[x, y] = weight
        return weight_matrix

    def initPopulationWithHueristics(self):

        size_of_matrix = self.genSize
        weight_matrix = self.parse_in()
        for z in range(self.popSize):
            cities_left = list(range(1, size_of_matrix + 1))
            origin_city = random.randint(1, size_of_matrix)
            
            current_city = origin_city
            order_of_city = [current_city]
            order_of_weight = list()
            cities_left.remove(current_city)
            
            for z in range(size_of_matrix - 1):
                current_slice = weight_matrix[current_city][:]
                lowest_wt = 10 ** 15
                for x in cities_left:
                    if x != current_city:
                        current_wt = current_slice[x]
                        
                        if current_wt < lowest_wt:
                            
                            lowest_wt = current_wt
                            current_city = x
                if current_city in cities_left:
                    cities_left.remove(current_city)

                order_of_city.append(current_city)
                order_of_weight.append(lowest_wt)

            order_of_weight.append(weight_matrix[current_city][origin_city])
            
            total_weight = 0
            for x in order_of_weight:
                total_weight += x
            
            individual = Individual(self.genSize, self.data)
            individual.setGene(order_of_city)
            individual.computeFitness()
            self.population.append(individual)
            

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[random.randint(0, self.popSize - 1)]
        indB = self.matingPool[random.randint(0, self.popSize - 1)]
        return [indA, indB]

    def stochasticUniversalSampling(self):
        lower_bound = 0
        i = 0
        selected_chromsomes = []
        fitness_dictionary = {}

        # Step1 : Let F be the sum of the fitness values of all chromosomes in the population.
        # Solving the Minimization problem
        z = self.calculateTotalFitness()

        f = (z * 10 ** 15)

        # Step 2 : Let N be the number of parents to select.
        n = self.parentCount

        # step 3 : p is the distance between each mark up on the ruler
        p = int(f / n)

        # Step 4: Randomly select the starting point of the ruler
        start_point_ofRuler = random.randint(0, p)
        list_ofRuler_cordinates = [start_point_ofRuler]

        # Since the list_ofRuler_cordinates has already been initialized with staring point range must be parent count-1
        # Step : 5 Append the rest of the mark ups to the co ordinate
        for x in range(self.parentCount - 1):
            list_ofRuler_cordinates.append(list_ofRuler_cordinates[-1] + p)

        # Step 6: Assign each chromosome a range equal in length to its fitness and a starting point that is after the end point of the previous chromosome
        for individual in self.matingPool:
            upper_bound = lower_bound + int((1 / individual.fitness) * 10 ** 15)
            
            fitness_dictionary[i] = [lower_bound, upper_bound]
            lower_bound = upper_bound + 1
            i += 1

        # Step 7: Select the chromosomes whose range contains a marker
        for x in list_ofRuler_cordinates:
            for y in range(self.popSize):
                [lower, upper] = fitness_dictionary[y]

                if x in range(lower, upper):
                    selected_chromsomes.append(y)

        indA = self.matingPool[selected_chromsomes[0]]
        indB = self.matingPool[selected_chromsomes[1]]
        return [indA, indB]

    def calculateTotalFitness(self):
        total_fitness = 0
        
        for individual in self.matingPool:
            individual.computeFitness()
            
            total_fitness += (1 / individual.getFitness())
            
        return total_fitness

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        '''
        Step 1:
        Randomly choose few index locations. 
        So that, the elements in the respective child locations remain fixed in the child chromosome.'''

        fixed_set_of_gene = random.sample(indA.genes, random.randint(0, self.genSize - 1))
        childA = np.zeros(self.genSize, dtype=int)

        '''Step 2:
        Starting from the second crossover site the genes that are not already 
        present in the offspring from the alternative parent (the parent other than 
        the one whose genes are copied by the offspring in the initial phase) in the 
        order they appear.
        '''
        for fixed_gene in fixed_set_of_gene:
            childA[indA.genes.index(fixed_gene)] = fixed_gene
        k = 0
        while k < self.genSize:
            if childA[k] != 0:
                pass
            else:
                for gene_in_b in indB.genes:
                    if gene_in_b in childA:
                        pass
                    else:
                        childA[k] = gene_in_b
            k += 1
        '''Step 3: Return the mutated child'''
        return list(childA)

    def pmxCrossover(self, individualA, individualB):
        child1 = []
        child2 = []
        slice_indexes = []
        non_slice_indexes = []
        indexes_ofC1_inConflict = []
        index4 = []
        slice1 = []
        slice2 = []
        genSize = self.genSize
        indA = individualA.genes
        indB = individualB.genes
        

        # Step1:Create a child with all x of length of gene
        for i in range(0, genSize):
            child1.append('x')
            child2.append('x')

        

        # Step2:Randomly choose the slice index locations
        start_of_slice = random.randint(0, genSize - 1)
        end_of_slice = random.randint(start_of_slice, genSize - 1)

        
        # Step3:Copy the slice of indB to childA and indA to childB
        for j in range(start_of_slice, end_of_slice + 1):
            slice_indexes.append(j)
            child1[j] = indB[j]
            child2[j] = indA[j]
            slice1.append(child1[j])
            slice2.append(child2[j])
        

        # Step4: Create a list of indexes those do not belong to the slice
        for i in range(0, genSize):
            if i in slice_indexes:
                i += 1
            else:
                non_slice_indexes.append(i)

        # Step5: Copy the no conflict elements from IndA to childA , indB to childB
        for k in non_slice_indexes:
            if indA[k] in child1:
                k += 1

            else:
                child1[k] = indA[k]
        
        for l in non_slice_indexes:
            if indB[l] in child2:
                l += 1
            else:
                child2[l] = indB[l]

        # Step6: Replace all the x in children with 0
        for i in range(len(child1)):
            if 'x' in child1:
                indexes_ofC1_inConflict.append(child1.index('x'))
                child1[child1.index('x')] = 0
            else:
                i += 1

        for i in range(len(child2)):
            if 'x' in child2:
                index4.append(child2.index('x'))
                child2[child2.index('x')] = 0
            else:
                i += 1
        # Step7: Now replace the conflicted locations with elements by using one to one relations between slice1 and slice2
        for i in indexes_ofC1_inConflict:
            n = indA[i]
            m = slice1.index(n)

            if not slice2[m] in slice1:
                child1[i] = slice2[m]
            else:
                p = slice1.index(slice2[m])

                flag = 1
                while flag == 1:
                    if not slice2[p] in slice1:
                        child1[i] = slice2[p]
                        flag = 0

                    else:
                        q = slice1.index(slice2[p])
                        p = q
                        continue

        for i in index4:
            n = indB[i]
            m = slice2.index(n)

            if not slice1[m] in slice2:
                child2[i] = slice1[m]
            else:
                p = slice2.index(slice1[m])

                flag = 1
                while flag == 1:
                    if not slice1[p] in slice2:
                        child2[i] = slice1[p]
                        flag = 0

                    else:
                        q = slice2.index(slice1[p])
                        p = q
                        continue

        return child1

    def reciprocalExchangeMutation(self, ind):
        """
       Your Reciprocal Exchange Mutation implementation
       """
        if random.random() < self.mutationRate:
            geneseq = ind.genes
            seq1 = random.randint(0, self.genSize - 1)
            seq2 = random.randint(0, self.genSize - 1)
            
            temp = ind.genes[seq1]
            ind.genes[seq1] = ind.genes[seq2]
            ind.genes[seq2] = temp
            
            ind.computeFitness()
            self.updateBest(ind)

    def inversionMutation(self, ind):
        if random.random() < self.mutationRate:
            geneseq = ind.genes

            # choose two random locations from the child chromosomes
            seq1 = random.randint(0, len(geneseq) - 1)
            seq2 = random.randint(seq1, len(geneseq) - 1)

            temp = geneseq[seq1:seq2]

            # Reverse the slice chosen between the random indexes
            temp.reverse()

            # Insert the reversed slice to the mutated child chromsome at the same location
            i = 0
            for x in range(seq1, seq2):
                geneseq[x] = temp[i]
                i += 1
            
            ind.setGene(geneseq)
            ind.computeFitness()
            self.updateBest(ind)

    def crossover(self, indA, indB):
        """
       Executes a 1 order crossover and returns a new individual
       """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize - 1)
        indexB = random.randint(0, self.genSize - 1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):

        
        indexA = random.randint(0, self.genSize - 1)
        indexB = random.randint(0, self.genSize - 1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp
        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        self.matingPool = []
        list_of_genes = []

        for ind_i in self.population:
            self.matingPool.append(ind_i.copy())

    def newGeneration(self):

        self.population = []
        for i in range(0, self.popSize):

            # selection
            if self.selection_type == 'random_selection':
                [indA, indB] = self.randomSelection()
            else:
                [indA, indB] = self.stochasticUniversalSampling()

            # crossover
            if self.crossover_type == 'uniform_crossover':
                child_gene = self.uniformCrossover(indA, indB)
            else:
                child_gene = self.pmxCrossover(indA, indB)

            child_individual = Individual(self.genSize, self.data)
            child_individual.setGene(child_gene)

            # mutation
            if self.mutation_type == 'inverse_mutation':
                self.inversionMutation(child_individual)
            else:
                self.reciprocalExchangeMutation(child_individual)

            # Update the population with the new mutated chromosome
            self.population.append(child_individual)
            
            list_of_genes = []
            for indi in self.population:
                list_of_genes.append(indi.genes)
            

    def GAStep(self):
        """
       One step in the GA main algorithm
       1. Updating mating pool with current population
       2. Creating a new Generation
       """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
       General search template.
       Iterates for a given number of steps
          """

        file = open(log_file_name, 'a+')
        self.iteration = 0

        while self.iteration < self.maxIterations:
            self.best_fitness_list.append(int(self.best.getFitness() / 10000))
            # print('The best fitness chromosome until now==>', self.best.getFitness()/10000 )
            print(
                '***********************************************ITERATION {} ***Config 5******Population Size:{}**************************************'.format(
                    self.iteration, self.popSize))
            self.GAStep()

            self.iteration += 1
            file.write(str(self.best.getFitness()) + ' , ')
        file.close()

        # print('The list of best fitness==>',self.best_fitness_list)
        # print("Total iterations: ", self.iteration)
        # print("Best Solution: ", self.best.getFitness())
        # print('Best Chromosome : ', self.best.genes)
        best_fitness_list_forHyperParams.append(int(self.best.getFitness() / 10000))


'''Use this method to run different configurations
        initilization_type= random_init / hueristics_init 
        crossover_type =    uniform_crossover / pmx_crossover
        mutation_type  =    inverse_mutation /  reciprocal_exchange
        selection_type =    random_selection /  stocastic_selection
   '''

# Configuration 4  :
log_file_name = 'Config5_poppulationVariation.txt'
problem_file = 'inst-4.tsp'
_popSize = 100
_mutationRate = 0.1
_maxIterations = 200
initilization_type = 'random_init'
selection_type = 'stocastic_selection'
crossover_type = 'pmx_crossover'
mutation_type = 'inverse_mutation'

print(
    '*****************************************************************CONFIGURATION 5***************************************************************************')
tic = time.time()

print('Start time is==>', tic)

population_list = [100, 200, 300, 400, 500]
for i in range(5):
    ga = BasicTSP(problem_file, _popSize, _mutationRate, _maxIterations, initilization_type, selection_type,
                  crossover_type, mutation_type)

    if (initilization_type == 'random_init'):
        ga.initPopulation()
    else:
        ga.initPopulationWithHueristics()

    ga.search()
    c5_fitness_list_per_iteration.append(ga.best_fitness_list)
    c5_fitness_list_per_execution.append(best_fitness_list_forHyperParams)

print('*******************************************c5_fitness_list_per_iteration******************************')
print(c5_fitness_list_per_iteration)
print('*******************************************c5_fitness_list_per_execution******************************')
print(c5_fitness_list_per_execution)

tac = time.time()
print('End time is==>', tac)

time_elapsed = tac - tic
print('Total time elapsed==>', time_elapsed)




'''Use below to generate plots'''

# my_list = [1, 2, 3, 4, 5]
# population_list = [100, 200, 300, 400, 500]
# for x in range(5):
#     list_of_cost_overTheIterations = c5_fitness_list_per_iteration[x]
#     plt.plot(list_of_cost_overTheIterations)
#     plt.xlabel('number of iterations')
#     plt.ylabel('cost')
#     plt.title('Config 5 - Population Size:{}  :200:0.1  min cost:{}'.format(population_list[x],
#                                                                             best_fitness_list_forHyperParams[x]))
#     plt.show()
#
# list_of_cost_overTheIterations = best_fitness_list_forHyperParams
# plt.plot(list_of_cost_overTheIterations)
# plt.xlabel('Number of Loops')
# plt.ylabel('Cost  per-10000')
# plt.title('Configuration 5: Cost Performance over multiple Population Size')
# plt.show()





'''
# Configuration 5  : 
    problem_file        ='inst-4.tsp'
    _popSize            =100
    _mutationRate       =0.05
    _maxIterations      =10
    initilization_type  ='random_init'
    selection_type      ='stocastic_selection'
    crossover_type      ='pmx_crossover'
    mutation_type       ='reciprocal_exchange'
    '''



# Configuration 6  :
'''
    problem_file        ='inst-4.tsp'
    _popSize            =100
    _mutationRate       =0.05
    _maxIterations      =10
    initilization_type  ='random_init'
    selection_type      ='stocastic_selection'
    crossover_type      ='uniform_crossover'
    mutation_type       ='inverse_mutation'
    '''

# Configuration 7  : 
'''
    problem_file        ='inst-0.tsp'
    _popSize            =100
    _mutationRate       =0.05
    _maxIterations      =10
    initilization_type  ='hueristics_init'
    selection_type      ='stocastic_selection'
    crossover_type      ='pmx_crossover'
    mutation_type       ='reciprocal_exchange'
    '''

# Configuration 8  : 
'''
    problem_file        ='inst-0.tsp'
    _popSize            =100

    _mutationRate       =0.05
    _maxIterations      =10
    initilization_type  ='hueristics_init'
    selection_type      ='stocastic_selection'
    crossover_type      ='uniform_crossover'
    mutation_type       ='inverse_mutation'
    '''
