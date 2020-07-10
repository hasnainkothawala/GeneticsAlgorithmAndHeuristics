

"""
Basic TSP Example
file: Individual.py
Author :
    first name: hasnain
    last  name: kothawala
"""

import random
import math


class Individual:

    def __init__(self, _size, _data):

        """
        Parameters and general variables
        """
        self.fitness    = 0
        self.genes      = []
        self.genSize    = _size
        self.data       = _data

        self.genes = list(self.data.keys())

        for i in range(0, self.genSize):
            n1 = random.randint(0, self.genSize-1)
            n2 = random.randint(0, self.genSize-1)
            tmp = self.genes[n2]
            self.genes[n2] = self.genes[n1]
            self.genes[n1] = tmp

    def setGene(self, genes):
        """
        Updating current choromosome
        """
        self.genes = []
        for gene_i in genes:
            self.genes.append(gene_i)


    def setGenSize(self, gene_size):
        """
        Updating current choromosome
        """
        self.genSize = gene_size


    def copy(self):
        """
        Creating a new individual
        """
        ind = Individual(self.genSize, self.data)
        for i in range(0, self.genSize):
            ind.genes[i] = self.genes[i]
        ind.fitness = self.getFitness()
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return int( math.sqrt( (d1[0]-d2[0])**2 + (d1[1]-d2[1])**2 ) )
    
    
    
    
    
    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        #print('Inside compute fitness and the gene sequence of this Indi is -->',self.genes)
        #print('Inside compute fitness and the Data for this Indi is -->',self.data)
        self.fitness    = self.euclideanDistance(self.genes[0], self.genes[self.genSize-1])
        for i in range(0, self.genSize-1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i+1])
        #print('calculated fitness for this mutated gene is ==>',self.fitness)
        
        
    


