import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage

class DnaEvolution(object):
    def __init__(self):
        self.code = ['A', 'T', 'C', 'G']
        self.num_parents = 3
        self.n_generations = 99
        self.n_offsprints_cutoff = 1000

    def generate_code(self,int_array):
        return [self.code[integer] for integer in int_array]

    def one_offspring(self,letter):
        p = np.random.random()
        if p < 0.001:
           return np.random.choice(self.code)
        return letter

    @staticmethod
    def similarity(offspr1, offspr2):
        return sum(i != j for i, j in zip(offspr1, offspr2))

    def generate_offsprint(self, ancestor):
        return [self.one_offspring(letter) for letter in ancestor]

    def parents(self):
        current_gen = []
        for i in range(self.num_parents):
            int_array_i = np.random.randint(4, size=1000)
            ancestor_i = self.generate_code(int_array_i)
            current_gen.append(ancestor_i)
        return current_gen

    def generation(self):
        current_gen = self.parents()
        for gen in range(self.n_generations):
            next_gen = []
            for ancestor in current_gen:
                offsprings = np.random.randint(3) + 1
                for offspr in range(offsprings):
                    child = self.generate_offsprint(ancestor)
                    next_gen.append(child)
            print(gen)
            current_gen = next_gen
            np.random.shuffle(current_gen)
            current_gen = current_gen[:self.n_offsprints_cutoff]
        return current_gen

    def distance_matrix(self):
        current_gen = self.generation()
        N = len(current_gen)
        dist_matrix = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                elif j > i:
                    childi = current_gen[i]
                    childj = current_gen[j]
                    sim_ij = DnaEvolution.similarity(childi, childj)
                    dist_matrix[i,j] = sim_ij
                else:
                    dist_matrix[i,j] = dist_matrix[j,i]
        self.distances = distance.squareform(dist_matrix)

    def ward_linkage(self):
        Z = linkage(self.distances, 'ward')
        plt.title("Ward linkage")
        dendrogram(Z)
        plt.savefig('Ward linkage.png')
        # plt.show()

    def single_linkage(self):
        Z = linkage(self.distances, 'single')
        plt.title("Single linkage")
        dendrogram(Z)
        plt.savefig('Single linkage.png')
        # plt.show()

    def complete_linkage(self):
        Z = linkage(self.distances, 'complete')
        plt.title("Complete linkage")
        dendrogram(Z)
        plt.savefig('Complete linkage.png')
        # plt.show()

    def run(self):
        # self.ward_linkage()
        self.single_linkage()
        # self.complete_linkage()

if __name__ == "__main__":
    evolution = DnaEvolution()
    evolution.distance_matrix()
    evolution.run()