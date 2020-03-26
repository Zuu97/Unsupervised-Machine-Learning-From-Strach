import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.cluster.hierarchy import dendrogram, linkage

class HierarchicalClustering(object):
    def __init__(self):
        self.f = 2
        self.N = 1200

    def create_data(self, m):
        X = np.zeros((self.N, self.f))

        mu1 = np.array([0,0])
        mu2 = np.array([m,m])
        mu3 = np.array([m,0])
        mu4 = np.array([0,m])

        X[:300,] = np.random.randn(300,self.f) + mu1
        X[300:600,] = np.random.randn(300,self.f) + mu2
        X[600:900,] = np.random.randn(300,self.f) + mu3
        X[900:1200,] = np.random.randn(300,self.f) + mu4

        plt.scatter(X[:,0],X[:,1])
        plt.title('before_clustering')
        plt.savefig('before_clustering.png')
        plt.show()
        self.X = X

    def ward_linkage(self):
        Z = linkage(self.X, 'ward')
        plt.title("Ward linkage")
        dendrogram(Z)
        plt.savefig('Ward linkage.png')
        plt.show()

    def single_linkage(self):
        Z = linkage(self.X, 'single')
        plt.title("Single linkage")
        dendrogram(Z)
        plt.savefig('Single linkage.png')
        plt.show()

    def complete_linkage(self):
        Z = linkage(self.X, 'complete')
        plt.title("Complete linkage")
        dendrogram(Z)
        plt.savefig('Complete linkage.png')
        plt.show()

    def run(self):
        self.ward_linkage()
        self.single_linkage()
        self.complete_linkage()

if __name__ == '__main__':
    hclustering = HierarchicalClustering()
    hclustering.create_data(4)
    hclustering.run()