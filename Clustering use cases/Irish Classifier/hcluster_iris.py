import numpy as np
import matplotlib.pyplot as plt
from util import get_data, dim_reduction
from scipy.cluster.hierarchy import dendrogram, linkage

class HclusterIris(object):
    def __init__(self):
        Xtrain, Xtest, Ytrain, Ytest = get_data()
        self.pca = dim_reduction(Xtrain)
        self.Xtrain = Xtrain
        self.Xtest = Xtest

    def pca_data(self):
        Xtrain_pca = self.pca.transform(self.Xtrain)
        Xtest_pca  = self.pca.transform(self.Xtest)

        plt.scatter(Xtrain_pca[:,0],Xtrain_pca[:,1], color='red')
        plt.title('Train data after applying PCA')
        plt.savefig('train_pca.png')
        plt.show()

        plt.scatter(Xtest_pca[:,0],Xtest_pca[:,1], color='green')
        plt.title('Test data after applying PCA')
        plt.savefig('test_pca.png')
        plt.show()

        self.Xtrain_pca = Xtrain_pca
        self.Xtest_pca = Xtest_pca

    def ward_linkage(self):
        Z = linkage(self.Xtrain_pca, 'ward')
        plt.title("Train Ward linkage")
        dendrogram(Z)
        plt.savefig('Train Ward linkage.png')
        plt.show()

        Z = linkage(self.Xtest_pca, 'ward')
        plt.title("Test Ward linkage")
        dendrogram(Z)
        plt.savefig('Test Ward linkage.png')
        plt.show()

    def single_linkage(self):
        Z = linkage(self.Xtrain_pca, 'single')
        plt.title("Train Single linkage")
        dendrogram(Z)
        plt.savefig('Train Single linkage.png')
        plt.show()

        Z = linkage(self.Xtest_pca, 'single')
        plt.title("Test Single linkage")
        dendrogram(Z)
        plt.savefig('Test Single linkage.png')
        plt.show()

    def complete_linkage(self):
        Z = linkage(self.Xtrain_pca, 'complete')
        plt.title("Train Complete linkage")
        dendrogram(Z)
        plt.savefig('Train Complete linkage.png')
        plt.show()

        Z = linkage(self.Xtest_pca, 'complete')
        plt.title("Test Complete linkage")
        dendrogram(Z)
        plt.savefig('Test Complete linkage.png')
        plt.show()

    def run(self):
        self.ward_linkage()
        self.single_linkage()
        self.complete_linkage()

if __name__ == "__main__":
    cluster = HclusterIris()
    cluster.pca_data()
    cluster.run()