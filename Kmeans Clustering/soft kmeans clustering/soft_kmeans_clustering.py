import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle

class HardKMeans(object):
    def __init__(self,K,beta):
        self.f = 2    # no: of features
        self.N = 1200 # dataset size
        self.K = K    # cluster centers
        self.beta = beta
        self.minimal_cost_defference = 10e-5

    def create_data(self, m):
        X = np.zeros((self.N, self.f))

        mu1 = np.array([0,0])
        mu2 = np.array([m,m])
        mu3 = np.array([0,m])

        X[:300,] = np.random.randn(300,self.f) + mu1
        X[300:600,] = np.random.randn(300,self.f) + mu2
        X[600:900,] = np.random.randn(300,self.f) + mu3

        plt.scatter(X[:,0],X[:,1])
        plt.title('before_clustering')
        plt.savefig('before_clustering.png')
        plt.show()
        self.X = X

    @staticmethod
    def distance(v1, v2):
        diff = v1 - v2
        return np.dot(diff,diff)

    def initialize_clusters(self):
        self.X = shuffle(self.X)
        Mu = np.zeros((self.K, self.f))
        for k in range(self.K):
            idx = np.random.choice(self.N)
            Mu[k] = self.X[idx]
        self.Mu = Mu

    def calculate_responsibilities(self):
        self.R = np.empty((self.N, self.K))
        for i,x in enumerate(self.X):
            for j,mu_k in enumerate(self.Mu):
                self.R[i,j] = np.exp(-self.beta * HardKMeans.distance(mu_k,x))
        self.R = self.R/ np.sum(self.R, axis=1, keepdims=True)

    def calculate_mean(self):
        Means = np.dot(self.R.T , self.X)
        self.Mu = Means / Means.sum(axis=0, keepdims=True)

    def loss_fnction(self):
        loss = 0
        for k,mu in enumerate(self.Mu):
            dist = ((self.X - mu) * (self.X - mu)).sum(axis=1)
            loss += (self.R[:,k] * dist).sum()
        return loss

    def plot_data(self,Total_loss):
        plt.plot(Total_loss)
        plt.title("Costs")
        plt.show()

        random_colors = np.random.random((self.K, 3))
        colors = self.R.dot(random_colors)
        plt.scatter(self.X[:,0], self.X[:,1], c=colors)
        plt.title('K = '+str(self.K))
        plt.savefig(str(self.K)+'.png')
        plt.show()

    def train(self,m):
        Total_loss = []
        self.create_data(m)
        self.initialize_clusters()
        a = 0
        while True:
            print(a)
            self.calculate_responsibilities()
            self.calculate_mean()
            loss = self.loss_fnction()
            Total_loss.append(loss)
            if len(Total_loss) > 2:
               if Total_loss[-2] - Total_loss[-1] <  self.minimal_cost_defference:
                self.plot_data(Total_loss)
                break
            a += 1
if __name__ == "__main__":
    cluster = HardKMeans(4,20.0)
    cluster.train(8)