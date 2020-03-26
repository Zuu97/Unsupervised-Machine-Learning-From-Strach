import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle

class HardKMeans(object):
    def __init__(self,K):
        self.f = 2    # no: of features
        self.N = 1200 # dataset size
        self.K = K    # cluster centers
        self.minimal_cost_defference = 10e-10

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

    @staticmethod
    def distance(v1, v2):
        return np.linalg.norm(v1-v2)

    def initialize_clusters(self):
        self.X = shuffle(self.X)
        Mu = np.zeros((self.K, self.f))
        for k in range(self.K):
            idx = np.random.choice(self.N)
            Mu[k] = self.X[idx]
        self.Mu = Mu

    def assingn_point(self,data_point):
        distances = np.array([HardKMeans.distance(mui,data_point) for mui in self.Mu]).squeeze()

        return np.argmin(distances),np.min(distances)

    def cluster_assiging_step(self,Total_costs):
        self.X = shuffle(self.X)
        cluster2point = {}
        cost = 0
        for data_point in self.X:
            cluster, distance2cluster = self.assingn_point(data_point)
            cost += distance2cluster
            if cluster in cluster2point:
                cluster2point[cluster].append(data_point)
            else:
                cluster2point[cluster] = [data_point]
        cost = cost/self.N
        Total_costs.append(cost)
        for k,v in cluster2point.items():
            self.Mu[k] = np.array(v).mean(axis=0)
        return cluster2point

    def train(self):
        self.initialize_clusters()
        Total_costs = []
        while True:
            cluster2point = self.cluster_assiging_step(Total_costs)
            if len(Total_costs) > 2 and Total_costs[-2] - Total_costs[-1] < self.minimal_cost_defference:
                self.plot_data(cluster2point,Total_costs)
                break

    def plot_data(self, cluster2point, Total_costs):
        cmap = cm.get_cmap("rainbow")
        colors = cmap(np.linspace(0, 1, self.K))
        for i,v in enumerate(cluster2point.values()):
            x = np.array(v)
            plt.scatter(x[:,0], x[:,1], c=colors[i])
        plt.title('after_clustering for K='+str(self.K))
        plt.savefig(str(self.K)+'.png')
        plt.show()


if __name__ == "__main__":
    for k in range(2,7):
        kmeans = HardKMeans(k)
        kmeans.create_data(4)
        kmeans.train()