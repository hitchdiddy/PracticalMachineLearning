import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification

class kmeans:
    def __init__(self):
        pass

    def fit(self,x,n,no_iter):
        # place w to random samples
        if not hasattr(self,'w'):
            self.w = x[np.random.choice(range(len(x)),n,False)]

        for i in range(no_iter):
            #assignment
            d = cdist(x,self.w)
            self.winner = np.argmin(d, axis=1)
            #update
            for k in range(len(self.w)):
                self.w[k] = np.mean(x[self.winner == k], axis=0)
        return self.w



if __name__ == '__main__':
    x, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=2)
    cluster = kmeans()

    plt.ion()

    for e in range(20):
        w = cluster.fit(x,2,1)
        plt.scatter(x[:,0],x[:,1])
        plt.scatter(w[:,0],w[:,1],marker='x')
        plt.pause(1)
        plt.show()
