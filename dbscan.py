import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class dbscan:

    def __init__(self,eps,min_pts):
        self.eps = eps
        self.min_pts = min_pts

    def fit_rec(self,x):
        self.x = x
        # calc dist matrix
        self.dists = cdist(x,x,'euclidean')

        self.visited = np.zeros(len(x),dtype='bool')
        self.clusters = np.zeros(len(x))
        self.cluster_no = 1
        for i in range(len(self.x)):
            if not self.visited[i]:
                self.visited[i] = True
                n = self.region_query(i)
                if len(n) < self.min_pts:
                    self.clusters[i] = -1  # mark as noise
                else:
                    self.expand_rec(i, n)
                    self.cluster_no += 1


    def expand_rec(self,i,n):
        self.clusters[i] = self.cluster_no
        for p in n:
            if not self.visited[p]:
                self.visited[p] = True
                n_new = self.region_query(p)
                if len(n_new) < self.min_pts:
                    self.clusters[i] = -1  # mark as noise
                else:
                    self.expand_rec(p, n_new)

    def fit(self,x):
        self.x = x
        # calc dist matrix
        self.dists = cdist(x,x,'euclidean')

        self.visited = np.zeros(len(x),dtype='bool')
        self.clusters = np.zeros(len(x))
        self.cluster_no = 1
        for i in range(len(self.x)):
            if not self.visited[i]:
                self.visited[i] = True
                n = self.region_query(i)
                if len(n) < self.min_pts:
                    self.clusters[i] = -1  # mark as noise
                else:
                    self.expand(i, n)
                    self.cluster_no += 1

    def expand(self,i,n):
        self.clusters[i] = self.cluster_no
        while len(n) > 0:
            p = n[0]
            n = np.delete(n,0)
            if not self.visited[p]:
                self.visited[p] = True
                n_new = self.region_query(p)
                if len(n_new) >= self.min_pts:
                    n = np.hstack((n,n_new))
            if self.clusters[p] == 0:
                self.clusters[p] = self.cluster_no


    def region_query(self,i):
        points = self.dists[i] < self.eps
        points[i] = False
        return np.where(points)[0]
    @staticmethod
    def region_query_static(i,dists,eps):
        points = dists[i] < eps
        points[i] = False
        return np.where(points)[0]


    @staticmethod
    def fit_single(x,i,eps,min_pts,dists=None ):
        # calc dist matrix
        dists = dists if dists else cdist(x,x,'euclidean')

        visited = np.zeros(len(x),dtype='bool')
        clusters = np.zeros(len(x))

        visited[i] = True
        n = dbscan.region_query_static(i,dists,eps)
        if len(n) < min_pts:
            clusters[i] = -1  # mark as noise
        else:
            clusters[i] = 1
            while len(n) > 0:
                p = n[0]
                n = np.delete(n, 0)
                if not visited[p]:
                    visited[p] = True
                    n_new = dbscan.region_query_static(p,dists,eps)
                    if len(n_new) >= min_pts:
                        n = np.hstack((n, n_new))
                if clusters[p] == 0:
                    clusters[p] = 1
        return clusters==1


if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=90, centers=2, n_features=2, random_state = 0, cluster_std=.3)
    

    db = dbscan(1.5,3)
    clusters = dbscan.fit_single(X,0,1.5,3)



    plt.scatter(X[:,0],X[:,1],c=clusters,label=clusters)
    plt.ioff()
    plt.legend()
    plt.show()
