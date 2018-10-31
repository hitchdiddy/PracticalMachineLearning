import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification

class knn:
    def __init__(self, k):
        self.k = k

    def fit(self,x,y):
        self.x = x
        self.y = y
        self.unique_classes = np.unique(self.y)

    def fit_partial(self,x,y):
        if not hasattr(self,'x'):
            self.x = x
            self.y = y
        else:
            self.x = np.vstack((self.x,x))
            self.y = np.hstack((self.y,y))
        self.unique_classes = np.unique(self.y)


    def _get_weight_from_distances(self,dists):
        total_dists = dists.sum()
        normalized_dists = dists / total_dists
        inverted_normalized_dists = 1-normalized_dists
        return inverted_normalized_dists


    def predict_proba(self,x):
        dists = cdist(self.x, x)
        min_dist_i = dists.argsort(axis=0)[:self.k].T
        y = []
        posteriors = np.zeros((x.shape[0], len(self.unique_classes)))

        for sample_i, inds in enumerate(min_dist_i):
            weights = self._get_weight_from_distances(dists[inds, sample_i])
            classes = self.y[inds]
            weighted_classes = [sum(weights[np.where(classes == c)[0]]) for c in self.unique_classes]
            posteriors[sample_i] = weighted_classes / sum(weighted_classes)
        return posteriors

    def predict(self,x):
        return np.array([self.unique_classes[i] for i in self.predict_proba(x).argmax(axis=1)])


if __name__ == '__main__':
    x, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=2)
    clf = knn(5)

    clf.fit(x,y)

    y_pred = clf.predict(x)

    wrong_i = np.where(y!=y_pred)[0]

    colors = ['green','blue','yellow']

    plt.scatter(x[:,0],x[:,1],c=[colors[yy] for yy in y])
    plt.scatter(x[wrong_i,0],x[wrong_i,1], c='red', alpha=0.3)

    plt.show()

    print(y)




