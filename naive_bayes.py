#conditional probability
#P(x & y) = P(x) * P(y | x)

#bayes rule
#P(Disease | Test-positive) = P(Test-positive|Disease) * P(Disease)
#                              _____________________________________
#                        P(Testing Positive(with or without the disease))

import numpy as np
from math import pi,exp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from common.classification import get_numerical_label_encoders

class NaiveBayes:
    def __init__(self):
        pass
    def p(self,mu,sigma,point):
        return sum([self.norm(m,s,p) for m,s,p in zip(mu,sigma,point)])
    def norm(self,mu,sigma,point):
        denom = (2 * pi * sigma) ** .5
        num = exp(-(point - mu) ** 2 / (2 * sigma))
        return num / denom
    #fit is just for calculating each mu and sigma
    def fit(self,x,y_names):
        y, self.label_encoder_forward, self.label_encoder_backward = get_numerical_label_encoders(y_names)
        self.num_train_data = len(y)
        y_unique = set(y)
        self.num_classes = len(y_unique)
        self.mu = np.zeros((len(y_unique),x.shape[1]))
        self.sigma = np.zeros((len(y_unique),x.shape[1]))
        self.py = np.zeros((len(y_unique)))



        for label in y_unique:
            label_feats = x[y == label]
            self.mu[label] = np.mean(label_feats,axis=0)
            n = len(label_feats)
            self.py[label] = n
            v = ((label_feats - self.mu[label]) ** 2).mean(axis=0)
            if len(label_feats) == 1:
                v = 0.1
            self.sigma[label] = v
            #1d
            #self.sigma[label] = np.mean(np.linalg.norm([(self.mu[label] - iterx) for iterx in label_feats],axis=1)**2)


    def fit_partial(self,x,y):
        if not hasattr(self,'x_partial'):
            self.x_partial = x
            self.y_partial = y
        else:
            self.x_partial = np.vstack((self.x_partial,x))
            self.y_partial = np.hstack((self.y_partial,y))
        self.fit(self.x_partial,self.y_partial)




    def predict_proba(self,x):
        if self.num_classes == 1:
            raise Exception('only 1 class')

        y_pred = np.zeros((len(x),self.num_classes))
        for i,xiter in enumerate(x):
            normalize = 0
            for mu, sigma, py in zip(self.mu, self.sigma, self.py):
                normalize += self.p(mu,sigma,xiter)*(py/self.num_train_data)
            probs = np.zeros(self.num_classes)
            for i2,(mu,sigma,py) in enumerate(zip(self.mu,self.sigma,self.py)):
                likelihood = self.p(mu,sigma,xiter)
                prior = py/self.num_train_data
                probs[i2] = (likelihood*prior)/normalize
            y_pred[i] = probs
        return y_pred

    def predict(self,x):
        probas = self.predict_proba(x)
        num_classes = np.argmax(probas,axis=1)
        return np.array([self.label_encoder_backward[i] for i in num_classes])

    def score(self,x,y_names):
        y = np.array([self.label_encoder_forward[i] for i in y_names])
        y_pred = self.predict(x)
        return float(len(np.where(y==y_pred)[0]))/len(x)


if __name__ == '__main__':
    x,y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_repeated=0,
                                         n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0,
                                         hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.33, random_state = 42)
    nb = NaiveBayes()
    nb.fit(x_train,y_train)


    print(nb.predict(x_test))
    print(y_test)
    print(nb.score(x_test,y_test))

    from math import sqrt
    ax = plt.gca()
    plt.scatter(x_train[:,0],x_train[:,1])
    for m,v in zip(nb.mu,nb.sigma):
        el = Ellipse(xy=m,width=2*sqrt(v[0]),height=2*sqrt(v[1]))
        ax.add_artist(el)
        el.set_alpha(0.3)

    print('score', nb.score(x,y))

    plt.show()