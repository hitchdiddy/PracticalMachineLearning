import numpy as np
from math import pi,exp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from math import exp
import matplotlib.pyplot as plt

class glvq():

    def __init__(self):
        self.max_prototypes_per_class = 5
        self.learning_rate = 1
        self.strech_factor = 20


    def fit(self,x,y):
        x = np.array(x)
        y = np.array(y)
        feat_dim = x.shape[1]
        if not hasattr(self,'x'):
            self.x = x
            self.y = y
            self.prototypes = np.zeros(shape=(0,feat_dim))
            self.labels = np.array([])
        else:
            self.x = np.vstack((self.x,x))
            self.y = np.hstack((self.y,y))

        for xi,yi in zip(x,y):
            num_prototypes_per_class = len(np.where(self.labels == yi)[0])
            if num_prototypes_per_class < self.max_prototypes_per_class: #add new
                self.prototypes = np.vstack((self.prototypes,xi))
                self.labels = np.hstack((self.labels,yi))
                print("adding prototype for class"+str(yi))
            elif len(set(self.labels)) > 1: #move prototype

                proto_dist = self.dist(np.array([xi]), self.prototypes)
                proto_dist = proto_dist[0]

                #find out nearest proto of same class and different class
                smallest_dist_wrong = float("inf")
                smallest_dist_right = float("inf")
                w1i = -1
                w2i = -1
                for i,p in enumerate(proto_dist):
                    if self.labels[i] == yi and smallest_dist_right > p:
                        smallest_dist_right = p
                        w1i = i
                    if self.labels[i] != yi and smallest_dist_wrong > p:
                        smallest_dist_wrong = p
                        w2i = i
                w1 = self.prototypes[w1i].copy()
                w2 = self.prototypes[w2i].copy()
                d1 = proto_dist[w1i]
                d2 = proto_dist[w2i]

                mu = (d1-d2)/(d1+d2)
                #sigm = (1/(1+exp(-mu)))
                derive = exp(mu*self.strech_factor)/((exp(mu*self.strech_factor)+1)*(exp(mu*self.strech_factor)+1))
                print('mu: '+str(mu)+' derive: '+str(derive))
                # GLVQ
                self.prototypes[w1i] = w1 + self.learning_rate * derive * (d2 / ((d1 + d2) * (d1 + d2))) * ( xi - w1)
                self.prototypes[w2i] = w2 - self.learning_rate * derive * (d1 / ((d1 + d2) * (d1 + d2))) * ( xi - w2)
                print('derive '+str(derive))
                print('move p1 from '+str(w1)+' to '+str(self.prototypes[w1i]))
                print('move p2 from '+str(w2)+' to '+str(self.prototypes[w2i]))

            else:
                print('cant move because only one labeled class')

    def dist(self,x,y):
        return cdist(x,y,'euclidean')

    def predict_proba(self,x):
        if len(set(self.y)) < 2:
            return [0]*len(x)
        ds = self.dist(x,self.prototypes)
        relsims = []
        for d in ds:
            winner,looser = self.get_win_loose_prototypes(d)
            relsims.append((d[looser]-d[winner])/(d[looser]+d[winner]))
        return np.array(relsims)

    def get_win_loose_prototypes(self,dists,n=2):
        ds = np.argsort(dists)
        # the classes already included into prototype list
        labels_included = []
        prototypes_i = []

        for id,d in enumerate(ds):
            if not self.labels[d] in labels_included:
                labels_included.append(self.labels[d])
                prototypes_i.append(d)
                if len(prototypes_i) >= n:
                    break
        return prototypes_i

    def predict(self,x):
            return np.array(self.labels[np.argmin(self.dist(x,self.prototypes), axis=1)],np.int)

    def score(self,x,y):
        y_pred = self.predict(x)
        return float(len(np.where(y==y_pred)[0]))/len(x)

    def visualize_2d(self,ax=None):
        if not hasattr(self,'pltCount'):
            self.pltCount = 0
        if ax is None:
            ax = plt.gca()
        plt.cla()
        plt.ion()
        some_colors = ['red','green','blue','yellow','orange','pink','black','brown']
        pred = self.predict(self.x)
        for x,y in zip(self.x,pred):
            plt.scatter(x[0],x[1],c='grey')#some_colors[int(y)]
        for p,l in zip(self.prototypes,self.labels):
            plt.scatter(p[0],p[1],c=some_colors[int(l)],marker='D',s=80,edgecolors='black')
        plt.pause(0.001)
        plt.savefig('./plt/plt'+str(self.pltCount)+'.png',format='png')
        self.pltCount +=1

        plt.ioff()



if __name__ == '__main__':
    x,y = make_classification(n_samples=5000, n_features=2, n_informative=1, n_redundant=0, n_repeated=0,
                                         n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=0.5,
                                         hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.33, random_state = 45)
    b = glvq()

    ax = plt.figure().gca()

    for x,y in zip(x_train,y_train):
        b.fit(x[np.newaxis],y[np.newaxis])

    b.visualize_2d(ax)
    plt.show()

    print(b.predict(x_test))
    print(b.predict_proba(x_test))
    print(y_test)
    print('score: '+str(b.score(x_test,y_test)))