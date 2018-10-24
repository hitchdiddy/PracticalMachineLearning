import matplotlib.pyplot as plt
import numpy as np
from common.data_handler.artificial_regression import linear_2d


class linear_regression:
    def __init__(self):
        pass

    def fit(self,x,y):
        x = np.array(x)
        y = np.array(y)

        mux = np.mean(x)
        muy = np.mean(y)

        self.m = 0
        for xx,yy in zip(x,y):
            self.m += (yy - muy)/(xx - mux)

        self.m /= len(x)
        self.b = muy-self.m*mux

    def predict(self,x):
        return np.array([self.m * xx + self.b for xx in x])

if __name__ == '__main__':
    data = linear_2d()
    x = data[:,0]
    y = data[:,1]

    lr = linear_regression()
    lr.fit(x,y)

    predicted_x = list(np.arange(0,5,0.01))
    predicted_y = lr.predict(predicted_x)


    plt.scatter(x,y,c='red')
    plt.scatter(predicted_x,predicted_y,c='blue')
    plt.show()
