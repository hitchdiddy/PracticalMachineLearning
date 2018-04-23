class dbscan:
    def __init__(self,eps,min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
    






if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state = 0)
    print(X.shape)