from common.load_data import load_outdoor_new

from common.classification import get_numerical_classes

DB_DIR = '/hri/storage/user/climberg/datasets/outdoor/dump'


from sklearn.utils import shuffle


features, labels, approaches, approach_imgnos, recognizable, images = load_outdoor_new(DB_DIR, load_images=False)

from sklearn.decomposition import PCA

pca = PCA(n_components=1000, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto',
          random_state=None)

features = pca.fit_transform(features)

features, labels, approaches, approach_imgnos, recognizable = shuffle(features, labels, approaches, approach_imgnos,
                                                                      recognizable)

labels_num = get_numerical_classes(labels)

from common.classification import separate_classes_by_subclasses

train_i, test_i = separate_classes_by_subclasses(labels, approaches, 0.3)

from glvq import glvq


scores = []
lrs = list(range(1,62,10))
for lr in lrs:
    score = []
    for repeat in range(5):
        cls = glvq(max_prototypes_per_class=None,learning_rate=lr,strech_factor=1,placement_strategy=None)
        cls.placement_strategy = cls.placement_certainty_adaptive
        cls.fit(features[train_i],labels_num[train_i])
        s = cls.score(features[test_i],labels_num[test_i])
        print('lr',lr,'stretch',1,)
        score.append(s)
    scores.append(score)

import matplotlib.pyplot as plt
import numpy as np
plt.plot(lrs,np.array(scores).mean(axis=1))
plt.show()

