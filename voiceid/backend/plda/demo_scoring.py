import os

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from scoring import *

T = lambda X: torch.tensor(X).float()

# generate 2D data
seed = 0
np.random.seed(seed)

n_classes = 3

n_samples = 200
X_all, y_all = make_blobs(n_samples=n_samples, centers=n_classes, cluster_std=1, random_state=seed)

X_enroll, X, y_enroll, y = train_test_split(X_all, y_all, test_size=0.95, random_state=seed)


centroid_list = []
# known classes (enrolled)
for c in np.unique(y_enroll):
    centroid_list += [np.mean(X_enroll[y_enroll==c], axis=0, keepdims=True)]
# unknown class (background)
centroid_list += [np.zeros((1, 2))]


c_e = T(np.concatenate(centroid_list))
c_t = T(X)

counts_list = [np.sum(y_enroll==c) for c in np.unique(y_enroll)] + [0] # known classes + background class
n_e = torch.tensor(counts_list)
n_t = torch.ones(X.shape[0],)



# run model
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

params = [(100, 0.01), (100, 0.1), (100, 1)]

for i, ax in enumerate(axes):

    b, w = params[i]

    # == log p(x | X) - log p(x) == log p(x, X) - log p(x) - log p(X)
    scores = plda_score_scalar_many2many((c_t, n_t), (c_e, n_e), T(b), T(w)).numpy()
    # == log p(x | X)
    scores2 = log_predictive(T(X), c_e, n_e, T(b), T(w)).numpy() # == scores + const(x)

    y_pred = np.argmax(scores, axis=1)
    y_pred[y_pred==n_classes] = -1

    ax.scatter(X[:, 0], X[:, 1], c=y);
    ax.scatter(X[y!=y_pred, 0], X[y!=y_pred, 1], marker='s', color='k');
    ax.scatter(X_enroll[:, 0], X_enroll[:, 1], marker='v', c=y_enroll, edgecolor='r');
    plt.title(f'Predictions');
    
    print('---')
    
plt.subplots_adjust(wspace=0.1, hspace=0);    
plt.show()

