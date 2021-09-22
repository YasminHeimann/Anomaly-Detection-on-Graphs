import numpy as np
from sklearn.neighbors import NearestNeighbors

import Dgl_pretask.train_dgl as dgl_task
import SSL_pretask.src.train_ssl as ssl_task


# https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    knn = NearestNeighbors(n_neighbours)
    knn.fit(train_set)
    D, I = knn.kneighbors(test_set)
    return np.sum(D, axis=1)


def clip_gradient(optimizer, grad_clip):
    assert grad_clip > 0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)


def get_pretrained_model(pre_task):
    if pre_task == 'dgl':
        return dgl_task.get_pre_trained_model()
    if pre_task == 'ssl':
        return ssl_task.get_pre_trained_model()

