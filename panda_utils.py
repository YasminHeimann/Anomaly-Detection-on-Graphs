import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

import Dgl_pretask.train_dgl as dgl_task
import SSL_pretask.src.train_ssl as ssl_task


# https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
def knn_score(train_set, test_set, n_neigh=2):
    """
    Calculates the KNN distance
    """
    knn = NearestNeighbors(n_neighbors=n_neigh)
    knn.fit(train_set)
    D, I = knn.kneighbors(test_set)
    return np.sum(D, axis=1)


def get_score(data, model, device, num_classes=2, multi_class=False):
    # data is GraphData obj
    with torch.no_grad():  # todo check if it applies still on another block
        train_feature_space = model.feature_space('train')
        test_feature_space = model.feature_space('test')

    train_np = train_feature_space.cpu().detach().numpy()
    test_np = test_feature_space.cpu().detach().numpy()
    distances = knn_score(train_np, test_np, num_classes)

    #auc
    # multi class
    # if multi_class:
    #     one_hot = np.zeros((labels.size, data.test_labels.max()+1))
    #     one_hot[np.arange(labels.size),labels] = 1
    #     auc = roc_auc_score(one_hot,distances.reshape(-1, 1), #average='weighted',
    #                         multi_class='ovr')
    # else:
    #     # todo: is it how to calc auc?
    #     # binary classifier
    #     # אם הנורמלי זה אפס אז המרחקים נמוכים והפוך
    #     # סיווג לחריג 0 מרחק גבוה
    #     # אם 0 נמוך ו1 גבוה
    auc = roc_auc_score(data.test_labels, distances)

    return auc, train_np


def clip_gradient(optimizer, grad_clip):
    assert grad_clip > 0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)


def get_pretrained_model(args, base_path, save_model=True):
    ssl_models = ["PairwiseDistance_citeseer"]  # ["PairwiseDistance_cora", "Base_cora"]
    end = ".pt"
    if args.pre_task == 'dgl':
        return dgl_task.get_pre_trained_model(args.label, model_path="", save_path="", to_save=False)
    if args.pre_task == 'ssl':
        if save_model:
            return ssl_task.get_pre_trained_model(args.label, model_path="", save_path=base_path, to_save=True)
        else:
            for ssl in ssl_models:
                saved = base_path + ssl + end
                # load existing model
                return ssl_task.get_pre_trained_model(args.label, model_path=saved, save_path="", to_save=False)

