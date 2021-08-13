import torch
import numpy as np
import faiss
import dgl.data

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


# TODO what to change?
def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

# https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_graph(dataset_name):
    dataset, g = None, None
    if dataset_name =='cora':
        dataset = dgl.data.CoraGraphDataset()
        # cora db has one graph
        g = dataset[0]
    elif dataset_name == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        # citeseer db has one graph todo ?
        g = dataset[0]
    else:
        print('Default Dataset is cora')
        dataset = dgl.data.CoraGraphDataset()
        # cora db has one graph
        g = dataset[0]
    print('\nNumber of categories:', dataset.num_classes)
    print('Node features')
    print(g.ndata)
    print('Edge features')
    print(g.edata)
    return g, dataset


def clip_gradient(optimizer, grad_clip):
    assert grad_clip > 0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)
