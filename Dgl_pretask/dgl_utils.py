import dgl
import torch
import numpy as np
from torch import nn


def get_dgl_graph(dataset_name):
    if dataset_name == 'cora':
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


def get_data_one2many(g, label):
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    full_labels = g.ndata['label']
    train_labels = full_labels[train_mask]
    test_labels = full_labels[test_mask]
    # full_labels = (test_labels.cpu().detach().numpy() > 0).astype(int)
    # 0 - 1, 0 - 1,2,...9

    test_labels_final = (test_labels.cpu().detach().numpy() != label).astype(int)
    train_labels_final = (train_labels.cpu().detach().numpy() == label).astype(int)

    labels_np = full_labels.cpu().detach().numpy()
    train_mask_final = [(labels_np[i] == label) and m.item() for i, m in enumerate(train_mask)]

    #full_train_labels_mask = full_labels.cpu().detach().numpy() and train_mask
    #train_mask_final = np.array([l != label for l in full_train_labels_mask])

    return torch.tensor(train_mask_final), train_labels_final, test_mask, test_labels_final


class GraphData:
    def __init__(self, g, dataset, args, method='one2many'):
        self._dataset = dataset
        self._graph = g
        # can choose the method of processing data
        self.train_mask, self.train_labels, self.test_mask, self.test_labels \
            = get_data_one2many(self._graph, args.label)

    @property
    def get_train_mask(self):
        return self.train_mask

    @property
    def get_train_labels(self):
        return self.train_labels

    @property
    def get_test_mask(self):
        return self.test_mask

    @property
    def get_test_labels(self):
        return self.test_labels

    @property
    def graph(self):  # todo - how to generalize?
        return self._graph

    @property
    def num_classes(self):
        return self._dataset.num_classes


class PretrainedModel:
    def __init__(self, model: nn.Module, data: GraphData, pre_task):
        self.dgl_model = model
        self._data = data
        self.graph = data.graph
        self.pre_task = pre_task

    def predict_logits(self):
        return self.dgl_model(self.graph)

    def loss(self, criterion, features, mode):
        if mode == 'train':
            #mask = self.graph.ndata['train_mask']
            mask = self._data.train_mask
        else:
            #mask = self.graph.ndata['test_mask']
            mask = self._data.test_mask
        return criterion(features[mask])

    def feature_space(self, mode):
        features = self.dgl_model(self.graph)
        if mode == 'train':
            #train_mask = self.graph.ndata['train_mask']
            return features[self._data.train_mask]
        elif mode == 'test':
            #test_mask = self.graph.ndata['test_mask']
            return features[self._data.test_mask]

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self.dgl_model

    def parameters(self):
        return self.dgl_model.parameters()

    def train(self):
        self.dgl_model.train()

    def eval(self):
        self.dgl_model.eval()

    def to(self, device):
        self.dgl_model.to(device)
