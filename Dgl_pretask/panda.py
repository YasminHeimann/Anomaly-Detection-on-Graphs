from gnn_pre_task import get_model

import torch
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import panda_utils
from copy import deepcopy

from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import numpy as np


def sklearn_knn(train_np, test_np, n_neighbors=2):
    knn = NearestNeighbors(n_neighbors)
    knn.fit(train_np)
    D, I = knn.kneighbors(test_np)
    return np.sum(D, axis=1)


def get_score(data, model, device, g, num_classes=2, multi_class=False):
    # data is GraphData obj
    with torch.no_grad():
        features = model(g) # todo
        train_feature_space = features[data.train_mask]
        test_feature_space = features[data.test_mask]

    train_np = train_feature_space.cpu().detach().numpy()
    test_np = test_feature_space.cpu().detach().numpy()
    distances = sklearn_knn(train_np, test_np, num_classes)

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


def run_epoch(model, graph, train_mask, optimizer, criterion, device, ewc, ewc_loss):
    # todo: images = imgs.to(device)
    optimizer.zero_grad()

    logits = model(graph)  # todo

    loss = criterion(logits[train_mask])  # todo

    if ewc:
        loss += ewc_loss(model)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

    optimizer.step()

    return loss.item()


def train_panda_model(data, model, graph, device, args, ewc_loss, num_classes):
    model.eval()  # todo
    auc, train_feature_space = get_score(data, model, device, graph, num_classes)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)

    #todo how many labels? two?
    center = torch.FloatTensor(train_feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    #train_mask = graph.ndata['train_mask']
    # todo: is the train mask on two labels? one label? in article?
    # todo: i assume only 2 labels in general.
    run_loss, auroc = [], []
    for epoch in range(args.epochs):
        model.train()
        # todo
        running_loss = run_epoch(model, graph, data.train_mask, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        run_loss.append(running_loss)
        model.eval()
        # todo
        auc, feature_space = get_score(data, model, device, graph, num_classes)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        auroc.append(auc)
    return run_loss, auroc


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = get_model(args)  # todo change to utils to choose the correct model later
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC todo
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        panda_utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    # panda_utils.freeze_parameters(model)  # todo - what does it mean here?
    # todo
    g, dataset = panda_utils.get_dgl_graph(dataset_name=args.dataset)
    train_model(model, g, device, args, ewc_loss, dataset.num_classes)


class PandaArguments:

    epocs = 50
    lr = 1e-2
    ewc = False
    pre_task = 'dgl'  # or ssl

    dataset = 'cora'
    model = 'gcn'
    task = 'node_class'
    layers = 1
    h_dim1 = 16
    label = 0


if __name__ == "__main__":
    panda_args = PandaArguments()
    # args = parse_args()

    main(panda_args)

