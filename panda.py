from gnn_pre_task import get_model

import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import panda_utils
from copy import deepcopy


def train_model(model, graph, device, args, ewc_loss, num_classes):
    model.eval()  # todo
    auc, feature_space = get_score(model, device, graph, num_classes)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)

    #todo how many labels? two?
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    train_mask = graph.ndata['train_mask']

    for epoch in range(args.epochs):
        model.train()
        running_loss = run_epoch(model, graph, train_mask, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        model.eval()
        auc, feature_space = get_score(model, device, graph, num_classes)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))


def run_epoch(model, graph, train_mask, optimizer, criterion, device, ewc, ewc_loss):
    # todo: images = imgs.to(device)
    optimizer.zero_grad()

    features = model(graph)

    loss = criterion(features[train_mask])

    if ewc:
        loss += ewc_loss(model)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

    optimizer.step()

    return loss.item()


def get_score(model, device, graph, num_classes):
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    with torch.no_grad():
        f = graph.ndata['feat']
        features = model(graph, f)
        train_feature_space = features[train_mask]
        test_feature_space = features[test_mask]
        # todo train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_labels = graph.ndata['label'][test_mask]

    distances = panda_utils.knn_score(train_feature_space,
                                      test_feature_space,
                                      num_classes
                                      )

    # todo - how to calculate the ROC for unknown anomalies?

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space


def main(args):
    print('Dataset: {}, Model Architecture: {}, LR: {}'.format(args.dataset, args.net, args.lr))
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

    g, dataset = panda_utils.get_graph(dataset_name=args.dataset)
    train_model(model, g, device, args, ewc_loss, dataset.num_classes)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', default=False, action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    #parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--model', default='gcn', type=str, help='which architecture to use')
    parser.add_argument('--task', default='node_class', type=str, help='which architecture to use')
    parser.add_argument('--layers', default=1, type=int, help='which architecture to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)

