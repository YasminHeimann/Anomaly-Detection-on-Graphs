import argparse
import torch
import torch.nn.functional as F

import Dgl_pretask.models as models
import Dgl_pretask.dgl_utils as dgl_utils


def train(g, model, args):  # epocs=100, lr=0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0
    best_test_acc = 0

    # features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # train set labels: takes only samples with label 0
    # train set data: takes only the TARGET labels. True iff i == label.
    # test set labels: takes 1 if not label, 0 o.w.
    # test set data: FULL
    label = 0
    train_labels = g.ndata['label'][train_mask]
    test_labels = g.ndata['label'][test_mask]

    masked_test_labels = (test_labels.cpu().detach().numpy() != label).astype(int)
    masked_train_set = train_labels.cpu().detach().numpy() == label  # if label = 0 True
    masked_train_labels = masked_train_set.astype(int)

    for e in range(args.epochs):
        # Forward
        logits = model(g)

        # Compute prediction
        pred = logits.argmax(1)
        # [-1, 5, 6, 2] -> index: 2

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 1 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))


def get_model_architecture(model, g, dataset):
    if model == 'gcn':
        return models.GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)  # .to('cuda')
    else:
        print('Default model architecture is gcn')
        return models.GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)  # .to('cuda')


def get_node_class_model(args):
    # Create the model with given dimensions
    g, dataset = dgl_utils.get_dgl_graph(args.dataset)
    print("features dim: ", g.ndata['feat'].shape[1])

    model = get_model_architecture(args.model, g, dataset)
    train(g, model, args)

    print("\nfinished training GCN\n")
    return model, g, dataset


def get_model(args):
    if args.task == 'node_class':
        return get_node_class_model(args)
    else:
        print('Default model is node classification with gcn')
        return get_node_class_model(args)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cora')

    # parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    # parser.add_argument('--ewc', default=False, action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=100, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')

    parser.add_argument('--model', default='gcn', type=str, help='which architecture to use')
    parser.add_argument('--task', default='node_class', type=str, help='which architecture to use')
    parser.add_argument('--layers', default=1, type=int, help='which architecture to use')
    return parser.parse_args()


def get_pre_trained_model():
    args = parse_args()
    print('Dataset: {}, Model Architecture: {}, LR: {}'.format(args.dataset, args.model, args.lr))
    model, g, dataset = get_model(args)
    data = dgl_utils.GraphData(g, dataset, args)
    return dgl_utils.PretrainedModel(model, data, args.task)

