import torch
import torch.nn.functional as F

import models
import panda_utils
import numpy as np


def train(g, model, args):  # epocs=100, lr=0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0
    best_test_acc = 0

    # features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # train set labels: takes onle samples with label 0
    # train set data: takes only the TARGET labels. True iff i == label.
    # test set labels: takes 1 if not label, 0 o.w.
    # test set data: FULL
    label = 0
    train_labels = g.ndata['label'][train_mask]
    test_labels = g.ndata['label'][test_mask]

    masked_test_labels = (test_labels.cpu().detach().numpy() != label).astype(int)
    masked_train_set = train_labels.cpu().detach().numpy() == label  # if label = 0 True
    masked_train_labels = masked_train_set.astype(int)

    for e in range(args.epocs):
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

        if e % 5 == 0:
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
    g, dataset = panda_utils.get_graph('cora')
    print("features dim: ", g.ndata['feat'].shape[1])

    model = get_model_architecture(args.model, g, dataset)
    train(g, model, args)

    print("hello gur")
    print("\nfinished training GCN\n")
    return model, g, dataset


def get_model(args):
    if args.task == 'node_class':
        return get_node_class_model(args)
    else:
        print('Default model is node classification with gcn')
        return get_node_class_model(args)


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
    train_mask_final = [(labels_np[i] == label) and m.item() for i,m in enumerate(train_mask)]


    #full_train_labels_mask = full_labels.cpu().detach().numpy() and train_mask
    #train_mask_final = np.array([l != label for l in full_train_labels_mask])

    return torch.tensor(train_mask_final), train_labels_final, test_mask, test_labels_final

class GraphData():

    def __init__(self, g, args, method='one2many'):
        self.graph = g
        # can choose the method of processing data
        self.train_mask, self.train_labels, self.test_mask, self.test_labels \
            = get_data_one2many(self.graph, args.label)
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


##run separatly
class Arguments:
    dataset = 'cora'
    epocs = 5
    lr = 0.001
    model = 'gcn'
    task = 'node_class'
    layers = 1
    h_dim1 = 16
    label = 0


args = Arguments()
model, g, dataset = get_model(args)
data = GraphData(g, args)
print(1)

