from __future__ import division
from __future__ import print_function

import time
import argparse
import json

import dgl
import torch.optim as optim
from tensorboardX import SummaryWriter

import ssl_utils
from metric import accuracy, roc_auc_compute_fn

from earlystopping import EarlyStopping
from sample import Sampler
import shutil
import os.path as osp
from models import *
from selfsl import *


def get_arguments():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1, help='alpha for label correction')
    parser.add_argument('--write_res', type=int, default=0, help='if write results')
    parser.add_argument('--param_searching', type=int, default=0, help='if plot')
    parser.add_argument('--train_size', type=int, default=0, help='if plot')
    parser.add_argument('--pca', type=int, default=0, help='if plot')

    parser.add_argument('--ssl', type=str, default=None, help='ssl agent')
    parser.add_argument('--lambda_', type=float, default=0, help='if lploss')
    parser.add_argument('--no_cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Disable validation during training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=400,  # 800
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--lradjust', action='store_true',
                        default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--mixmode", action="store_true",
                        default=False, help="Enable CPU GPU mixing mode.")
    parser.add_argument("--warm_start", default="",
                        help="The model name to be loaded for warm start.")
    parser.add_argument('--debug', action='store_true',
                        default=False, help="Enable the detialed training output.")
    parser.add_argument('--dataset', default="cora", help="The data set")
    parser.add_argument('--datapath', default="data/", help="The data path.")
    parser.add_argument("--early_stopping", type=int,
                        default=0,
                        help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
    parser.add_argument("--no_tensorboard", default=True, help="Disable writing logs to tensorboard")

    # Model parameter
    # TODO CHANGE TYPE
    parser.add_argument('--type', default='resgcn',
                        help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
    parser.add_argument('--inputlayer', default='gcn',
                        help="The input layer of the model.")
    parser.add_argument('--outputlayer', default='gcn',
                        help="The output layer of the model.")
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--withbn', action='store_true', default=False,
                        help='Enable Bath Norm GCN')
    parser.add_argument('--withloop', action="store_true", default=False,
                        help="Enable loop layer GCN")
    parser.add_argument('--nhiddenlayer', type=int, default=1,
                        help='The number of hidden layers.')
    parser.add_argument("--normalization", default="AugNormAdj",
                        help="The normalization on the adj matrix.")
    parser.add_argument("--sampling_percent", type=float, default=1.0,
                        help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
    # parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
    parser.add_argument("--nbaseblocklayer", type=int, default=1,
                        help="The number of layers in each baseblock")
    parser.add_argument("--aggrmethod", default="default",
                        help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
    parser.add_argument("--task_type", default="full",
                        help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

    args = parser.parse_args()
    return args


def pre_setting(args):
    # pre setting
    if args.ssl == 'AttributeMask':
        args.pca = 1

    if args.debug:
        print(args)

    if args.lambda_ != 0 and args.ssl != 'Base' and not args.param_searching:
        from configs import lambda_config
        args.lambda_ = lambda_config[args.ssl][args.dataset]

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
    if args.aggrmethod == "default":
        if args.type == "resgcn":
            args.aggrmethod = "add"
        else:
            args.aggrmethod = "concat"

    if args.fastmode and args.early_stopping > 0:
        args.early_stopping = 0
        print("In the fast mode, early_stopping is not valid option. Setting early_stopping = 0.")
    if args.type == "mutigcn":
        print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
        args.nhiddenlayer = 1
        args.aggrmethod = "nores"

    # random seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda or args.mixmode:
        torch.cuda.manual_seed(args.seed)


def set_sampler(args) -> Sampler:
    return Sampler(args.dataset, args, args.datapath, args.task_type)


def set_model(args, sampler, path=""):
    nfeat = sampler.nfeat
    nclass = sampler.nclass
    print("nclass: %d\tnfeat:%d" % (nclass, nfeat))
    model = GCNModel(nfeat=nfeat,
                     nhid=args.hidden,
                     nclass=nclass,
                     nhidlayer=args.nhiddenlayer,
                     dropout=args.dropout,
                     baseblock=args.type,
                     inputlayer=args.inputlayer,
                     outputlayer=args.outputlayer,
                     nbaselayer=args.nbaseblocklayer,
                     activation=F.relu,
                     withbn=args.withbn,
                     withloop=args.withloop,
                     aggrmethod=args.aggrmethod,
                     mixmode=args.mixmode)
    if path != "":
        # load existing model
        print("Loading existing model: ", path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def set_optimizer_scheduler(model, args, sampler):
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=1)

    # convert to cuda
    if args.cuda:
        model.cuda()
    if args.ssl is None or args.lambda_ == 0:
        args.ssl = 'Base'

    # For the mix mode, lables and indexes are in cuda.
    labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
    if args.cuda or args.mixmode:
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    if args.warm_start is not None and args.warm_start != "":
        early_stopping = EarlyStopping(fname=args.warm_start, verbose=False)
        print("Restore checkpoint from %s" % (early_stopping.fname))
        model.load_state_dict(early_stopping.load_checkpoint())

    return optimizer, scheduler


def set_early_stopping(args):
    # set early_stopping
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, verbose=False)
        print("Model is saving to: %s" % (early_stopping.fname))

    tb_writer = None
    if not args.no_tensorboard:
        dirpath = f"./runs"
        if osp.exists(dirpath):
            shutil.rmtree(dirpath)
        tb_writer = SummaryWriter(logdir=dirpath)

    return early_stopping, tb_writer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_epoch(args, model, sampler, labels, early_stopping, scheduler, optimizer, ssl_agent,
                epoch, train_adj, train_fea, idx_train, idx_val, val_adj=None, val_fea=None):
    if val_adj is None:
        val_adj = train_adj
        val_fea = train_fea

    t = time.time()
    model.train()
    optimizer.zero_grad()

    if 'ContextLabel' in args.ssl:
        output, embeddings = model.myforward(train_fea, train_adj, layer=1)
    else:
        output, embeddings = model.myforward(train_fea, train_adj, layer=1.5)

    # special for reddit
    if sampler.learning_type == "inductive":
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_ssl = args.lambda_ * ssl_agent.make_loss(embeddings)
    loss_total = loss_train + loss_ssl

    loss_total.backward()
    optimizer.step()
    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the reddit dataset.
    # if args.early_stopping > 0 and sampler.dataset != "reddit":
    #     loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
    #     early_stopping(loss_val, model)

    if not args.fastmode and args.early_stopping > 0:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        model.eval()
        output = model(val_fea, val_adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        early_stopping(acc_val, model)
        # early_stopping(-loss_val, model)
    else:
        loss_val = 0
        acc_val = 0

    if args.lradjust:
        scheduler.step()

    val_t = time.time() - val_t

    try:
        return loss_train.item(), acc_train.item(), loss_val, acc_val, loss_ssl.item(), loss_total.item(), train_t
    except:
        return loss_train.item(), acc_train.item(), loss_val, acc_val, loss_ssl, loss_total.item(), train_t


def test(args, model, test_adj, test_fea, labels, idx_test):
    model.eval()
    # output = model(test_fea, test_adj)
    output, embeddings = model.myforward(test_fea, test_adj)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])

    if args.pca:
        args.ssl += '~PCA'

    if args.write_res:
        probs = torch.exp(output)
        np.save(f'preds/{args.dataset}_{args.seed}_pred.npy', probs.detach().cpu().numpy())

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "auc= {:.4f}".format(auc_test),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))
    return (loss_test.item(), acc_test.item())


def get_agent_by_task(args, sampler, model, optimizer):
    labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
    ssl_agent = None
    nclass = max(labels).item() + 1

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.ssl is None or args.lambda_ == 0 or args.ssl == 'Base':
        ssl_agent = Base(sampler.adj, sampler.features, device=device)
        args.ssl = 'Base'

    if args.ssl == 'EdgeMask':
        ssl_agent = EdgeMask(sampler.adj, sampler.features, device=device, nhid=args.hidden)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    # # TODO
    # if args.ssl == 'EdgeMelt':
    #     ssl_agent = EdgeMelt(sampler.adj, sampler.features, device='cuda')

    if args.ssl == 'DistanceCluster':
        ssl_agent = DistanceCluster(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden, device=device,
                                    args=args)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    # TODO
    # if args.ssl == 'SpectralPredictor':
    #     ssl_agent = SpectralPredictor(sampler.adj, sampler.features, nhid=args.hidden, args=args, device='cuda')
    #     optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
    #                            lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'PairwiseDistance':
        ssl_agent = PairwiseDistance(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden,
                                     device=device)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'PairwiseAttrSim':
        ssl_agent = PairwiseAttrSim(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden, args=args,
                                    device=device)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'Distance2Labeled':
        ssl_agent = Distance2Labeled(sampler.adj, sampler.features, sampler.labels, nclass=nclass, idx_train=idx_train,
                                     nhid=args.hidden, device=device)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'ICAContextLabel':
        ssl_agent = ICAContextLabel(sampler.adj, sampler.features, sampler.labels, nclass=nclass, idx_train=idx_train,
                                    nhid=args.hidden, device=device, args=args)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'LPContextLabel':
        ssl_agent = LPContextLabel(sampler.adj, sampler.features, sampler.labels, nclass=nclass, idx_train=idx_train,
                                   nhid=args.hidden, device=device, args=args)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'CombinedContextLabel':
        ssl_agent = CombinedContextLabel(sampler.adj, sampler.features, sampler.labels, nclass=nclass,
                                         idx_train=idx_train, nhid=args.hidden, device=device, args=args)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'AttributeMask':
        ssl_agent = AttributeMask(sampler.adj, sampler.features, idx_train=idx_train, nhid=args.hidden, device=device)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    if args.ssl == 'NodeProperty':
        ssl_agent = NodeProperty(sampler.adj, sampler.features, nhid=args.hidden, device=device)
        optimizer = optim.Adam(list(model.parameters()) + list(ssl_agent.linear.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

    return ssl_agent, optimizer


def train(args, ssl_agent, model, optimizer, sampler, scheduler, early_stopping, tb_writer):
    # Train model
    t_total = time.time()
    loss_train = np.zeros((args.epochs,))
    acc_train = np.zeros((args.epochs,))
    loss_val = np.zeros((args.epochs,))
    acc_val = np.zeros((args.epochs,))
    loss_ssl = np.zeros((args.epochs,))
    labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)

    sampling_t = 0
    # dataset = dgl.data.CoraGraphDataset()
    # graph = dataset[0]
    for epoch in range(args.epochs):
        if args.alpha != 0:
            ssl_agent.label_correction = True

        input_idx_train = idx_train
        sampling_t = time.time()
        # no sampling
        # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.

        # train_adj_2 = graph.adj()
        # train_fea_2 = graph.ndata['feat']

        train_adj, train_fea = ssl_agent.transform_data()
        # for loading the dgl dataset
        # train_adj, train_fea = train_adj_2,train_fea_2

        # (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization, cuda=args.cuda)
        if args.mixmode:
            train_adj = train_adj.cuda()

        sampling_t = time.time() - sampling_t

        # The validation set is controlled by idx_val
        # if sampler.learning_type == "transductive":
        # if False:
        #     outputs = train_epoch(args, model, sampler, labels, early_stopping, scheduler, optimizer, ssl_agent,
        #                           epoch, train_adj, train_fea, input_idx_train, idx_val)
        # else:
        (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        # val_adj, val_fea = ssl_agent.transform_data()

        if args.mixmode:
            val_adj = val_adj.cuda()
        outputs = train_epoch(args, model, sampler, labels, early_stopping, scheduler, optimizer, ssl_agent,
                              epoch, train_adj, train_fea, input_idx_train, idx_val, val_adj, val_fea)

        if args.debug and epoch % 1 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(outputs[0]),
                  'acc_train: {:.4f}'.format(outputs[1]),
                  'loss_val: {:.4f}'.format(outputs[2]),
                  'acc_val: {:.4f}'.format(outputs[3]),
                  'loss_ssl: {:.4f}'.format(outputs[4]),
                  'loss_total: {:.4f}'.format(outputs[5]),
                  't_time: {:.4f}s'.format(outputs[6]))

        if args.no_tensorboard is False:
            tb_writer.add_scalars('Loss',
                                  {'class': outputs[0], 'ssl': outputs[4], 'total': outputs[5], 'val': outputs[2]},
                                  epoch)
            tb_writer.add_scalars('Accuracy', {'train': outputs[1], 'val': outputs[3]}, epoch)

        if args.early_stopping > 0 and early_stopping.early_stop:
            print("Early stopping.")
            model.load_state_dict(early_stopping.load_checkpoint())
            break

        loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch], loss_ssl[epoch] = outputs[0], outputs[1], \
                                                                                                outputs[2], outputs[3], \
                                                                                                outputs[4]

    if args.early_stopping > 0:
        model.load_state_dict(early_stopping.load_checkpoint())
        # print('=== best score: %s, epoch %s ===' % (early_stopping.best_score, early_stopping.best_epoch))
        print('=== best score: %s, loss_val: %s, epoch %s ===' % (
            early_stopping.best_score, loss_val[early_stopping.best_epoch], early_stopping.best_epoch))
        print('For this epoch, val loss: %s, val acc: %s' % (
            loss_val[early_stopping.best_epoch], acc_val[early_stopping.best_epoch]))

    if args.debug:
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        x = list(range(1, len(loss_train) + 1))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(x, loss_train)

        title = 'ssl loss, task={}, epochs={}'.format(args.ssl, str(args.epochs))
        ax.set(xlabel='ephocs', ylabel='loss',
               title=title)
        ax.grid()
        plt.show()
        plt.clf()
        print(loss_train)

    return loss_train, acc_train, loss_val, acc_val, loss_ssl


def test_model(args, model, sampler, tb_writer, loss_train, loss_val, acc_train, acc_val):
    (test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)

    if args.mixmode:
        test_adj = test_adj.cuda()

    (loss_test, acc_test) = test(args, model, test_adj, test_fea, labels, idx_test)
    print("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (
        loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test))
    print('Self-Supervised Type: %s' % args.ssl)

    print(args)
    nnodes = sampler.adj.shape[0]
    print('len(idx_train)/len(adj.shape[0])= ', len(idx_train) / nnodes)

    if not args.no_tensorboard:
        tb_writer.close()

    return loss_test, acc_test


def check_one_task_per_dataset(args, datasets):
    results = []
    for ds in datasets:
        args.dataset = ds
        print(args.dataset)
        model, ssl_agent, sampler, loss_test, acc_test = run_ssl_model(args, model_path="",
                                                                       save_path="", to_save=False)
        results.append([ds, loss_test, acc_test])

    for res in results:
        print("task={}, dataset={}, test loss={}, accuracy test={}".format(args.ssl, res[0], res[1], res[2]))


def save_models_for_best_params(args, datasets, tasks, save_path):
    results = []
    for ds in datasets:
        args.dataset = ds
        for task in tasks:
            args.ssl = task
            print("running on: ", args.dataset, args.ssl)
            model, ssl_agent, sampler, loss_test, acc_test = run_ssl_model(args, model_path="",
                                                                           save_path=save_path, to_save=True)
            results.append([task, ds, loss_test, acc_test])

    print(results)
    for res in results:
        print("task={}, dataset={}, test loss={}, accuracy test={}".format(res[0], res[1], res[2], res[3]))


def analyse_ssl_tasks(save_path=""):
    print("Analysing ssl tasks")
    # all_tasks = ["EdgeMask", "DistanceCluster", "PairwiseDistance", "PairwiseAttrSim",
    #          "Distance2Labeled", "ICAContextLabel", "LPContextLabel", "CombinedContextLabel",
    #          "AttributeMask", "NodeProperty"]

    best_tasks = ["EdgeMask", "PairwiseDistance", "PairwiseAttrSim",
                  "Distance2Labeled", "AttributeMask", "NodeProperty"]
    datasets = ['pubmed', 'cora', 'citeseer']

    args = get_arguments()
    # check_one_task_per_dataset(args, datasets)
    save_models_for_best_params(args, datasets, best_tasks, save_path)


def load_args(path):
    # load model arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args_file = path.split('.pt')[0] + "_cmd_args.txt"
    with open(args_file, 'r') as f:
        args.__dict__ = json.load(f)
    print("dataset: ", args.dataset, ". ssl task: ", args.ssl)
    return args


def save_model(base_path, args, model, optimizer, loss_train):
    print("saving model:", args.ssl, "on dataset: ", args.dataset)
    # save model
    save_path = base_path + args.ssl + "_" + args.dataset
    model_path = save_path + ".pt"
    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': loss_train},
               model_path)
    # save arguments
    args_file_name = save_path + "_cmd_args.txt"
    with open(args_file_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def run_ssl_model(args, model_path, save_path, to_save):
    pre_setting(args)
    sampler = set_sampler(args)

    early_stopping, tb_writer = set_early_stopping(args)

    model = set_model(args, sampler, path=model_path)
    optimizer, scheduler = set_optimizer_scheduler(model, args, sampler)

    ssl_agent, optimizer = get_agent_by_task(args, sampler, model, optimizer)
    if model_path == "":  # the model needs to be trained and was not loaded
        loss_train, acc_train, loss_val, acc_val, loss_ssl = train(args, ssl_agent, model,
                                                                   optimizer, sampler, scheduler,
                                                                   early_stopping, tb_writer)
        loss_test, acc_test = test_model(args, model, sampler, tb_writer, loss_train, loss_val, acc_train, acc_val)
        if to_save:
            save_model(save_path, args, model, optimizer, loss_train)
    else:
        loss_test, acc_test = 0, 0  # load from a log file
    return model, ssl_agent, sampler, loss_test, acc_test


def get_pre_trained_model(label, model_path="", save_path="", to_save=True):
    if model_path == "":
        args = get_arguments()
    else:
        args = load_args(model_path)
        args.ssl = args.ssl.split("~")[0]
    model, ssl_agent, sampler, loss_test, acc_test = run_ssl_model(args, model_path, save_path, to_save)
    data = ssl_utils.GraphData(sampler, label, args.cuda)
    return ssl_utils.PretrainedModel(model, data, ssl_agent, args.ssl)

