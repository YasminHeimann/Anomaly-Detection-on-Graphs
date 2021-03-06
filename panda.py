from copy import deepcopy

import pandas as pd
import torch
import torch.optim as optim
import wandb

from losses import CompactnessLoss, EWCLoss
import panda_utils


def run_epoch(model, optimizer, criterion, device, ewc, ewc_loss):
    optimizer.zero_grad()

    logits = model.predict_logits()

    loss = model.loss(criterion, logits, 'train')

    if ewc:
        loss += ewc_loss(model.model)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

    optimizer.step()

    return loss.item()


def get_optim(args, model):
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    return optimizer


def train_panda_model(data, model, device, args, ewc_loss, num_classes):
    model.eval()  # todo
    auc, train_feature_space = panda_utils.get_score(data, model, device, num_classes)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = get_optim(args, model)

    center = torch.FloatTensor(train_feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    run_loss, auroc = [], [auc]
    for epoch in range(args.epochs):
        model.train()
        running_loss = run_epoch(model, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        run_loss.append(running_loss)
        model.eval()
        auc, feature_space = panda_utils.get_score(data, model, device, num_classes)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        auroc.append(auc)
    return run_loss, auroc


def plot_results(args, dataset, tasks: list, auroc: list, run_loss: list, to_save=False,
                 title_add="", file_name_add="", loss_plot=False):
    import matplotlib.pyplot as plt
    # Data for plotting
    x = list(range(0, args.epochs + 1))

    fig, ax = plt.subplots()
    for auc, task in zip(auroc, tasks):
        ax.plot(x, auc, label=task.split("_")[0])
    plt.legend()
    title = 'PANDA AUROC, dataset={}, optim={}, {}'.format(dataset, args.optim, title_add)
    ax.set(xlabel='ephocs', ylabel='AUROC',
           title=title)
    ax.grid()
    auc_name = "{}_{}_{}_{}_l{}_{}_auc.png".format(args.base_path, args.pre_task,
                                                   dataset, args.optim, args.label, file_name_add)

    fig.savefig(auc_name)
    plt.show()

    if to_save:
        import json
        json_name = "{}_{}_{}_{}_l{}_{}_data.json".format(args.base_path, args.pre_task,
                                                       dataset, args.optim, args.label, file_name_add)
        data = {"tasks": tasks, "auc": auroc, "loss": run_loss}
        with open(json_name, 'w') as f:
            json.dump(data, f, indent=2)

    # if loss_plot:
    #     plt.clf()
    #     fig1, ax1 = plt.subplots()
    #     ax1.plot(x, run_loss)
    #
    #     ax1.set(xlabel='ephocs', ylabel='AUROC',
    #             title='GNN+PANDA Training Loss')
    #     ax1.grid()
    #     loss_name = "{}_{}_{}_loss.png".format(args.base_path, dataset, args.optim)
    #     fig1.savefig(loss_name)
    #     plt.show()


def run_labels_analysis(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    for ds in args.datasets:
        if args.pre_task == 'dgl':
            run_basic_model(args, device, ds)
            return
        labels, panda_auc, panda_loss = [], [], []
        ssl = args.best_model_per_task[ds]
        ssl = "{}_{}".format(ssl, ds)
        model = panda_utils.get_pretrained_model(args, args.base_path,
                                                 ssl_model=ssl,
                                                 save_model=args.save_model)

        for label in range(model.data.num_classes):
            args.label = label
            model = panda_utils.get_pretrained_model(args, args.base_path,
                                                     ssl_model=ssl,
                                                     save_model=args.save_model)
            model.to(device)
            run_loss, auroc = train_panda_model(model.data, model, device, args, None, model.data.num_classes)
            labels.append(str(label)), panda_auc.append(auroc), panda_loss.append(run_loss)
        title = "task={}".format(ssl.split("_")[0])
        file_name_add = "labels_test"
        avg_auc = sum([auc[-1] for auc in panda_auc]) / float(model.data.num_classes)
        print("\n==avg auc on all labels: ", avg_auc, "==")
        plot_results(args, ds, labels, panda_auc, panda_loss, to_save=True,
                     title_add=title, file_name_add=file_name_add)


def run_basic_model(args, device, ds):
    model = panda_utils.get_pretrained_model(args, args.base_path)
    model.to(device)
    run_loss, auroc = train_panda_model(model.data, model, device, args, None, model.data.num_classes)
    plot_results(args, ds, ['dgl'], [auroc], [run_loss], to_save=False)


def run_panda_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    for ds in args.datasets:
        if args.pre_task == 'dgl':
            run_basic_model(args, device, ds)
            return
        tasks, panda_auc, panda_loss = [], [], []
        for ssl in args.ssl_models:
            ssl = "{}_{}".format(ssl, ds)
            model = panda_utils.get_pretrained_model(args, args.base_path,
                                                     ssl_model=ssl,
                                                     save_model=args.save_model)
            model.to(device)

            ewc_loss = None
            #Freezing Pre-trained model for EWC
            if args.ewc:
                frozen_model = deepcopy(model.model).to(device)
                frozen_model.eval()
                panda_utils.freeze_model(frozen_model)
                fisher = torch.load(args.diag_path, map_location=torch.device('cpu'))
                ewc_loss = EWCLoss(frozen_model, fisher)

            run_loss, auroc = train_panda_model(model.data, model, device, args, ewc_loss, model.data.num_classes)
            tasks.append(ssl), panda_auc.append(auroc), panda_loss.append(run_loss)
        title_add = "label={}".format(str(args.label))
        print("models: ", args.ssl_models)
        print("auroc: ", [auc[-1] for auc in panda_auc])
        plot_results(args, ds, tasks, panda_auc, panda_loss, to_save=True, title_add=title_add)


def main(args):
    if args.analyse_ssl:
        import SSL_pretask.src.train_ssl as ssl_task
        ssl_task.analyse_ssl_tasks(args.base_path)
    elif args.test_label:
        run_labels_analysis(args)
    else:
        run_panda_model(args)


class PandaArguments:
    # panda parameters
    epochs = 125
    lr = 1e-1
    ewc = False
    diag_path = './data/fisher_diagonal.pth'
    optim = 'sgd'  # supports: adam, sgd
    label = 2
    best_model_per_task = {'cora': 'EdgeMask'}
    # supports: {'cora': 'EdgeMask', 'citeseer': 'NodeProperty', 'pubmed': 'PairwiseDistance'}

    # pre-task params: change to your needs
    pre_task = 'ssl'  # change between: ssl / dgl
    datasets = ['cora']  # supports: ['cora', 'citeseer', 'pubmed']
    ssl_models = ["EdgeMask"]
    # supports: ["PairwiseDistance", "EdgeMask", "PairwiseAttrSim",
    # "Distance2Labeled", "AttributeMask", "NodeProperty"]

    # saving models / analysis modes
    save_model = False
    base_path = "./pre_trained_models/100_epochs/model_"
    analyse_ssl = False
    test_label = False


def get_arguments():
    # edit the csv file with arguments
    # args explanations can be found in the README file
    args = PandaArguments()
    data = pd.read_csv("./args.csv")
    ds = ['cora', 'pubmed', 'citeseer']
    if data['dataset'][0] not in ds:
        print("Available datasets: cora, pubmed or citeseer")
        exit(1)
    else:
        args.datasets = [data['dataset'][0]]
    task = data['task'][0]
    avail_tasks = ["PairwiseDistance", "EdgeMask", "PairwiseAttrSim",
                    "Distance2Labeled", "AttributeMask", "NodeProperty", "best"]
    if task not in avail_tasks:
        print("Unknown task: ", task)
        print("Available tasks: PairwiseDistance, EdgeMask, PairwiseAttrSim, Distance2Labeled, AttributeMask, NodeProperty")
        print("For the best task enter: best")
        exit(1)
    best_models = {'cora': 'EdgeMask', 'citeseer': 'NodeProperty', 'pubmed': 'PairwiseDistance'}
    if task == 'best':
        best_task = best_models[args.datasets[0]]
        args.ssl_models = [best_task]
    else:
        args.ssl_models = [task]
    args.best_model_per_task = {args.datasets[0]:
                                args.ssl_models[0]}
    args.label = data['label'][0]
    args.epochs = data['epochs'][0]
    args.lr = data['lr'][0]
    return args


if __name__ == "__main__":
    panda_args = get_arguments()
    print("Running PANDA on task={}, dataset={}, epoch={}, lr={}".format(
        panda_args.ssl_models[0], panda_args.datasets[0], panda_args.epochs, panda_args.lr))
    main(panda_args)

