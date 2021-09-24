import torch
import torch.optim as optim
from losses import CompactnessLoss, EWCLoss
import panda_utils


def run_epoch(model, optimizer, criterion, device, ewc, ewc_loss):
    # todo: images = imgs.to(device)
    optimizer.zero_grad()

    logits = model.predict_logits()

    loss = model.loss(criterion, logits, 'train')

    if ewc:
        loss += ewc_loss(model.model)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

    optimizer.step()

    return loss.item()


def train_panda_model(data, model, device, args, ewc_loss, num_classes):
    model.eval()  # todo
    auc, train_feature_space = panda_utils.get_score(data, model, device, num_classes)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    # todo
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    center = torch.FloatTensor(train_feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    run_loss, auroc = [], []
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


def plot_results(args, task, auroc, run_loss):
    import matplotlib.pyplot as plt
    # Data for plotting
    x = list(range(1, args.epochs + 1))

    fig, ax = plt.subplots()
    ax.plot(x, auroc)

    ax.set(xlabel='ephocs', ylabel='AUROC',
           title='GNN+PANDA AUROC')
    ax.grid()

    auc_name = args.base_path + task + "_" + "auc.png"
    fig.savefig(auc_name)
    plt.show()
    plt.clf()

    fig1, ax1 = plt.subplots()
    ax1.plot(x, run_loss)

    ax1.set(xlabel='ephocs', ylabel='AUROC',
            title='GNN+PANDA Training Loss')
    ax1.grid()
    loss_name = args.base_path + task + "_" + "loss_curve.png"
    fig1.savefig(loss_name)
    plt.show()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = panda_utils.get_pretrained_model(args, args.base_path, save_model=args.save_model)
    model.to(device)  # todo, should it be model = model.to(device)

    ewc_loss = None
    # Freezing Pre-trained model for EWC todo
    # if args.ewc:
    #     frozen_model = deepcopy(model).to(device)
    #     frozen_model.eval()
    #     panda_utils.freeze_model(frozen_model)
    #     fisher = torch.load(args.diag_path)
    #     ewc_loss = EWCLoss(frozen_model, fisher)

    # panda_utils.freeze_parameters(model)  # todo - what does it mean here?

    run_loss, auroc = train_panda_model(model.data, model, device, args, ewc_loss, 2) #model.data.num_classes)
    plot_results(args, model.task, auroc, run_loss)


class PandaArguments:
    epochs = 50
    lr = 1e-2
    ewc = False
    pre_task = 'ssl'  # ssl/ dgl # todo
    label = 0
    save_model = False  # todo
    base_path = "./pre_trained_models/400_epochs/model_"
    # dataset = 'cora'
    # model = 'gcn'
    # task = 'node_class'
    # layers = 1
    # h_dim1 = 16
    # label = 0


if __name__ == "__main__":
    panda_args = PandaArguments()
    main(panda_args)

