import torch
import torchvision
import torchvision.transforms as transforms
import ssl
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pathlib
import time
import os
import random
import json
ssl._create_default_https_context = ssl._create_unverified_context
device = "cuda"

#python worker.py --random 0 --max_rank 1000
#python worker.py --random 0 --max_rank 900
#python worker.py --random 0 --max_rank 800
#python worker.py --random 0 --max_rank 700
#python worker.py --random 0 --max_rank 600

#python worker.py --random 0 --max_rank 500
#python worker.py --random 0 --max_rank 400
#python worker.py --random 0 --max_rank 300
#python worker.py --random 0 --max_rank 200
#python worker.py --random 0 --max_rank 100

parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--n_layers', type=int,
                    help='dim_recurrent', default=12)
parser.add_argument('--m_per_layer', type=int,
                    help='index of this trial', default=1000)
parser.add_argument('--max_rank', type=int,
                    help='init random to', default=-1)
parser.add_argument("--verbose", action="store_true")
parser.add_argument('--random', type=str,
                    help='init random to', default="X")
parser.add_argument('--n_epochs', type=int,
                    help='dim_recurrent', default=50)
parser.add_argument('--batch_size', type=int,
                    help='dim_recurrent', default=128)
parser.add_argument('--lr', type=float,
                    help='dim_recurrent', default=0.005)
parser.add_argument("--just_net", action="store_true")
parser.add_argument("--no_fancy_init", action="store_true")

parser.add_argument('--reglam', type=float,
                    help='regularization lambda?', default=0)
parser.add_argument('--regnorm', type=int,
                    help='regulatization norm to use?', default=0)
parser.add_argument("--sqloss", action="store_true")

args = parser.parse_args()
init_random = abs(hash(args.random)) % 10**8
n_layers = args.n_layers
m_per_layer = args.m_per_layer
max_rank = args.max_rank if args.max_rank>0 else None
n_epochs = args.n_epochs
batch_size = args.batch_size
lr = args.lr
just_net = args.just_net
no_fancy_init = args.no_fancy_init
reg_lam = args.reglam
reg_norm = args.regnorm
square_loss = args.sqloss

random.seed(init_random)
torch.manual_seed(init_random)
np.random.seed(init_random)


class Net(nn.Module):
    def __init__(self, N_layers, M_per_layer, max_rank=None, activation=F.relu):
        super().__init__()
        if max_rank is None: max_rank = M_per_layer

        self.layers_a = []
        for i in range(N_layers):
            in_n = 32 ** 2 * 3 if i == 0 else M_per_layer
            out_n = 10 if i == N_layers - 1 else M_per_layer
            self.layers_a.append(nn.Linear(in_n, out_n, bias=False))
            with torch.no_grad():
                Wahh = self.layers_a[i].weight
                Wahh = np.random.randn(Wahh.shape[0], Wahh.shape[1])
                u, s, vT = np.linalg.svd(Wahh, full_matrices=False)  # np.linalg.svd returns v transpose!
                if not no_fancy_init:
                    Wahh = u @ vT  # make the eigenvalues large so they decay slowly
                Wahh = torch.tensor(Wahh, dtype=torch.float32)
                self.layers_a[i].weight[:, :] = Wahh[:, :]
        self.layers_a = nn.ModuleList(self.layers_a)
        self.activation = F.relu
        self.N_layers = N_layers

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i in range(self.N_layers):
            x = self.layers_a[i](x)
            if i < self.N_layers - 1:
                x = self.activation(x)
        return x


class Net_Rank(nn.Module):
    def __init__(self, N_layers, M_per_layer, max_rank=None, activation=F.relu):
        super().__init__()
        if max_rank is None: max_rank = M_per_layer

        self.layers_a = []
        self.layers_b = []
        for i in range(N_layers):
            in_n = 32 ** 2 * 3 if i == 0 else M_per_layer
            out_n = 10 if i == N_layers - 1 else M_per_layer
            self.layers_a.append(nn.Linear(in_n, max_rank, bias=False))
            self.layers_b.append(nn.Linear(max_rank, out_n, bias=False))
            with torch.no_grad():
                Wahh = self.layers_a[i].weight
                Wahh = np.random.randn(Wahh.shape[0], Wahh.shape[1]) / (max(Wahh.shape[0], Wahh.shape[1])**0.5)
                u, s, vT = np.linalg.svd(Wahh, full_matrices=False)  # np.linalg.svd returns v transpose!
                if not no_fancy_init:
                    Wahh = u @ vT  # make the eigenvalues large so they decay slowly
                Wahh = torch.tensor(Wahh, dtype=torch.float32)
                self.layers_a[i].weight[:, :] = Wahh[:, :]
            with torch.no_grad():
                Wahh = self.layers_b[i].weight
                Wahh = np.random.randn(Wahh.shape[0], Wahh.shape[1]) / (max(Wahh.shape[0], Wahh.shape[1])**0.5)
                u, s, vT = np.linalg.svd(Wahh, full_matrices=False)  # np.linalg.svd returns v transpose!
                if not no_fancy_init:
                    Wahh = u @ vT  # make the eigenvalues large so they decay slowly
                Wahh = torch.tensor(Wahh, dtype=torch.float32)
                self.layers_b[i].weight[:, :] = Wahh[:, :]
        self.layers_a = nn.ModuleList(self.layers_a)
        self.layers_b = nn.ModuleList(self.layers_b)
        self.activation = F.relu
        self.N_layers = N_layers

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i in range(self.N_layers):
            x = self.layers_a[i](x)
            x = self.layers_b[i](x)
            if i < self.N_layers - 1:
                x = self.activation(x)
        return x


def accuracy(net, test=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader if test else trainloader_acc:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
trainloader_acc = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

_path = pathlib.Path(f"results/megabatch_tuningdata.pt")
_path.parent.mkdir(parents=True, exist_ok=True)
filename = f"r{max_rank}_i{args.random}_n{n_layers}_m{m_per_layer}_e{n_epochs}_b{batch_size}_lr{lr}"

if just_net:
    net = Net(N_layers=n_layers, M_per_layer=m_per_layer)
else:
    net = Net_Rank(N_layers=n_layers, M_per_layer=m_per_layer, max_rank=max_rank)
if square_loss:
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)#, momentum=0.8)

test_acc = [accuracy(net, test=True)]
train_acc = [accuracy(net)]
losses = []
losses_nr = []
start_time = time.time()
min_running_loss = 1e10
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_loss_nr = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        error = 0
        if reg_norm == 1:
            for param in net.parameters():
                if param.requires_grad is True:
                    error += reg_lam * torch.sum(torch.abs(param))
        if reg_norm == 2:
            for param in net.parameters():
                if param.requires_grad is True:
                    error += reg_lam * torch.sum(param ** 2)
        running_loss_nr += loss.item()
        loss += error
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.10f}, nr {running_loss_nr / 20:.10f}')
            losses.append(running_loss)
            losses_nr.append(running_loss_nr)
            #print(accuracy(net))
            if running_loss_nr < min_running_loss:
                min_running_loss = running_loss_nr
                torch.save({'model_state_dict': net.state_dict()}, f"results/{filename}_best.pth")

            running_loss = 0.0
            running_loss_nr = 0.0


    train_acc.append(accuracy(net))
    test_acc.append(accuracy(net, test=True))
    print("============")
    print(train_acc[-1], test_acc[-1])

print('Finished Training')

end_time = time.time()


result = {
    "start_time": start_time,
    "end_time": end_time,
    "random": args.random,
    "n_layers": n_layers,
    "m_per_layer": m_per_layer,
    "max_rank": max_rank,
    "n_epochs": n_epochs,
    "batch_size": batch_size,
    "lr": lr,
    "test_acc": test_acc,
    "train_acc": train_acc,
    "losses": losses,
    "losses_nr": losses_nr,
    "reg_norm": reg_norm,
    "reg_lam": reg_lam
}

if just_net:
    filename += "_jn"
if no_fancy_init:
    filename += "_nfi"
if reg_norm>0:
    filename += f"_rn{reg_norm}_rl{reg_lam}"

with open(f"results/{filename}.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

torch.save({'model_state_dict': net.state_dict()}, f"results/{filename}.pth")




