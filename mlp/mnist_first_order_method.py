from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from nn_models import *
import logging, os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer-type', type=str, default='sgd',
                    help='optimizer type')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--decay-epoch', type=int, nargs='+', default=[299, 599, 899], 
                    help='learning rate decay epoch')
parser.add_argument('--hidden', type=int, default=128, 
                    help='hidden size of the mlp')
parser.add_argument('--saving-folder',
                    type=str,
                    default='checkpoints/',
                    help='choose saving name')
                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.saving_folder == '':
    raise ('you must give a position and name to save your model')
if args.saving_folder[-1] != '/':
    args.saving_folder += '/'
if not os.path.isdir(args.saving_folder):
    os.makedirs(args.saving_folder)

log = f"{args.saving_folder}log.log"
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(filename=log, filemode='w', format=FORMAT, level=logging.DEBUG)
logging.debug(args)
# print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
transform_train = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transform_train),
    batch_size=args.batch_size, shuffle=True, **kwargs)

n = len(train_loader.dataset)
batch_size = train_loader.batch_size

criterion = nn.CrossEntropyLoss()
model = mlp(hidden=args.hidden)

if args.cuda:
    model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    
if args.optimizer_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
elif args.optimizer_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer_type == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
else:
    raise("We did not support else yet")

def closure():
    model.eval()
    loss = 0.0
    count = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        count = count + 1
    model.train()
    return loss / count

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

start_epoch = -1
num_props = 1

train_loss = closure()
logging.debug("[Epoch {}] No Prop: {}, Train loss: {}".format(0, num_props, train_loss))


for epoch in range(start_epoch+1, args.epochs):  # loop over the dataset multiple times
    if epoch in args.decay_epoch:
        optimizer.param_groups[0]['lr'] /= 10.0
    for i, (inputs, labels) in enumerate(train_loader):
        model.train()
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    num_props += 60000 * 2
    train_loss = closure()
    logging.debug("[Epoch {}] No Prop: {}, Train loss: {}".format(epoch, num_props, train_loss))
        
print('Finished Training')
