from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from optm import *
from nn_models import *
import logging, os
from copy import deepcopy

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--delta', type=float, default=1.0, metavar='D',
                    help='delta0 (default: 1.0)')
parser.add_argument('--cg', type=int, default=250, metavar='CG',
                    help='maximum cg iterations (default: 250)')
parser.add_argument('--gamma1', type=float, default=2.0, metavar='G1',
                    help='gamma1 (default: 2.0)')
parser.add_argument('--rho1', type=float, default=0.8, metavar='R1',
                    help='rho1 (default: 0.8)')
parser.add_argument('--gamma2', type=float, default=1.2, metavar='G2',
                    help='gamma2 (default: 1.2)')
parser.add_argument('--rho2', type=float, default=1e-4, metavar='R2',
                    help='rho2 (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--grad-size', type=str, default='full',
                    help='use full gradient or not')
parser.add_argument('--grad-batch-size', type=int, default=5000,
                    help='data used to compute gradient')
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
logging.basicConfig(filename=log, filemode='a', format=FORMAT, level=logging.DEBUG)
logging.debug(args)
print(args)

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

trainset = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transform_train),
    batch_size=1000, shuffle=True, **kwargs)

train_full_data = torch.randn(60000, 1, 28, 28)
train_full_label = torch.zeros(60000).type(torch.LongTensor)

for i, (d, l) in enumerate(trainset):
    train_full_data[1000*i:1000*(i+1), :] = d
    train_full_label[1000*i:1000*(i+1)] = l

n = len(train_loader.dataset)
batch_size = train_loader.batch_size

criterion = nn.CrossEntropyLoss()

model = mlp(hidden=args.hidden)

if args.cuda:
    model.cuda()
    
optimizer = STR(model.parameters(), delta=args.delta, max_iters=args.cg, tol=1e-8, gamma1=args.gamma1, gamma2=args.gamma2, rho1=args.rho1, rho2=args.rho2)

def full_pass():
    model.eval()
    loss = 0.0
    count = 0.0
    grad_sample_list = []
    model.zero_grad()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        # forward + backward
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        batch_loss.backward()
        count = count + 1

    grads = [deepcopy(p.grad.data) / count for p in model.parameters()]
    model.train()
    return loss / count, grads, inputs, labels

def sub_pass(size=1000):
    model.eval()
    loss = 0.0
    count = 0.0
    A = np.arange(60000)
    np.random.shuffle(A)
    sample_list = A[:size]
    model.zero_grad()
    d = train_full_data[sample_list, :].cuda()
    l = train_full_label[sample_list].cuda()
    outputs = model(d)
    batch_loss = criterion(outputs, l)
    loss += batch_loss.item()
    # batch_loss.backward()
    # grads = [p.grad.data for p in model.parameters()]
    _grads = torch.autograd.grad(batch_loss, model.parameters())
    grads = [t.detach().data for t in _grads]
    # print(math.sqrt(group_product(grads, grads)))
    model.train()
    return loss, grads, d, l

def closure():
    model.eval()
    loss = 0
    count = 0.0
    for data in train_loader:
        inputs, labels = data
        if args.cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        # forward
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        count = count + 1
    model.train()
    return loss / count


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
piter = args.log_interval

start_epoch = -1
num_props = 1

grads_old_norm = 0

if args.grad_size == 'sub':
    max_size = int(60000)
    min_size = int(1000)
else:
    max_size = int(60000)
    min_size = int(60000)

train_loss = closure()
logging.debug("[Epoch {}] No Prop: {}, Train loss: {}".format(0, num_props, train_loss))

grads_sampling_size = int(args.grad_batch_size)

for epoch in range(start_epoch+1, args.epochs):  # loop over the dataset multiple times

    for i, data in enumerate(train_loader):
        model.train()
        
        # compute full or sub gradient
        if args.grad_size == 'full':
            num_props += 2 * n
        else:
            num_props += 2 * grads_sampling_size

        if args.grad_size == 'full':
            loss, grads, _data, _label = full_pass()
        else:
            loss, grads, _data, _label = sub_pass(size=grads_sampling_size)
        
        # compute hessian
        hess_data = _data[:1000].cuda()
        hess_label = _label[:1000].cuda()
        model.zero_grad()
        outputs = model(hess_data)
        batch_loss = criterion(outputs, hess_label)
        gradsH = torch.autograd.grad(batch_loss, model.parameters(), create_graph=True)

        gnorm = math.sqrt(group_product(grads, grads))
        print(f"For checking: {grads_sampling_size}")
        if grads_old_norm == 0:
            grads_old_norm = gnorm
        else:
            if gnorm > 1.2 * grads_old_norm:
                grads_old_norm = gnorm
                grads_sampling_size = int(max(min_size, grads_sampling_size / 1.2))
            elif gnorm > grads_old_norm/1.2:
                grads_old_norm = gnorm
                grads_sampling_size = int(min(max_size, grads_sampling_size * 1.2))
            else:
                grads_sampling_size = grads_sampling_size

        def Batch_Loss():
            model.eval()
            _outputs = model(_data)
            batch_loss = criterion(_outputs, _label)
            model.train()
            return batch_loss.cpu().item()

        if args.grad_size == 'full':
            train_loss, m, num_cg, num_feval = optimizer.step(gradsH, grads, closure, loss)
        else:
            train_loss, m, num_cg, num_feval = optimizer.step(gradsH, grads, Batch_Loss, loss)

        num_props += num_feval * grads_sampling_size + num_cg * 2 * 1000 # hessian size is 1000

        if i % piter == piter - 1:  
            
            train_loss = closure()
            logging.debug("[Epoch {}] No Prop: {}, Train loss: {}".format(0, num_props, train_loss))
    
        
print('Finished Training')
