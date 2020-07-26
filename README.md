# Inexact_Newton_Method

This repository contains Python and Matlab code that produces all the experimental results in the paper: [Inexact non-convex Newton-type methods](https://arxiv.org/pdf/1802.06925.pdf).

Specifically, multilayer perceptron(MLP) networks and non-linear least squares(NLS) are the two non-convex problems considered.

## Usage

### MLP networks

#### Example: Mnist Classification
- <code>[mlp](./mlp)</code>: This folder contains all the source code for implementing the mnist classification task. 
- <code>[mlp/optim](.mlp/optim.py)</code> contains the implementation of (sub-sampled) TR algorithm.

Run the first order methods:
```
export CUDA_VISIBLE_DEVICES=0; export CUDA_VISIBLE_DEVICES=1; python mnist_first_order_method.py --optimizer-type sgd --lr 0.01 --hidden 128 --saving-folder checkpoints/128_sgd_0.01 

--optimizer-type: sgd, adagrad, adam
--lr: 0.1, 0.01, 0.001, 0.0001
--hidden: 1024, 128, 16
--saving-folder: set properly
```

Run the trust region mehods:
```
export CUDA_VISIBLE_DEVICES=0; python mnist_str.py --grad-size sub --grad-batch-size 5000 --hidden 128 --saving-folder checkpoints/128_sub

--grad-size: sub (means Inexact TR), full (means SubH TR)
--hidden: 1024, 128, 16
--saving-folder: set properly
```

The reuslts are saved in the --saving-folder/log.log

### NLS
- <code>[nls](./nls)</code>: This folder contains all the source code for implementing the binary linear classification task using square loss (which gives a non-linear square problem). 
- <code>[nls/algorithms](./nls/algorithms)</code> contains the implementation of (sub-sampled) TR, ARC algorithms for non-linear least squares.

#### Example: NLS on ijcnn1
```
Download 'ijcnn1' dataset from: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1
```
or run the command
```
bash download_ijcnn1.sh
```
In the Matlab Command Window, run
```
# this will generate the plots of all algorithms.
# check the details of the function for more options.
>> blc_demo('ijcnn1')
```


## References
- Zhewei Yao, Peng Xu, Farbod Roosta-Khorasani and Michael W. Mahoney, [Inexact non-convex Newton-type methods](https://arxiv.org/pdf/1802.06925.pdf), 2018.

