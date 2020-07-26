# Inexact_Newton_Method

This repository contains Python and Matlab code that produces all the experimental results in the paper: [Inexact non-convex Newton-type methods](https://arxiv.org/pdf/1802.06925.pdf).

Specifically, multilayer perceptron(MLP) networks and non-linear least squares(NLS) are the two non-convex problems considered.

## Usage

### MLP networks

#### Example: Mnist Classification

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

